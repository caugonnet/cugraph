/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "experimental/stf/segment_launchers.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/prims/detail/sample_and_compute_local_nbr_indices.cuh>
#include <cugraph/prims/detail/transform_v_frontier_e.cuh>
#include <cugraph/prims/transform_reduce_e.cuh>
#include <cugraph/sampling_functions.hpp>
#include <cugraph/utilities/dataframe_buffer.hpp>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/bit>
#include <cuda/std/optional>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string_view>
#include <vector>

namespace {

using vertex_t = int32_t;
using edge_t   = int32_t;
using result_t = uint64_t;

std::size_t iteration_count()
{
  auto const* value = std::getenv("CUGRAPH_STF_ITERATIONS");
  if (value == nullptr) { return 9; }
  auto parsed = std::strtoull(value, nullptr, 10);
  return parsed == 0 ? 9 : static_cast<std::size_t>(parsed);
}

struct timing {
  double median_ms{};
  double min_ms{};
  result_t result{};
};

template <typename Function>
timing measure(std::size_t iterations, Function&& function)
{
  auto result = function();

  std::vector<double> milliseconds{};
  milliseconds.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    auto const start = std::chrono::steady_clock::now();
    result           = function();
    auto const stop  = std::chrono::steady_clock::now();
    milliseconds.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }

  auto sorted = milliseconds;
  std::sort(sorted.begin(), sorted.end());
  return timing{sorted[sorted.size() / 2], sorted.front(), result};
}

struct real_edge_op {
  template <typename Src, typename Dst, typename SrcValue, typename DstValue>
  __device__ result_t operator()(Src src, Dst dst, SrcValue, DstValue, float edge_value) const
  {
    auto src_bits = static_cast<uint32_t>(src);
    auto dst_bits = static_cast<uint32_t>(dst);
    return static_cast<result_t>(cuda::std::bit_cast<uint32_t>(edge_value) ^
                                 (src_bits * uint32_t{0x9e3779b9}) ^
                                 (dst_bits * uint32_t{0x85ebca6b}));
  }
};

struct realistic_bias_op {
  template <typename Src, typename Dst, typename SrcValue, typename DstValue>
  __device__ float operator()(Src src, Dst dst, SrcValue, DstValue, float edge_value) const
  {
    auto endpoint_perturbation =
      static_cast<float>((static_cast<uint32_t>(src) ^ static_cast<uint32_t>(dst)) & 0xff) *
      0x1.0p-24f;
    return edge_value + endpoint_perturbation + 0x1.0p-24f;
  }
};

template <typename GraphViewType, typename EdgeValueInputWrapper, typename SegmentLauncher>
result_t run_with_launcher(raft::handle_t const& handle,
                           GraphViewType const& graph_view,
                           EdgeValueInputWrapper edge_value_input,
                           SegmentLauncher& launcher)
{
  return cugraph::detail::transform_reduce_e_with_segment_launcher(
    handle,
    graph_view,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    edge_value_input,
    real_edge_op{},
    result_t{0},
    launcher);
}

template <typename GraphViewType, typename EdgeValueInputWrapper>
result_t run_baseline(raft::handle_t const& handle,
                      GraphViewType const& graph_view,
                      EdgeValueInputWrapper edge_value_input)
{
  return cugraph::transform_reduce_e(handle,
                                     graph_view,
                                     cugraph::edge_src_dummy_property_t{}.view(),
                                     cugraph::edge_dst_dummy_property_t{}.view(),
                                     edge_value_input,
                                     real_edge_op{},
                                     result_t{0});
}

void print(std::string_view arm, timing const& value)
{
  std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
            << " primitive=transform_reduce_e"
            << " arm=" << arm << " median_ms=" << value.median_ms << " min_ms=" << value.min_ms
            << " result=" << value.result << '\n';
}

struct frontier_timing {
  double median_ms{};
  double min_ms{};
  std::size_t output_size{};
  result_t output_sum{};
  std::size_t final_offset{};
};

template <typename Function>
frontier_timing measure_frontier(raft::handle_t const& handle,
                                 std::size_t iterations,
                                 Function&& function)
{
  {
    auto output = function();
    handle.sync_stream();
  }

  std::vector<double> milliseconds{};
  milliseconds.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    auto const start = std::chrono::steady_clock::now();
    auto output      = function();
    handle.sync_stream();
    auto const stop = std::chrono::steady_clock::now();
    milliseconds.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }

  auto output = function();
  handle.sync_stream();
  auto& values            = std::get<0>(output);
  auto& offsets           = std::get<1>(output);
  auto const sum          = thrust::reduce(rmm::exec_policy(handle.get_stream()),
                                  cugraph::get_dataframe_buffer_begin(values),
                                  cugraph::get_dataframe_buffer_end(values),
                                  result_t{0});
  auto const final_offset = offsets.back_element(handle.get_stream());

  auto sorted = milliseconds;
  std::sort(sorted.begin(), sorted.end());
  return frontier_timing{
    sorted[sorted.size() / 2], sorted.front(), values.size(), sum, final_offset};
}

template <typename GraphViewType, typename EdgeValueInputWrapper, typename SegmentLauncher>
auto run_frontier_with_launcher(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                vertex_t const* frontier_first,
                                EdgeValueInputWrapper edge_value_input,
                                raft::host_span<size_t const> frontier_offsets,
                                SegmentLauncher& launcher)
{
  return cugraph::detail::transform_v_frontier_e_with_segment_launcher(
    handle,
    graph_view,
    frontier_first,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    edge_value_input,
    real_edge_op{},
    frontier_offsets,
    launcher);
}

template <typename GraphViewType, typename EdgeValueInputWrapper>
auto run_frontier_baseline(raft::handle_t const& handle,
                           GraphViewType const& graph_view,
                           vertex_t const* frontier_first,
                           EdgeValueInputWrapper edge_value_input,
                           raft::host_span<size_t const> frontier_offsets)
{
  return cugraph::detail::transform_v_frontier_e(handle,
                                                 graph_view,
                                                 frontier_first,
                                                 cugraph::edge_src_dummy_property_t{}.view(),
                                                 cugraph::edge_dst_dummy_property_t{}.view(),
                                                 edge_value_input,
                                                 real_edge_op{},
                                                 frontier_offsets);
}

void print(std::string_view arm, frontier_timing const& value)
{
  std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
            << " primitive=transform_v_frontier_e"
            << " arm=" << arm << " median_ms=" << value.median_ms << " min_ms=" << value.min_ms
            << " output_size=" << value.output_size << " output_sum=" << value.output_sum
            << " final_offset=" << value.final_offset << '\n';
}

struct sampling_timing {
  double median_ms{};
  double min_ms{};
  std::size_t output_size{};
  result_t index_sum{};
  std::size_t final_offset{};
};

template <typename Function>
sampling_timing measure_sampling(raft::handle_t const& handle,
                                 std::size_t iterations,
                                 Function&& function)
{
  {
    auto output = function();
    handle.sync_stream();
  }

  std::vector<double> milliseconds{};
  milliseconds.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    auto const start = std::chrono::steady_clock::now();
    auto output      = function();
    handle.sync_stream();
    auto const stop = std::chrono::steady_clock::now();
    milliseconds.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }

  auto output = function();
  handle.sync_stream();
  auto& indices  = std::get<0>(output);
  auto const sum = thrust::reduce(
    rmm::exec_policy(handle.get_stream()), indices.begin(), indices.end(), result_t{0});
  auto const final_offset = std::get<2>(output).back();

  auto sorted = milliseconds;
  std::sort(sorted.begin(), sorted.end());
  return sampling_timing{
    sorted[sorted.size() / 2], sorted.front(), indices.size(), sum, final_offset};
}

template <typename GraphViewType, typename EdgeValueInputWrapper, typename SegmentLauncher>
auto run_sampling_with_launcher(raft::handle_t const& handle,
                                GraphViewType const& graph_view,
                                vertex_t const* frontier_first,
                                EdgeValueInputWrapper edge_value_input,
                                raft::host_span<size_t const> frontier_offsets,
                                raft::random::RngState& rng_state,
                                SegmentLauncher& launcher)
{
  return cugraph::detail::
    homogeneous_biased_sample_and_compute_local_nbr_indices_with_segment_launcher(
      handle,
      graph_view,
      frontier_first,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_value_input,
      realistic_bias_op{},
      frontier_offsets,
      &rng_state,
      std::size_t{10},
      false,
      false,
      launcher);
}

template <typename GraphViewType, typename EdgeValueInputWrapper>
auto run_sampling_baseline(raft::handle_t const& handle,
                           GraphViewType const& graph_view,
                           vertex_t const* frontier_first,
                           EdgeValueInputWrapper edge_value_input,
                           raft::host_span<size_t const> frontier_offsets,
                           raft::random::RngState& rng_state)
{
  return cugraph::detail::homogeneous_biased_sample_and_compute_local_nbr_indices(
    handle,
    graph_view,
    frontier_first,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    edge_value_input,
    realistic_bias_op{},
    frontier_offsets,
    &rng_state,
    std::size_t{10},
    false,
    false);
}

void print(std::string_view arm, sampling_timing const& value)
{
  std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
            << " primitive=homogeneous_biased_local_sampling"
            << " arm=" << arm << " median_ms=" << value.median_ms << " min_ms=" << value.min_ms
            << " output_size=" << value.output_size << " index_sum=" << value.index_sum
            << " final_offset=" << value.final_offset << '\n';
}

template <typename Function>
sampling_timing measure_full_sampling(raft::handle_t const& handle,
                                      std::size_t iterations,
                                      Function&& function)
{
  {
    auto output = function();
    handle.sync_stream();
  }

  std::vector<double> milliseconds{};
  milliseconds.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    auto const start = std::chrono::steady_clock::now();
    auto output      = function();
    handle.sync_stream();
    auto const stop = std::chrono::steady_clock::now();
    milliseconds.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }

  auto output = function();
  handle.sync_stream();
  auto& sources  = std::get<0>(output);
  auto const sum = thrust::reduce(
    rmm::exec_policy(handle.get_stream()), sources.begin(), sources.end(), result_t{0});

  auto sorted = milliseconds;
  std::sort(sorted.begin(), sorted.end());
  return sampling_timing{
    sorted[sorted.size() / 2], sorted.front(), sources.size(), sum, sources.size()};
}

void print_full_sampling(sampling_timing const& value)
{
  std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
            << " primitive=homogeneous_biased_neighbor_sample"
            << " arm=baseline median_ms=" << value.median_ms << " min_ms=" << value.min_ms
            << " output_size=" << value.output_size << " source_sum=" << value.index_sum << '\n';
}

TEST(CudaStfTransformReduceEExperiment, SegmentScheduling)
{
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(8);
  raft::handle_t handle{rmm::cuda_stream_per_thread, stream_pool};

  auto usecase = cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(
    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false, 0));
  auto [graph, edge_weights, renumber_map] =
    cugraph::test::construct_graph<vertex_t, edge_t, float, false, false>(
      handle, usecase, true, true);
  ASSERT_TRUE(edge_weights.has_value());
  renumber_map.reset();

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights->view();
  auto const repeats    = iteration_count();
  auto const baseline =
    measure(repeats, [&] { return run_baseline(handle, graph_view, edge_weight_view); });
  auto const explicit_streams = measure(repeats, [&] {
    cugraph::test::experimental::explicit_stream_segment_launcher launcher{handle};
    return run_with_launcher(handle, graph_view, edge_weight_view, launcher);
  });
  auto const stf              = measure(repeats, [&] {
    cugraph::test::experimental::stf_segment_launcher launcher{handle};
    return run_with_launcher(handle, graph_view, edge_weight_view, launcher);
  });

  EXPECT_NE(baseline.result, result_t{0});
  EXPECT_EQ(explicit_streams.result, baseline.result);
  EXPECT_EQ(stf.result, baseline.result);

  print("baseline", baseline);
  print("explicit_streams", explicit_streams);
  print("cuda_stf", stf);
}

TEST(CudaStfTransformVFrontierEExperiment, SegmentScheduling)
{
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(8);
  raft::handle_t handle{rmm::cuda_stream_per_thread, stream_pool};

  auto usecase = cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(
    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false, 0));
  auto [graph, edge_weights, renumber_map] =
    cugraph::test::construct_graph<vertex_t, edge_t, float, false, false>(
      handle, usecase, true, true);
  ASSERT_TRUE(edge_weights.has_value());
  renumber_map.reset();

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights->view();
  raft::random::RngState frontier_rng{0};
  auto frontier =
    cugraph::select_random_vertices(handle,
                                    graph_view,
                                    std::optional<raft::device_span<vertex_t const>>{std::nullopt},
                                    frontier_rng,
                                    std::max<std::size_t>(graph_view.number_of_vertices() / 20, 1),
                                    false,
                                    false);
  std::array<std::size_t, 2> frontier_offsets{0, frontier.size()};
  auto offsets = raft::host_span<size_t const>{frontier_offsets.data(), frontier_offsets.size()};
  auto const repeats = iteration_count();

  auto const baseline         = measure_frontier(handle, repeats, [&] {
    return run_frontier_baseline(handle, graph_view, frontier.data(), edge_weight_view, offsets);
  });
  auto const explicit_streams = measure_frontier(handle, repeats, [&] {
    cugraph::test::experimental::explicit_stream_segment_launcher launcher{handle};
    return run_frontier_with_launcher(
      handle, graph_view, frontier.data(), edge_weight_view, offsets, launcher);
  });
  auto const stf              = measure_frontier(handle, repeats, [&] {
    cugraph::test::experimental::stf_segment_launcher launcher{handle};
    return run_frontier_with_launcher(
      handle, graph_view, frontier.data(), edge_weight_view, offsets, launcher);
  });

  EXPECT_GT(baseline.output_size, std::size_t{0});
  EXPECT_EQ(baseline.final_offset, baseline.output_size);
  EXPECT_EQ(explicit_streams.output_size, baseline.output_size);
  EXPECT_EQ(explicit_streams.output_sum, baseline.output_sum);
  EXPECT_EQ(explicit_streams.final_offset, baseline.final_offset);
  EXPECT_EQ(stf.output_size, baseline.output_size);
  EXPECT_EQ(stf.output_sum, baseline.output_sum);
  EXPECT_EQ(stf.final_offset, baseline.final_offset);

  print("baseline", baseline);
  print("explicit_streams", explicit_streams);
  print("cuda_stf", stf);
}

TEST(CudaStfBiasedSamplingExperiment, PartialFrontier)
{
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(8);
  raft::handle_t handle{rmm::cuda_stream_per_thread, stream_pool};

  auto usecase = cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(
    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false, 0));
  auto [graph, edge_weights, renumber_map] =
    cugraph::test::construct_graph<vertex_t, edge_t, float, false, false>(
      handle, usecase, true, true);
  ASSERT_TRUE(edge_weights.has_value());
  renumber_map.reset();

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights->view();
  raft::random::RngState frontier_rng{0};
  auto frontier =
    cugraph::select_random_vertices(handle,
                                    graph_view,
                                    std::optional<raft::device_span<vertex_t const>>{std::nullopt},
                                    frontier_rng,
                                    std::max<std::size_t>(graph_view.number_of_vertices() / 20, 1),
                                    false,
                                    false);
  std::array<std::size_t, 2> frontier_offsets{0, frontier.size()};
  auto offsets = raft::host_span<size_t const>{frontier_offsets.data(), frontier_offsets.size()};
  auto const repeats = iteration_count();

  auto const baseline         = measure_sampling(handle, repeats, [&] {
    raft::random::RngState rng_state{1};
    return run_sampling_baseline(
      handle, graph_view, frontier.data(), edge_weight_view, offsets, rng_state);
  });
  auto const explicit_streams = measure_sampling(handle, repeats, [&] {
    raft::random::RngState rng_state{1};
    cugraph::test::experimental::explicit_stream_segment_launcher launcher{handle};
    return run_sampling_with_launcher(
      handle, graph_view, frontier.data(), edge_weight_view, offsets, rng_state, launcher);
  });
  auto const stf              = measure_sampling(handle, repeats, [&] {
    raft::random::RngState rng_state{1};
    cugraph::test::experimental::stf_segment_launcher launcher{handle};
    return run_sampling_with_launcher(
      handle, graph_view, frontier.data(), edge_weight_view, offsets, rng_state, launcher);
  });
  std::array<int32_t, 1> fanout{10};
  auto const full_sampling = measure_full_sampling(handle, repeats, [&] {
    raft::random::RngState rng_state{1};
    return cugraph::homogeneous_biased_neighbor_sample(
      handle,
      rng_state,
      graph_view,
      std::make_optional(edge_weight_view),
      std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
      std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
      edge_weight_view,
      raft::device_span<vertex_t const>{frontier.data(), frontier.size()},
      std::optional<raft::device_span<int32_t const>>{std::nullopt},
      std::optional<raft::device_span<int32_t const>>{std::nullopt},
      raft::host_span<int32_t const>{fanout.data(), fanout.size()},
      cugraph::sampling_flags_t{cugraph::prior_sources_behavior_t{0},
                                false,
                                false,
                                false,
                                cugraph::temporal_sampling_comparison_t::STRICTLY_INCREASING,
                                false});
  });

  EXPECT_GT(baseline.output_size, std::size_t{0});
  EXPECT_EQ(baseline.final_offset, baseline.output_size);
  EXPECT_EQ(explicit_streams.output_size, baseline.output_size);
  EXPECT_EQ(explicit_streams.index_sum, baseline.index_sum);
  EXPECT_EQ(explicit_streams.final_offset, baseline.final_offset);
  EXPECT_EQ(stf.output_size, baseline.output_size);
  EXPECT_EQ(stf.index_sum, baseline.index_sum);
  EXPECT_EQ(stf.final_offset, baseline.final_offset);
  EXPECT_GT(full_sampling.output_size, std::size_t{0});
  EXPECT_LE(full_sampling.output_size, baseline.output_size);

  print("baseline", baseline);
  print("explicit_streams", explicit_streams);
  print("cuda_stf", stf);
  print_full_sampling(full_sampling);
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
