/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/traversal/sssp_impl.cuh"
#include "experimental/stf/segment_launchers.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/prims/transform_reduce_e.cuh>

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/tuple>
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using vertex_t = int32_t;
using edge_t   = int32_t;
using weight_t = float;

enum class arm : std::size_t { baseline = 0, explicit_streams = 1, cuda_stf = 2 };

constexpr std::array<std::string_view, 3> arm_names{"baseline", "explicit_streams", "cuda_stf"};

std::size_t iteration_count()
{
  auto const* value = std::getenv("CUGRAPH_STF_ITERATIONS");
  if (value == nullptr) { return 9; }
  auto parsed = std::strtoull(value, nullptr, 10);
  return parsed == 0 ? 9 : static_cast<std::size_t>(parsed);
}

struct edge_weight_op {
  __device__ weight_t
  operator()(vertex_t, vertex_t, cuda::std::nullopt_t, cuda::std::nullopt_t, weight_t weight) const
  {
    return weight;
  }
};

struct timing {
  double median_ms{};
  double min_ms{};
};

timing summarize(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return timing{values[values.size() / 2], values.front()};
}

template <typename Function>
double elapsed_ms(raft::handle_t const& handle, Function&& function)
{
  auto const start = std::chrono::steady_clock::now();
  function();
  handle.sync_stream();
  auto const stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

template <typename GraphViewType, typename EdgeWeightView>
void run_sssp_arm(arm selected_arm,
                  raft::handle_t const& handle,
                  GraphViewType const& graph_view,
                  EdgeWeightView edge_weight_view,
                  vertex_t source,
                  weight_t* distances,
                  vertex_t* predecessors)
{
  if (selected_arm == arm::baseline) {
    cugraph::sssp(handle,
                  graph_view,
                  edge_weight_view,
                  distances,
                  predecessors,
                  source,
                  std::numeric_limits<weight_t>::max(),
                  false);
  } else if (selected_arm == arm::explicit_streams) {
    cugraph::test::experimental::explicit_stream_segment_launcher launcher{handle};
    cugraph::detail::sssp_with_segment_launcher(handle,
                                                graph_view,
                                                edge_weight_view,
                                                distances,
                                                predecessors,
                                                source,
                                                std::numeric_limits<weight_t>::max(),
                                                false,
                                                launcher);
  } else {
    cugraph::test::experimental::stf_segment_launcher launcher{handle};
    cugraph::detail::sssp_with_segment_launcher(handle,
                                                graph_view,
                                                edge_weight_view,
                                                distances,
                                                predecessors,
                                                source,
                                                std::numeric_limits<weight_t>::max(),
                                                false,
                                                launcher);
  }
}

template <typename GraphViewType, typename EdgeWeightView>
weight_t run_reduce_arm(arm selected_arm,
                        raft::handle_t const& handle,
                        GraphViewType const& graph_view,
                        EdgeWeightView edge_weight_view)
{
  if (selected_arm == arm::baseline) {
    return cugraph::transform_reduce_e(handle,
                                       graph_view,
                                       cugraph::edge_src_dummy_property_t{}.view(),
                                       cugraph::edge_dst_dummy_property_t{}.view(),
                                       edge_weight_view,
                                       edge_weight_op{},
                                       weight_t{0});
  } else if (selected_arm == arm::explicit_streams) {
    cugraph::test::experimental::explicit_stream_segment_launcher launcher{handle};
    return cugraph::detail::transform_reduce_e_with_segment_launcher(
      handle,
      graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_weight_view,
      edge_weight_op{},
      weight_t{0},
      launcher);
  } else {
    cugraph::test::experimental::stf_segment_launcher launcher{handle};
    return cugraph::detail::transform_reduce_e_with_segment_launcher(
      handle,
      graph_view,
      cugraph::edge_src_dummy_property_t{}.view(),
      cugraph::edge_dst_dummy_property_t{}.view(),
      edge_weight_view,
      edge_weight_op{},
      weight_t{0},
      launcher);
  }
}

std::size_t count_distance_mismatches(raft::handle_t const& handle,
                                      rmm::device_uvector<weight_t> const& lhs,
                                      rmm::device_uvector<weight_t> const& rhs)
{
  auto pair_first = thrust::make_zip_iterator(lhs.begin(), rhs.begin());
  return thrust::count_if(rmm::exec_policy(handle.get_stream()),
                          pair_first,
                          pair_first + lhs.size(),
                          [] __device__(auto pair) {
                            auto lhs_value = cuda::std::get<0>(pair);
                            auto rhs_value = cuda::std::get<1>(pair);
                            auto invalid   = std::numeric_limits<weight_t>::max();
                            if ((lhs_value == invalid) || (rhs_value == invalid)) {
                              return lhs_value != rhs_value;
                            }
                            auto difference =
                              lhs_value > rhs_value ? lhs_value - rhs_value : rhs_value - lhs_value;
                            auto magnitude = lhs_value > weight_t{1} ? lhs_value : weight_t{1};
                            return difference > magnitude * weight_t{1e-5};
                          });
}

std::size_t count_invalid_predecessors(raft::handle_t const& handle,
                                       rmm::device_uvector<vertex_t> const& predecessors,
                                       vertex_t number_of_vertices)
{
  return thrust::count_if(
    rmm::exec_policy(handle.get_stream()),
    predecessors.begin(),
    predecessors.end(),
    [number_of_vertices] __device__(vertex_t predecessor) {
      return predecessor != cugraph::invalid_vertex_id<vertex_t>::value &&
             ((predecessor < vertex_t{0}) || (predecessor >= number_of_vertices));
    });
}

TEST(CudaStfSsspExperiment, FullAlgorithm)
{
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(8);
  raft::handle_t handle{rmm::cuda_stream_per_thread, stream_pool};

  auto usecase = cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(
    cugraph::test::Rmat_Usecase(20, 16, 0.57, 0.19, 0.19, 0, false, false, 0));
  auto [graph, edge_weights, renumber_map] =
    cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
      handle, usecase, true, true);
  ASSERT_TRUE(edge_weights.has_value());
  renumber_map.reset();

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights->view();
  auto source           = vertex_t{0};
  auto const repeats    = iteration_count();

  std::array<rmm::device_uvector<weight_t>, 3> distances{
    rmm::device_uvector<weight_t>(graph_view.number_of_vertices(), handle.get_stream()),
    rmm::device_uvector<weight_t>(graph_view.number_of_vertices(), handle.get_stream()),
    rmm::device_uvector<weight_t>(graph_view.number_of_vertices(), handle.get_stream())};
  std::array<rmm::device_uvector<vertex_t>, 3> predecessors{
    rmm::device_uvector<vertex_t>(graph_view.number_of_vertices(), handle.get_stream()),
    rmm::device_uvector<vertex_t>(graph_view.number_of_vertices(), handle.get_stream()),
    rmm::device_uvector<vertex_t>(graph_view.number_of_vertices(), handle.get_stream())};

  for (std::size_t i = 0; i < arm_names.size(); ++i) {
    run_sssp_arm(static_cast<arm>(i),
                 handle,
                 graph_view,
                 edge_weight_view,
                 source,
                 distances[i].data(),
                 predecessors[i].data());
  }
  handle.sync_stream();

  std::array<std::vector<double>, 3> sssp_samples{};
  std::array<std::vector<double>, 3> reduce_samples{};
  std::array<weight_t, 3> reduce_results{};
  for (auto& samples : sssp_samples) {
    samples.reserve(repeats);
  }
  for (auto& samples : reduce_samples) {
    samples.reserve(repeats);
  }

  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    for (std::size_t offset = 0; offset < arm_names.size(); ++offset) {
      auto index        = (repeat + offset) % arm_names.size();
      auto selected_arm = static_cast<arm>(index);
      sssp_samples[index].push_back(elapsed_ms(handle, [&] {
        run_sssp_arm(selected_arm,
                     handle,
                     graph_view,
                     edge_weight_view,
                     source,
                     distances[index].data(),
                     predecessors[index].data());
      }));
    }
  }

  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    for (std::size_t offset = 0; offset < arm_names.size(); ++offset) {
      auto index        = (repeat + offset) % arm_names.size();
      auto selected_arm = static_cast<arm>(index);
      reduce_samples[index].push_back(elapsed_ms(handle, [&] {
        reduce_results[index] = run_reduce_arm(selected_arm, handle, graph_view, edge_weight_view);
      }));
    }
  }

  auto explicit_mismatches = count_distance_mismatches(handle, distances[0], distances[1]);
  auto stf_mismatches      = count_distance_mismatches(handle, distances[0], distances[2]);
  EXPECT_EQ(explicit_mismatches, std::size_t{0});
  EXPECT_EQ(stf_mismatches, std::size_t{0});
  for (auto const& values : predecessors) {
    EXPECT_EQ(count_invalid_predecessors(
                handle, values, static_cast<vertex_t>(graph_view.number_of_vertices())),
              std::size_t{0});
  }

  std::array<timing, 3> sssp_timings{};
  std::array<timing, 3> reduce_timings{};
  for (std::size_t i = 0; i < arm_names.size(); ++i) {
    sssp_timings[i]   = summarize(std::move(sssp_samples[i]));
    reduce_timings[i] = summarize(std::move(reduce_samples[i]));
    auto fraction     = reduce_timings[i].median_ms / sssp_timings[i].median_ms;
    std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
              << " algorithm=sssp"
              << " arm=" << arm_names[i] << " median_ms=" << sssp_timings[i].median_ms
              << " min_ms=" << sssp_timings[i].min_ms
              << " initial_reduce_ms=" << reduce_timings[i].median_ms
              << " initial_reduce_fraction=" << fraction << " reduce_result=" << reduce_results[i]
              << '\n';
  }
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
