/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "experimental/stf/iterative_graph_common.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/experimental/stf.cuh>
#include <thrust/count.h>
#include <thrust/reduce.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string_view>
#include <vector>

namespace {

using namespace cugraph::test::experimental;
namespace stf = cuda::experimental::stf;

constexpr stf_result_t alpha{0.85f};
constexpr stf_result_t epsilon{1.0e-6f};
constexpr int max_iterations{500};

struct native_pagerank_state {
  native_pagerank_state(std::size_t size, rmm::cuda_stream_view stream)
    : current(size, stream),
      next(size, stream),
      dangling_sum(1, stream),
      difference_sum(1, stream),
      iteration(1, stream)
  {
  }

  rmm::device_uvector<stf_result_t> current;
  rmm::device_uvector<stf_result_t> next;
  rmm::device_uvector<stf_result_t> dangling_sum;
  rmm::device_uvector<stf_result_t> difference_sum;
  rmm::device_uvector<int> iteration;
};

struct dangling_sum_op {
  template <typename Current, typename OutDegrees>
  __device__ void operator()(std::size_t i,
                             Current current,
                             OutDegrees out_degrees,
                             stf_result_t& sum) const
  {
    if (out_degrees(i) == 0) { sum += current(i); }
  }
};

struct pagerank_update_op {
  std::size_t vertices;

  template <typename Offsets,
            typename Indices,
            typename OutDegrees,
            typename Current,
            typename Next,
            typename Dangling>
  __device__ void operator()(std::size_t vertex,
                             Offsets offsets,
                             Indices indices,
                             OutDegrees out_degrees,
                             Current current,
                             Next next,
                             Dangling dangling,
                             stf_result_t& difference) const
  {
    stf_result_t incoming{0};
    for (auto edge = offsets(vertex); edge < offsets(vertex + 1); ++edge) {
      auto const source = indices(edge);
      auto const degree = out_degrees(source);
      if (degree > 0) { incoming += current(source) / static_cast<stf_result_t>(degree); }
    }
    auto const uniform = stf_result_t{1} / static_cast<stf_result_t>(vertices);
    auto const updated =
      alpha * incoming + (alpha * (*dangling) + (stf_result_t{1} - alpha)) * uniform;
    next(vertex) = updated;
    difference += fabsf(updated - current(vertex));
  }
};

struct pagerank_continue_op {
  template <typename Difference, typename Iteration>
  __device__ bool operator()(Difference difference, Iteration iteration) const
  {
    ++(*iteration);
    return (*difference >= epsilon) && (*iteration < max_iterations);
  }
};

template <typename Current, typename Next, typename Iteration>
void add_initialization(stf::stackable_ctx& context,
                        Current& current,
                        Next& next,
                        Iteration& iteration,
                        std::size_t vertices)
{
  auto const initial = stf_result_t{1} / static_cast<stf_result_t>(vertices);
  context.parallel_for(stf::box(vertices), current.write(), next.write())
      .set_symbol("initialize ranks")
      ->*[initial] __device__(std::size_t i, auto current_values, auto next_values) {
            current_values(i) = initial;
            next_values(i)    = stf_result_t{0};
          };
  context.parallel_for(stf::box(1), iteration.write()).set_symbol("initialize iteration")
      ->*[] __device__(std::size_t, auto iteration_value) { *iteration_value = 0; };
}

template <typename Current, typename Next>
void add_commit(stf::stackable_ctx& context, Current& current, Next& next, std::size_t vertices)
{
  context.parallel_for(stf::box(vertices), current.write(), next.read()).set_symbol("commit ranks")
      ->*[] __device__(std::size_t i, auto current_values, auto next_values) {
            current_values(i) = next_values(i);
          };
}

std::size_t mismatch_count(raft::handle_t const& handle,
                           rmm::device_uvector<stf_result_t> const& reference,
                           rmm::device_uvector<stf_result_t> const& candidate)
{
  return static_cast<std::size_t>(
    thrust::count_if(handle.get_thrust_policy(),
                     thrust::make_counting_iterator<std::size_t>(0),
                     thrust::make_counting_iterator(reference.size()),
                     [lhs = reference.data(), rhs = candidate.data()] __device__(std::size_t i) {
                       auto const absolute = fabsf(lhs[i] - rhs[i]);
                       auto const scale    = fmaxf(fabsf(lhs[i]), fabsf(rhs[i]));
                       return absolute > (2.0e-6f + 2.0e-4f * scale);
                     }));
}

TEST(CudaStfNativePageRankExperiment, RootRmmStorageAndDeviceConvergence)
{
  raft::handle_t handle{};
  auto usecase = cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(
    cugraph::test::Rmat_Usecase(18, 16, 0.57, 0.19, 0.19, 0, false, false, 0));
  auto [graph, edge_weights, renumber_map] =
    cugraph::test::construct_graph<stf_vertex_t, stf_edge_t, stf_result_t, true, false>(
      handle, usecase, true, true);
  edge_weights.reset();
  renumber_map.reset();

  auto graph_view     = graph.view();
  auto edge_partition = graph_view.local_edge_partition_view();
  auto offsets        = edge_partition.offsets().data();
  auto indices        = edge_partition.indices().data();
  auto const vertices = static_cast<std::size_t>(graph_view.number_of_vertices());
  auto const edges    = static_cast<std::size_t>(edge_partition.number_of_edges());
  auto out_degrees    = graph_view.compute_out_degrees(handle);

  native_pagerank_state state{vertices, handle.get_stream()};
  stf::stackable_ctx context;
  auto l_offsets =
    context.logical_data(stf::make_slice(offsets, vertices + 1), stf::data_place::current_device());
  auto l_indices =
    context.logical_data(stf::make_slice(indices, edges), stf::data_place::current_device());
  auto l_out_degrees = context.logical_data(stf::make_slice(out_degrees.data(), vertices),
                                            stf::data_place::current_device());
  auto l_current     = context.logical_data(stf::make_slice(state.current.data(), vertices),
                                        stf::data_place::current_device());
  auto l_next        = context.logical_data(stf::make_slice(state.next.data(), vertices),
                                     stf::data_place::current_device());
  auto l_dangling = context.logical_data(stf::scalar_view<stf_result_t>{state.dangling_sum.data()},
                                         stf::data_place::current_device());
  auto l_difference = context.logical_data(
    stf::scalar_view<stf_result_t>{state.difference_sum.data()}, stf::data_place::current_device());
  auto l_iteration = context.logical_data(stf::scalar_view<int>{state.iteration.data()},
                                          stf::data_place::current_device());

  stf::stackable_ctx::launchable_graph_scope graph_scope{context};
  l_offsets.push(stf::access_mode::read, stf::data_place::current_device());
  l_indices.push(stf::access_mode::read, stf::data_place::current_device());
  l_out_degrees.push(stf::access_mode::read, stf::data_place::current_device());
  l_current.push(stf::access_mode::rw, stf::data_place::current_device());
  l_next.push(stf::access_mode::rw, stf::data_place::current_device());
  l_dangling.push(stf::access_mode::rw, stf::data_place::current_device());
  l_difference.push(stf::access_mode::rw, stf::data_place::current_device());
  l_iteration.push(stf::access_mode::rw, stf::data_place::current_device());

  add_initialization(context, l_current, l_next, l_iteration, vertices);

  {
    auto loop = context.while_graph_scope(1);
    context
        .parallel_for(stf::box(vertices),
                      l_current.read(),
                      l_out_degrees.read(),
                      l_dangling.reduce(stf::reducer::sum<stf_result_t>{}))
        .set_symbol("dangling sum")
        ->*dangling_sum_op{};
    context
        .parallel_for(stf::box(vertices),
                      l_offsets.read(),
                      l_indices.read(),
                      l_out_degrees.read(),
                      l_current.read(),
                      l_next.write(),
                      l_dangling.read(),
                      l_difference.reduce(stf::reducer::sum<stf_result_t>{}))
        .set_symbol("pagerank update and difference")
        ->*pagerank_update_op{vertices};
    add_commit(context, l_current, l_next, vertices);
    loop.update_cond(l_difference.read(), l_iteration.rw())->*pagerank_continue_op{};
  }
  graph_scope.exec();

  rmm::device_uvector<stf_result_t> reference(vertices, handle.get_stream());
  cugraph::centrality_algorithm_metadata_t reference_metadata{};
  auto run_reference = [&] {
    auto [result, metadata] =
      cugraph::pagerank<stf_vertex_t, stf_edge_t, stf_result_t, stf_result_t, false>(
        handle,
        graph_view,
        std::optional<cugraph::edge_property_view_t<stf_edge_t, stf_result_t const*>>{std::nullopt},
        std::optional<raft::device_span<stf_result_t const>>{std::nullopt},
        std::optional<
          std::tuple<raft::device_span<stf_vertex_t const>, raft::device_span<stf_result_t const>>>{
          std::nullopt},
        std::optional<raft::device_span<stf_result_t const>>{std::nullopt},
        alpha,
        epsilon,
        static_cast<std::size_t>(max_iterations),
        false);
    reference          = std::move(result);
    reference_metadata = metadata;
  };

  run_reference();
  graph_scope.launch();
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  std::array<std::vector<double>, 2> samples;
  auto const repetitions = experiment_repetitions();
  for (auto& values : samples) {
    values.reserve(repetitions);
  }
  for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
    for (std::size_t offset = 0; offset < samples.size(); ++offset) {
      auto arm = (repeat + offset) % samples.size();
      if (arm == 0) {
        samples[arm].push_back(wall_time_ms(run_reference));
      } else {
        samples[arm].push_back(wall_time_ms([&] { graph_scope.launch(); }));
      }
    }
  }

  EXPECT_EQ(mismatch_count(handle, reference, state.current), std::size_t{0});
  auto rank_sum = thrust::reduce(
    handle.get_thrust_policy(), state.current.begin(), state.current.end(), stf_result_t{0});
  EXPECT_NEAR(rank_sum, stf_result_t{1}, stf_result_t{2.0e-3});
  int stf_iterations{};
  RAFT_CUDA_TRY(
    cudaMemcpy(&stf_iterations, state.iteration.data(), sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(static_cast<std::size_t>(stf_iterations), reference_metadata.number_of_iterations_);

  constexpr std::array<std::string_view, 2> names{"production_cugraph", "native_cuda_stf"};
  for (std::size_t arm = 0; arm < names.size(); ++arm) {
    auto result = summarize(std::move(samples[arm]));
    std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
              << " algorithm=native_pagerank"
              << " arm=" << names[arm] << " median_ms=" << result.median_ms
              << " min_ms=" << result.min_ms << " iterations=" << stf_iterations
              << " vertices=" << vertices << " edges=" << edges << '\n';
  }

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
