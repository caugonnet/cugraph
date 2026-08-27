/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "experimental/stf/iterative_graph_common.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>

#include <raft/core/handle.hpp>
#include <raft/core/resource/custom_resource.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/experimental/stf.cuh>
#include <thrust/count.h>
#include <thrust/reduce.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

using namespace cugraph::test::experimental;
namespace stf = cuda::experimental::stf;

constexpr std::size_t stream_count{4};
constexpr stf_result_t alpha{0.85f};

std::size_t query_count()
{
  auto const* value = std::getenv("CUGRAPH_STF_PPR_QUERIES");
  if (value == nullptr) { return 4; }
  auto parsed = std::strtoull(value, nullptr, 10);
  return parsed == 0 ? 4 : static_cast<std::size_t>(parsed);
}

int pagerank_iteration_count()
{
  auto const* value = std::getenv("CUGRAPH_STF_PPR_STEPS");
  if (value == nullptr) { return 20; }
  auto parsed = std::strtol(value, nullptr, 10);
  return parsed <= 0 ? 20 : static_cast<int>(parsed);
}

struct query_state {
  query_state(std::size_t size, rmm::cuda_stream_view stream)
    : current(size, stream),
      next(size, stream),
      personalization(size, stream),
      dangling_sum(1, stream)
  {
  }

  rmm::device_uvector<stf_result_t> current;
  rmm::device_uvector<stf_result_t> next;
  rmm::device_uvector<stf_result_t> personalization;
  rmm::device_uvector<stf_result_t> dangling_sum;
};

using state_set = std::vector<std::unique_ptr<query_state>>;

state_set make_states(std::size_t queries, std::size_t vertices, rmm::cuda_stream_view stream)
{
  state_set states;
  states.reserve(queries);
  for (std::size_t i = 0; i < queries; ++i) {
    states.push_back(std::make_unique<query_state>(vertices, stream));
  }
  return states;
}

void enqueue_query(cudaStream_t stream,
                   stf_edge_t const* offsets,
                   stf_vertex_t const* indices,
                   stf_edge_t const* out_degrees,
                   std::size_t vertices,
                   stf_vertex_t source,
                   int iterations,
                   query_state& state)
{
  initialize_pagerank_query<<<grid_size(vertices), block_size, 0, stream>>>(
    state.current.data(),
    state.next.data(),
    state.personalization.data(),
    state.dangling_sum.data(),
    vertices,
    source);
  enqueue_pagerank_iterations(stream,
                              offsets,
                              indices,
                              out_degrees,
                              state.current.data(),
                              state.next.data(),
                              state.personalization.data(),
                              state.dangling_sum.data(),
                              vertices,
                              alpha,
                              iterations);
}

void run_serial(cudaStream_t stream,
                stf_edge_t const* offsets,
                stf_vertex_t const* indices,
                stf_edge_t const* out_degrees,
                std::size_t vertices,
                int iterations,
                state_set& states)
{
  for (std::size_t query = 0; query < states.size(); ++query) {
    enqueue_query(stream,
                  offsets,
                  indices,
                  out_degrees,
                  vertices,
                  static_cast<stf_vertex_t>(query % vertices),
                  iterations,
                  *states[query]);
  }
}

class explicit_query_scheduler {
 public:
  explicit_query_scheduler(raft::handle_t const& handle) : caller_stream_(handle.get_stream())
  {
    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&start_, cudaEventDisableTiming));
    for (std::size_t i = 0; i < stream_count; ++i) {
      streams_[i] = handle.get_stream_from_stream_pool(i);
      RAFT_CUDA_TRY(cudaEventCreateWithFlags(&done_[i], cudaEventDisableTiming));
    }
  }

  ~explicit_query_scheduler()
  {
    for (auto event : done_) {
      if (event != nullptr) { cudaEventDestroy(event); }
    }
    if (start_ != nullptr) { cudaEventDestroy(start_); }
  }

  void run(stf_edge_t const* offsets,
           stf_vertex_t const* indices,
           stf_edge_t const* out_degrees,
           std::size_t vertices,
           int iterations,
           state_set& states)
  {
    RAFT_CUDA_TRY(cudaEventRecord(start_, caller_stream_));
    std::array<bool, stream_count> used{};
    for (std::size_t query = 0; query < states.size(); ++query) {
      auto index  = query % stream_count;
      auto stream = streams_[index];
      if (!used[index]) {
        RAFT_CUDA_TRY(cudaStreamWaitEvent(stream, start_));
        used[index] = true;
      }
      enqueue_query(stream,
                    offsets,
                    indices,
                    out_degrees,
                    vertices,
                    static_cast<stf_vertex_t>(query % vertices),
                    iterations,
                    *states[query]);
    }
    for (std::size_t i = 0; i < stream_count; ++i) {
      if (!used[i]) { continue; }
      RAFT_CUDA_TRY(cudaEventRecord(done_[i], streams_[i]));
      RAFT_CUDA_TRY(cudaStreamWaitEvent(caller_stream_, done_[i]));
    }
  }

 private:
  cudaStream_t caller_stream_{};
  cudaEvent_t start_{};
  std::array<cudaStream_t, stream_count> streams_{};
  std::array<cudaEvent_t, stream_count> done_{};
};

stf::stream_ctx make_stf_context(raft::handle_t const& handle)
{
  auto* resources = raft::resource::get_custom_resource<stf::async_resources_handle>(handle);
  return stf::stream_ctx(static_cast<cudaStream_t>(handle.get_stream()), *resources);
}

void run_stf(raft::handle_t const& handle,
             stf_edge_t const* offsets,
             stf_vertex_t const* indices,
             stf_edge_t const* out_degrees,
             std::size_t edges,
             std::size_t vertices,
             int iterations,
             state_set& states)
{
  auto context = make_stf_context(handle);
  auto l_offsets =
    context.logical_data(stf::make_slice(offsets, vertices + 1), stf::data_place::current_device());
  auto l_indices =
    context.logical_data(stf::make_slice(indices, edges), stf::data_place::current_device());
  auto l_out_degrees =
    context.logical_data(stf::make_slice(out_degrees, vertices), stf::data_place::current_device());

  for (std::size_t query = 0; query < states.size(); ++query) {
    auto& state            = *states[query];
    auto l_current         = context.logical_data(stf::make_slice(state.current.data(), vertices),
                                          stf::data_place::current_device());
    auto l_next            = context.logical_data(stf::make_slice(state.next.data(), vertices),
                                       stf::data_place::current_device());
    auto l_personalization = context.logical_data(
      stf::make_slice(state.personalization.data(), vertices), stf::data_place::current_device());
    auto l_dangling =
      context.logical_data(stf::make_slice(state.dangling_sum.data(), std::size_t{1}),
                           stf::data_place::current_device());

    context.task(l_offsets.read(),
                 l_indices.read(),
                 l_out_degrees.read(),
                 l_current.rw(),
                 l_next.rw(),
                 l_personalization.rw(),
                 l_dangling.rw())
        ->*[vertices, iterations, source = static_cast<stf_vertex_t>(query % vertices)](
             cudaStream_t stream,
             auto task_offsets,
             auto task_indices,
             auto task_out_degrees,
             auto current,
             auto next,
             auto personalization,
             auto dangling) {
              initialize_pagerank_query<<<grid_size(vertices), block_size, 0, stream>>>(
                current.data_handle(),
                next.data_handle(),
                personalization.data_handle(),
                dangling.data_handle(),
                vertices,
                source);
              enqueue_pagerank_iterations(stream,
                                          task_offsets.data_handle(),
                                          task_indices.data_handle(),
                                          task_out_degrees.data_handle(),
                                          current.data_handle(),
                                          next.data_handle(),
                                          personalization.data_handle(),
                                          dangling.data_handle(),
                                          vertices,
                                          alpha,
                                          iterations);
            };
  }
  context.finalize();
}

std::size_t mismatch_count(raft::handle_t const& handle,
                           query_state const& reference,
                           query_state const& candidate)
{
  return static_cast<std::size_t>(thrust::count_if(
    handle.get_thrust_policy(),
    thrust::make_counting_iterator<std::size_t>(0),
    thrust::make_counting_iterator(reference.current.size()),
    [lhs = reference.current.data(), rhs = candidate.current.data()] __device__(std::size_t i) {
      auto const absolute = fabsf(lhs[i] - rhs[i]);
      auto const scale    = fmaxf(fabsf(lhs[i]), fabsf(rhs[i]));
      return absolute > (2.0e-6f + 2.0e-5f * scale);
    }));
}

TEST(CudaStfConcurrentPageRankExperiment, IndependentPersonalizations)
{
  auto stream_pool = std::make_shared<rmm::cuda_stream_pool>(stream_count);
  raft::handle_t handle{rmm::cuda_stream_per_thread, stream_pool};
  auto stream  = static_cast<cudaStream_t>(handle.get_stream());
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
  auto vertices       = static_cast<std::size_t>(graph_view.number_of_vertices());
  auto edges          = static_cast<std::size_t>(edge_partition.number_of_edges());
  auto out_degrees    = graph_view.compute_out_degrees(handle);
  auto queries        = query_count();
  auto iterations     = pagerank_iteration_count();

  std::array<state_set, 3> states{make_states(queries, vertices, handle.get_stream()),
                                  make_states(queries, vertices, handle.get_stream()),
                                  make_states(queries, vertices, handle.get_stream())};
  explicit_query_scheduler explicit_scheduler{handle};

  run_serial(stream, offsets, indices, out_degrees.data(), vertices, iterations, states[0]);
  explicit_scheduler.run(offsets, indices, out_degrees.data(), vertices, iterations, states[1]);
  run_stf(handle, offsets, indices, out_degrees.data(), edges, vertices, iterations, states[2]);
  handle.sync_stream();

  std::array<std::vector<double>, 3> samples{};
  auto repetitions = experiment_repetitions();
  for (auto& values : samples) {
    values.reserve(repetitions);
  }
  for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
    for (std::size_t offset = 0; offset < 3; ++offset) {
      auto arm = (repeat + offset) % 3;
      if (arm == 0) {
        samples[arm].push_back(wall_time_ms([&] {
          run_serial(stream, offsets, indices, out_degrees.data(), vertices, iterations, states[0]);
        }));
      } else if (arm == 1) {
        samples[arm].push_back(wall_time_ms([&] {
          explicit_scheduler.run(
            offsets, indices, out_degrees.data(), vertices, iterations, states[1]);
        }));
      } else {
        samples[arm].push_back(wall_time_ms([&] {
          run_stf(
            handle, offsets, indices, out_degrees.data(), edges, vertices, iterations, states[2]);
        }));
      }
    }
  }

  for (std::size_t query = 0; query < queries; ++query) {
    EXPECT_EQ(mismatch_count(handle, *states[0][query], *states[1][query]), std::size_t{0});
    EXPECT_EQ(mismatch_count(handle, *states[0][query], *states[2][query]), std::size_t{0});
    auto rank_sum = thrust::reduce(handle.get_thrust_policy(),
                                   states[2][query]->current.begin(),
                                   states[2][query]->current.end(),
                                   stf_result_t{0});
    EXPECT_NEAR(rank_sum, stf_result_t{1}, stf_result_t{2.0e-3});
  }

  query_state long_run{vertices, handle.get_stream()};
  enqueue_query(
    stream, offsets, indices, out_degrees.data(), vertices, stf_vertex_t{0}, 100, long_run);
  rmm::device_uvector<stf_vertex_t> reference_vertex(1, handle.get_stream());
  rmm::device_uvector<stf_result_t> reference_value(1, handle.get_stream());
  rmm::device_uvector<stf_result_t> reference_ranks(vertices, handle.get_stream());
  stf_vertex_t host_vertex{0};
  stf_result_t host_value{1};
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    reference_vertex.data(), &host_vertex, sizeof(host_vertex), cudaMemcpyHostToDevice, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    reference_value.data(), &host_value, sizeof(host_value), cudaMemcpyHostToDevice, stream));
  cugraph::pagerank<stf_vertex_t, stf_edge_t, stf_result_t, stf_result_t, false>(
    handle,
    graph_view,
    std::optional<cugraph::edge_property_view_t<stf_edge_t, stf_result_t const*>>{std::nullopt},
    std::optional<stf_result_t const*>{std::nullopt},
    std::optional<stf_vertex_t const*>{reference_vertex.data()},
    std::optional<stf_result_t const*>{reference_value.data()},
    std::optional<stf_vertex_t>{stf_vertex_t{1}},
    reference_ranks.data(),
    alpha,
    stf_result_t{1.0e-6},
    std::size_t{500},
    false,
    false);
  query_state reference_state{vertices, handle.get_stream()};
  RAFT_CUDA_TRY(cudaMemcpyAsync(reference_state.current.data(),
                                reference_ranks.data(),
                                vertices * sizeof(stf_result_t),
                                cudaMemcpyDeviceToDevice,
                                stream));
  EXPECT_EQ(mismatch_count(handle, reference_state, long_run), std::size_t{0});

  constexpr std::array<std::string_view, 3> names{"serial", "explicit_streams", "cuda_stf"};
  for (std::size_t arm = 0; arm < names.size(); ++arm) {
    auto result = summarize(std::move(samples[arm]));
    std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
              << " algorithm=concurrent_pagerank"
              << " arm=" << names[arm] << " median_ms=" << result.median_ms
              << " min_ms=" << result.min_ms << " queries=" << queries
              << " iterations=" << iterations << " vertices=" << vertices << " edges=" << edges
              << " throughput_qps=" << (1000.0 * static_cast<double>(queries) / result.median_ms)
              << '\n';
  }
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
