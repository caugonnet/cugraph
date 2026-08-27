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
#include <rmm/exec_policy.hpp>

#include <cuda/experimental/stf.cuh>
#include <thrust/count.h>
#include <thrust/extrema.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using namespace cugraph::test::experimental;
namespace stf = cuda::experimental::stf;

constexpr stf_result_t beta{1.0f};
constexpr stf_result_t epsilon{1.0e-6f};
constexpr int max_iterations{100};

struct katz_continue_op {
  template <typename Diff, typename Iteration>
  __device__ bool operator()(Diff diff, Iteration iteration) const
  {
    return (++iteration(0) < max_iterations) && (diff(0) >= epsilon);
  }
};

struct katz_state {
  explicit katz_state(std::size_t size, rmm::cuda_stream_view stream)
    : current(size, stream), next(size, stream), diff(1, stream), iteration(1, stream)
  {
  }

  rmm::device_uvector<stf_result_t> current;
  rmm::device_uvector<stf_result_t> next;
  rmm::device_uvector<stf_result_t> diff;
  rmm::device_uvector<int> iteration;
};

void enqueue_initialization(cudaStream_t stream, katz_state& state)
{
  initialize_values<<<grid_size(state.current.size()), block_size, 0, stream>>>(
    state.current.data(), state.next.data(), state.current.size(), stf_result_t{0});
  initialize_scalar<<<1, 1, 0, stream>>>(state.diff.data(), state.iteration.data());
}

int run_host_loop(cudaStream_t stream,
                  stf_edge_t const* offsets,
                  stf_vertex_t const* indices,
                  std::size_t size,
                  stf_result_t alpha,
                  katz_state& state)
{
  enqueue_initialization(stream, state);
  stf_result_t host_diff{};
  int iteration{};
  do {
    enqueue_katz_iteration(stream,
                           offsets,
                           indices,
                           state.current.data(),
                           state.next.data(),
                           state.diff.data(),
                           size,
                           alpha,
                           beta);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      &host_diff, state.diff.data(), sizeof(host_diff), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ++iteration;
  } while (iteration < max_iterations && host_diff >= epsilon);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    state.iteration.data(), &iteration, sizeof(iteration), cudaMemcpyHostToDevice, stream));
  return iteration;
}

class conditional_katz_graph {
 public:
  conditional_katz_graph(stf_edge_t const* offsets,
                         stf_vertex_t const* indices,
                         std::size_t size,
                         stf_result_t alpha,
                         katz_state& state)
  {
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
    RAFT_CUDA_TRY(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));
    enqueue_initialization(capture_stream_, state);

    cudaStreamCaptureStatus status{};
    cudaGraph_t captured_graph{};
    cudaGraphNode_t const* dependencies{};
    std::size_t dependency_count{};
    RAFT_CUDA_TRY(cudaStreamGetCaptureInfo(capture_stream_,
                                           &status,
                                           nullptr,
                                           &captured_graph,
                                           &dependencies,
                                           nullptr,
                                           &dependency_count));
    RAFT_CUDA_TRY(cudaGraphConditionalHandleCreate(&condition_, captured_graph));
    set_graph_condition<<<1, 1, 0, capture_stream_>>>(condition_, 1u);

    RAFT_CUDA_TRY(cudaStreamGetCaptureInfo(capture_stream_,
                                           &status,
                                           nullptr,
                                           &captured_graph,
                                           &dependencies,
                                           nullptr,
                                           &dependency_count));
    cudaGraphNodeParams parameters{};
    parameters.type                    = cudaGraphNodeTypeConditional;
    parameters.conditional.handle      = condition_;
    parameters.conditional.type        = cudaGraphCondTypeWhile;
    parameters.conditional.size        = 1;
    parameters.conditional.phGraph_out = nullptr;
    RAFT_CUDA_TRY(cudaGraphAddNode(
      &conditional_node_, captured_graph, dependencies, nullptr, dependency_count, &parameters));
    body_graph_ = parameters.conditional.phGraph_out[0];
    RAFT_CUDA_TRY(cudaStreamUpdateCaptureDependencies(
      capture_stream_, &conditional_node_, nullptr, 1, cudaStreamSetCaptureDependencies));
    RAFT_CUDA_TRY(cudaStreamEndCapture(capture_stream_, &graph_));

    cudaStream_t body_stream{};
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&body_stream, cudaStreamNonBlocking));
    RAFT_CUDA_TRY(cudaStreamBeginCaptureToGraph(
      body_stream, body_graph_, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
    enqueue_katz_iteration(body_stream,
                           offsets,
                           indices,
                           state.current.data(),
                           state.next.data(),
                           state.diff.data(),
                           size,
                           alpha,
                           beta);
    update_katz_condition<<<1, 1, 0, body_stream>>>(
      state.diff.data(), state.iteration.data(), epsilon, max_iterations, condition_);
    RAFT_CUDA_TRY(cudaStreamEndCapture(body_stream, nullptr));
    RAFT_CUDA_TRY(cudaStreamDestroy(body_stream));
    RAFT_CUDA_TRY(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0));
  }

  conditional_katz_graph(conditional_katz_graph const&)            = delete;
  conditional_katz_graph& operator=(conditional_katz_graph const&) = delete;

  ~conditional_katz_graph()
  {
    if (executable_ != nullptr) { cudaGraphExecDestroy(executable_); }
    if (graph_ != nullptr) { cudaGraphDestroy(graph_); }
    if (capture_stream_ != nullptr) { cudaStreamDestroy(capture_stream_); }
  }

  void launch(cudaStream_t stream) { RAFT_CUDA_TRY(cudaGraphLaunch(executable_, stream)); }

 private:
  cudaStream_t capture_stream_{};
  cudaGraph_t graph_{};
  cudaGraph_t body_graph_{};
  cudaGraphExec_t executable_{};
  cudaGraphNode_t conditional_node_{};
  cudaGraphConditionalHandle condition_{};
};

std::size_t mismatch_count(raft::handle_t const& handle,
                           rmm::device_uvector<stf_result_t> const& reference,
                           rmm::device_uvector<stf_result_t> const& candidate,
                           stf_result_t absolute_tolerance = 1.0e-6f,
                           stf_result_t relative_tolerance = 1.0e-5f)
{
  return static_cast<std::size_t>(
    thrust::count_if(handle.get_thrust_policy(),
                     thrust::make_counting_iterator<std::size_t>(0),
                     thrust::make_counting_iterator(reference.size()),
                     [lhs = reference.data(),
                      rhs = candidate.data(),
                      absolute_tolerance,
                      relative_tolerance] __device__(std::size_t i) {
                       auto const absolute = fabsf(lhs[i] - rhs[i]);
                       auto const scale    = fmaxf(fabsf(lhs[i]), fabsf(rhs[i]));
                       return absolute > (absolute_tolerance + relative_tolerance * scale);
                     }));
}

TEST(CudaStfConditionalKatzExperiment, DeviceControlledIteration)
{
  raft::handle_t handle{};
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
  auto edges          = static_cast<std::size_t>(edge_partition.number_of_edges());
  auto size           = static_cast<std::size_t>(graph_view.number_of_vertices());
  std::vector<stf_edge_t> host_offsets(size + 1);
  RAFT_CUDA_TRY(cudaMemcpy(host_offsets.data(),
                           offsets,
                           host_offsets.size() * sizeof(stf_edge_t),
                           cudaMemcpyDeviceToHost));
  stf_edge_t max_degree{};
  for (std::size_t vertex = 0; vertex < size; ++vertex) {
    max_degree = std::max(max_degree, host_offsets[vertex + 1] - host_offsets[vertex]);
  }
  auto alpha = stf_result_t{0.9f} / static_cast<stf_result_t>(max_degree + 1);

  std::array<katz_state, 3> states{katz_state{size, handle.get_stream()},
                                   katz_state{size, handle.get_stream()},
                                   katz_state{size, handle.get_stream()}};

  conditional_katz_graph manual_graph{offsets, indices, size, alpha, states[1]};

  stf::stackable_ctx stf_context;
  auto l_offsets =
    stf_context.logical_data(stf::make_slice(offsets, size + 1), stf::data_place::current_device());
  auto l_indices =
    stf_context.logical_data(stf::make_slice(indices, edges), stf::data_place::current_device());
  auto l_current = stf_context.logical_data(stf::make_slice(states[2].current.data(), size),
                                            stf::data_place::current_device());
  auto l_next    = stf_context.logical_data(stf::make_slice(states[2].next.data(), size),
                                         stf::data_place::current_device());
  auto l_diff    = stf_context.logical_data(stf::make_slice(states[2].diff.data(), std::size_t{1}),
                                         stf::data_place::current_device());
  auto l_iteration = stf_context.logical_data(
    stf::make_slice(states[2].iteration.data(), std::size_t{1}), stf::data_place::current_device());

  std::vector<double> host_samples;
  std::vector<double> manual_samples;
  std::vector<double> stf_samples;
  auto const repetitions = experiment_repetitions();
  host_samples.reserve(repetitions);
  manual_samples.reserve(repetitions);
  stf_samples.reserve(repetitions);

  stf::stackable_ctx::launchable_graph_scope stf_graph{stf_context};
  l_offsets.push(stf::access_mode::read, stf::data_place::current_device());
  l_indices.push(stf::access_mode::read, stf::data_place::current_device());
  l_current.push(stf::access_mode::rw, stf::data_place::current_device());
  l_next.push(stf::access_mode::rw, stf::data_place::current_device());
  l_diff.push(stf::access_mode::rw, stf::data_place::current_device());
  l_iteration.push(stf::access_mode::rw, stf::data_place::current_device());
  stf_context.task(l_current.write(), l_next.write(), l_diff.write(), l_iteration.write())
      ->*
    [size](cudaStream_t task_stream, auto current, auto next, auto diff, auto iteration) {
      initialize_values<<<grid_size(size), block_size, 0, task_stream>>>(
        current.data_handle(), next.data_handle(), size, stf_result_t{0});
      initialize_scalar<<<1, 1, 0, task_stream>>>(diff.data_handle(), iteration.data_handle());
    };
  {
    auto loop = stf_context.while_graph_scope(1);
    stf_context.task(l_offsets.read(), l_indices.read(), l_current.rw(), l_next.rw(), l_diff.rw())
        ->*[size, alpha](cudaStream_t task_stream,
                         auto task_offsets,
                         auto task_indices,
                         auto current,
                         auto next,
                         auto diff) {
              enqueue_katz_iteration(task_stream,
                                     task_offsets.data_handle(),
                                     task_indices.data_handle(),
                                     current.data_handle(),
                                     next.data_handle(),
                                     diff.data_handle(),
                                     size,
                                     alpha,
                                     beta);
            };
    loop.update_cond(l_diff.read(), l_iteration.rw())->*katz_continue_op{};
  }
  stf_graph.exec();

  run_host_loop(stream, offsets, indices, size, alpha, states[0]);
  manual_graph.launch(stream);
  stf_graph.launch();
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
    for (std::size_t offset = 0; offset < 3; ++offset) {
      auto arm = (repeat + offset) % 3;
      if (arm == 0) {
        host_samples.push_back(
          wall_time_ms([&] { run_host_loop(stream, offsets, indices, size, alpha, states[0]); }));
      } else if (arm == 1) {
        manual_samples.push_back(wall_time_ms([&] { manual_graph.launch(stream); }));
      } else {
        stf_samples.push_back(wall_time_ms([&] { stf_graph.launch(); }));
      }
    }
  }

  EXPECT_EQ(mismatch_count(handle, states[0].current, states[1].current), std::size_t{0});
  EXPECT_EQ(mismatch_count(handle, states[0].current, states[2].current), std::size_t{0});
  rmm::device_uvector<stf_result_t> cugraph_reference(size, handle.get_stream());
  cugraph::katz_centrality<stf_vertex_t, stf_edge_t, stf_result_t, stf_result_t, false>(
    handle,
    graph_view,
    std::optional<cugraph::edge_property_view_t<stf_edge_t, stf_result_t const*>>{std::nullopt},
    nullptr,
    cugraph_reference.data(),
    alpha,
    beta,
    epsilon,
    max_iterations,
    false,
    false,
    false);
  EXPECT_EQ(
    mismatch_count(
      handle, cugraph_reference, states[0].current, stf_result_t{1.0e-5}, stf_result_t{1.0e-4}),
    std::size_t{0});

  std::array<int, 3> iterations{};
  for (std::size_t i = 0; i < states.size(); ++i) {
    RAFT_CUDA_TRY(
      cudaMemcpy(&iterations[i], states[i].iteration.data(), sizeof(int), cudaMemcpyDeviceToHost));
  }
  EXPECT_EQ(iterations[0], iterations[1]);
  EXPECT_EQ(iterations[0], iterations[2]);

  auto host_timing   = summarize(std::move(host_samples));
  auto manual_timing = summarize(std::move(manual_samples));
  auto stf_timing    = summarize(std::move(stf_samples));
  std::array<std::string_view, 3> names{"host_loop", "manual_conditional_graph", "cuda_stf"};
  std::array<timing, 3> timings{host_timing, manual_timing, stf_timing};
  for (std::size_t i = 0; i < names.size(); ++i) {
    std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
              << " algorithm=katz_conditional"
              << " arm=" << names[i] << " median_ms=" << timings[i].median_ms
              << " min_ms=" << timings[i].min_ms << " iterations=" << iterations[i]
              << " vertices=" << size << " edges=" << edges << '\n';
  }

  // Ensure all imported allocations remain alive until the popped graph is released.
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
