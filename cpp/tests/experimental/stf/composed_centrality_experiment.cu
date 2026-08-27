/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "experimental/stf/iterative_graph_common.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/experimental/stf.cuh>
#include <thrust/count.h>
#include <thrust/reduce.h>

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

constexpr stf_result_t pagerank_alpha{0.85f};
constexpr stf_result_t convergence_epsilon{1.0e-5f};
constexpr int max_iterations{200};
constexpr std::size_t algorithm_count{3};

enum class algorithm : std::size_t { pagerank = 0, hits = 1, eigenvector = 2 };

constexpr std::array<std::string_view, 3> arm_names{
  "serial_host_loops", "concurrent_conditional_graphs", "composed_cuda_stf"};

// These kernels intentionally are test-local, capture-ready approximations rather than calls to
// the production cuGraph centrality APIs. PageRank uses the standard unweighted update; HITS uses
// two-phase L1-normalized authority/hub updates; eigenvector centrality uses an L1-normalized
// (A + I) power iteration. All arms below call exactly these allocation-free raw-CUDA iteration
// bodies.

struct centrality_state {
  centrality_state(std::size_t size, rmm::cuda_stream_view stream)
    : current(size, stream),
      next(size, stream),
      secondary(size, stream),
      secondary_next(size, stream),
      diff(1, stream),
      scalar0(1, stream),
      scalar1(1, stream),
      iteration(1, stream)
  {
  }

  rmm::device_uvector<stf_result_t> current;
  rmm::device_uvector<stf_result_t> next;
  rmm::device_uvector<stf_result_t> secondary;
  rmm::device_uvector<stf_result_t> secondary_next;
  rmm::device_uvector<stf_result_t> diff;
  rmm::device_uvector<stf_result_t> scalar0;
  rmm::device_uvector<stf_result_t> scalar1;
  rmm::device_uvector<int> iteration;
};

__global__ void initialize_centrality(stf_result_t* current,
                                      stf_result_t* next,
                                      stf_result_t* secondary,
                                      stf_result_t* secondary_next,
                                      stf_result_t* diff,
                                      stf_result_t* scalar0,
                                      stf_result_t* scalar1,
                                      int* iteration,
                                      std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    auto const initial = stf_result_t{1} / static_cast<stf_result_t>(size);
    current[i]         = initial;
    next[i]            = stf_result_t{0};
    secondary[i]       = initial;
    secondary_next[i]  = stf_result_t{0};
  }
  if (i == 0) {
    *diff      = stf_result_t{0};
    *scalar0   = stf_result_t{0};
    *scalar1   = stf_result_t{0};
    *iteration = 0;
  }
}

__global__ void clear_iteration_scalars(stf_result_t* diff,
                                        stf_result_t* scalar0,
                                        stf_result_t* scalar1)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *diff    = stf_result_t{0};
    *scalar0 = stf_result_t{0};
    *scalar1 = stf_result_t{0};
  }
}

__global__ void clear_values(stf_result_t* first, stf_result_t* second, std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    first[i]  = stf_result_t{0};
    second[i] = stf_result_t{0};
  }
}

__global__ void pagerank_step(stf_edge_t const* offsets,
                              stf_vertex_t const* indices,
                              stf_edge_t const* out_degrees,
                              stf_result_t const* current,
                              stf_result_t* next,
                              stf_result_t const* dangling_sum,
                              std::size_t size)
{
  auto v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= size) { return; }
  stf_result_t incoming{0};
  for (auto edge = offsets[v]; edge < offsets[v + 1]; ++edge) {
    auto const source = indices[edge];
    auto const degree = out_degrees[source];
    if (degree > 0) { incoming += current[source] / static_cast<stf_result_t>(degree); }
  }
  auto const uniform = stf_result_t{1} / static_cast<stf_result_t>(size);
  next[v]            = pagerank_alpha * incoming +
            (pagerank_alpha * (*dangling_sum) + (stf_result_t{1} - pagerank_alpha)) * uniform;
}

__global__ void hits_authority_step(stf_edge_t const* offsets,
                                    stf_vertex_t const* indices,
                                    stf_result_t const* hubs,
                                    stf_result_t* next_authorities,
                                    std::size_t size)
{
  auto v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= size) { return; }
  stf_result_t authority{0};
  for (auto edge = offsets[v]; edge < offsets[v + 1]; ++edge) {
    authority += hubs[indices[edge]];
  }
  next_authorities[v] = authority;
}

__global__ void hits_hub_step(stf_edge_t const* offsets,
                              stf_vertex_t const* indices,
                              stf_result_t const* next_authorities,
                              stf_result_t* next_hubs,
                              std::size_t size)
{
  auto destination = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (destination >= size) { return; }
  for (auto edge = offsets[destination]; edge < offsets[destination + 1]; ++edge) {
    atomicAdd(next_hubs + indices[edge], next_authorities[destination]);
  }
}

__global__ void eigenvector_step(stf_edge_t const* offsets,
                                 stf_vertex_t const* indices,
                                 stf_result_t const* current,
                                 stf_result_t* next,
                                 std::size_t size)
{
  auto v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= size) { return; }
  stf_result_t value{current[v]};
  for (auto edge = offsets[v]; edge < offsets[v + 1]; ++edge) {
    value += current[indices[edge]];
  }
  next[v] = value;
}

__global__ void sum_values(stf_result_t const* first,
                           stf_result_t const* second,
                           stf_result_t* first_sum,
                           stf_result_t* second_sum,
                           std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    atomicAdd(first_sum, first[i]);
    if (second != nullptr) { atomicAdd(second_sum, second[i]); }
  }
}

__global__ void normalize_commit(stf_result_t* current,
                                 stf_result_t const* next,
                                 stf_result_t* secondary,
                                 stf_result_t const* secondary_next,
                                 stf_result_t const* first_sum,
                                 stf_result_t const* second_sum,
                                 stf_result_t* max_diff,
                                 std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size) { return; }
  auto const first =
    (*first_sum > stf_result_t{0}) ? next[i] / (*first_sum) : stf_result_t{1} / size;
  atomic_max_nonnegative(max_diff, fabsf(first - current[i]));
  current[i] = first;
  if (secondary != nullptr) {
    auto const second =
      (*second_sum > stf_result_t{0}) ? secondary_next[i] / (*second_sum) : stf_result_t{1} / size;
    atomic_max_nonnegative(max_diff, fabsf(second - secondary[i]));
    secondary[i] = second;
  }
}

__global__ void update_condition(stf_result_t const* max_diff,
                                 int* iteration,
                                 cudaGraphConditionalHandle handle)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto const keep_going = (++(*iteration) < max_iterations) && (*max_diff >= convergence_epsilon);
    cudaGraphSetConditional(handle, keep_going ? 1u : 0u);
  }
}

struct continue_op {
  template <typename Diff, typename Iteration>
  __device__ bool operator()(Diff diff, Iteration iteration) const
  {
    return (++iteration(0) < max_iterations) && (diff(0) >= convergence_epsilon);
  }
};

void enqueue_initialization(cudaStream_t stream, centrality_state& state, std::size_t size)
{
  initialize_centrality<<<grid_size(size), block_size, 0, stream>>>(state.current.data(),
                                                                    state.next.data(),
                                                                    state.secondary.data(),
                                                                    state.secondary_next.data(),
                                                                    state.diff.data(),
                                                                    state.scalar0.data(),
                                                                    state.scalar1.data(),
                                                                    state.iteration.data(),
                                                                    size);
}

struct centrality_state_view {
  stf_result_t* current;
  stf_result_t* next;
  stf_result_t* secondary;
  stf_result_t* secondary_next;
  stf_result_t* diff;
  stf_result_t* scalar0;
  stf_result_t* scalar1;
};

centrality_state_view make_view(centrality_state& state)
{
  return {state.current.data(),
          state.next.data(),
          state.secondary.data(),
          state.secondary_next.data(),
          state.diff.data(),
          state.scalar0.data(),
          state.scalar1.data()};
}

void enqueue_iteration(cudaStream_t stream,
                       algorithm selected,
                       stf_edge_t const* offsets,
                       stf_vertex_t const* indices,
                       stf_edge_t const* out_degrees,
                       centrality_state_view state,
                       std::size_t size)
{
  clear_iteration_scalars<<<1, 1, 0, stream>>>(state.diff, state.scalar0, state.scalar1);
  if (selected == algorithm::pagerank) {
    dangling_sum_kernel<<<grid_size(size), block_size, 0, stream>>>(
      state.current, out_degrees, state.scalar0, size);
    pagerank_step<<<grid_size(size), block_size, 0, stream>>>(
      offsets, indices, out_degrees, state.current, state.next, state.scalar0, size);
    commit_and_find_max_diff<<<grid_size(size), block_size, 0, stream>>>(
      state.current, state.next, state.diff, size);
  } else if (selected == algorithm::hits) {
    clear_values<<<grid_size(size), block_size, 0, stream>>>(
      state.next, state.secondary_next, size);
    hits_authority_step<<<grid_size(size), block_size, 0, stream>>>(
      offsets, indices, state.secondary, state.next, size);
    hits_hub_step<<<grid_size(size), block_size, 0, stream>>>(
      offsets, indices, state.next, state.secondary_next, size);
    sum_values<<<grid_size(size), block_size, 0, stream>>>(
      state.next, state.secondary_next, state.scalar0, state.scalar1, size);
    normalize_commit<<<grid_size(size), block_size, 0, stream>>>(state.current,
                                                                 state.next,
                                                                 state.secondary,
                                                                 state.secondary_next,
                                                                 state.scalar0,
                                                                 state.scalar1,
                                                                 state.diff,
                                                                 size);
  } else {
    eigenvector_step<<<grid_size(size), block_size, 0, stream>>>(
      offsets, indices, state.current, state.next, size);
    sum_values<<<grid_size(size), block_size, 0, stream>>>(
      state.next, nullptr, state.scalar0, nullptr, size);
    normalize_commit<<<grid_size(size), block_size, 0, stream>>>(
      state.current, state.next, nullptr, nullptr, state.scalar0, nullptr, state.diff, size);
  }
}

void enqueue_iteration(cudaStream_t stream,
                       algorithm selected,
                       stf_edge_t const* offsets,
                       stf_vertex_t const* indices,
                       stf_edge_t const* out_degrees,
                       centrality_state& state,
                       std::size_t size)
{
  enqueue_iteration(stream, selected, offsets, indices, out_degrees, make_view(state), size);
}

int run_host_loop(cudaStream_t stream,
                  algorithm selected,
                  stf_edge_t const* offsets,
                  stf_vertex_t const* indices,
                  stf_edge_t const* out_degrees,
                  centrality_state& state,
                  std::size_t size)
{
  enqueue_initialization(stream, state, size);
  stf_result_t host_diff{};
  int iteration{};
  do {
    enqueue_iteration(stream, selected, offsets, indices, out_degrees, state, size);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      &host_diff, state.diff.data(), sizeof(host_diff), cudaMemcpyDeviceToHost, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ++iteration;
  } while (iteration < max_iterations && host_diff >= convergence_epsilon);
  RAFT_CUDA_TRY(cudaMemcpyAsync(
    state.iteration.data(), &iteration, sizeof(iteration), cudaMemcpyHostToDevice, stream));
  return iteration;
}

class conditional_centrality_graph {
 public:
  conditional_centrality_graph(algorithm selected,
                               stf_edge_t const* offsets,
                               stf_vertex_t const* indices,
                               stf_edge_t const* out_degrees,
                               centrality_state& state,
                               std::size_t size)
  {
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
    RAFT_CUDA_TRY(cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeGlobal));
    enqueue_initialization(capture_stream_, state, size);

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
    enqueue_iteration(body_stream, selected, offsets, indices, out_degrees, state, size);
    update_condition<<<1, 1, 0, body_stream>>>(
      state.diff.data(), state.iteration.data(), condition_);
    RAFT_CUDA_TRY(cudaStreamEndCapture(body_stream, nullptr));
    RAFT_CUDA_TRY(cudaStreamDestroy(body_stream));
    RAFT_CUDA_TRY(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0));
  }

  conditional_centrality_graph(conditional_centrality_graph const&)            = delete;
  conditional_centrality_graph& operator=(conditional_centrality_graph const&) = delete;

  ~conditional_centrality_graph()
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

class explicit_composed_runner {
 public:
  explicit_composed_runner(cudaStream_t caller_stream,
                           stf_edge_t const* offsets,
                           stf_vertex_t const* indices,
                           stf_edge_t const* out_degrees,
                           std::array<centrality_state, algorithm_count>& states,
                           std::size_t size)
    : caller_stream_(caller_stream),
      graphs_{conditional_centrality_graph{
                algorithm::pagerank, offsets, indices, out_degrees, states[0], size},
              conditional_centrality_graph{
                algorithm::hits, offsets, indices, out_degrees, states[1], size},
              conditional_centrality_graph{
                algorithm::eigenvector, offsets, indices, out_degrees, states[2], size}}
  {
    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&start_, cudaEventDisableTiming));
    for (std::size_t i = 0; i < algorithm_count; ++i) {
      RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
      RAFT_CUDA_TRY(cudaEventCreateWithFlags(&done_[i], cudaEventDisableTiming));
    }
  }

  ~explicit_composed_runner()
  {
    for (std::size_t i = 0; i < algorithm_count; ++i) {
      if (done_[i] != nullptr) { cudaEventDestroy(done_[i]); }
      if (streams_[i] != nullptr) { cudaStreamDestroy(streams_[i]); }
    }
    if (start_ != nullptr) { cudaEventDestroy(start_); }
  }

  void launch()
  {
    RAFT_CUDA_TRY(cudaEventRecord(start_, caller_stream_));
    for (std::size_t i = 0; i < algorithm_count; ++i) {
      RAFT_CUDA_TRY(cudaStreamWaitEvent(streams_[i], start_));
      graphs_[i].launch(streams_[i]);
      RAFT_CUDA_TRY(cudaEventRecord(done_[i], streams_[i]));
      RAFT_CUDA_TRY(cudaStreamWaitEvent(caller_stream_, done_[i]));
    }
  }

 private:
  cudaStream_t caller_stream_{};
  cudaEvent_t start_{};
  std::array<cudaStream_t, algorithm_count> streams_{};
  std::array<cudaEvent_t, algorithm_count> done_{};
  std::array<conditional_centrality_graph, algorithm_count> graphs_;
};

std::size_t mismatch_count(raft::handle_t const& handle,
                           rmm::device_uvector<stf_result_t> const& reference,
                           rmm::device_uvector<stf_result_t> const& candidate)
{
  return static_cast<std::size_t>(
    thrust::count_if(handle.get_thrust_policy(),
                     thrust::make_counting_iterator<std::size_t>(0),
                     thrust::make_counting_iterator(reference.size()),
                     [lhs = reference.data(), rhs = candidate.data()] __device__(std::size_t i) {
                       auto const difference = fabsf(lhs[i] - rhs[i]);
                       auto const magnitude  = fmaxf(fabsf(lhs[i]), fabsf(rhs[i]));
                       return difference > (2.0e-5f + 2.0e-4f * magnitude);
                     }));
}

std::size_t invalid_count(raft::handle_t const& handle,
                          rmm::device_uvector<stf_result_t> const& values)
{
  return static_cast<std::size_t>(thrust::count_if(
    handle.get_thrust_policy(), values.begin(), values.end(), [] __device__(stf_result_t value) {
      return !isfinite(value) || value < stf_result_t{0};
    }));
}

TEST(CudaStfComposedCentralityExperiment, IndependentDeviceConvergence)
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
  auto const vertices = static_cast<std::size_t>(graph_view.number_of_vertices());
  auto const edges    = static_cast<std::size_t>(edge_partition.number_of_edges());
  auto out_degrees    = graph_view.compute_out_degrees(handle);

  using state_array = std::array<centrality_state, algorithm_count>;
  std::array<state_array, 3> states{state_array{centrality_state{vertices, handle.get_stream()},
                                                centrality_state{vertices, handle.get_stream()},
                                                centrality_state{vertices, handle.get_stream()}},
                                    state_array{centrality_state{vertices, handle.get_stream()},
                                                centrality_state{vertices, handle.get_stream()},
                                                centrality_state{vertices, handle.get_stream()}},
                                    state_array{centrality_state{vertices, handle.get_stream()},
                                                centrality_state{vertices, handle.get_stream()},
                                                centrality_state{vertices, handle.get_stream()}}};

  explicit_composed_runner explicit_runner{
    stream, offsets, indices, out_degrees.data(), states[1], vertices};

  stf::stackable_ctx stf_context;
  auto l_offsets = stf_context.logical_data(stf::make_slice(offsets, vertices + 1),
                                            stf::data_place::current_device());
  auto l_indices =
    stf_context.logical_data(stf::make_slice(indices, edges), stf::data_place::current_device());
  auto l_out_degrees = stf_context.logical_data(stf::make_slice(out_degrees.data(), vertices),
                                                stf::data_place::current_device());

  struct logical_state {
    stf::stackable_logical_data<stf::slice<stf_result_t>> current;
    stf::stackable_logical_data<stf::slice<stf_result_t>> next;
    stf::stackable_logical_data<stf::slice<stf_result_t>> secondary;
    stf::stackable_logical_data<stf::slice<stf_result_t>> secondary_next;
    stf::stackable_logical_data<stf::slice<stf_result_t>> diff;
    stf::stackable_logical_data<stf::slice<stf_result_t>> scalar0;
    stf::stackable_logical_data<stf::slice<stf_result_t>> scalar1;
    stf::stackable_logical_data<stf::slice<int>> iteration;
  };
  std::vector<logical_state> logical_states;
  logical_states.reserve(algorithm_count);
  for (auto& state : states[2]) {
    logical_states.push_back(
      {stf_context.logical_data(stf::make_slice(state.current.data(), vertices),
                                stf::data_place::current_device()),
       stf_context.logical_data(stf::make_slice(state.next.data(), vertices),
                                stf::data_place::current_device()),
       stf_context.logical_data(stf::make_slice(state.secondary.data(), vertices),
                                stf::data_place::current_device()),
       stf_context.logical_data(stf::make_slice(state.secondary_next.data(), vertices),
                                stf::data_place::current_device()),
       stf_context.logical_data(stf::make_slice(state.diff.data(), std::size_t{1}),
                                stf::data_place::current_device()),
       stf_context.logical_data(stf::make_slice(state.scalar0.data(), std::size_t{1}),
                                stf::data_place::current_device()),
       stf_context.logical_data(stf::make_slice(state.scalar1.data(), std::size_t{1}),
                                stf::data_place::current_device()),
       stf_context.logical_data(stf::make_slice(state.iteration.data(), std::size_t{1}),
                                stf::data_place::current_device())});
  }

  stf::stackable_ctx::launchable_graph_scope stf_graph{stf_context};
  l_offsets.push(stf::access_mode::read, stf::data_place::current_device());
  l_indices.push(stf::access_mode::read, stf::data_place::current_device());
  l_out_degrees.push(stf::access_mode::read, stf::data_place::current_device());
  for (auto& state : logical_states) {
    state.current.push(stf::access_mode::rw, stf::data_place::current_device());
    state.next.push(stf::access_mode::rw, stf::data_place::current_device());
    state.secondary.push(stf::access_mode::rw, stf::data_place::current_device());
    state.secondary_next.push(stf::access_mode::rw, stf::data_place::current_device());
    state.diff.push(stf::access_mode::rw, stf::data_place::current_device());
    state.scalar0.push(stf::access_mode::rw, stf::data_place::current_device());
    state.scalar1.push(stf::access_mode::rw, stf::data_place::current_device());
    state.iteration.push(stf::access_mode::rw, stf::data_place::current_device());
    stf_context.task(state.current.write(),
                     state.next.write(),
                     state.secondary.write(),
                     state.secondary_next.write(),
                     state.diff.write(),
                     state.scalar0.write(),
                     state.scalar1.write(),
                     state.iteration.write())
        ->*[vertices](cudaStream_t task_stream,
                      auto current,
                      auto next,
                      auto secondary,
                      auto secondary_next,
                      auto diff,
                      auto scalar0,
                      auto scalar1,
                      auto iteration) {
              initialize_centrality<<<grid_size(vertices), block_size, 0, task_stream>>>(
                current.data_handle(),
                next.data_handle(),
                secondary.data_handle(),
                secondary_next.data_handle(),
                diff.data_handle(),
                scalar0.data_handle(),
                scalar1.data_handle(),
                iteration.data_handle(),
                vertices);
            };
  }

  for (std::size_t index = 0; index < algorithm_count; ++index) {
    auto selected = static_cast<algorithm>(index);
    auto& state   = logical_states[index];
    auto loop     = stf_context.while_graph_scope(1);
    stf_context.task(l_offsets.read(),
                     l_indices.read(),
                     l_out_degrees.read(),
                     state.current.rw(),
                     state.next.rw(),
                     state.secondary.rw(),
                     state.secondary_next.rw(),
                     state.diff.rw(),
                     state.scalar0.rw(),
                     state.scalar1.rw())
        ->*[selected, vertices](cudaStream_t task_stream,
                                auto task_offsets,
                                auto task_indices,
                                auto task_out_degrees,
                                auto current,
                                auto next,
                                auto secondary,
                                auto secondary_next,
                                auto diff,
                                auto scalar0,
                                auto scalar1) {
              enqueue_iteration(task_stream,
                                selected,
                                task_offsets.data_handle(),
                                task_indices.data_handle(),
                                task_out_degrees.data_handle(),
                                centrality_state_view{current.data_handle(),
                                                      next.data_handle(),
                                                      secondary.data_handle(),
                                                      secondary_next.data_handle(),
                                                      diff.data_handle(),
                                                      scalar0.data_handle(),
                                                      scalar1.data_handle()},
                                vertices);
            };
    loop.update_cond(state.diff.read(), state.iteration.rw())->*continue_op{};
  }
  stf_graph.exec();

  auto run_serial = [&] {
    for (std::size_t i = 0; i < algorithm_count; ++i) {
      run_host_loop(stream,
                    static_cast<algorithm>(i),
                    offsets,
                    indices,
                    out_degrees.data(),
                    states[0][i],
                    vertices);
    }
  };

  run_serial();
  explicit_runner.launch();
  stf_graph.launch();
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  std::array<std::vector<double>, 3> samples;
  auto const repetitions = experiment_repetitions();
  for (auto& arm_samples : samples) {
    arm_samples.reserve(repetitions);
  }
  for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
    for (std::size_t offset = 0; offset < arm_names.size(); ++offset) {
      auto const arm = (repeat + offset) % arm_names.size();
      if (arm == 0) {
        samples[arm].push_back(wall_time_ms(run_serial));
      } else if (arm == 1) {
        samples[arm].push_back(wall_time_ms([&] { explicit_runner.launch(); }));
      } else {
        samples[arm].push_back(wall_time_ms([&] { stf_graph.launch(); }));
      }
    }
  }

  std::array<std::array<int, algorithm_count>, 3> iterations{};
  for (std::size_t arm = 0; arm < states.size(); ++arm) {
    for (std::size_t selected = 0; selected < algorithm_count; ++selected) {
      RAFT_CUDA_TRY(cudaMemcpy(&iterations[arm][selected],
                               states[arm][selected].iteration.data(),
                               sizeof(int),
                               cudaMemcpyDeviceToHost));
      EXPECT_EQ(mismatch_count(handle, states[0][selected].current, states[arm][selected].current),
                std::size_t{0});
      EXPECT_EQ(invalid_count(handle, states[arm][selected].current), std::size_t{0});
      auto sum = thrust::reduce(handle.get_thrust_policy(),
                                states[arm][selected].current.begin(),
                                states[arm][selected].current.end(),
                                stf_result_t{0});
      EXPECT_NEAR(sum, stf_result_t{1}, stf_result_t{2.0e-3});
      if (selected == static_cast<std::size_t>(algorithm::hits)) {
        EXPECT_EQ(
          mismatch_count(handle, states[0][selected].secondary, states[arm][selected].secondary),
          std::size_t{0});
        EXPECT_EQ(invalid_count(handle, states[arm][selected].secondary), std::size_t{0});
        auto hub_sum = thrust::reduce(handle.get_thrust_policy(),
                                      states[arm][selected].secondary.begin(),
                                      states[arm][selected].secondary.end(),
                                      stf_result_t{0});
        EXPECT_NEAR(hub_sum, stf_result_t{1}, stf_result_t{2.0e-3});
      }
    }
  }
  for (std::size_t selected = 0; selected < algorithm_count; ++selected) {
    EXPECT_EQ(iterations[0][selected], iterations[1][selected]);
    EXPECT_EQ(iterations[0][selected], iterations[2][selected]);
  }

  for (std::size_t arm = 0; arm < arm_names.size(); ++arm) {
    auto result = summarize(std::move(samples[arm]));
    std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
              << " algorithm=composed_centrality"
              << " arm=" << arm_names[arm] << " median_ms=" << result.median_ms
              << " min_ms=" << result.min_ms << " pagerank_iterations=" << iterations[arm][0]
              << " hits_iterations=" << iterations[arm][1]
              << " eigenvector_iterations=" << iterations[arm][2] << " vertices=" << vertices
              << " edges=" << edges << '\n';
  }

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
