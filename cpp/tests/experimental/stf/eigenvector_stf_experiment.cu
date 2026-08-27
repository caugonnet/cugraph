/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "experimental/stf/iterative_graph_common.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_src_dst_property.hpp>
#include <cugraph/prims/per_v_transform_reduce_incoming_outgoing_e.cuh>
#include <cugraph/prims/reduce_op.cuh>
#include <cugraph/prims/update_edge_src_dst_property.cuh>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_reduce.cuh>
#include <cuda/experimental/stf.cuh>
#include <thrust/count.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

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

constexpr stf_result_t epsilon{1.0e-6f};
constexpr int max_iterations{500};

// Asynchrony-only STF spelling of cuGraph's eigenvector centrality (the
// (A + I) power iteration in centrality/eigenvector_centrality_impl.cuh):
// tasks over the production per_v_transform_reduce_incoming_e traversal, CUB
// reductions with caller-owned workspaces, and raw kernels. Three arms:
//
// - the production public API (per-call allocation, host-controlled loop);
// - a host loop over the same preallocated kernels (isolates what
//   preallocation alone buys);
// - the STF while_graph_scope (adds device-side iteration and replay).

struct eigenvector_state {
  eigenvector_state(std::size_t size, rmm::cuda_stream_view stream)
    : current(size, stream),
      old(size, stream),
      norm_sq(1, stream),
      difference_sum(1, stream),
      iteration(1, stream)
  {
  }

  rmm::device_uvector<stf_result_t> current;
  rmm::device_uvector<stf_result_t> old;
  rmm::device_uvector<stf_result_t> norm_sq;
  rmm::device_uvector<stf_result_t> difference_sum;
  rmm::device_uvector<int> iteration;
};

__global__ void initialize_centralities_kernel(stf_result_t* current,
                                               std::size_t size,
                                               stf_result_t initial)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) { current[i] = initial; }
}

__global__ void initialize_iteration_kernel(int* iteration)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { *iteration = 0; }
}

__global__ void add_values_kernel(stf_result_t* current, stf_result_t const* old, std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) { current[i] += old[i]; }
}

__global__ void normalize_by_l2_kernel(stf_result_t* current,
                                       stf_result_t const* norm_sq,
                                       std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) { current[i] /= sqrtf(*norm_sq); }
}

// Ordinary CUB transform-reduce inputs; library usage, not STF codegen.
struct square_op {
  stf_result_t const* values{};

  __device__ stf_result_t operator()(std::size_t i) const { return values[i] * values[i]; }
};

struct absolute_difference_op {
  stf_result_t const* current{};
  stf_result_t const* old{};

  __device__ stf_result_t operator()(std::size_t i) const { return fabsf(current[i] - old[i]); }
};

template <typename Op>
auto make_reduce_input(Op op)
{ return thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), op); }

struct eigenvector_continue_op {
  stf_result_t threshold;

  template <typename Difference, typename Iteration>
  __device__ bool operator()(Difference difference, Iteration iteration) const
  {
    ++(*iteration);
    return (*difference >= threshold) && (*iteration < max_iterations);
  }
};

struct source_value_op {
  template <typename Src, typename Dst, typename SrcValue, typename DstValue, typename EdgeValue>
  __device__ stf_result_t operator()(Src, Dst, SrcValue src_value, DstValue, EdgeValue) const
  { return src_value; }
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
                       auto const absolute = fabsf(lhs[i] - rhs[i]);
                       auto const scale    = fmaxf(fabsf(lhs[i]), fabsf(rhs[i]));
                       return absolute > (2.0e-6f + 2.0e-4f * scale);
                     }));
}

TEST(CudaStfEigenvectorExperiment, CapturedProductionTraversal)
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
  auto const vertices = static_cast<std::size_t>(graph_view.number_of_vertices());
  auto const edges    = static_cast<std::size_t>(edge_partition.number_of_edges());

  auto const initial   = stf_result_t{1} / static_cast<stf_result_t>(vertices);
  auto const threshold = static_cast<stf_result_t>(vertices) * epsilon;

  // Separate state per non-production arm so iteration counts and outputs are
  // independently checkable; the edge source property is shared because arms
  // never run concurrently.
  eigenvector_state host_state{vertices, handle.get_stream()};
  eigenvector_state stf_state{vertices, handle.get_stream()};
  cugraph::edge_src_property_t<stf_vertex_t, stf_result_t> edge_src_centralities(handle,
                                                                                 graph_view);

  // Caller-owned CUB workspaces, sized before graph construction.
  std::size_t norm_temp_bytes{};
  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(nullptr,
                                       norm_temp_bytes,
                                       make_reduce_input(square_op{}),
                                       stf_state.norm_sq.data(),
                                       vertices,
                                       handle.get_stream()));
  std::size_t difference_temp_bytes{};
  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(nullptr,
                                       difference_temp_bytes,
                                       make_reduce_input(absolute_difference_op{}),
                                       stf_state.difference_sum.data(),
                                       vertices,
                                       handle.get_stream()));
  rmm::device_uvector<std::byte> norm_temp(norm_temp_bytes, handle.get_stream());
  rmm::device_uvector<std::byte> difference_temp(difference_temp_bytes, handle.get_stream());

  auto enqueue_iteration = [&](cudaStream_t stream, eigenvector_state& state) {
    copy_values<<<grid_size(vertices), block_size, 0, stream>>>(
      state.old.data(), state.current.data(), vertices);
    raft::handle_t step_handle{rmm::cuda_stream_view{stream}};
    cugraph::update_edge_src_property(
      step_handle, graph_view, state.old.data(), edge_src_centralities.mutable_view());
    cugraph::per_v_transform_reduce_incoming_e(step_handle,
                                               graph_view,
                                               edge_src_centralities.view(),
                                               cugraph::edge_dst_dummy_property_t{}.view(),
                                               cugraph::edge_dummy_property_t{}.view(),
                                               source_value_op{},
                                               stf_result_t{0},
                                               cugraph::reduce_op::plus<stf_result_t>{},
                                               state.current.data());
    add_values_kernel<<<grid_size(vertices), block_size, 0, stream>>>(
      state.current.data(), state.old.data(), vertices);
    auto norm_bytes = norm_temp_bytes;
    RAFT_CUDA_TRY(cub::DeviceReduce::Sum(norm_temp.data(),
                                         norm_bytes,
                                         make_reduce_input(square_op{state.current.data()}),
                                         state.norm_sq.data(),
                                         vertices,
                                         stream));
    normalize_by_l2_kernel<<<grid_size(vertices), block_size, 0, stream>>>(
      state.current.data(), state.norm_sq.data(), vertices);
    auto difference_bytes = difference_temp_bytes;
    RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
      difference_temp.data(),
      difference_bytes,
      make_reduce_input(absolute_difference_op{state.current.data(), state.old.data()}),
      state.difference_sum.data(),
      vertices,
      stream));
  };

  auto run_host_loop = [&](cudaStream_t stream) {
    initialize_centralities_kernel<<<grid_size(vertices), block_size, 0, stream>>>(
      host_state.current.data(), vertices, initial);
    stf_result_t host_difference{};
    int iteration{};
    do {
      enqueue_iteration(stream, host_state);
      RAFT_CUDA_TRY(cudaMemcpyAsync(&host_difference,
                                    host_state.difference_sum.data(),
                                    sizeof(host_difference),
                                    cudaMemcpyDeviceToHost,
                                    stream));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      ++iteration;
    } while (host_difference >= threshold && iteration < max_iterations);
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      host_state.iteration.data(), &iteration, sizeof(iteration), cudaMemcpyHostToDevice, stream));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  };

  stf::stackable_ctx context;
  auto l_current = context.logical_data(stf::make_slice(stf_state.current.data(), vertices),
                                        stf::data_place::current_device());
  auto l_old     = context.logical_data(stf::make_slice(stf_state.old.data(), vertices),
                                        stf::data_place::current_device());
  auto l_norm    = context.logical_data(stf::scalar_view<stf_result_t>{stf_state.norm_sq.data()},
                                        stf::data_place::current_device());
  auto l_difference =
    context.logical_data(stf::scalar_view<stf_result_t>{stf_state.difference_sum.data()},
                         stf::data_place::current_device());
  auto l_iteration = context.logical_data(stf::scalar_view<int>{stf_state.iteration.data()},
                                          stf::data_place::current_device());
  stf::stackable_ctx::launchable_graph_scope graph_scope{context};
  l_current.push(stf::access_mode::rw, stf::data_place::current_device());
  l_old.push(stf::access_mode::rw, stf::data_place::current_device());
  l_norm.push(stf::access_mode::rw, stf::data_place::current_device());
  l_difference.push(stf::access_mode::rw, stf::data_place::current_device());
  l_iteration.push(stf::access_mode::rw, stf::data_place::current_device());

  context.task(l_current.write(), l_iteration.write()).set_symbol("initialize state")
      ->*[vertices, initial](cudaStream_t task_stream, auto current, auto iteration) {
            initialize_centralities_kernel<<<grid_size(vertices), block_size, 0, task_stream>>>(
              current.data_handle(), vertices, initial);
            initialize_iteration_kernel<<<1, 1, 0, task_stream>>>(iteration.addr);
          };

  {
    auto loop = context.while_graph_scope(1);
    context.task(l_old.write(), l_current.read()).set_symbol("copy to old")
        ->*[vertices](cudaStream_t task_stream, auto old, auto current) {
              copy_values<<<grid_size(vertices), block_size, 0, task_stream>>>(
                old.data_handle(), current.data_handle(), vertices);
            };
    context.task(l_old.read(), l_current.write()).set_symbol("optimized incoming traversal")
        ->*
      [graph_view, &edge_src_centralities, vertices](
        cudaStream_t task_stream, auto old, auto current) {
        raft::handle_t task_handle{rmm::cuda_stream_view{task_stream}};
        cugraph::update_edge_src_property(
          task_handle, graph_view, old.data_handle(), edge_src_centralities.mutable_view());
        cugraph::per_v_transform_reduce_incoming_e(task_handle,
                                                   graph_view,
                                                   edge_src_centralities.view(),
                                                   cugraph::edge_dst_dummy_property_t{}.view(),
                                                   cugraph::edge_dummy_property_t{}.view(),
                                                   source_value_op{},
                                                   stf_result_t{0},
                                                   cugraph::reduce_op::plus<stf_result_t>{},
                                                   current.data_handle());
      };
    context.task(l_current.rw(), l_old.read()).set_symbol("add identity term")
        ->*[vertices](cudaStream_t task_stream, auto current, auto old) {
              add_values_kernel<<<grid_size(vertices), block_size, 0, task_stream>>>(
                current.data_handle(), old.data_handle(), vertices);
            };
    context.task(l_current.read(), l_norm.write()).set_symbol("norm (cub)")
        ->*
      [temp = norm_temp.data(), temp_bytes = norm_temp_bytes, vertices](
        cudaStream_t task_stream, auto current, auto norm) {
        auto bytes = temp_bytes;
        RAFT_CUDA_TRY(cub::DeviceReduce::Sum(temp,
                                             bytes,
                                             make_reduce_input(square_op{current.data_handle()}),
                                             norm.addr,
                                             vertices,
                                             task_stream));
      };
    context.task(l_current.rw(), l_norm.read()).set_symbol("normalize")
        ->*[vertices](cudaStream_t task_stream, auto current, auto norm) {
              normalize_by_l2_kernel<<<grid_size(vertices), block_size, 0, task_stream>>>(
                current.data_handle(), norm.addr, vertices);
            };
    context.task(l_current.read(), l_old.read(), l_difference.write())
        .set_symbol("difference (cub)")
        ->*[temp = difference_temp.data(), temp_bytes = difference_temp_bytes, vertices](
             cudaStream_t task_stream, auto current, auto old, auto difference) {
              auto bytes = temp_bytes;
              RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
                temp,
                bytes,
                make_reduce_input(absolute_difference_op{current.data_handle(), old.data_handle()}),
                difference.addr,
                vertices,
                task_stream));
            };
    loop.update_cond(l_difference.read(), l_iteration.rw())->*eigenvector_continue_op{threshold};
  }
  graph_scope.exec();

  rmm::device_uvector<stf_result_t> reference(vertices, handle.get_stream());
  auto run_reference = [&] {
    reference = cugraph::eigenvector_centrality<stf_vertex_t, stf_edge_t, stf_result_t, false>(
      handle,
      graph_view,
      std::optional<cugraph::edge_property_view_t<stf_edge_t, stf_result_t const*>>{std::nullopt},
      std::optional<raft::device_span<stf_result_t const>>{std::nullopt},
      epsilon,
      static_cast<std::size_t>(max_iterations),
      false);
  };

  auto stream = static_cast<cudaStream_t>(handle.get_stream());
  run_reference();
  run_host_loop(stream);
  graph_scope.launch();
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  std::array<std::vector<double>, 3> samples;
  auto const repetitions = experiment_repetitions();
  for (auto& values : samples) {
    values.reserve(repetitions);
  }
  for (std::size_t repeat = 0; repeat < repetitions; ++repeat) {
    for (std::size_t offset = 0; offset < samples.size(); ++offset) {
      auto arm = (repeat + offset) % samples.size();
      if (arm == 0) {
        samples[arm].push_back(wall_time_ms(run_reference));
      } else if (arm == 1) {
        samples[arm].push_back(wall_time_ms([&] { run_host_loop(stream); }));
      } else {
        samples[arm].push_back(wall_time_ms([&] { graph_scope.launch(); }));
      }
    }
  }

  auto const host_mismatches = mismatch_count(handle, reference, host_state.current);
  auto const stf_mismatches  = mismatch_count(handle, reference, stf_state.current);
  EXPECT_EQ(host_mismatches, std::size_t{0});
  EXPECT_EQ(stf_mismatches, std::size_t{0});
  std::array<int, 2> iterations{};
  RAFT_CUDA_TRY(
    cudaMemcpy(&iterations[0], host_state.iteration.data(), sizeof(int), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(
    cudaMemcpy(&iterations[1], stf_state.iteration.data(), sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(iterations[0], iterations[1]);
  EXPECT_GT(iterations[1], 0);

  constexpr std::array<std::string_view, 3> names{
    "production_cugraph", "preallocated_host_loop", "captured_cuda_stf"};
  for (std::size_t arm = 0; arm < names.size(); ++arm) {
    auto result = summarize(std::move(samples[arm]));
    std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
              << " algorithm=eigenvector_centrality"
              << " arm=" << names[arm] << " median_ms=" << result.median_ms
              << " min_ms=" << result.min_ms << " iterations=" << iterations[1]
              << " host_mismatches=" << host_mismatches << " stf_mismatches=" << stf_mismatches
              << " vertices=" << vertices << " edges=" << edges << '\n';
  }

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
