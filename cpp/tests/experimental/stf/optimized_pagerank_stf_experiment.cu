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

// This experiment restricts CUDASTF to its asynchrony facilities: tasks over
// existing cuGraph, CUB, and plain CUDA kernels, logical-data dependencies,
// and conditional graph scopes. No STF kernel-generation facility
// (parallel_for/launch or reduce access modes) is used; update_cond is the one
// accepted boundary case because it replaces hand-written
// cudaGraphConditionalHandle plumbing rather than compute.

struct optimized_pagerank_state {
  optimized_pagerank_state(std::size_t size, rmm::cuda_stream_view stream)
    : current(size, stream),
      normalized(size, stream),
      next(size, stream),
      dangling_sum(1, stream),
      difference_sum(1, stream),
      iteration(1, stream)
  {
  }

  rmm::device_uvector<stf_result_t> current;
  rmm::device_uvector<stf_result_t> normalized;
  rmm::device_uvector<stf_result_t> next;
  rmm::device_uvector<stf_result_t> dangling_sum;
  rmm::device_uvector<stf_result_t> difference_sum;
  rmm::device_uvector<int> iteration;
};

__global__ void initialize_state_kernel(stf_result_t* current,
                                        stf_result_t* normalized,
                                        stf_result_t* next,
                                        std::size_t size,
                                        stf_result_t initial)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    current[i]    = initial;
    normalized[i] = stf_result_t{0};
    next[i]       = stf_result_t{0};
  }
}

__global__ void initialize_iteration_kernel(int* iteration)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { *iteration = 0; }
}

__global__ void normalize_ranks_kernel(stf_result_t const* current,
                                       stf_edge_t const* out_degrees,
                                       stf_result_t* normalized,
                                       std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    auto const degree = out_degrees[i];
    normalized[i] = degree == 0 ? stf_result_t{0} : current[i] / static_cast<stf_result_t>(degree);
  }
}

__global__ void add_uniform_kernel(stf_result_t* next,
                                   stf_result_t const* dangling,
                                   std::size_t size,
                                   stf_result_t uniform)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) { next[i] += (alpha * (*dangling) + (stf_result_t{1} - alpha)) * uniform; }
}

// Ordinary CUB transform-reduce inputs; these are library usage, not STF code
// generation.
struct dangling_value_op {
  stf_result_t const* current{};
  stf_edge_t const* out_degrees{};

  __device__ stf_result_t operator()(std::size_t i) const
  { return out_degrees[i] == 0 ? current[i] : stf_result_t{0}; }
};

struct absolute_difference_op {
  stf_result_t const* current{};
  stf_result_t const* next{};

  __device__ stf_result_t operator()(std::size_t i) const { return fabsf(next[i] - current[i]); }
};

template <typename Op>
auto make_reduce_input(Op op)
{ return thrust::make_transform_iterator(thrust::make_counting_iterator(std::size_t{0}), op); }

struct pagerank_continue_op {
  template <typename Difference, typename Iteration>
  __device__ bool operator()(Difference difference, Iteration iteration) const
  {
    ++(*iteration);
    return (*difference >= epsilon) && (*iteration < max_iterations);
  }
};

struct scaled_source_op {
  template <typename Src, typename Dst, typename SrcValue, typename DstValue, typename EdgeValue>
  __device__ stf_result_t operator()(Src, Dst, SrcValue src_value, DstValue, EdgeValue) const
  { return alpha * src_value; }
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

TEST(CudaStfOptimizedPageRankExperiment, CapturedProductionTraversal)
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
  auto out_degrees    = graph_view.compute_out_degrees(handle);

  optimized_pagerank_state state{vertices, handle.get_stream()};
  cugraph::edge_src_property_t<stf_vertex_t, stf_result_t> edge_src_pageranks(handle, graph_view);

  // Caller-owned CUB workspaces, sized before graph construction so the
  // captured tasks never allocate.
  std::size_t dangling_temp_bytes{};
  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(nullptr,
                                       dangling_temp_bytes,
                                       make_reduce_input(dangling_value_op{}),
                                       state.dangling_sum.data(),
                                       vertices,
                                       handle.get_stream()));
  std::size_t difference_temp_bytes{};
  RAFT_CUDA_TRY(cub::DeviceReduce::Sum(nullptr,
                                       difference_temp_bytes,
                                       make_reduce_input(absolute_difference_op{}),
                                       state.difference_sum.data(),
                                       vertices,
                                       handle.get_stream()));
  rmm::device_uvector<std::byte> dangling_temp(dangling_temp_bytes, handle.get_stream());
  rmm::device_uvector<std::byte> difference_temp(difference_temp_bytes, handle.get_stream());

  stf::stackable_ctx context;
  auto l_out_degrees = context.logical_data(stf::make_slice(out_degrees.data(), vertices),
                                            stf::data_place::current_device());
  auto l_current     = context.logical_data(stf::make_slice(state.current.data(), vertices),
                                            stf::data_place::current_device());
  auto l_normalized  = context.logical_data(stf::make_slice(state.normalized.data(), vertices),
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
  l_out_degrees.push(stf::access_mode::read, stf::data_place::current_device());
  l_current.push(stf::access_mode::rw, stf::data_place::current_device());
  l_normalized.push(stf::access_mode::rw, stf::data_place::current_device());
  l_next.push(stf::access_mode::rw, stf::data_place::current_device());
  l_dangling.push(stf::access_mode::rw, stf::data_place::current_device());
  l_difference.push(stf::access_mode::rw, stf::data_place::current_device());
  l_iteration.push(stf::access_mode::rw, stf::data_place::current_device());

  auto const uniform = stf_result_t{1} / static_cast<stf_result_t>(vertices);

  context.task(l_current.write(), l_normalized.write(), l_next.write(), l_iteration.write())
      .set_symbol("initialize state")
      ->*
    [vertices, uniform](
      cudaStream_t task_stream, auto current, auto normalized, auto next, auto iteration) {
      initialize_state_kernel<<<grid_size(vertices), block_size, 0, task_stream>>>(
        current.data_handle(), normalized.data_handle(), next.data_handle(), vertices, uniform);
      initialize_iteration_kernel<<<1, 1, 0, task_stream>>>(iteration.addr);
    };

  {
    auto loop = context.while_graph_scope(1);
    context.task(l_current.read(), l_out_degrees.read(), l_dangling.write())
        .set_symbol("dangling sum (cub)")
        ->*[temp = dangling_temp.data(), temp_bytes = dangling_temp_bytes, vertices](
             cudaStream_t task_stream, auto current, auto degrees, auto dangling) {
              auto bytes = temp_bytes;
              RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
                temp,
                bytes,
                make_reduce_input(dangling_value_op{current.data_handle(), degrees.data_handle()}),
                dangling.addr,
                vertices,
                task_stream));
            };
    context.task(l_current.read(), l_out_degrees.read(), l_normalized.write())
        .set_symbol("normalize source ranks")
        ->*[vertices](cudaStream_t task_stream, auto current, auto degrees, auto normalized) {
              normalize_ranks_kernel<<<grid_size(vertices), block_size, 0, task_stream>>>(
                current.data_handle(), degrees.data_handle(), normalized.data_handle(), vertices);
            };
    context.task(l_normalized.read(), l_next.write()).set_symbol("optimized incoming traversal")
        ->*
      [graph_view, &edge_src_pageranks](cudaStream_t task_stream, auto normalized, auto next) {
        raft::handle_t task_handle{rmm::cuda_stream_view{task_stream}};
        cugraph::update_edge_src_property(
          task_handle, graph_view, normalized.data_handle(), edge_src_pageranks.mutable_view());
        cugraph::per_v_transform_reduce_incoming_e(task_handle,
                                                   graph_view,
                                                   edge_src_pageranks.view(),
                                                   cugraph::edge_dst_dummy_property_t{}.view(),
                                                   cugraph::edge_dummy_property_t{}.view(),
                                                   scaled_source_op{},
                                                   stf_result_t{0},
                                                   cugraph::reduce_op::plus<stf_result_t>{},
                                                   next.data_handle());
      };
    context.task(l_next.rw(), l_dangling.read()).set_symbol("uniform update")
        ->*[vertices, uniform](cudaStream_t task_stream, auto next, auto dangling) {
              add_uniform_kernel<<<grid_size(vertices), block_size, 0, task_stream>>>(
                next.data_handle(), dangling.addr, vertices, uniform);
            };
    context.task(l_current.read(), l_next.read(), l_difference.write())
        .set_symbol("difference (cub)")
        ->*
      [temp = difference_temp.data(), temp_bytes = difference_temp_bytes, vertices](
        cudaStream_t task_stream, auto current, auto next, auto difference) {
        auto bytes = temp_bytes;
        RAFT_CUDA_TRY(cub::DeviceReduce::Sum(
          temp,
          bytes,
          make_reduce_input(absolute_difference_op{current.data_handle(), next.data_handle()}),
          difference.addr,
          vertices,
          task_stream));
      };
    context.task(l_current.write(), l_next.read()).set_symbol("commit ranks")
        ->*[vertices](cudaStream_t task_stream, auto current, auto next) {
              copy_values<<<grid_size(vertices), block_size, 0, task_stream>>>(
                current.data_handle(), next.data_handle(), vertices);
            };
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

  auto const mismatches = mismatch_count(handle, reference, state.current);
  EXPECT_EQ(mismatches, std::size_t{0});
  auto rank_sum = thrust::reduce(
    handle.get_thrust_policy(), state.current.begin(), state.current.end(), stf_result_t{0});
  EXPECT_NEAR(rank_sum, stf_result_t{1}, stf_result_t{2.0e-3});
  int stf_iterations{};
  RAFT_CUDA_TRY(
    cudaMemcpy(&stf_iterations, state.iteration.data(), sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(static_cast<std::size_t>(stf_iterations), reference_metadata.number_of_iterations_);

  constexpr std::array<std::string_view, 2> names{"production_cugraph", "optimized_cuda_stf"};
  for (std::size_t arm = 0; arm < names.size(); ++arm) {
    auto result = summarize(std::move(samples[arm]));
    std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
              << " algorithm=optimized_pagerank"
              << " arm=" << names[arm] << " median_ms=" << result.median_ms
              << " min_ms=" << result.min_ms << " iterations=" << stf_iterations
              << " reference_iterations=" << reference_metadata.number_of_iterations_
              << " mismatches=" << mismatches << " rank_sum=" << rank_sum
              << " vertices=" << vertices << " edges=" << edges << '\n';
  }

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
