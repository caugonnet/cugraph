/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cudart_utils.hpp>

#include <cuda/experimental/stf.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>

namespace cugraph::test::experimental {

using stf_vertex_t = int32_t;
using stf_edge_t   = int32_t;
using stf_result_t = float;

constexpr int block_size = 256;

inline int grid_size(std::size_t size)
{
  return static_cast<int>((size + block_size - 1) / block_size);
}

__device__ inline void atomic_max_nonnegative(float* address, float value)
{
  atomicMax(reinterpret_cast<unsigned int*>(address), __float_as_uint(value));
}

__global__ void initialize_values(stf_result_t* values,
                                  stf_result_t* scratch,
                                  std::size_t size,
                                  stf_result_t value)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    values[i]  = value;
    scratch[i] = value;
  }
}

__global__ void initialize_scalar(stf_result_t* value, int* iteration)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *value     = stf_result_t{0};
    *iteration = 0;
  }
}

__global__ void katz_iteration(stf_edge_t const* offsets,
                               stf_vertex_t const* indices,
                               stf_result_t const* current,
                               stf_result_t* next,
                               std::size_t size,
                               stf_result_t alpha,
                               stf_result_t beta)
{
  auto v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= size) { return; }

  stf_result_t sum{0};
  for (auto edge = offsets[v]; edge < offsets[v + 1]; ++edge) {
    sum += current[indices[edge]];
  }
  next[v] = alpha * sum + beta;
}

__global__ void commit_and_find_max_diff(stf_result_t* current,
                                         stf_result_t const* next,
                                         stf_result_t* max_diff,
                                         std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size) { return; }
  auto difference = fabsf(next[i] - current[i]);
  current[i]      = next[i];
  atomic_max_nonnegative(max_diff, difference);
}

__global__ void update_katz_condition(stf_result_t const* max_diff,
                                      int* iteration,
                                      stf_result_t epsilon,
                                      int max_iterations,
                                      cudaGraphConditionalHandle handle)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto const keep_going = (++(*iteration) < max_iterations) && (*max_diff >= epsilon);
    cudaGraphSetConditional(handle, keep_going ? 1u : 0u);
  }
}

__global__ void clear_scalar(stf_result_t* value)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { *value = stf_result_t{0}; }
}

__global__ void increment_iteration(int* iteration)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { ++(*iteration); }
}

__global__ void set_graph_condition(cudaGraphConditionalHandle handle, unsigned int value)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { cudaGraphSetConditional(handle, value); }
}

__global__ void dangling_sum_kernel(stf_result_t const* ranks,
                                    stf_edge_t const* out_degrees,
                                    stf_result_t* dangling_sum,
                                    std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size && out_degrees[i] == 0) { atomicAdd(dangling_sum, ranks[i]); }
}

__global__ void pagerank_iteration(stf_edge_t const* offsets,
                                   stf_vertex_t const* indices,
                                   stf_edge_t const* out_degrees,
                                   stf_result_t const* current,
                                   stf_result_t* next,
                                   stf_result_t const* personalization,
                                   stf_result_t const* dangling_sum,
                                   std::size_t size,
                                   stf_result_t alpha)
{
  auto v = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= size) { return; }

  stf_result_t incoming{0};
  for (auto edge = offsets[v]; edge < offsets[v + 1]; ++edge) {
    auto const source = indices[edge];
    auto const degree = out_degrees[source];
    if (degree > 0) { incoming += current[source] / static_cast<stf_result_t>(degree); }
  }
  next[v] =
    alpha * incoming + (alpha * (*dangling_sum) + (stf_result_t{1} - alpha)) * personalization[v];
}

__global__ void copy_values(stf_result_t* output, stf_result_t const* input, std::size_t size)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) { output[i] = input[i]; }
}

__global__ void initialize_pagerank_query(stf_result_t* current,
                                          stf_result_t* next,
                                          stf_result_t* personalization,
                                          stf_result_t* dangling_sum,
                                          std::size_t size,
                                          stf_vertex_t source)
{
  auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    current[i] = stf_result_t{1} / static_cast<stf_result_t>(size);
    next[i]    = stf_result_t{0};
    personalization[i] =
      (i == static_cast<std::size_t>(source)) ? stf_result_t{1} : stf_result_t{0};
  }
  if (i == 0) { *dangling_sum = stf_result_t{0}; }
}

inline void enqueue_katz_iteration(cudaStream_t stream,
                                   stf_edge_t const* offsets,
                                   stf_vertex_t const* indices,
                                   stf_result_t* current,
                                   stf_result_t* next,
                                   stf_result_t* max_diff,
                                   std::size_t size,
                                   stf_result_t alpha,
                                   stf_result_t beta)
{
  clear_scalar<<<1, 1, 0, stream>>>(max_diff);
  katz_iteration<<<grid_size(size), block_size, 0, stream>>>(
    offsets, indices, current, next, size, alpha, beta);
  commit_and_find_max_diff<<<grid_size(size), block_size, 0, stream>>>(
    current, next, max_diff, size);
}

inline void enqueue_pagerank_iterations(cudaStream_t stream,
                                        stf_edge_t const* offsets,
                                        stf_vertex_t const* indices,
                                        stf_edge_t const* out_degrees,
                                        stf_result_t* current,
                                        stf_result_t* next,
                                        stf_result_t const* personalization,
                                        stf_result_t* dangling_sum,
                                        std::size_t size,
                                        stf_result_t alpha,
                                        int iterations)
{
  for (int iteration = 0; iteration < iterations; ++iteration) {
    clear_scalar<<<1, 1, 0, stream>>>(dangling_sum);
    dangling_sum_kernel<<<grid_size(size), block_size, 0, stream>>>(
      current, out_degrees, dangling_sum, size);
    pagerank_iteration<<<grid_size(size), block_size, 0, stream>>>(
      offsets, indices, out_degrees, current, next, personalization, dangling_sum, size, alpha);
    copy_values<<<grid_size(size), block_size, 0, stream>>>(current, next, size);
  }
}

struct timing {
  double median_ms{};
  double min_ms{};
};

inline timing summarize(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return timing{values[values.size() / 2], values.front()};
}

inline std::size_t experiment_repetitions()
{
  auto const* value = std::getenv("CUGRAPH_STF_ITERATIONS");
  if (value == nullptr) { return 9; }
  auto parsed = std::strtoull(value, nullptr, 10);
  return parsed == 0 ? 9 : static_cast<std::size_t>(parsed);
}

template <typename Function>
double wall_time_ms(Function&& function)
{
  auto const start = std::chrono::steady_clock::now();
  function();
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  auto const stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

}  // namespace cugraph::test::experimental
