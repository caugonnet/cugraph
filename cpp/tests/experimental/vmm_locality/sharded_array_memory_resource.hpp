/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/experimental/sharded.cuh>
#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <rmm/aligned.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cugraph::test::experimental {

struct sharded_resource_statistics {
  std::size_t allocation_calls{};
  std::size_t deallocation_calls{};
  std::size_t cumulative_requested_bytes{};
  std::size_t cumulative_vm_bytes{};
  std::size_t live_allocations{};
  std::size_t live_requested_bytes{};
  std::size_t live_vm_bytes{};
  std::size_t peak_live_requested_bytes{};
  std::size_t peak_live_vm_bytes{};
  std::size_t placement_granularity_bytes{};
  std::size_t locality_domains{};
  std::map<std::string, std::size_t> live_logical_bytes_per_place{};
};

/**
 * @brief Experiment-only RMM/CCCL resource backed by contiguous sharded arrays.
 *
 * Each RMM allocation becomes one `sharded_array<std::byte>::allocate_contiguous`
 * allocation, evenly split over device 0's locality domains. The returned pointer
 * is an ordinary contiguous device pointer, so existing graph containers and
 * kernels require no changes.
 *
 * The resource is intentionally synchronous. In particular, deallocation waits
 * for its caller stream before unmapping VMM pages. This is conservative but
 * correct for construction-time use and keeps that overhead out of steady-state
 * algorithm measurements.
 */
class sharded_array_memory_resource {
 private:
  using array_type = cuda::experimental::sharded::sharded_array<std::byte>;

  struct allocation_record {
    std::size_t requested_bytes{};
    std::size_t vm_bytes{};
    std::vector<std::pair<std::string, std::size_t>> logical_placements{};
    array_type array{};
  };

  struct state {
    state()
      : group(cuda::experimental::places::place_group::by_locality_domains({0})),
        placement_granularity(cuda::experimental::places::default_placement_block_size())
    {
    }

    cuda::experimental::places::place_group group;
    std::size_t placement_granularity{};
    std::mutex mutex{};
    std::unordered_map<void*, allocation_record> allocations{};
    sharded_resource_statistics statistics{};
  };

 public:
  sharded_array_memory_resource() : state_(std::make_shared<state>())
  {
    state_->statistics.placement_granularity_bytes = state_->placement_granularity;
    state_->statistics.locality_domains            = state_->group.size();
  }

  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    if (bytes == 0) { return nullptr; }
    if (alignment > state_->placement_granularity) {
      throw std::invalid_argument(
        "sharded_array_memory_resource: requested alignment exceeds VMM granularity");
    }

    // Guard both the place_group stream cache and accounting. The benchmark
    // constructs a graph on one host thread, but making that constraint explicit
    // prevents accidental races if the experiment is extended.
    std::lock_guard lock(state_->mutex);
    auto array = array_type::allocate_contiguous(state_->group, bytes);
    array.join_into(stream.get());
    auto* pointer = static_cast<void*>(array.contiguous_data());

    auto const vm_bytes =
      rmm::align_up(bytes, static_cast<std::size_t>(state_->placement_granularity));
    std::vector<std::pair<std::string, std::size_t>> placements{};
    placements.reserve(array.num_shards());
    for (std::size_t i = 0; i < array.num_shards(); ++i) {
      auto const& shard = array.shard(i);
      placements.emplace_back(shard.place.to_string(), shard.size_bytes());
    }

    auto [position, inserted] =
      state_->allocations.emplace(pointer,
                                  allocation_record{bytes,
                                                    vm_bytes,
                                                    std::move(placements),
                                                    std::move(array)});
    if (!inserted) {
      throw std::runtime_error("sharded_array_memory_resource: duplicate allocation pointer");
    }

    auto& stats = state_->statistics;
    ++stats.allocation_calls;
    stats.cumulative_requested_bytes += bytes;
    stats.cumulative_vm_bytes += vm_bytes;
    ++stats.live_allocations;
    stats.live_requested_bytes += bytes;
    stats.live_vm_bytes += vm_bytes;
    stats.peak_live_requested_bytes =
      std::max(stats.peak_live_requested_bytes, stats.live_requested_bytes);
    stats.peak_live_vm_bytes = std::max(stats.peak_live_vm_bytes, stats.live_vm_bytes);
    for (auto const& [place, logical_bytes] : position->second.logical_placements) {
      stats.live_logical_bytes_per_place[place] += logical_bytes;
    }
    return pointer;
  }

  void deallocate(cuda::stream_ref stream,
                  void* pointer,
                  std::size_t,
                  std::size_t = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    if (pointer == nullptr) { return; }
    (void)cudaStreamSynchronize(stream.get());

    std::lock_guard lock(state_->mutex);
    auto const found = state_->allocations.find(pointer);
    if (found == state_->allocations.end()) { std::terminate(); }

    auto& stats = state_->statistics;
    ++stats.deallocation_calls;
    --stats.live_allocations;
    stats.live_requested_bytes -= found->second.requested_bytes;
    stats.live_vm_bytes -= found->second.vm_bytes;
    for (auto const& [place, logical_bytes] : found->second.logical_placements) {
      auto& bytes = stats.live_logical_bytes_per_place[place];
      bytes -= logical_bytes;
      if (bytes == 0) { stats.live_logical_bytes_per_place.erase(place); }
    }
    state_->allocations.erase(found);
  }

  [[nodiscard]] void* allocate_sync(
    std::size_t bytes, std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    return allocate(cuda::stream_ref{cudaStream_t{}}, bytes, alignment);
  }

  void deallocate_sync(void* pointer,
                       std::size_t bytes,
                       std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    (void)cudaDeviceSynchronize();
    deallocate(cuda::stream_ref{cudaStream_t{}}, pointer, bytes, alignment);
  }

  [[nodiscard]] sharded_resource_statistics statistics() const
  {
    std::lock_guard lock(state_->mutex);
    return state_->statistics;
  }

  /// Locality-domain place group backing this resource's allocations; exposes
  /// the per-domain green-context streams for confined-execution experiments.
  [[nodiscard]] cuda::experimental::places::place_group& group() const { return state_->group; }

  [[nodiscard]] bool contiguous_backing_supported() const
  {
    return cuda::experimental::sharded::contiguous_backing_supported(0);
  }

  friend void get_property(sharded_array_memory_resource const&,
                           cuda::mr::device_accessible) noexcept
  {
  }

  [[nodiscard]] bool operator==(sharded_array_memory_resource const& other) const noexcept
  {
    return state_ == other.state_;
  }

 private:
  std::shared_ptr<state> state_;
};

static_assert(cuda::mr::resource_with<sharded_array_memory_resource, cuda::mr::device_accessible>);
static_assert(
  cuda::mr::synchronous_resource_with<sharded_array_memory_resource, cuda::mr::device_accessible>);

}  // namespace cugraph::test::experimental
