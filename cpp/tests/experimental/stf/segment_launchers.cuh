/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/handle.hpp>
#include <raft/core/resource/custom_resource.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda/experimental/stf.cuh>

#include <array>
#include <cstddef>
#include <utility>

namespace cugraph::test::experimental {

class explicit_stream_segment_launcher {
 public:
  explicit_stream_segment_launcher(raft::handle_t const& handle)
    : caller_stream_(handle.get_stream()), start_event_(make_event())
  {
    RAFT_CUDA_TRY(cudaEventRecord(start_event_, caller_stream_));
    for (std::size_t i = 0; i < streams_.size(); ++i) {
      streams_[i] = handle.get_stream_from_stream_pool(i);
      RAFT_CUDA_TRY(cudaStreamWaitEvent(streams_[i], start_event_));
      completion_events_[i] = make_event();
    }
  }

  ~explicit_stream_segment_launcher()
  {
    for (auto event : completion_events_) {
      if (event != nullptr) { cudaEventDestroy(event); }
    }
    if (start_event_ != nullptr) { cudaEventDestroy(start_event_); }
  }

  explicit_stream_segment_launcher(explicit_stream_segment_launcher const&)            = delete;
  explicit_stream_segment_launcher& operator=(explicit_stream_segment_launcher const&) = delete;

  template <typename Launch>
  void submit(char const*, Launch&& launch)
  {
    auto const index = submitted_++ % streams_.size();
    used_[index]     = true;
    launch(streams_[index]);
  }

  void join()
  {
    for (std::size_t i = 0; i < streams_.size(); ++i) {
      if (!used_[i]) { continue; }
      RAFT_CUDA_TRY(cudaEventRecord(completion_events_[i], streams_[i]));
      RAFT_CUDA_TRY(cudaStreamWaitEvent(caller_stream_, completion_events_[i]));
    }
  }

 private:
  static cudaEvent_t make_event()
  {
    cudaEvent_t event{};
    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
  }

  static constexpr std::size_t stream_count = 4;
  cudaStream_t caller_stream_{};
  cudaEvent_t start_event_{};
  std::array<cudaStream_t, stream_count> streams_{};
  std::array<cudaEvent_t, stream_count> completion_events_{};
  std::array<bool, stream_count> used_{};
  std::size_t submitted_{};
};

class stf_segment_launcher {
 public:
  explicit stf_segment_launcher(raft::handle_t const& handle) : ctx_(make_context(handle)) {}

  template <typename Launch>
  void submit(char const* symbol, Launch&& launch)
  {
    auto token = ctx_.token();
    auto task  = ctx_.task(token.write());
    task.set_symbol(symbol);
    task->*[launch = std::forward<Launch>(launch)](cudaStream_t stream) mutable { launch(stream); };
  }

  void join() { ctx_.finalize(); }

 private:
  static cuda::experimental::stf::stream_ctx make_context(raft::handle_t const& handle)
  {
    auto stream = static_cast<cudaStream_t>(handle.get_stream());
    cudaStreamCaptureStatus capture_status{};
    RAFT_CUDA_TRY(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      return cuda::experimental::stf::stream_ctx(stream);
    }

    auto* resources =
      raft::resource::get_custom_resource<cuda::experimental::stf::async_resources_handle>(handle);
    return cuda::experimental::stf::stream_ctx(stream, *resources);
  }

  cuda::experimental::stf::stream_ctx ctx_;
};

}  // namespace cugraph::test::experimental
