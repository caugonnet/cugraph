/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "experimental/stf/iterative_graph_common.cuh"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace cugraph::test::experimental;

// ECG (community/ecg_impl.cuh:66) runs ensemble_size independent
// single-level Louvain instances over one shared read-only graph in a strictly
// serial loop. This experiment measures the cross-instance concurrency
// opportunity: the same ensemble on one stream versus one host thread + stream
// per member.
//
// There is deliberately no STF arm. cugraph::louvain synchronizes the host
// inside its iteration (modularity readbacks, allocation joins), so submitting
// members as STF tasks from one thread would serialize on those internal
// syncs. The measured opportunity motivates the asynchronous algorithm-step
// seam that any STF (or plain stream) composition of production Louvain
// requires.

constexpr stf_result_t louvain_threshold{1.0e-7f};
constexpr stf_result_t louvain_resolution{1.0f};
constexpr std::size_t louvain_max_level{1};  // ECG ensemble members use one level

struct member_result {
  stf_result_t modularity{};
  std::size_t levels{};
};

member_result run_member(
  raft::handle_t const& handle,
  cugraph::graph_view_t<stf_vertex_t, stf_edge_t, false, false> const& graph_view,
  cugraph::edge_property_view_t<stf_edge_t, stf_result_t const*> edge_weight_view,
  rmm::device_uvector<stf_vertex_t>& cluster_assignments,
  std::uint64_t seed)
{
  raft::random::RngState rng_state{seed};
  auto [levels, modularity] =
    cugraph::louvain(handle,
                     std::make_optional(std::reference_wrapper<raft::random::RngState>(rng_state)),
                     graph_view,
                     std::make_optional(edge_weight_view),
                     cluster_assignments.data(),
                     louvain_max_level,
                     louvain_threshold,
                     louvain_resolution);
  return member_result{modularity, levels};
}

TEST(CudaStfEcgConcurrencyExperiment, EnsembleLouvain)
{
  raft::handle_t handle{};
  auto usecase = cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(
    cugraph::test::Rmat_Usecase(18, 16, 0.57, 0.19, 0.19, 0, false, false, 0));
  auto [graph, edge_weights, renumber_map] =
    cugraph::test::construct_graph<stf_vertex_t, stf_edge_t, stf_result_t, false, false>(
      handle, usecase, true, true);
  renumber_map.reset();
  ASSERT_TRUE(edge_weights.has_value());

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights->view();
  auto const vertices   = static_cast<std::size_t>(graph_view.number_of_vertices());
  auto const edges =
    static_cast<std::size_t>(graph_view.local_edge_partition_view().number_of_edges());

  constexpr std::size_t ensemble_size = 8;

  // Per-member state for both arms, so same-seed members are comparable.
  std::vector<rmm::device_uvector<stf_vertex_t>> serial_assignments;
  std::vector<rmm::device_uvector<stf_vertex_t>> concurrent_assignments;
  for (std::size_t i = 0; i < ensemble_size; ++i) {
    serial_assignments.emplace_back(vertices, handle.get_stream());
    concurrent_assignments.emplace_back(vertices, handle.get_stream());
  }

  // One handle + stream per concurrent member, created outside the timed
  // region.
  std::vector<cudaStream_t> member_streams(ensemble_size);
  std::vector<std::unique_ptr<raft::handle_t>> member_handles;
  for (std::size_t i = 0; i < ensemble_size; ++i) {
    RAFT_CUDA_TRY(cudaStreamCreateWithFlags(&member_streams[i], cudaStreamNonBlocking));
    member_handles.push_back(
      std::make_unique<raft::handle_t>(rmm::cuda_stream_view{member_streams[i]}));
  }

  std::vector<member_result> serial_results(ensemble_size);
  std::vector<member_result> concurrent_results(ensemble_size);

  auto run_serial = [&] {
    for (std::size_t i = 0; i < ensemble_size; ++i) {
      serial_results[i] =
        run_member(handle, graph_view, edge_weight_view, serial_assignments[i], i + 1);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(handle.get_stream()));
  };

  auto run_concurrent = [&] {
    std::vector<std::thread> workers;
    workers.reserve(ensemble_size);
    for (std::size_t i = 0; i < ensemble_size; ++i) {
      workers.emplace_back([&, i] {
        RAFT_CUDA_TRY(cudaSetDevice(0));
        concurrent_results[i] = run_member(
          *member_handles[i], graph_view, edge_weight_view, concurrent_assignments[i], i + 1);
        RAFT_CUDA_TRY(cudaStreamSynchronize(member_streams[i]));
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  };

  // Warm-up: fills the RMM pool and JIT/module caches for both arms.
  run_serial();
  run_concurrent();
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
        samples[arm].push_back(wall_time_ms(run_serial));
      } else {
        samples[arm].push_back(wall_time_ms(run_concurrent));
      }
    }
  }

  // Same-seed members must produce comparable modularity across arms. Louvain
  // uses atomics internally, so allow a small tolerance rather than equality.
  for (std::size_t i = 0; i < ensemble_size; ++i) {
    EXPECT_GT(serial_results[i].levels, std::size_t{0});
    EXPECT_TRUE(std::isfinite(serial_results[i].modularity));
    EXPECT_TRUE(std::isfinite(concurrent_results[i].modularity));
    EXPECT_NEAR(serial_results[i].modularity, concurrent_results[i].modularity, 5.0e-2);
  }

  constexpr std::array<std::string_view, 2> names{"serial_ensemble", "concurrent_ensemble"};
  std::array<timing, 2> timings{summarize(std::move(samples[0])), summarize(std::move(samples[1]))};
  for (std::size_t arm = 0; arm < names.size(); ++arm) {
    std::cout << std::fixed << std::setprecision(3) << "CUGRAPH_STF_RESULT"
              << " algorithm=ecg_ensemble_louvain"
              << " arm=" << names[arm] << " median_ms=" << timings[arm].median_ms
              << " min_ms=" << timings[arm].min_ms << " ensemble_size=" << ensemble_size
              << " vertices=" << vertices << " edges=" << edges << '\n';
  }
  for (std::size_t i = 0; i < ensemble_size; ++i) {
    std::cout << std::fixed << std::setprecision(4) << "CUGRAPH_STF_MEMBER member=" << i
              << " serial_modularity=" << serial_results[i].modularity
              << " concurrent_modularity=" << concurrent_results[i].modularity << '\n';
  }

  for (auto stream : member_streams) {
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
