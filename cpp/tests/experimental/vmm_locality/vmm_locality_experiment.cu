/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "experimental/vmm_locality/sharded_array_memory_resource.hpp"
#include "utilities/base_fixture.hpp"
#include "utilities/test_graphs.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/sampling_functions.hpp>

#include <raft/core/handle.hpp>
#include <raft/random/rng_state.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda/memory_resource>

#include <gtest/gtest.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using vertex_t = int32_t;
using edge_t   = int32_t;
using value_t  = float;
using cugraph::test::experimental::sharded_array_memory_resource;
using cugraph::test::experimental::sharded_resource_statistics;

enum class backing_kind { pool, sharded };

struct measurement {
  double construction_ms{};
  double steady_state_mean_ms{};
  double steady_state_median_ms{};
  double steady_state_min_ms{};
  std::size_t vertices{};
  std::size_t edges{};
  std::size_t result_size{};
  double result_sum{};
  sharded_resource_statistics allocation{};
};

class scoped_current_resource {
 public:
  explicit scoped_current_resource(sharded_array_memory_resource const& resource)
    : previous_(rmm::mr::set_current_device_resource(
        cuda::mr::any_resource<cuda::mr::device_accessible>{resource}))
  {
  }

  ~scoped_current_resource()
  {
    [[maybe_unused]] auto discarded =
      rmm::mr::set_current_device_resource(std::move(previous_));
  }

  scoped_current_resource(scoped_current_resource const&)            = delete;
  scoped_current_resource& operator=(scoped_current_resource const&) = delete;

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
};

std::size_t env_or(char const* name, std::size_t default_value)
{
  auto const* value = std::getenv(name);
  if (value == nullptr) { return default_value; }
  auto parsed = std::strtoull(value, nullptr, 10);
  return parsed == 0 ? default_value : static_cast<std::size_t>(parsed);
}

std::size_t iteration_count() { return env_or("CUGRAPH_VMM_ITERATIONS", 5); }

// Work-size knobs so sampling can be pushed from the latency-bound default
// into a bandwidth-bound regime without recompiling.
std::size_t source_count_limit() { return env_or("CUGRAPH_VMM_SOURCES", 4096); }

std::vector<int32_t> fanout_from_env()
{
  auto const per_hop = static_cast<int32_t>(env_or("CUGRAPH_VMM_FANOUT", 10));
  auto const hops    = env_or("CUGRAPH_VMM_HOPS", 2);
  return std::vector<int32_t>(hops, per_hop);
}

template <bool store_transposed>
auto construct_graph(raft::handle_t const& handle,
                     cugraph::test::Rmat_Usecase const& usecase,
                     backing_kind backing,
                     sharded_array_memory_resource const& resource)
{
  if (backing == backing_kind::sharded) {
    scoped_current_resource use_sharded_resource(resource);
    return cugraph::test::construct_graph<vertex_t, edge_t, value_t, store_transposed, false>(
      handle, usecase, false, true);
  }
  return cugraph::test::construct_graph<vertex_t, edge_t, value_t, store_transposed, false>(
    handle, usecase, false, true);
}

struct steady_state_timing {
  double mean_ms{};
  double median_ms{};
  double min_ms{};
};

template <typename Function>
steady_state_timing time_steady_state(raft::handle_t const& handle,
                                      std::size_t iterations,
                                      Function&& function)
{
  function();  // warm-up
  handle.sync_stream();

  std::vector<double> milliseconds{};
  milliseconds.reserve(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    handle.sync_stream();
    auto const start = std::chrono::steady_clock::now();
    function();
    handle.sync_stream();
    auto const stop = std::chrono::steady_clock::now();
    milliseconds.push_back(
      std::chrono::duration<double, std::milli>(stop - start).count());
  }

  auto sorted = milliseconds;
  std::sort(sorted.begin(), sorted.end());
  auto const median = sorted.size() % 2 == 1
                        ? sorted[sorted.size() / 2]
                        : (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0;
  return {std::accumulate(milliseconds.begin(), milliseconds.end(), 0.0) /
            static_cast<double>(milliseconds.size()),
          median,
          sorted.front()};
}

measurement run_sampling(backing_kind backing,
                         cugraph::test::Rmat_Usecase const& usecase,
                         std::size_t iterations)
{
  raft::handle_t handle{};
  sharded_array_memory_resource resource{};

  handle.sync_stream();
  auto const construction_start = std::chrono::steady_clock::now();
  auto [graph, edge_weights, renumber_map] =
    construct_graph<false>(handle, usecase, backing, resource);
  handle.sync_stream();
  auto const construction_stop = std::chrono::steady_clock::now();

  EXPECT_FALSE(edge_weights.has_value());
  renumber_map.reset();
  handle.sync_stream();

  auto graph_view       = graph.view();
  auto const num_edges  = graph_view.compute_number_of_edges(handle);
  auto const source_count = std::min<std::size_t>(
    static_cast<std::size_t>(graph_view.number_of_vertices()), source_count_limit());
  rmm::device_uvector<vertex_t> sources(source_count, handle.get_stream());
  thrust::sequence(rmm::exec_policy(handle.get_stream()), sources.begin(), sources.end());
  auto const fanout = fanout_from_env();
  std::size_t sampled_edges{};

  auto sample_once = [&] {
    raft::random::RngState rng_state{0};
    auto result = cugraph::neighbor_sample<vertex_t, edge_t, value_t, int32_t, int32_t, false, false>(
      handle,
      rng_state,
      graph_view,
      std::optional<cugraph::edge_property_view_t<edge_t, value_t const*>>{std::nullopt},
      std::optional<cugraph::edge_property_view_t<edge_t, edge_t const*>>{std::nullopt},
      std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
      std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
      std::optional<cugraph::edge_property_view_t<edge_t, int32_t const*>>{std::nullopt},
      std::optional<cugraph::edge_property_view_t<edge_t, value_t const*>>{std::nullopt},
      raft::device_span<vertex_t const>{sources.data(), sources.size()},
      std::optional<raft::device_span<int32_t const>>{std::nullopt},
      std::optional<raft::device_span<int32_t const>>{std::nullopt},
      std::optional<raft::device_span<int32_t const>>{std::nullopt},
      std::optional<raft::device_span<int32_t const>>{std::nullopt},
      raft::host_span<int32_t const>{fanout.data(), fanout.size()},
      std::optional<int32_t>{std::nullopt},
      cugraph::sampling_options_t{cugraph::prior_sources_behavior_t{0},
                                  false,
                                  false,
                                  false,
                                  std::nullopt,
                                  false});
    sampled_edges = std::get<0>(result).size();
  };

  auto const timing = time_steady_state(handle, iterations, sample_once);
  auto const allocation =
    backing == backing_kind::sharded ? resource.statistics() : sharded_resource_statistics{};

  if (backing == backing_kind::sharded) {
    EXPECT_EQ(allocation.live_allocations, 2);
    EXPECT_EQ(allocation.live_requested_bytes,
              (static_cast<std::size_t>(graph_view.number_of_vertices()) + 1) * sizeof(edge_t) +
                static_cast<std::size_t>(num_edges) * sizeof(vertex_t));
  }

  return measurement{
    std::chrono::duration<double, std::milli>(construction_stop - construction_start).count(),
    timing.mean_ms,
    timing.median_ms,
    timing.min_ms,
    static_cast<std::size_t>(graph_view.number_of_vertices()),
    static_cast<std::size_t>(num_edges),
    sampled_edges,
    0.0,
    allocation};
}

measurement run_pagerank(backing_kind backing,
                         cugraph::test::Rmat_Usecase const& usecase,
                         std::size_t iterations)
{
  raft::handle_t handle{};
  sharded_array_memory_resource resource{};

  handle.sync_stream();
  auto const construction_start = std::chrono::steady_clock::now();
  auto [graph, edge_weights, renumber_map] =
    construct_graph<true>(handle, usecase, backing, resource);
  handle.sync_stream();
  auto const construction_stop = std::chrono::steady_clock::now();

  EXPECT_FALSE(edge_weights.has_value());
  renumber_map.reset();
  handle.sync_stream();

  auto graph_view      = graph.view();
  auto const num_edges = graph_view.compute_number_of_edges(handle);
  double rank_sum{};
  bool compute_rank_sum{false};
  auto pagerank_once = [&] {
    auto result = cugraph::pagerank<vertex_t, edge_t, value_t, value_t, false>(
      handle,
      graph_view,
      std::optional<cugraph::edge_property_view_t<edge_t, value_t const*>>{std::nullopt},
      std::nullopt,
      std::nullopt,
      std::optional<raft::device_span<value_t const>>{std::nullopt},
      value_t{0.85},
      value_t{1e-6},
      std::numeric_limits<std::size_t>::max(),
      false);
    if (compute_rank_sum) {
      handle.sync_stream();
      auto& ranks = std::get<0>(result);
      rank_sum    = thrust::reduce(
        rmm::exec_policy(handle.get_stream()), ranks.begin(), ranks.end(), value_t{0.0});
    }
  };

  auto const timing = time_steady_state(handle, iterations, pagerank_once);
  compute_rank_sum  = true;
  pagerank_once();
  handle.sync_stream();
  auto const allocation =
    backing == backing_kind::sharded ? resource.statistics() : sharded_resource_statistics{};

  EXPECT_NEAR(rank_sum, 1.0, 1e-4);
  if (backing == backing_kind::sharded) {
    EXPECT_EQ(allocation.live_allocations, 2);
    EXPECT_EQ(allocation.live_requested_bytes,
              (static_cast<std::size_t>(graph_view.number_of_vertices()) + 1) * sizeof(edge_t) +
                static_cast<std::size_t>(num_edges) * sizeof(vertex_t));
  }

  return measurement{
    std::chrono::duration<double, std::milli>(construction_stop - construction_start).count(),
    timing.mean_ms,
    timing.median_ms,
    timing.min_ms,
    static_cast<std::size_t>(graph_view.number_of_vertices()),
    static_cast<std::size_t>(num_edges),
    static_cast<std::size_t>(graph_view.number_of_vertices()),
    rank_sum,
    allocation};
}

void print_measurement(std::string const& algorithm,
                       std::string const& backing,
                       measurement const& value)
{
  std::cout << std::fixed << std::setprecision(3) << "VMM_LOCALITY_RESULT"
            << " algorithm=" << algorithm << " backing=" << backing
            << " vertices=" << value.vertices << " edges=" << value.edges
            << " construction_ms=" << value.construction_ms
            << " steady_state_mean_ms=" << value.steady_state_mean_ms
            << " steady_state_median_ms=" << value.steady_state_median_ms
            << " steady_state_min_ms=" << value.steady_state_min_ms
            << " result_size=" << value.result_size << " result_sum=" << value.result_sum
            << " allocation_calls=" << value.allocation.allocation_calls
            << " deallocation_calls=" << value.allocation.deallocation_calls
            << " live_allocations=" << value.allocation.live_allocations
            << " live_requested_bytes=" << value.allocation.live_requested_bytes
            << " live_vm_bytes=" << value.allocation.live_vm_bytes
            << " peak_live_requested_bytes=" << value.allocation.peak_live_requested_bytes
            << " peak_live_vm_bytes=" << value.allocation.peak_live_vm_bytes
            << " placement_granularity_bytes="
            << value.allocation.placement_granularity_bytes
            << " locality_domains=" << value.allocation.locality_domains << '\n';
  for (auto const& [place, bytes] : value.allocation.live_logical_bytes_per_place) {
    std::cout << "VMM_LOCALITY_PLACEMENT"
              << " algorithm=" << algorithm << " backing=" << backing << " place=\"" << place
              << "\" logical_bytes=" << bytes << '\n';
  }
}

TEST(VmmLocalityExperiment, PoolVersusShardedGraphStorage)
{
  sharded_array_memory_resource capability_probe{};
  if (!capability_probe.contiguous_backing_supported()) {
    GTEST_SKIP() << "CUDA VMM is not supported on device 0";
  }
  // Comparing two whole-device allocations would silently measure nothing:
  // require at least two real locality domains (needs a CUDA >= 13.4 toolkit
  // at compile time and a driver/device exposing the domain attribute).
  auto const domains = capability_probe.statistics().locality_domains;
  if (domains < 2) {
    GTEST_SKIP() << "device 0 exposes " << domains
                 << " locality domain(s); sharded placement would degrade to whole-device";
  }

  auto usecase = cugraph::test::override_Rmat_Usecase_with_cmd_line_arguments(
    cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, false, false, 0));
  auto const iterations = iteration_count();

  auto const sampling_pool    = run_sampling(backing_kind::pool, usecase, iterations);
  auto const sampling_sharded = run_sampling(backing_kind::sharded, usecase, iterations);
  EXPECT_EQ(sampling_pool.vertices, sampling_sharded.vertices);
  EXPECT_EQ(sampling_pool.edges, sampling_sharded.edges);
  EXPECT_EQ(sampling_pool.result_size, sampling_sharded.result_size);

  auto const pagerank_pool    = run_pagerank(backing_kind::pool, usecase, iterations);
  auto const pagerank_sharded = run_pagerank(backing_kind::sharded, usecase, iterations);
  EXPECT_EQ(pagerank_pool.vertices, pagerank_sharded.vertices);
  EXPECT_EQ(pagerank_pool.edges, pagerank_sharded.edges);
  EXPECT_NEAR(pagerank_pool.result_sum, pagerank_sharded.result_sum, 1e-5);

  print_measurement("uniform_neighbor_sampling", "rmm_pool", sampling_pool);
  print_measurement("uniform_neighbor_sampling", "cccl_sharded_array", sampling_sharded);
  print_measurement("pagerank", "rmm_pool", pagerank_pool);
  print_measurement("pagerank", "cccl_sharded_array", pagerank_sharded);
}

}  // namespace

CUGRAPH_TEST_PROGRAM_MAIN()
