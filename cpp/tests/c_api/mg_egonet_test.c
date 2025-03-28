/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mg_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int generic_egonet_test(const cugraph_resource_handle_t* resource_handle,
                        vertex_t* h_src,
                        vertex_t* h_dst,
                        weight_t* h_wgt,
                        vertex_t* h_seeds,
                        vertex_t* h_expected_src,
                        vertex_t* h_expected_dst,
                        size_t* h_expected_offsets,
                        size_t num_vertices,
                        size_t num_edges,
                        size_t num_seeds,
                        size_t radius,
                        bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* graph                              = NULL;
  cugraph_type_erased_device_array_t* seeds           = NULL;
  cugraph_type_erased_device_array_view_t* seeds_view = NULL;
  cugraph_induced_subgraph_result_t* result           = NULL;

  int rank = cugraph_resource_handle_get_rank(resource_handle);

  ret_code = create_mg_test_graph_with_properties(resource_handle,
                                                  h_src,
                                                  h_dst,
                                                  NULL,
                                                  NULL,
                                                  h_wgt,
                                                  num_edges,
                                                  store_transposed,
                                                  FALSE,
                                                  &graph,
                                                  &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  size_t num_seeds_to_use = num_seeds;
  if (rank != 0) { num_seeds_to_use = 0; }

  ret_code = cugraph_type_erased_device_array_create(
    resource_handle, num_seeds_to_use, INT32, &seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "seeds create failed.");

  seeds_view = cugraph_type_erased_device_array_view(seeds);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    resource_handle, seeds_view, (byte_t*)h_seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code =
    cugraph_extract_ego(resource_handle, graph, seeds_view, radius, FALSE, &result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_egonet failed.");

  if (test_ret_value == 0) {
    cugraph_type_erased_device_array_view_t* src;
    cugraph_type_erased_device_array_view_t* dst;
    cugraph_type_erased_device_array_view_t* wgt;
    cugraph_type_erased_device_array_view_t* offsets;

    src     = cugraph_induced_subgraph_get_sources(result);
    dst     = cugraph_induced_subgraph_get_destinations(result);
    wgt     = cugraph_induced_subgraph_get_edge_weights(result);
    offsets = cugraph_induced_subgraph_get_subgraph_offsets(result);

    size_t num_result_edges   = cugraph_type_erased_device_array_view_size(src);
    size_t num_result_offsets = cugraph_type_erased_device_array_view_size(offsets);

    vertex_t h_result_src[num_result_edges];
    vertex_t h_result_dst[num_result_edges];
    weight_t h_result_wgt[num_result_edges];
    size_t h_result_offsets[num_result_offsets];

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      resource_handle, (byte_t*)h_result_src, src, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      resource_handle, (byte_t*)h_result_dst, dst, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    if (h_wgt != NULL) {
      ret_code = cugraph_type_erased_device_array_view_copy_to_host(
        resource_handle, (byte_t*)h_result_wgt, wgt, &ret_error);
      TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");
    }

    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      resource_handle, (byte_t*)h_result_offsets, offsets, &ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

    printf("rank = %d, num_result_offsets = %lu, num_seeds = %lu\n",
           rank,
           num_result_offsets,
           num_seeds);

    TEST_ASSERT(
      test_ret_value, (num_seeds + 1) == num_result_offsets, "number of offsets doesn't match");

#if 0
    for (int i = 0; (i < num_result_offsets) && (test_ret_value == 0); ++i) {
      TEST_ASSERT(
        test_ret_value, h_result_offsets[i] == h_expected_offsets[i], "offsets don't match");
    }
#endif

    weight_t M[num_vertices][num_vertices];

    for (int i = 0; (i < num_seeds) && (test_ret_value == 0); ++i) {
      for (int r = 0; r < num_vertices; ++r)
        for (int c = 0; c < num_vertices; ++c)
          M[r][c] = 0;

      for (size_t e = h_expected_offsets[i]; e < h_expected_offsets[i + 1]; ++e)
        M[h_expected_src[e]][h_expected_dst[e]] = 1;

      for (size_t e = h_result_offsets[i]; (e < h_result_offsets[i + 1]) && (test_ret_value == 0);
           ++e) {
        TEST_ASSERT(
          test_ret_value, (M[h_result_src[e]][h_result_dst[e]] > 0), "found different edges");
      }
    }

    cugraph_type_erased_device_array_view_free(src);
    cugraph_type_erased_device_array_view_free(dst);
    cugraph_type_erased_device_array_view_free(wgt);
    cugraph_type_erased_device_array_view_free(offsets);
    cugraph_induced_subgraph_result_free(result);
  }

  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_egonet(const cugraph_resource_handle_t* resource_handle)
{
  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t radius       = 2;
  size_t num_seeds    = 2;

  vertex_t h_src[]   = {0, 1, 1, 2, 2, 2, 3, 3, 4};
  vertex_t h_dst[]   = {1, 3, 4, 0, 1, 3, 4, 5, 5};
  weight_t h_wgt[]   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 6.1f};
  vertex_t h_seeds[] = {0, 1};

  vertex_t h_result_src[]   = {0, 1, 1, 3, 1, 1, 3, 3, 4};
  vertex_t h_result_dst[]   = {1, 3, 4, 4, 3, 4, 4, 5, 5};
  size_t h_result_offsets[] = {0, 4, 9};

  // Egonet wants store_transposed = FALSE
  return generic_egonet_test(resource_handle,
                             h_src,
                             h_dst,
                             h_wgt,
                             h_seeds,
                             h_result_src,
                             h_result_dst,
                             h_result_offsets,
                             num_vertices,
                             num_edges,
                             num_seeds,
                             radius,
                             FALSE);
}

int test_egonet_no_weights(const cugraph_resource_handle_t* resource_handle)
{
  size_t num_edges    = 9;
  size_t num_vertices = 6;
  size_t radius       = 2;
  size_t num_seeds    = 2;

  vertex_t h_src[]   = {0, 1, 1, 2, 2, 2, 3, 3, 4};
  vertex_t h_dst[]   = {1, 3, 4, 0, 1, 3, 4, 5, 5};
  vertex_t h_seeds[] = {0, 1};

  vertex_t h_result_src[]   = {0, 1, 1, 3, 1, 1, 3, 3, 4};
  vertex_t h_result_dst[]   = {1, 3, 4, 4, 3, 4, 4, 5, 5};
  size_t h_result_offsets[] = {0, 4, 9};

  // Egonet wants store_transposed = FALSE
  return generic_egonet_test(resource_handle,
                             h_src,
                             h_dst,
                             NULL,
                             h_seeds,
                             h_result_src,
                             h_result_dst,
                             h_result_offsets,
                             num_vertices,
                             num_edges,
                             num_seeds,
                             radius,
                             FALSE);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_egonet, handle);
  result |= RUN_MG_TEST(test_egonet_no_weights, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
