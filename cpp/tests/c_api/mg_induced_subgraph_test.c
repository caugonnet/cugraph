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

#include <cugraph_c/graph.h>
#include <cugraph_c/graph_functions.h>

#include <stdio.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

/*
 * Simple check of creating a graph from a COO on device memory.
 */
int generic_induced_subgraph_test(const cugraph_resource_handle_t* handle,
                                  vertex_t* h_src,
                                  vertex_t* h_dst,
                                  weight_t* h_wgt,
                                  size_t num_vertices,
                                  size_t num_edges,
                                  bool_t store_transposed,
                                  size_t* h_subgraph_offsets,
                                  vertex_t* h_subgraph_vertices,
                                  size_t num_subgraph_offsets,
                                  vertex_t* h_result_src,
                                  vertex_t* h_result_dst,
                                  weight_t* h_result_wgt,
                                  size_t* h_result_offsets,
                                  size_t num_results)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_graph_t* graph                                          = NULL;
  cugraph_type_erased_device_array_t* subgraph_offsets            = NULL;
  cugraph_type_erased_device_array_t* subgraph_vertices           = NULL;
  cugraph_type_erased_device_array_view_t* subgraph_offsets_view  = NULL;
  cugraph_type_erased_device_array_view_t* subgraph_vertices_view = NULL;

  cugraph_induced_subgraph_result_t* result = NULL;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t size_t_tid = SIZE_T;
  size_t num_subgraph_vertices      = h_subgraph_offsets[num_subgraph_offsets - 1];

  ret_code = create_mg_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, &graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  if (cugraph_resource_handle_get_rank(handle) != 0) {
    num_subgraph_vertices = 0;
    for (int i = 0; i < num_subgraph_offsets; ++i) {
      h_subgraph_offsets[i] = 0;
    }
  }

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_subgraph_offsets, size_t_tid, &subgraph_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "subgraph_offsets create failed.");

  ret_code = cugraph_type_erased_device_array_create(
    handle, num_subgraph_vertices, vertex_tid, &subgraph_vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "subgraph_offsets create failed.");

  subgraph_offsets_view  = cugraph_type_erased_device_array_view(subgraph_offsets);
  subgraph_vertices_view = cugraph_type_erased_device_array_view(subgraph_vertices);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, subgraph_offsets_view, (byte_t*)h_subgraph_offsets, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "subgraph_offsets copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, subgraph_vertices_view, (byte_t*)h_subgraph_vertices, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "subgraph_vertices copy_from_host failed.");

  ret_code = cugraph_extract_induced_subgraph(handle,
                                              graph,
                                              subgraph_offsets_view,
                                              subgraph_vertices_view,
                                              // FALSE,
                                              TRUE,
                                              &result,
                                              &ret_error);
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_extract_induced_subgraph failed.");

  cugraph_type_erased_device_array_view_t* extracted_src;
  cugraph_type_erased_device_array_view_t* extracted_dst;
  cugraph_type_erased_device_array_view_t* extracted_wgt;
  cugraph_type_erased_device_array_view_t* extracted_graph_offsets;

  extracted_src           = cugraph_induced_subgraph_get_sources(result);
  extracted_dst           = cugraph_induced_subgraph_get_destinations(result);
  extracted_wgt           = cugraph_induced_subgraph_get_edge_weights(result);
  extracted_graph_offsets = cugraph_induced_subgraph_get_subgraph_offsets(result);

  size_t extracted_size = cugraph_type_erased_device_array_view_size(extracted_src);

  vertex_t h_extracted_src[extracted_size];
  vertex_t h_extracted_dst[extracted_size];
  weight_t h_extracted_wgt[extracted_size];
  vertex_t h_extracted_graph_offsets[extracted_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_extracted_src, extracted_src, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_extracted_dst, extracted_dst, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_extracted_wgt, extracted_wgt, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_extracted_graph_offsets, extracted_graph_offsets, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (size_t i = 0; (i < extracted_size) && (test_ret_value == 0); ++i) {
    bool_t found = FALSE;
    for (size_t j = 0; (j < num_results) && !found; ++j) {
      if ((h_extracted_src[i] == h_result_src[j]) && (h_extracted_dst[i] == h_result_dst[j]) &&
          (h_extracted_wgt[i] == h_result_wgt[j]))
        found = TRUE;
    }
    TEST_ASSERT(test_ret_value, found, "extracted an edge that doesn't match");
  }

  cugraph_type_erased_device_array_free(subgraph_offsets);
  cugraph_type_erased_device_array_free(subgraph_vertices);
  cugraph_induced_subgraph_result_free(result);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);
  return test_ret_value;
}

int test_induced_subgraph(const cugraph_resource_handle_t* handle)
{
  size_t num_edges            = 8;
  size_t num_vertices         = 6;
  size_t num_subgraph_offsets = 2;
  size_t num_results          = 5;

  vertex_t h_src[]               = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]               = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]               = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  size_t h_subgraph_offsets[]    = {0, 4};
  vertex_t h_subgraph_vertices[] = {0, 1, 2, 3};
  vertex_t h_result_src[]        = {0, 1, 2, 2, 2};
  vertex_t h_result_dst[]        = {1, 3, 0, 1, 3};
  weight_t h_result_wgt[]        = {0.1f, 2.1f, 5.1f, 3.1f, 4.1f};
  size_t h_result_offsets[]      = {0, 5};

  // Pagerank wants store_transposed = TRUE
  return generic_induced_subgraph_test(handle,
                                       h_src,
                                       h_dst,
                                       h_wgt,
                                       num_vertices,
                                       num_edges,
                                       TRUE,
                                       h_subgraph_offsets,
                                       h_subgraph_vertices,
                                       num_subgraph_offsets,
                                       h_result_src,
                                       h_result_dst,
                                       h_result_wgt,
                                       h_result_offsets,
                                       num_results);
}

/******************************************************************************/

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_induced_subgraph, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
