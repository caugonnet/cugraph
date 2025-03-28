/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

int generic_uniform_random_walks_test(const cugraph_resource_handle_t* handle,
                                      vertex_t* h_src,
                                      vertex_t* h_dst,
                                      weight_t* h_wgt,
                                      size_t num_vertices,
                                      size_t num_edges,
                                      vertex_t* h_start,
                                      size_t num_starts,
                                      size_t max_depth,
                                      bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_graph_t* graph               = NULL;
  cugraph_random_walk_result_t* result = NULL;

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;

  ret_code = create_mg_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, &graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  int rank = cugraph_resource_handle_get_rank(handle);
  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_uniform_random_walks(
    handle, rng_state, graph, d_start_view, max_depth, &result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_random_walks failed.");

  cugraph_type_erased_device_array_view_t* verts;
  cugraph_type_erased_device_array_view_t* wgts;

  verts = cugraph_random_walk_result_get_paths(result);
  wgts  = cugraph_random_walk_result_get_weights(result);

  size_t verts_size = cugraph_type_erased_device_array_view_size(verts);
  size_t wgts_size  = cugraph_type_erased_device_array_view_size(wgts);

  vertex_t h_result_verts[verts_size];
  weight_t h_result_wgts[wgts_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_verts, verts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_wgts, wgts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  weight_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = -1;

  for (int i = 0; i < num_edges; ++i)
    M[h_src[i]][h_dst[i]] = h_wgt[i];

  TEST_ASSERT(test_ret_value,
              cugraph_random_walk_result_get_max_path_length(result) == max_depth,
              "path length does not match");

  for (int i = 0; (i < num_starts) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(
      test_ret_value, h_start[i] == h_result_verts[i * (max_depth + 1)], "start of path not found");
    for (size_t j = 0; j < max_depth; ++j) {
      int src_index = i * (max_depth + 1) + j;
      int dst_index = src_index + 1;
      if (h_result_verts[dst_index] < 0) {
        if (h_result_verts[src_index] >= 0) {
          int departing_count = 0;
          for (int k = 0; k < num_vertices; ++k) {
            if (M[h_result_verts[src_index]][k] >= 0) departing_count++;
          }
          TEST_ASSERT(test_ret_value,
                      departing_count == 0,
                      "uniform_random_walks found no edge when an edge exists");
        }
      } else {
        // printf("\na_ = %f, b_ = %f\n", M[h_result_verts[src_index]][h_result_verts[dst_index]],
        // h_result_wgts[i * max_depth + j]);
        TEST_ASSERT(test_ret_value,
                    M[h_result_verts[src_index]][h_result_verts[dst_index]] ==
                      h_result_wgts[i * max_depth + j],
                    "uniform_random_walks got edge that doesn't exist");
      }
    }
  }

  cugraph_random_walk_result_free(result);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int generic_biased_random_walks_test(const cugraph_resource_handle_t* handle,
                                     vertex_t* h_src,
                                     vertex_t* h_dst,
                                     weight_t* h_wgt,
                                     size_t num_vertices,
                                     size_t num_edges,
                                     vertex_t* h_start,
                                     size_t num_starts,
                                     size_t max_depth,
                                     bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_graph_t* graph               = NULL;
  cugraph_random_walk_result_t* result = NULL;

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;

  ret_code = create_mg_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, &graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  int rank = cugraph_resource_handle_get_rank(handle);
  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_biased_random_walks(
    handle, rng_state, graph, d_start_view, max_depth, &result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "uniform_random_walks failed.");

  cugraph_type_erased_device_array_view_t* verts;
  cugraph_type_erased_device_array_view_t* wgts;

  verts = cugraph_random_walk_result_get_paths(result);
  wgts  = cugraph_random_walk_result_get_weights(result);

  size_t verts_size = cugraph_type_erased_device_array_view_size(verts);
  size_t wgts_size  = cugraph_type_erased_device_array_view_size(wgts);

  vertex_t h_result_verts[verts_size];
  weight_t h_result_wgts[wgts_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_verts, verts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_wgts, wgts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  weight_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = -1;

  for (int i = 0; i < num_edges; ++i)
    M[h_src[i]][h_dst[i]] = h_wgt[i];

  TEST_ASSERT(test_ret_value,
              cugraph_random_walk_result_get_max_path_length(result) == max_depth,
              "path length does not match");

  for (int i = 0; (i < num_starts) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(
      test_ret_value, h_start[i] == h_result_verts[i * (max_depth + 1)], "start of path not found");
    for (size_t j = 0; j < max_depth; ++j) {
      int src_index = i * (max_depth + 1) + j;
      int dst_index = src_index + 1;
      if (h_result_verts[dst_index] < 0) {
        if (h_result_verts[src_index] >= 0) {
          int departing_count = 0;
          for (int k = 0; k < num_vertices; ++k) {
            // edges with weight/bias value less than 0 will not be sampled.
            if (M[h_result_verts[src_index]][k] > 0) departing_count++;
          }
          TEST_ASSERT(test_ret_value,
                      departing_count == 0,
                      "biased_random_walks found no edge when an edge exists");
        }
      } else {
        TEST_ASSERT(test_ret_value,
                    M[h_result_verts[src_index]][h_result_verts[dst_index]] ==
                      h_result_wgts[i * max_depth + j],
                    "biased_random_walks got edge that doesn't exist");
      }
    }
  }

  cugraph_random_walk_result_free(result);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int generic_node2vec_random_walks_test(const cugraph_resource_handle_t* handle,
                                       vertex_t* h_src,
                                       vertex_t* h_dst,
                                       weight_t* h_wgt,
                                       size_t num_vertices,
                                       size_t num_edges,
                                       vertex_t* h_start,
                                       size_t num_starts,
                                       size_t max_depth,
                                       float p,
                                       float q,
                                       bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_graph_t* graph               = NULL;
  cugraph_random_walk_result_t* result = NULL;

  cugraph_type_erased_device_array_t* d_start           = NULL;
  cugraph_type_erased_device_array_view_t* d_start_view = NULL;

  ret_code = create_mg_test_graph(
    handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, &graph, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  ret_code =
    cugraph_type_erased_device_array_create(handle, num_starts, INT32, &d_start, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "d_start create failed.");

  d_start_view = cugraph_type_erased_device_array_view(d_start);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    handle, d_start_view, (byte_t*)h_start, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "start copy_from_host failed.");

  int rank = cugraph_resource_handle_get_rank(handle);
  cugraph_rng_state_t* rng_state;
  ret_code = cugraph_rng_state_create(handle, rank, &rng_state, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "rng_state create failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_node2vec_random_walks(
    handle, rng_state, graph, d_start_view, max_depth, p, q, &result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "node2vec_random_walks failed.");

  cugraph_type_erased_device_array_view_t* verts;
  cugraph_type_erased_device_array_view_t* wgts;

  verts = cugraph_random_walk_result_get_paths(result);
  wgts  = cugraph_random_walk_result_get_weights(result);

  size_t verts_size = cugraph_type_erased_device_array_view_size(verts);
  size_t wgts_size  = cugraph_type_erased_device_array_view_size(wgts);

  vertex_t h_result_verts[verts_size];
  weight_t h_result_wgts[wgts_size];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_verts, verts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_result_wgts, wgts, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  //  NOTE:  The C++ tester does a more thorough validation.  For our purposes
  //  here we will do a simpler validation, merely checking that all edges
  //  are actually part of the graph
  weight_t M[num_vertices][num_vertices];

  for (int i = 0; i < num_vertices; ++i)
    for (int j = 0; j < num_vertices; ++j)
      M[i][j] = -1;

  for (int i = 0; i < num_edges; ++i)
    M[h_src[i]][h_dst[i]] = h_wgt[i];

  TEST_ASSERT(test_ret_value,
              cugraph_random_walk_result_get_max_path_length(result) == max_depth,
              "path length does not match");

  for (int i = 0; (i < num_starts) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(
      test_ret_value, h_start[i] == h_result_verts[i * (max_depth + 1)], "start of path not found");
    for (size_t j = 0; j < max_depth; ++j) {
      int src_index = i * (max_depth + 1) + j;
      int dst_index = src_index + 1;
      if (h_result_verts[dst_index] < 0) {
        if (h_result_verts[src_index] >= 0) {
          int departing_count = 0;
          for (int k = 0; k < num_vertices; ++k) {
            if (M[h_result_verts[src_index]][k] >= 0) departing_count++;
          }
          TEST_ASSERT(test_ret_value,
                      departing_count == 0,
                      "node2vec_random_walks found no edge when an edge exists");
        }
      } else {
        TEST_ASSERT(test_ret_value,
                    M[h_result_verts[src_index]][h_result_verts[dst_index]] ==
                      h_result_wgts[i * max_depth + j],
                    "node2vec_random_walks got edge that doesn't exist");
      }
    }
  }

  cugraph_random_walk_result_free(result);
  cugraph_graph_free(graph);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_uniform_random_walks(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t num_starts   = 2;
  size_t max_depth    = 3;

  vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]   = {0, 1, 2, 3, 4, 5, 6, 7};
  vertex_t start[] = {2, 2};

  return generic_uniform_random_walks_test(
    handle, src, dst, wgt, num_vertices, num_edges, start, num_starts, max_depth, FALSE);
}

int test_biased_random_walks(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t num_starts   = 2;
  size_t max_depth    = 3;

  vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]   = {0, 1, 2, 3, 4, 5, 6, 7};
  vertex_t start[] = {2, 2};

  return generic_biased_random_walks_test(
    handle, src, dst, wgt, num_vertices, num_edges, start, num_starts, max_depth, FALSE);
}

int test_node2vec_random_walks(const cugraph_resource_handle_t* handle)
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t num_starts   = 2;
  size_t max_depth    = 3;

  vertex_t src[]   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]   = {0, 1, 2, 3, 4, 5, 6, 7};
  vertex_t start[] = {2, 2};

  weight_t p = 5;
  weight_t q = 8;

  return generic_node2vec_random_walks_test(
    handle, src, dst, wgt, num_vertices, num_edges, start, num_starts, p, q, max_depth, FALSE);
}

int main(int argc, char** argv)
{
  void* raft_handle                 = create_mg_raft_handle(argc, argv);
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(raft_handle);

  int result = 0;
  result |= RUN_MG_TEST(test_uniform_random_walks, handle);
  result |= RUN_MG_TEST(test_biased_random_walks, handle);
  result |= RUN_MG_TEST(test_node2vec_random_walks, handle);

  cugraph_free_resource_handle(handle);
  free_mg_raft_handle(raft_handle);

  return result;
}
