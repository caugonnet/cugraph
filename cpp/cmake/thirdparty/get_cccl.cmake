# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds CCCL and sets any additional necessary environment variables.
function(find_and_configure_cccl)
  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  if(BUILD_CUGRAPH_STF_EXPERIMENT)
    # Keep cudax on the component list when CMake reuses an existing CPM checkout.
    set(CCCL_ENABLE_UNSTABLE ON)
  endif()
  if(BUILD_CUGRAPH_STF_EXPERIMENT AND NOT CPM_CCCL_SOURCE)
    rapids_cpm_cccl(
      BUILD_EXPORT_SET cugraph-exports
      INSTALL_EXPORT_SET cugraph-exports
      ENABLE_UNSTABLE
      GIT_REPOSITORY https://github.com/caugonnet/cccl.git
      GIT_TAG 46c1ec41a8cb8196e5ca2becddd5bd388286d66f)
  elseif(BUILD_CUGRAPH_STF_EXPERIMENT)
    rapids_cpm_cccl(
      BUILD_EXPORT_SET cugraph-exports
      INSTALL_EXPORT_SET cugraph-exports
      ENABLE_UNSTABLE)
  else()
    rapids_cpm_cccl(BUILD_EXPORT_SET cugraph-exports INSTALL_EXPORT_SET cugraph-exports)
  endif()
  if(BUILD_CUGRAPH_STF_EXPERIMENT AND NOT TARGET cudax::cudax)
    find_package(cudax REQUIRED CONFIG
                 PATHS "${CCCL_SOURCE_DIR}/lib/cmake/cudax"
                 NO_DEFAULT_PATH)
  endif()
endfunction()

find_and_configure_cccl()
