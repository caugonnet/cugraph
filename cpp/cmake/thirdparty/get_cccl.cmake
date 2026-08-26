# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2020-2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

# This function finds CCCL and sets any additional necessary environment variables.
function(find_and_configure_cccl)
  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  if(BUILD_CUGRAPH_VMM_LOCALITY_EXPERIMENT AND NOT CPM_CCCL_SOURCE)
    # sharded_array::allocate_contiguous is currently available only on this
    # experiment branch. Pin the commit so benchmark results remain reproducible.
    rapids_cpm_cccl(
      BUILD_EXPORT_SET cugraph-exports
      INSTALL_EXPORT_SET cugraph-exports
      ENABLE_UNSTABLE
      GIT_REPOSITORY https://github.com/caugonnet/cccl.git
      GIT_TAG 46c1ec41a8cb8196e5ca2becddd5bd388286d66f)
  elseif(BUILD_CUGRAPH_VMM_LOCALITY_EXPERIMENT)
    # CPM's standard source override is useful for iterating on sharded-core.
    rapids_cpm_cccl(
      BUILD_EXPORT_SET cugraph-exports
      INSTALL_EXPORT_SET cugraph-exports
      ENABLE_UNSTABLE)
  else()
    rapids_cpm_cccl(BUILD_EXPORT_SET cugraph-exports INSTALL_EXPORT_SET cugraph-exports)
  endif()
endfunction()

find_and_configure_cccl()
