# =============================================================================
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# This function finds CCCL and sets any additional necessary environment variables.
function(find_and_configure_cccl)
  include(${rapids-cmake-dir}/cpm/cccl.cmake)
  include(${rapids-cmake-dir}/cpm/package_override.cmake)

  rapids_cpm_package_override("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cccl_override.json")

  # Enable cudax namespace install
  set(CCCL_ENABLE_UNSTABLE ON)

  rapids_cpm_cccl(BUILD_EXPORT_SET cugraph-exports INSTALL_EXPORT_SET cugraph-exports)
endfunction()

find_and_configure_cccl()
