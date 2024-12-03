#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cugraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the libcugraph and pylibcugraph wheesl built in the previous steps and make them
# available for pip to find.
#
# ensure 'cugraph' wheel builds always use the 'libcugraph' and 'pylibcugraph' just built in the same CI run
#
# using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment
LIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcugraph_dist)
PYLIBCUGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python /tmp/pylibcugraph_dist)

cat > ./constraints.txt <<EOF
libcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUGRAPH_WHEELHOUSE}/libcugraph_*.whl)
pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBCUGRAPH_WHEELHOUSE}/pylibcugraph_*.whl)
EOF

export PIP_CONSTRAINT="${PWD}/constraints.txt"

PARALLEL_LEVEL=$(python -c \
  "from math import ceil; from multiprocessing import cpu_count; print(ceil(cpu_count()/4))")
case "${RAPIDS_CUDA_VERSION}" in
  12.*)
    EXTRA_CMAKE_ARGS=";-DUSE_CUDA_MATH_WHEELS=ON"
  ;;
  11.*)
    EXTRA_CMAKE_ARGS=";-DUSE_CUDA_MATH_WHEELS=OFF"
  ;;
esac

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DFIND_CUGRAPH_CPP=OFF${EXTRA_CMAKE_ARGS}"
export SKBUILD_BUILD_TOOL_ARGS="-j${PARALLEL_LEVEL};-l${PARALLEL_LEVEL}"

./ci/build_wheel.sh cugraph ${package_dir} python
./ci/validate_wheel.sh ${package_dir} final_dist
