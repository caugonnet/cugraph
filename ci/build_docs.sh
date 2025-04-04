#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

export RAPIDS_VERSION="$(rapids-version)"
export RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_NUMBER="$RAPIDS_VERSION_MAJOR_MINOR"

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

if [[ "${RAPIDS_CUDA_VERSION}" == "11.8.0" ]]; then
  CONDA_CUDA_VERSION="11.8"
  DGL_CHANNEL="dglteam/label/th23_cu118"
else
  CONDA_CUDA_VERSION="12.1"
  DGL_CHANNEL="dglteam/label/th23_cu121"
fi

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  --channel conda-forge \
  --channel nvidia \
  --channel "${DGL_CHANNEL}" \
  "libcugraph=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "pylibcugraph=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "cugraph=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "cugraph-pyg=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "cugraph-dgl=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "cugraph-service-server=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "cugraph-service-client=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "libcugraph_etl=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  "pylibwholegraph=${RAPIDS_VERSION_MAJOR_MINOR}.*" \
  'pytorch>=2.3' \
  "cuda-version=${CONDA_CUDA_VERSION}"

export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
export XML_DIR_LIBCUGRAPH="$(pwd)/xml"
mkdir -p "${RAPIDS_DOCS_DIR}/libcugraph/xml_tar"
tar -czf "${RAPIDS_DOCS_DIR}/libcugraph/xml_tar"/xml.tar.gz -C xml .
popd

rapids-upload-docs
