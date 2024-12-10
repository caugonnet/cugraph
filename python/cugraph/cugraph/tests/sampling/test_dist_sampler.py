# Copyright (c) 2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import os
import shutil

import cupy
import cudf

from cugraph.datasets import karate
from cugraph.gnn import UniformNeighborSampler, DistSampleWriter
from cugraph.gnn.data_loading.bulk_sampler_io import create_df_from_disjoint_arrays

from pylibcugraph import SGGraph, ResourceHandle, GraphProperties

from cugraph.utilities.utils import (
    create_directory_with_overwrite,
    import_optional,
    MissingModule,
)


torch = import_optional("torch")
if not isinstance(torch, MissingModule):
    if torch.cuda.is_available():
        from rmm.allocators.torch import rmm_torch_allocator

        torch.cuda.change_current_allocator(rmm_torch_allocator)
    else:
        pytest.skip("CUDA-enabled PyTorch is unavailable", allow_module_level=True)


@pytest.fixture
def karate_graph() -> SGGraph:
    el = karate.get_edgelist().reset_index().rename(columns={"index": "eid"})
    G = SGGraph(
        ResourceHandle(),
        GraphProperties(is_multigraph=True, is_symmetric=False),
        el.src.astype("int64"),
        el.dst.astype("int64"),
        edge_id_array=el.eid,
    )

    return G


@pytest.mark.sg
@pytest.mark.parametrize("equal_input_size", [True, False])
@pytest.mark.parametrize("fanout", [[2, 2], [4, 4], [4, 2, 1]])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_dist_sampler_simple(
    scratch_dir, karate_graph, batch_size, fanout, equal_input_size
):
    G = karate_graph

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_simple")
    create_directory_with_overwrite(samples_path)

    writer = DistSampleWriter(samples_path)

    sampler = UniformNeighborSampler(G, writer, fanout=fanout)

    seeds = cupy.array([0, 5, 10, 15], dtype="int64")

    sampler.sample_from_nodes(
        seeds, batch_size=batch_size, assume_equal_input_size=equal_input_size
    )

    recovered_samples = cudf.read_parquet(samples_path)
    original_el = karate.get_edgelist()

    for b in range(len(seeds) // batch_size):
        el_start = recovered_samples.label_hop_offsets.iloc[b * len(fanout)]
        el_end = recovered_samples.label_hop_offsets.iloc[(b + 1) * len(fanout)]

        src = recovered_samples.majors.iloc[el_start:el_end]
        dst = recovered_samples.minors.iloc[el_start:el_end]
        edge_id = recovered_samples.edge_id.iloc[el_start:el_end]

        map_start = recovered_samples.renumber_map_offsets[b]
        map_end = recovered_samples.renumber_map_offsets[b + 1]
        renumber_map = recovered_samples["map"].iloc[map_start:map_end]

        src = renumber_map.iloc[src.values]
        dst = renumber_map.iloc[dst.values]

        for i in range(len(edge_id)):
            assert original_el.src.iloc[edge_id.iloc[i]] == src.iloc[i]
            assert original_el.dst.iloc[edge_id.iloc[i]] == dst.iloc[i]

    shutil.rmtree(samples_path)


@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("seeds_per_call", [4, 5, 10])
@pytest.mark.parametrize("compression", ["CSR", "COO"])
def test_dist_sampler_buffered_in_memory(
    scratch_dir: str, karate_graph: SGGraph, seeds_per_call: int, compression: str
):
    G = karate_graph

    samples_path = os.path.join(scratch_dir, "test_bulk_sampler_buffered_in_memory")
    create_directory_with_overwrite(samples_path)

    seeds = cupy.arange(10, dtype="int64")

    unbuffered_sampler = UniformNeighborSampler(
        G,
        writer=DistSampleWriter(samples_path),
        local_seeds_per_call=seeds_per_call,
        compression=compression,
    )

    buffered_sampler = UniformNeighborSampler(
        G,
        writer=None,
        local_seeds_per_call=seeds_per_call,
        compression=compression,
    )

    unbuffered_results = unbuffered_sampler.sample_from_nodes(
        seeds,
        batch_size=4,
    )

    unbuffered_results = [
        (create_df_from_disjoint_arrays(r[0]), r[1], r[2]) for r in unbuffered_results
    ]

    buffered_results = buffered_sampler.sample_from_nodes(seeds, batch_size=4)
    buffered_results = [
        (create_df_from_disjoint_arrays(r[0]), r[1], r[2]) for r in buffered_results
    ]

    print([r[1] for r in unbuffered_results])
    print("\n\n")
    print([r[1] for r in buffered_results])

    assert len(buffered_results) == len(unbuffered_results)

    for k in range(len(buffered_results)):
        br, bs, be = buffered_results[k]
        ur, us, ue = unbuffered_results[k]

        assert (be - bs) == (ue - us)

        for col in ur.columns:
            assert (br[col].dropna() == ur[col].dropna()).all()

    shutil.rmtree(samples_path)


@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_dist_sampler_hetero_from_nodes():
    props = GraphProperties(
        is_symmetric=False,
        is_multigraph=True,
    )

    handle = ResourceHandle()

    graph = SGGraph(
        handle,
        props,
        cupy.array([4, 5, 6, 7, 8, 9, 8, 9, 8, 7, 6, 5, 4, 5]),
        cupy.array([0, 1, 2, 3, 3, 0, 4, 5, 6, 8, 7, 8, 9, 9]),
        vertices_array=cupy.arange(10),
        edge_id_array=cupy.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7]),
        edge_type_array=cupy.array(
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype="int32"
        ),
        weight_array=cupy.ones((14,), dtype="float32"),
    )

    sampler = UniformNeighborSampler(
        graph,
        writer=None,
        compression="COO",
    )

    out = sampler.sample_from_nodes(
        nodes=cupy.array([4, 5]),
        input_id=cupy.array([5, 10]),
    )

    print(out)
