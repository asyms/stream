from math import prod
from typing import Dict, List, Tuple, Union

from zigzag.datatypes import LayerDim, LayerOperand, UnrollFactor

from stream.hardware.architecture.accelerator import Accelerator
from stream.utils import CostModelEvaluationLUT


def convert_id(i: int, j: int) -> int:
    k = 1000 * i + j
    return k


def invert_id(k: int) -> Tuple[int, int]:
    i, j = divmod(k, 1000)
    return i, j


def convert_ids(nodes: List) -> Dict:
    ids = {}
    for node in nodes:
        i, j = node.id, node.sub_id
        new_id = convert_id(i, j)
        ids[node] = new_id
    return ids


def invert_ids_list(ids_list: list[tuple[int, int, int]]) -> list[tuple[int, int, Tuple[int, int]]]:
    new_l = []
    for slot, core, k in ids_list:
        new_l.append((slot, core, invert_id(k)))
    return new_l


def invert_ids_dict(d: dict[int, Union[int, float]]) -> dict[Tuple[int, int], Union[int, float]]:
    new_d = {}
    for id, val in d.items():
        new_d[invert_id(id)] = val
    return new_d


def get_loop_size(loops: list[tuple[LayerDim, UnrollFactor]], dims: list[LayerDim]) -> int:
    return int(prod([tl[1] for tl in loops if tl[0] in dims]))


def get_latencies(
    nodes: list,
    core_ids: list[int],
    accelerator: Accelerator,
    node_hw_performances: CostModelEvaluationLUT,
    impossible_lat: float = 1e11,
    ids: Dict = {},
) -> tuple[dict[tuple[int, str, int], int], dict]:
    if not ids:
        ids = {node: node.id for node in nodes}
    core_names = [f"Core {id}" for id in core_ids]
    latencies = {(ids[node], core_name): impossible_lat for node in nodes for core_name in core_names}
    possible_allocations = {}
    k_g_sizes = {}

    for node in nodes:
        node_id = ids[node]
        possible_allocations[node_id] = []
        for core_id, core_name in zip(core_ids, core_names):
            core = accelerator.get_core(core_id)
            try:
                equal_node = node_hw_performances.get_equal_node(node)
                assert equal_node, f"No equal node for {node} found in CostModelEvaluationLUT"
                cme = node_hw_performances.get_cme(equal_node, core)
                output_operand = LayerOperand("O")
                temporal_loops = [
                    i for tm_level in cme.temporal_mapping.mapping_dic_stationary[output_operand] for i in tm_level
                ]
                k_g_size = get_loop_size(temporal_loops, [LayerDim("K"), LayerDim("G")])
                k_g_sizes[(node_id, core_name)] = k_g_size
                lat = cme.latency_total1
                possible_allocations[node_id].append(core_name)
            except ValueError:
                lat = impossible_lat
            latencies[(node_id, core_name)] = lat

    latencies_with_split = {}
    possible_allocation_splits = {}
    k_max = len(core_names)

    for node_id in ids.values():
        possible_allocation_splits[node_id] = {}
        for core_name in core_names:
            possible_allocation_splits[node_id][core_name] = {}
            if core_name in possible_allocations[node_id]:
                k_t = int(k_g_sizes[node_id, core_name])
                for k in range(1, k_max + 1):
                    if divmod(k_t, k)[1] == 0 and k <= len(possible_allocations[node_id]):
                        lat = int(latencies[(node_id, core_name)] / min(k_t, k))
                        possible_allocation_splits[node_id][core_name][k] = 1
                    else:
                        lat = impossible_lat
                        possible_allocation_splits[node_id][core_name][k] = 0
                    latencies_with_split[(node_id, core_name, k)] = lat
            else:
                for k in range(1, k_max + 1):
                    latencies_with_split[(node_id, core_name, k)] = impossible_lat
                    possible_allocation_splits[node_id][core_name][k] = 0

    return latencies_with_split, possible_allocation_splits


def get_energies(
    nodes: List,
    core_ids: List[int],
    accelerator: Accelerator,
    node_hw_performances: CostModelEvaluationLUT,
    impossible_energy: float = 1e11,
    ids: Dict = {},
) -> Dict[Tuple[int, str], float]:
    if not ids:
        ids = {node.id: node.id for node in nodes}
    core_names = [f"Core {id}" for id in core_ids]
    energies = {(ids[node], core_name): impossible_energy for node in nodes for core_name in core_names}

    for node in nodes:
        for core_id, core_name in zip(core_ids, core_names):
            core = accelerator.get_core(core_id)
            try:
                cme = node_hw_performances.get_cme(node, core)
                en = getattr(cme, "energy_total")
            except ValueError:
                en = impossible_energy
            energies[(ids[node], core_name)] = en

    return energies
