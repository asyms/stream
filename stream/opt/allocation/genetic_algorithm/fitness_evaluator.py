from zigzag.datatypes import LayerOperand
from zigzag.mapping.data_movement import MemoryAccesses
from zigzag.utils import pickle_deepcopy, pickle_save

from stream.cost_model.cost_model import StreamCostModelEvaluation
from stream.hardware.architecture.accelerator import Accelerator
from stream.utils import CostModelEvaluationLUT, get_too_large_operands
from stream.workload.computation_node import ComputationNode
from stream.workload.onnx_workload import ComputationNodeWorkload

# MODEL_ID = 7
# if MODEL_ID == -1:  # old experiment with model_latest.onnx
#     EENN_BLOCK_IDS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
# elif MODEL_ID == 4:   # for 0 2 4 model
#     EENN_BLOCK_IDS = [(0,), (1, 2), (3, 4), (5, 6, 7, 8, 9, 10, 11)]
# elif MODEL_ID == 5:   # for 0 2 5 model
#     EENN_BLOCK_IDS = [(0,), (1, 2), (3, 4, 5), (6, 7, 8, 9, 10, 11)]
# elif MODEL_ID == 6:   # for 0 2 6 model
#     EENN_BLOCK_IDS = [(0,), (1, 2), (3, 4, 5, 6), (7, 8, 9, 10, 11)]
# elif MODEL_ID == 7:   # for 0 2 7 model
#     EENN_BLOCK_IDS = [(0,), (1, 2), (3, 4, 5, 6, 7), (8, 9, 10, 11)]


# if MODEL_ID == -1:
#     DATA_PATH = "outputs-eenn/mounting_points_data/model_latest.pickle"
# else:
#     DATA_PATH = f"outputs-eenn/mounting_points_data/model_0_2_{MODEL_ID}.pickle"


class FitnessEvaluator:
    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        node_hw_performances: CostModelEvaluationLUT,
    ) -> None:
        self.workload = workload
        self.accelerator = accelerator
        self.node_hw_performances = node_hw_performances
        # self.num_cores = len(inputs.accelerator.cores)

    def get_fitness(self):
        raise NotImplementedError


class StandardFitnessEvaluator(FitnessEvaluator):
    """The standard fitness evaluator considers latency, max buffer occupancy and energy equally."""

    def __init__(
        self,
        workload: ComputationNodeWorkload,
        accelerator: Accelerator,
        node_hw_performances: CostModelEvaluationLUT,
        layer_groups_flexible,
        operands_to_prefetch: list[LayerOperand],
        scheduling_order: list[tuple[int, int]],
        model_id: list[int],
        precision_backbone: int,
        precision_classifier: int,
        model_path: str,
    ) -> None:
        super().__init__(workload, accelerator, node_hw_performances)

        self.weights = (-1.0, -1.0)
        self.metrics = ["energy", "latency"]

        self.layer_groups_flexible = layer_groups_flexible
        self.operands_to_prefetch = operands_to_prefetch
        self.scheduling_order = scheduling_order

        # Added to save different allocations data
        self.model_id = model_id
        self.stage_ids = list(range(len(model_id)))
        self.classifier_ids = list((i,) for i in range(len(model_id)))

        self.eenn_block_ids = []
        start = 0
        for end in self.model_id:
            self.eenn_block_ids.append(tuple(range(start, end + 1)))
            start = end + 1
        model_id_str = "_".join(map(str, self.model_id))
        self.data_save_path = f"{model_path}/model_{model_id_str}.pickle"
        self.data = []

    def get_fitness(self, core_allocations: list[int], return_scme: bool = False):
        """Get the fitness of the given core_allocations

        Args:
            core_allocations (list): core_allocations
        """
        self.set_node_core_allocations(core_allocations)
        scme = StreamCostModelEvaluation(
            pickle_deepcopy(self.workload),
            pickle_deepcopy(self.accelerator),
            self.operands_to_prefetch,
            self.scheduling_order,
        )
        scme.run()
        energy = scme.energy
        latency = scme.latency
        if not return_scme:
            self.save_eenn_data(scme)
            return energy, latency
        self.write_eenn_data()
        return energy, latency, scme

    def save_eenn_data(self, scme: StreamCostModelEvaluation):
        """! Save the latency and energy overhead of each eenn stage for this allocation."""
        all_block_nodes = {}
        all_classifier_nodes = {}
        all_block_starts = {}
        all_block_ends = {}
        all_classifier_starts = {}
        all_classifier_ends = {}
        for stage_id in self.stage_ids:
            block_ids = self.eenn_block_ids[stage_id]
            classifier_ids = self.classifier_ids[stage_id]
            block_nodes = []
            block_patterns = {f"blocks.{x}/" for x in block_ids}
            classifier_patterns = {f"classifiers.{x}/" for x in classifier_ids}
            if stage_id == 0:
                block_patterns.add("/first_conv/conv/Conv")
            block_nodes = [n for n in scme.workload.node_list if any((p in n.name for p in block_patterns))]
            classifier_nodes = [n for n in scme.workload.node_list if any((p in n.name for p in classifier_patterns))]
            all_block_nodes[stage_id] = block_nodes
            all_classifier_nodes[stage_id] = classifier_nodes

            block_start = min((n.start for n in block_nodes))
            classifier_start = min((n.start for n in classifier_nodes))
            block_end = max((n.end for n in block_nodes))
            classifier_end = max((n.end for n in classifier_nodes))
            all_block_starts[stage_id] = block_start
            all_block_ends[stage_id] = block_end
            all_classifier_starts[stage_id] = classifier_start
            all_classifier_ends[stage_id] = classifier_end
            assert classifier_end >= block_end, "Classifier end should be after block end"
            assert classifier_start >= block_start, "Classifier start should be after block start"

        cum_energy = 0
        cum_macs = 0
        total_macs = sum((n.total_mac_count for n in scme.workload.node_list))
        for stage_id in self.stage_ids:
            stage_start = all_block_starts[stage_id]
            stage_end = all_classifier_ends[stage_id]
            block_latency = all_block_ends[stage_id] - all_block_starts[stage_id]
            classifier_latency = all_classifier_ends[stage_id] - all_classifier_starts[stage_id]
            latency = stage_end - stage_start
            # CN energy
            cn_energy = sum(
                (n.onchip_energy + n.offchip_energy for n in all_block_nodes[stage_id] + all_classifier_nodes[stage_id])
            )
            cum_energy += cn_energy
            # Communication energy
            pass
            comm_energy = 0
            for event in scme.accelerator.communication_manager.events:
                if event.tasks[0].tensors[0].origin in all_block_nodes[stage_id] + all_classifier_nodes[stage_id]:
                    comm_energy += event.energy + event.memory_energy
            cum_energy += comm_energy
            if stage_id != self.stage_ids[-1]:
                next_block_nodes = all_block_nodes[stage_id + 1]
                overlapping_next_block_nodes = [n for n in next_block_nodes if n.start < all_classifier_ends[stage_id]]
                energy_overhead = sum((n.onchip_energy + n.offchip_energy for n in overlapping_next_block_nodes))
            else:
                energy_overhead = 0
            # macs on this stage
            macs = sum((n.total_mac_count for n in all_block_nodes[stage_id] + all_classifier_nodes[stage_id]))
            cum_macs += macs
            stage_data = {
                "stage_id": stage_id + 1,
                "block_latency": block_latency,
                "classifier_latency": classifier_latency,
                "latency": latency,
                "cn_energy": cn_energy,
                "comm_energy": comm_energy,
                "energy": cn_energy + comm_energy,
                "energy_overhead": energy_overhead,
                "cum_latency_fraction": all_classifier_ends[stage_id] / scme.latency,
                "energy_fraction": (cn_energy + comm_energy) / scme.energy,
                "cum_energy_fraction": cum_energy / scme.energy,
                "cum_macs_fraction": cum_macs / total_macs,
            }
            self.data.append(stage_data)

    def write_eenn_data(self):
        pickle_save(self.data, self.data_save_path)

    def set_node_core_allocations(self, core_allocations: list[int]):
        """Sets the core allocation of all nodes in self.workload according to core_allocations.
        This will only set the energy, runtime and core_allocation of the nodes which are flexible in their core
        allocation.
        We assume the energy, runtime and core_allocation of the other nodes are already set.

        Args:
            core_allocations (list): list of the node-core allocations
        """
        for i, core_allocation in enumerate(core_allocations):
            core = self.accelerator.get_core(core_allocation)
            (layer_id, group_id) = self.layer_groups_flexible[i]
            # Find all nodes of this coarse id and set their core_allocation, energy and runtime
            nodes = (
                node
                for node in self.workload.node_list
                if isinstance(node, ComputationNode) and node.id == layer_id and node.group == group_id
            )
            for node in nodes:
                equal_unique_node = self.node_hw_performances.get_equal_node(node)
                assert equal_unique_node is not None, "Node not found in node_hw_performances"
                cme = self.node_hw_performances.get_cme(equal_unique_node, core)
                onchip_energy = cme.energy_total  # Initialize on-chip energy as total energy
                latency = cme.latency_total1
                too_large_operands = get_too_large_operands(cme, self.accelerator, core_id=core_allocation)
                # If there is a too_large_operand, we separate the off-chip energy.
                offchip_energy = 0
                for too_large_operand in too_large_operands:
                    layer_operand = next(
                        (k for (k, v) in cme.layer.memory_operand_links.data.items() if v == too_large_operand)
                    )
                    layer_operand_offchip_energy = cme.mem_energy_breakdown[layer_operand][-1]
                    offchip_energy += layer_operand_offchip_energy
                    onchip_energy -= layer_operand_offchip_energy
                # If there was offchip memory added for too_large_operands, get the offchip bandwidth
                if self.accelerator.offchip_core_id is not None:
                    offchip_core = self.accelerator.get_core(self.accelerator.offchip_core_id)
                    offchip_instance = next(v for k, v in offchip_core.mem_hierarchy_dict.items())[-1].memory_instance
                    offchip_bw = cme.get_total_inst_bandwidth(offchip_instance)
                else:
                    offchip_bw = MemoryAccesses(0, 0, 0, 0)
                node.set_onchip_energy(onchip_energy)
                node.set_offchip_energy(offchip_energy)
                node.set_runtime(int(latency))
                node.set_chosen_core_allocation(core_allocation)
                node.set_too_large_operands(too_large_operands)
                node.set_offchip_bandwidth(offchip_bw)
