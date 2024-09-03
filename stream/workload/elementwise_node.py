from zigzag.datatypes import LayerOperand

from stream.workload.node import Node


class ElementwiseNode(Node):
    """Class that represents an onnx node that has elementwise dependencies node."""

    def __init__(
        self, node_id: int, node_name: str, predecessors: list[int], input_names: list[str], output_names: list[str]
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_name=node_name,
            type="elementwise",
            onchip_energy=0,
            offchip_energy=0,
            runtime=0,
            possible_core_allocation=[-1],
        )
        self.input_operand_source = {LayerOperand("I"): predecessors[0]}
        if len(predecessors) > 1:
            self.input_operand_source[LayerOperand("W")] = predecessors[1]

    def join(self, *tensors):
        """Join each position in the two tensors to propagate the dependencies (each position should contain a set).

        Args:
            tensor1 (np.ndarray): The first input tensor
            tensor2 (np.ndarray): The second input tensor
        """
        if len(tensors) == 1:
            return tensors[0].copy()
        elif len(tensors) == 2:
            return tensors[0] | tensors[1]
        else:
            raise ValueError("Maximum of two tensors are allowed for elementwise join operation")
