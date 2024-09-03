from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.lpnormalization_node import LpNormalizationNode


class LpNormalizationParser(OnnxOperatorParser):
    """Parses an onnx reshape operator into a LpNormalizationNode."""

    def __init__(self, node_id, node, nodes_outputs, mapping, onnx_model) -> None:
        # raise NotImplementedError

        super().__init__(node_id, node, nodes_outputs, onnx_model)

    def run(self):
        return self.generate_lpnormalization_node()

    def generate_lpnormalization_node(self):
        predecessors = self.get_node_predecessors()
        assert len(predecessors) == 1, "An ONNX LpNormalization node with multiple input nodes is not supported"
        predecessor = predecessors.pop()
        input_names = [self.node.input[0]]
        output_names = [self.node.output[0]]

        return LpNormalizationNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessor=predecessor, 
            input_names=input_names,
            output_names=output_names
        )
