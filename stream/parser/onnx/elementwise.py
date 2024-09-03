from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.elementwise_node import ElementwiseNode


class ElementwiseParser(OnnxOperatorParser):
    """Parser for onnx operators that perform an elementwise operation on two input tensors into a single output tensor.
    For example, an Add operator adds two tensors together in every position into one output tensor.
    """

    def __init__(self, node_id, node, nodes_outputs, onnx_model) -> None:
        super().__init__(node_id, node, nodes_outputs, onnx_model)
        self.type = node.op_type.lower()
        self.name = node.name

    def run(self):
        return self.generate_elementwise_node()

    def generate_elementwise_node(self):
        predecessors = self.get_node_predecessors()
        assert 0 < len(predecessors) <= 2, f"An ONNX Elementwise node of type {self.type} with {len(predecessors)} input nodes is not supported"
        input_names = self.node.input[:len(predecessors)]
        output_names = [self.node.output[0]]


        node_obj = ElementwiseNode(
            node_id=self.node_id,
            node_name=self.node.name,
            predecessors=predecessors,
            input_names=input_names,
            output_names=output_names,
        )
        return node_obj
