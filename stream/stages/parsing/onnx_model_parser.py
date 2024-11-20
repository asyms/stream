import logging
from typing import Any

from zigzag.datatypes import Constants
from zigzag.stages.parser.workload_parser import WorkloadParserStage as ZigZagWorkloadParserStage
from zigzag.stages.stage import Stage, StageCallable

from stream.hardware.architecture.accelerator import Accelerator
from stream.parser.onnx.model import ONNXModelParser
from stream.parser.workload_factory import WorkloadFactoryStream
from stream.workload.computation_node import ComputationNode
from stream.workload.dnn_workload import DNNWorkloadStream
from stream.workload.onnx_workload import ONNXWorkload

logger = logging.getLogger(__name__)


class ONNXModelParserStage(Stage):
    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload_path: str,
        mapping_path: str,
        accelerator: Accelerator,
        precision_backbone: int,
        precision_classifier: int,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables, **kwargs)
        self.accelerator = accelerator
        self.mapping_path = mapping_path
        self.onnx_model_parser = ONNXModelParser(workload_path, mapping_path, accelerator)
        self.precision_backbone = precision_backbone
        self.precision_classifier = precision_classifier

    def run(self):
        self.onnx_model_parser.run()
        onnx_model = self.onnx_model_parser.get_onnx_model()
        workload = self.onnx_model_parser.get_workload()
        self.override_precision_workload(workload)

        self.kwargs["accelerator"] = self.accelerator
        self.kwargs["mapping_path"] = self.mapping_path
        self.kwargs["precision_backbone"] = self.precision_backbone
        self.kwargs["precision_classifier"] = self.precision_classifier
        sub_stage = self.list_of_callables[0](
            self.list_of_callables[1:],
            onnx_model=onnx_model,
            workload=workload,
            **self.kwargs,
        )
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def override_precision_workload(self, workload: ONNXWorkload):
        for n in workload.nodes():
            if not isinstance(n, ComputationNode):
                continue
            if "classifier" in n.name:
                precision = self.precision_classifier
            else:
                precision = self.precision_backbone
            self.override_precision_node(n, precision)

    def override_precision_node(self, node: ComputationNode, precision: int):
        # Input operands
        for input_operand in node.input_operands:
            original_precision = node.operand_precision[input_operand]
            if original_precision == 0:  # pooling weights
                continue
            precision_factor = precision / original_precision
            assert not divmod(precision, original_precision)[1] or not divmod(original_precision, precision)[1]
            node.operand_precision.data[input_operand] = precision
            node.operand_size_bit[input_operand] = int(node.operand_size_bit[input_operand] * precision_factor)
            node.operand_tensors[input_operand].size = int(node.operand_tensors[input_operand].size * precision_factor)
        # Output operand
        original_precision = node.operand_precision.data[Constants.FINAL_OUTPUT_LAYER_OP]
        precision_factor = precision / original_precision
        assert not divmod(precision, original_precision)[1] or not divmod(original_precision, precision)[1]
        node.operand_precision.data[Constants.FINAL_OUTPUT_LAYER_OP] = precision
        if precision == 32:
            node.operand_precision.data[Constants.OUTPUT_LAYER_OP] = 32
        else:
            node.operand_precision.data[Constants.OUTPUT_LAYER_OP] = int(precision * 2)
        node.operand_size_bit[Constants.OUTPUT_LAYER_OP] = int(
            node.operand_size_bit[Constants.OUTPUT_LAYER_OP] * precision_factor
        )
        node.operand_tensors[Constants.OUTPUT_LAYER_OP].size = int(
            node.operand_tensors[Constants.OUTPUT_LAYER_OP].size * precision_factor
        )


class UserDefinedModelParserStage(ZigZagWorkloadParserStage):
    """Parses a user-provided workload from a yaml file.
    This class is very similar to WorkloadParserStage from ZigZag, the main difference being that this class creates a
    (Stream)DNNWorkload of ComputationNodes, while the ZigZag variant creates a (ZigZag) DNNWorkload of LayerNodes
    """

    def __init__(
        self,
        list_of_callables: list[StageCallable],
        *,
        workload_path: str,
        mapping_path: str,
        accelerator: Accelerator,
        **kwargs: Any,
    ):
        super().__init__(list_of_callables=list_of_callables, workload=workload_path, mapping=mapping_path, **kwargs)
        self.accelerator = accelerator

    def run(self):
        workload = self.parse_workload_stream()
        self.kwargs["accelerator"] = self.accelerator
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], workload=workload, **self.kwargs)
        for cme, extra_info in sub_stage.run():
            yield cme, extra_info

    def parse_workload_stream(self) -> DNNWorkloadStream:
        workload_data = self._parse_workload_data()
        mapping_data = self._parse_mapping_data()
        factory = WorkloadFactoryStream(workload_data, mapping_data)
        return factory.create()
