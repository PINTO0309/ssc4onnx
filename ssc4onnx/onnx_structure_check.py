#! /usr/bin/env python

import sys
import onnx
import onnxruntime
import numpy as np
from typing import Optional
from rich.table import Table
from rich import print as rich_print
from argparse import ArgumentParser
from collections import defaultdict


class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'


ONNX_DTYPES_TO_NUMPY_DTYPES: dict = {
    f'{onnx.TensorProto.FLOAT}': np.float32,
    f'{onnx.TensorProto.DOUBLE}': np.float64,
    f'{onnx.TensorProto.INT32}': np.int32,
    f'{onnx.TensorProto.INT64}': np.int64,
}


def human_readable_size(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


class ModelInfo:
    """
    Model info contains:
    1. Num of every op
    2. Model size
    """
    def __init__(self, model: onnx.ModelProto):
        self.op_nums = defaultdict(int)
        for node in model.graph.node:
            self.op_nums[node.op_type] += 1
        self.model_size = model.ByteSize()


def structure_check(
    input_onnx_file_path: Optional[str] = '',
    onnx_graph: Optional[onnx.ModelProto] = None,
) -> None:
    """

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.\n\
        Either input_onnx_file_path or onnx_graph must be specified.\n\
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.
    """

    # Unspecified check for input_onnx_file_path and onnx_graph
    if not input_onnx_file_path and not onnx_graph:
        print(
            f'{Color.RED}ERROR:{Color.RESET} '+
            f'One of input_onnx_file_path or onnx_graph must be specified.'
        )
        sys.exit(1)

    # Loading Graphs
    # onnx_graph If specified, onnx_graph is processed first
    if not onnx_graph:
        onnx_graph = onnx.load(input_onnx_file_path)

    onnx_session = onnxruntime.InferenceSession(
        input_onnx_file_path,
        providers=['CPUExecutionProvider'],
    )

    # Generation of dict for onnxruntime input
    ort_inputs = onnx_session.get_inputs()
    ort_outputs = onnx_session.get_outputs()
    onnx_inputs = onnx_graph.graph.input
    onnx_outputs = onnx_graph.graph.output

    ort_input_names = [
        ort_input.name for ort_input in ort_inputs
    ]
    ort_output_names = [
        ort_output.name for ort_output in ort_outputs
    ]

    ort_input_shapes = [ort_input.shape for ort_input in ort_inputs]
    ort_output_shapes = [ort_output.shape for ort_output in ort_outputs]

    onnx_input_types = [
        ONNX_DTYPES_TO_NUMPY_DTYPES[f'{onnx_input.type.tensor_type.elem_type}'] for onnx_input in onnx_inputs
    ]
    onnx_output_types = [
        ONNX_DTYPES_TO_NUMPY_DTYPES[f'{onnx_output.type.tensor_type.elem_type}'] for onnx_output in onnx_outputs
    ]

    # Print info
    model_info = ModelInfo(onnx_graph)
    table = Table()
    table.add_column('OP Type')
    table.add_column('OPs')
    _ = [table.add_row(key, str(model_info.op_nums[key])) for key in sorted(list(set(model_info.op_nums.keys())))]
    table.add_row('----------------------', '----------')
    table.add_row('Model Size', human_readable_size(model_info.model_size))
    rich_print(table)
    print(\
        f'{Color.GREEN}INFO:{Color.RESET} '+ \
        f'{Color.BLUE}file:{Color.RESET} {input_onnx_file_path}'
    )
    for idx, ort_input_name, ort_input_shape, onnx_input_type in zip(range(1, len(ort_input_names)+1), ort_input_names, ort_input_shapes, onnx_input_types):
        print(\
            f'{Color.GREEN}INFO:{Color.RESET} '+ \
            f'{Color.BLUE}input_name.{idx}:{Color.RESET} {ort_input_name} '+ \
            f'{Color.BLUE}shape:{Color.RESET} {ort_input_shape} '+ \
            f'{Color.BLUE}dtype:{Color.RESET} {onnx_input_type.__name__}'
        )

    for idx, ort_output_name, ort_output_shape, onnx_output_type in zip(range(1, len(ort_output_names)+1), ort_output_names, ort_output_shapes, onnx_output_types):
        print(\
            f'{Color.GREEN}INFO:{Color.RESET} '+ \
            f'{Color.BLUE}output_name.{idx}:{Color.RESET} {ort_output_name} '+ \
            f'{Color.BLUE}shape:{Color.RESET} {ort_output_shape} '+ \
            f'{Color.BLUE}dtype:{Color.RESET} {onnx_output_type.__name__}'
        )
    print(f'{Color.GREEN}INFO:{Color.RESET} Finish!')


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '--input_onnx_file_path',
        type=str,
        required=True,
        help='Input onnx file path.'
    )
    args = parser.parse_args()

    input_onnx_file_path = args.input_onnx_file_path

    # structure check
    structure_check(
        input_onnx_file_path=input_onnx_file_path,
    )


if __name__ == '__main__':
    main()