# ssc4onnx
Checker with simple ONNX model structure. **S**imple **S**tructure **C**hecker for **ONNX**.

https://github.com/PINTO0309/simple-onnx-processing-tools

[![Downloads](https://static.pepy.tech/personalized-badge/ssc4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/ssc4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/ssc4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/ssc4onnx?color=2BAF2B)](https://pypi.org/project/ssc4onnx/) [![CodeQL](https://github.com/PINTO0309/ssc4onnx/workflows/CodeQL/badge.svg)](https://github.com/PINTO0309/ssc4onnx/actions?query=workflow%3ACodeQL)

<p align="center">
  <img src="https://user-images.githubusercontent.com/33194443/170718388-a30d9c72-be08-4d13-b3e6-d089fe3f93da.png" />
</p>

# Key concept
- Analyzes and displays the structure of huge size models that cannot be displayed by Netron.

## 1. Setup

### 1-1. HostPC
```bash
### option
$ echo export PATH="~/.local/bin:$PATH" >> ~/.bashrc \
&& source ~/.bashrc

### run
$ pip install -U onnx rich \
&& pip install -U ssc4onnx
```
### 1-2. Docker
https://github.com/PINTO0309/simple-onnx-processing-tools#docker

## 2. CLI Usage
```bash
$ ssc4onnx -h

usage:
    ssc4onnx [-h]
    --input_onnx_file_path INPUT_ONNX_FILE_PATH

optional arguments:
  -h, --help
        show this help message and exit.

  --input_onnx_file_path INPUT_ONNX_FILE_PATH
        Input onnx file path.
```

## 3. In-script Usage
```python
>>> from ssc4onnx import structure_check
>>> help(structure_check)

Help on function structure_check in module ssc4onnx.onnx_structure_check:

structure_check(
    input_onnx_file_path: Union[str, NoneType] = '',
    onnx_graph: Union[onnx.onnx_ml_pb2.ModelProto, NoneType] = None
) -> None

    Parameters
    ----------
    input_onnx_file_path: Optional[str]
        Input onnx file path.
        Either input_onnx_file_path or onnx_graph must be specified.
        Default: ''

    onnx_graph: Optional[onnx.ModelProto]
        onnx.ModelProto.
        Either input_onnx_file_path or onnx_graph must be specified.
        onnx_graph If specified, ignore input_onnx_file_path and process onnx_graph.
```

## 4. CLI Execution
```bash
$ ssc4onnx --input_onnx_file_path deqflow_b_things_opset12_192x320.onnx
```

## 5. In-script Execution
```python
from ssc4onnx import structure_check

structure_check(
  input_onnx_file_path="deqflow_b_things_opset12_192x320.onnx",
)
```

## 6. Sample
![yeuq7-3pab9](https://user-images.githubusercontent.com/33194443/170716241-1b0aaf0d-ea36-4508-b8ba-1e076e648a2e.gif)

## 7. Reference
1. https://github.com/onnx/onnx/blob/main/docs/Operators.md
2. https://docs.nvidia.com/deeplearning/tensorrt/onnx-graphsurgeon/docs/index.html
3. https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon
4. https://github.com/PINTO0309/simple-onnx-processing-tools
5. https://github.com/PINTO0309/PINTO_model_zoo

## 8. Issues
https://github.com/PINTO0309/simple-onnx-processing-tools/issues
