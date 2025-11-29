# YOLOv8 + TensorRT (yolo-tensorrt)

Minimal project to run a YOLOv8 model with TensorRT for fast inference on CUDA-enabled GPUs.

## References

Some of the sites I used when writing the code:
- https://medium.com/@kachari.bikram42/cuda-for-python-programmers-a-beginners-guide-using-pycuda-part-2-4a82a7453c6d
- https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration-guide.html
- https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html
- https://forums.developer.nvidia.com/t/inference-multiple-images-tensorrt/129054

## Prerequisites
- CUDA and TensorRT installed and configured for your GPU.
- Python >= 3.13 (see [pyproject.toml](pyproject.toml)).
- Optional: the "uv" dependency manager used by this project.

## Setup

### Install dependencies
- Using uv (recommended if you use this toolchain): use uv to install the dependencies defined in [pyproject.toml](pyproject.toml).
- Or create a virtual environment and install matching packages manually (ensure versions compatible with your CUDA/TensorRT).

### Convert ONNX model to TensorRT engine
1. Export or obtain the ONNX model: [model/v8s416.onnx](model/v8s416.onnx)
2. Convert using trtexec (example):
   ```bash
   trtexec --onnx="model/v8s416.onnx" --saveEngine="model/v8s416.engine"
   ```
   You can look at the [documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html) for more flags.

## Running the model

You can run the model with the command:
```bash
uv run run.py --engine "path/to/model" --video "path/to/video.mp4" --out "path/to/outputs"
```

Other available flags are:

| Flag | Type | Default | Description | Example |
|------|------|---------|-------------|---------|
| --engine PATH | string | yolov8_fp16.engine | Path to the TensorRT engine file (.engine) to load for inference. | --engine yolov8_fp16.engine |
| --video PATH | string | input.mp4 | Input source for OpenCV VideoCapture (file path or camera index as string). | --video videos/test.mp4 or --video 0 |
| --out PATH | string | out.mp4 | Output video file where annotated frames are written. | --out out.mp4 |
| --iou FLOAT | float | 0.45 | IoU threshold used by NMS to suppress overlapping boxes. | --iou 0.5 |
| --display | flag | False | Show a real-time GUI window with the annotated frames. | --display |
| --display-interval N | int | 1 | Update/display the GUI every N frames (useful to reduce display overhead). | --display-interval 5 |
| --max-frames N | int | 0 | Stop after N frames (0 = process the entire input). | --max-frames 500 |

