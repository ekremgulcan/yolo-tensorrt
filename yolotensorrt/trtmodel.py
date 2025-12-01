from typing import Tuple
import numpy as np
import tensorrt as trt
import torch
import pycuda.autoinit
import pycuda.driver as cuda
import torchvision
from .utils import IMAGE_W, IMAGE_H, OUTPUT_DIM, NUM_DETECTIONS

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTModel:
    def __init__(self, engine_path: str):
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.buffers = self._allocate_buffers(self.engine, self.context)

    def _load_engine(self, engine_path: str):
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, "rb") as f:
            model_data = f.read()

            engine = runtime.deserialize_cuda_engine(model_data)
            if not engine:
                raise RuntimeError("Failed to load the engine")
            return engine
        
    def _allocate_buffers(self, engine, context):
        inputs = []
        outputs = []
        stream = cuda.Stream()

        for idx in range(engine.num_io_tensors):
            name = engine.get_tensor_name(idx)
            size = trt.volume(engine.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            host_mem =  cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
                context.set_tensor_address(name, device_mem)
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                context.set_tensor_address(name, device_mem)
        
        return inputs, outputs, stream
    
    def _inference(self, context, inputs, outputs, stream):
        for inp in inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, stream)

        context.execute_async_v3(stream.handle)

        for out in outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, stream)

        stream.synchronize()
        return [out.host for out in outputs]

    
    def detect(self, img_input):
        inputs, outputs, stream = self.buffers
        np.copyto(inputs[0].host, img_input.ravel())
        trt_outputs = self._inference(context=self.context, inputs=inputs, outputs=outputs, stream=stream)

        return trt_outputs

def decode_trt_output(output_flat: np.ndarray, original_frame_shape: Tuple[int,int]):

    # Reshape to (1,5,3549)
    out = output_flat.reshape((1, OUTPUT_DIM, NUM_DETECTIONS))
    # Squeeze batch
    out = out[0]
    
    # Center x-coordinates
    cx = out[0, :]
    # Center y-coordinates
    cy = out[1, :]
    # Width
    w = out[2, :]
    # Height
    h = out[3, :]
    # Confidence score
    score = out[4, :]

    # Adjust the coordinates to original frame size
    orig_h, orig_w = original_frame_shape
    scale_x = orig_w / IMAGE_W
    scale_y = orig_h / IMAGE_H

    x1 = (cx - w / 2.0) * scale_x
    y1 = (cy - h / 2.0) * scale_y

    x2 = (cx + w / 2.0) * scale_x
    y2 = (cy + h / 2.0) * scale_y

    boxes = np.stack([x1, y1, x2, y2, score], axis=1)

    # Filter out boxes with very low confidences to reduce NMS load
    conf_mask = boxes[:, 4] > 0.05
    boxes = boxes[conf_mask]

    return boxes.tolist()

def non_maximum_suppression(boxes_np: np.ndarray, iou_thresh: float = 0.5, topk: int = 200):
    if boxes_np.shape[0] == 0:
        return np.array([], dtype=np.int64)
    
    boxes = torch.from_numpy(boxes_np[:, :4]).float().cuda()
    scores = torch.from_numpy(boxes_np[:, 4]).float().cuda()
    keep = torchvision.ops.nms(boxes, scores, iou_thresh)
    if topk > 0:
        keep = keep[:topk]

    return keep.cpu().numpy()

