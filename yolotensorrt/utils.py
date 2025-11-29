import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Constants
NUM_DETECTIONS = 3549
OUTPUT_DIM = 5
IMAGE_W = 416
IMAGE_H = 416

def preprocess_frame(frame: np.ndarray, device: torch.device):
    # This converts HWC tensor to BCWH
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()

    # Move to GPU
    img_t = img_t.to(device, non_blocking=True)

    # Resize frame to 416x416
    img_t = F.interpolate(img_t, size=(IMAGE_H, IMAGE_W), mode='bilinear', align_corners=False)

    # Normalization
    img_t = img_t / 255.0

    # Make tensor memory contiguous
    img_t = img_t.contiguous()

    # Move back to CPU
    img_np = img_t.cpu().numpy().astype(np.float32) 

    return img_np

def draw_boxes(frame: np.ndarray, boxes_np: np.ndarray, color=(0,255,0), thickness=2):
    for box in boxes_np:
        x1,y1,x2,y2,score = box
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cv2.rectangle(frame, (x1i,y1i), (x2i,y2i), color, thickness)
        cv2.putText(frame, f"{score:.2f}", (x1i, max(0,y1i-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)