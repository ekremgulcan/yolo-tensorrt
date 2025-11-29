import argparse
import time

import cv2
import numpy as np
import torch

from yolotensorrt.trtmodel import TRTModel, decode_trt_output, non_maximum_suppression
from yolotensorrt.utils import draw_boxes, preprocess_frame

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script expects a CUDA GPU available for GPU preprocessing and NMS.")

    # Load TRT engine
    trt_model = TRTModel(args.engine)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    # Set up the video writer
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(args.out, fourcc, fps, (orig_w, orig_h))

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Preprocess on GPU and get numpy input for TRT (shape [1,3,H,W])
        input_np = preprocess_frame(frame, device)  # returns (1,3,416,416) numpy float32

        # The actual detection
        outputs = trt_model.detect(input_np)
        out0 = outputs[0]

        # Decode boxes to original frame scale
        boxes = decode_trt_output(out0, (orig_h, orig_w))  # returns list of [x1,y1,x2,y2,score]
        if len(boxes) == 0:
            out_vid.write(frame)
            if args.display and (frame_idx % args.display_interval == 0):
                cv2.imshow("out", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        boxes_np = np.array(boxes, dtype=np.float32)
        keep_indices = non_maximum_suppression(boxes_np, iou_thresh=args.iou)
        kept = boxes_np[keep_indices] if keep_indices.size > 0 else np.zeros((0,5), dtype=np.float32)

        draw_boxes(frame, kept, color=(0,255,0), thickness=2)
        out_vid.write(frame)

        if args.display and (frame_idx % args.display_interval == 0):
            cv2.imshow("out", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if args.max_frames and frame_idx >= args.max_frames:
            print("[i] Reached max frames limit")
            break

    t1 = time.time()
    print(f"[i] Processed {frame_idx} frames in {t1-t0:.2f} s ({frame_idx/(t1-t0):.2f} FPS)")
    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=str, default="yolov8_fp16.engine", help="TensorRT engine file")
    parser.add_argument("--video", type=str, default="input.mp4", help="Input video path")
    parser.add_argument("--out", type=str, default="out.mp4", help="Output video path")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--display", action="store_true", help="Show GUI windows")
    parser.add_argument("--display-interval", dest="display_interval", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames (0 = all)")
    args = parser.parse_args()

    main(args)