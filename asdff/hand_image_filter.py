from __future__ import annotations
import os

from pathlib import Path
from utils import (
    bbox_padding,
    mask_dilate,
    mask_gaussian_blur,
)
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("Please install ultralytics using `pip install ultralytics`")
    raise


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


def create_mask_from_bbox(
        bboxes: np.ndarray, shape: tuple[int, int]
) -> list[Image.Image]:
    """
    Parameters
    ----------
        bboxes: list[list[float]]
            list of [x1, y1, x2, y2]
            bounding boxes
        shape: tuple[int, int]
            shape of the image (width, height)

    Returns
    -------
        masks: list[Image.Image]
        A list of masks

    """
    masks = []
    for bbox in bboxes:
        mask = Image.new("L", shape, "black")
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(bbox, fill="white")
        masks.append(mask)
    return masks


def mask_to_pil(masks: torch.Tensor, shape: tuple[int, int]) -> list[Image.Image]:
    """
    Parameters
    ----------
    masks: torch.Tensor, dtype=torch.float32, shape=(N, H, W).
        The device can be CUDA, but `to_pil_image` takes care of that.

    shape: tuple[int, int]
        (width, height) of the original image

    Returns
    -------
    images: list[Image.Image]
    """
    n = masks.shape[0]
    return [to_pil_image(masks[i], mode="L").resize(shape) for i in range(n)]


def yolo_detector(
        image: Image.Image, model_path: str | Path | None = None, confidence: float = 0.5
) -> list[Image.Image] | None:
    if not model_path:
        model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt")
    model = YOLO(model_path)
    pred = model(image, conf=confidence)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    if bboxes.size == 0:
        return None

    if pred[0].masks is None:
        masks = create_mask_from_bbox(bboxes, image.size)
    else:
        masks = mask_to_pil(pred[0].masks.data, image.size)

    return masks


for filename in os.listdir("/data/wangyuxuan/Hagrid"):
    if filename == "call":
        continue
    for file in os.listdir("/data/wangyuxuan/Hagrid/" + filename):
        # import pdb;
        # pdb.set_trace()
        image = Image.open("/data/wangyuxuan/Hagrid/" + filename + '/' + file)
        masks = yolo_detector(image, '/data/wangyuxuan/adetailer/hand_yolov8s.pt', confidence=0.8)
        index = 0
        if masks is None:
            continue
        for k, mask in enumerate(masks):
            mask = mask.convert("L")
            mask = mask_dilate(mask, 4)
            bbox = mask.getbbox()
            if bbox is None:
                print(f"No object in {ordinal(k + 1)} mask.")
                continue
            index += 1
            mask = mask_gaussian_blur(mask, 4)
            bbox_padded = bbox_padding(bbox, image.size, 32)
            crop_image = image.crop(bbox_padded)
            crop_image.save('/data/wangyuxuan/hands/Hagrid/' + filename + "/" + str(index) + "_" + file)
