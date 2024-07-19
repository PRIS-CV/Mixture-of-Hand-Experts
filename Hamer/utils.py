from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import matplotlib


def calculate_area(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    return (x2 - x1) * (y2 - y1)


def calculate_iou(box1, box2):
    """
    computing the IoU of two boxes.
    Args:
        box: [x1, y1, x2, y2],通过左上和右下两个顶点坐标来确定矩形
    Return:
        IoU: IoU of box1 and box2.
    """
    px1 = box1[0]
    py1 = box1[1]
    px2 = box1[2]
    py2 = box1[3]

    gx1 = box2[0]
    gy1 = box2[1]
    gx2 = box2[2]
    gy2 = box2[3]

    parea = calculate_area(box1)  # 计算P的面积
    garea = calculate_area(box2)  # 计算G的面积

    # 求相交矩形的左上和右下顶点坐标(x1, y1, x2, y2)
    x1 = max(px1, gx1)  # 得到左上顶点的横坐标
    y1 = max(py1, gy1)  # 得到左上顶点的纵坐标
    x2 = min(px2, gx2)  # 得到右下顶点的横坐标
    y2 = min(py2, gy2)  # 得到右下顶点的纵坐标
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return 0

    area = w * h  # G∩P的面积
    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)

    return IoU


def is_overlapping(rect1, rect2):
    return not (rect1[2] <= rect2[0] or rect1[0] >= rect2[2] or
                rect1[3] <= rect2[1] or rect1[1] >= rect2[3])


def cal_laplacian(image: np.ndarray):
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    return sharpness


def filter_bboxes(bboxes, min_ratio=0.125, max_face_num=6, max_area=0, image=None):
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in bboxes]
    filted_bboxes = []
    for bbox, area in zip(bboxes, areas):
        if max(areas) * min_ratio < area < max_area:
            filted_bboxes.append(bbox)

    # -------加入模糊过滤逻辑--------
    sharpnesses = []
    for bbox in filted_bboxes:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        bbox_shrink = (x1 + w // 4, y1 + h // 4, x2 - w // 4, y2 - h // 4)
        cropped_image = image.crop(bbox_shrink)
        cropped_image = cv2.cvtColor(np.asarray(cropped_image), cv2.COLOR_RGB2GRAY)
        sharpness = cal_laplacian(cropped_image)
        sharpnesses.append(sharpness)

    rt_bboxes, rt_sharpnesses = [], []
    for bbox, sharpness in zip(filted_bboxes, sharpnesses):
        if sharpness > 0 and sharpness / max(sharpnesses) > 0:
            rt_bboxes.append(bbox)
            rt_sharpnesses.append(sharpness)
    return rt_bboxes, rt_sharpnesses


def create_mask_from_bbox(
        bbox: np.ndarray, shape: Tuple[int, int]
) -> List[Image.Image]:
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
    mask = Image.new("L", shape, "black")
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle(bbox, fill="white")
    return mask


def mask_dilate(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    arr = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (value, value))
    dilated = cv2.dilate(arr, kernel, iterations=1)
    return Image.fromarray(dilated)


def refine_mask(mask):
    mask_np = np.array(mask)
    _, binary_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.zeros_like(binary_mask)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    filled_mask_pil = Image.fromarray(filled_mask)
    return filled_mask_pil


def mask_gaussian_blur(image: Image.Image, value: int = 4) -> Image.Image:
    if value <= 0:
        return image

    blur = ImageFilter.GaussianBlur(value)
    return image.filter(blur)


def get_rays(W, H, fx, fy, cx, cy, c2w_t, center_pixels):  # rot = I

    j, i = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32))
    if center_pixels:
        i = i.copy() + 0.5
        j = j.copy() + 0.5

    directions = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], -1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    rays_o = np.expand_dims(c2w_t, 0).repeat(H * W, 0)

    rays_d = directions  # (H, W, 3)
    rays_d = (rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)).reshape(-1, 3)

    return rays_o, rays_d


def draw_handpose(canvas, all_hand_peaks):
    eps = 0.01
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    # print(all_hand_peaks)
    for peaks in all_hand_peaks:
        peaks = np.array(peaks)

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2),
                         matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def get_bounding_box(image):
    np_image = np.asarray(image)
    non_zero_coords = np.argwhere(np_image)
    if non_zero_coords.size > 0:
        top_left = non_zero_coords.min(axis=0)
        bottom_right = non_zero_coords.max(axis=0)

        return (top_left[1], top_left[0], bottom_right[1], bottom_right[0])
    else:
        return None


def scale_rectangle(rect, n):
    x1, y1, x2, y2 = rect
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    # scale
    half_width = (x2 - x1) / 2 * n
    half_height = (y2 - y1) / 2 * n
    new_x1 = int(center_x - half_width)
    new_y1 = int(center_y - half_height)
    new_x2 = int(center_x + half_width)
    new_y2 = int(center_y + half_height)

    new_rect = (new_x1, new_y1, new_x2, new_y2)
    return new_rect


def scale_to_square(rect, n):
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    new_width = int(width * n)
    new_height = int(height * n)
    side_length = max(new_width, new_height)
    new_x1 = cx - side_length // 2
    new_y1 = cy - side_length // 2
    new_x2 = cx + side_length // 2
    new_y2 = cy + side_length // 2

    rel_x1 = x1 - new_x1
    rel_y1 = y1 - new_y1
    rel_x2 = x2 - new_x1
    rel_y2 = y2 - new_y1

    new_rect = (new_x1, new_y1, new_x2, new_y2)
    relative_rect = (rel_x1, rel_y1, rel_x2, rel_y2)
    return new_rect, relative_rect
