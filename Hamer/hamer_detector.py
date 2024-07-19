from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import matplotlib
from typing import Tuple, List, Any
from skimage.filters import gaussian
from trimesh.ray.ray_pyembree import RayMeshIntersector
from trimesh import Trimesh
import trimesh
from .hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from .hamer.utils import recursive_to
from .hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from .hamer.utils.renderer import Renderer, cam_crop_to_full
from PIL import Image, ImageDraw, ImageFilter
from .hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from .vitpose_model import ViTPoseModel
from os.path import join, dirname
from diffusers.utils import load_image
from glob import glob
from tqdm import tqdm
from .utils import scale_to_square, scale_rectangle, create_mask_from_bbox, get_rays, draw_handpose, get_bounding_box, \
    is_overlapping, calculate_iou, calculate_area, cal_laplacian, filter_bboxes, mask_dilate, mask_gaussian_blur, \
    refine_mask

COLOR = (1.0, 1.0, 0.9)


class HamerDetector:
    def __init__(self, model_dir, body_detector, rescale_factor, device):
        # HaMeR model
        self.model, self.model_cfg = load_hamer(join(model_dir, "hamer/hamer_ckpts/checkpoints/hamer.ckpt"))
        self.model.to(device)
        self.model.eval()

        # body detector
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            cfg_path = join(dirname(__file__), "hamer/configs/cascade_mask_rcnn_vitdet_h_75ep.py")
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = join(model_dir, "hamer/vitdet_ckpts/model_final_f05665.pkl")
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif self.body_detector == 'regnety':
            from detectron2 import model_zoo
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py',
                                                  trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        self.detector.model.to(device)

        # keypoint detector
        self.cpm = ViTPoseModel(join(model_dir, "hamer/vitpose_ckpts/vitpose+_huge/wholebody.pth"), device)

        # renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

        self.rescale_factor = rescale_factor
        self.device = device

    @torch.no_grad()
    def __call__(self, image: Image.Image, bbox_scale_factor, mask_scale_factor):

        # Detect humans in image
        img_cv2 = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        det_out = self.detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []
        sum_valid = []
        mean_valid = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
                sum_valid.append(sum(valid))
                mean_valid.append(np.mean(keyp[:, 2]))

            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)
                sum_valid.append(sum(valid))
                mean_valid.append(np.mean(keyp[:, 2]))

        dropped = [False] * len(bboxes)
        # do over-lapping
        for i in range(len(bboxes)):
            for j in range(len(bboxes)):
                if i == j:
                    continue
                if is_overlapping(bboxes[i], bboxes[j]) and mean_valid[i] - mean_valid[j] < -0.1:
                    dropped[i] = True
        # filter with size
        for i in range(len(bboxes)):
            if calculate_area(bboxes[i]) <= 100:
                dropped[i] = True
        # filter with laps
        if len(bboxes) > 2:
            rt_bboxes, sharpnesses = filter_bboxes(
                bboxes,
                min_ratio=0.0,
                max_face_num=10,
                max_area=image.size[0] * image.size[1],
                image=image,
            )
            for k, sharpness in enumerate(sharpnesses):
                if sharpness < 100 or sharpness > 5000:
                    dropped[k] = True

        bboxes = [x for i, x in enumerate(bboxes) if not dropped[i]]
        is_right = [x for i, x in enumerate(is_right) if not dropped[i]]
        sum_valid = [x for i, x in enumerate(sum_valid) if not dropped[i]]
        mean_valid = [x for i, x in enumerate(mean_valid) if not dropped[i]]

        if bboxes == []:
            return None, None, None, None

        bboxes = np.array(bboxes).astype(int)

        depth_condition, pose_condition, mesh_condition = self.inference(
            image,
            bboxes,
            is_right
        )
        global_mask = Image.fromarray(np.zeros((image.size[0], image.size[1]))).convert('L')
        for bbox in bboxes:
            bbox_padded, _ = scale_to_square(bbox, bbox_scale_factor)
            crop_depth_condition = depth_condition.crop(bbox_padded)
            bbox_from_depth = get_bounding_box(crop_depth_condition)
            bbox_for_mask = [
                min(bbox[0], bbox_from_depth[0] + bbox_padded[0]),
                min(bbox[1], bbox_from_depth[1] + bbox_padded[1]),
                max(bbox[2], bbox_from_depth[2] + bbox_padded[0]),
                max(bbox[3], bbox_from_depth[3] + bbox_padded[1]),
            ]
            mask = create_mask_from_bbox(scale_rectangle(bbox_for_mask, mask_scale_factor), image.size)
            nonzero_y, nonzero_x = np.asarray(mask).nonzero()
            ymin = min(nonzero_y)
            ymax = max(nonzero_y)
            xmin = min(nonzero_x)
            xmax = max(nonzero_x)
            crop_mask = mask.crop([xmin, ymin, xmax, ymax])
            global_mask.paste(crop_mask, [xmin, ymin, xmax, ymax])
        mask = global_mask
        mask = mask_dilate(mask, 4)
        # --再一次修复和膨胀--
        mask = refine_mask(mask)
        mask = mask_dilate(mask, 4)
        # -------------------
        mask = mask_gaussian_blur(mask, 4)
        return depth_condition, pose_condition, mesh_condition, mask

    def inference(self, patch: Image.Image, bbox, right):
        img_cv2 = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)
        H, W, C = img_cv2.shape
        dataset = ViTDetDataset(self.model_cfg, img_cv2, np.stack(bbox), np.stack(right),
                                rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_hand_peaks = []
        all_box_size = []
        all_box_center = []

        depth_condition, pose_condition, mesh_condition = None, None, None

        padded_depthmap = np.zeros((2 * H, 2 * W))
        padded_posemap = np.zeros((2 * H, 2 * W, 3))
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            multiplier = (2 * batch['right'] - 1)
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                               scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:, None, None] / 255) + (
                        DEFAULT_MEAN[:, None, None] / 255)
                input_patch = input_patch.permute(1, 2, 0).numpy()

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                keyp2d = out['pred_keypoints_2d'][n].detach().cpu().numpy()
                box_size = batch["box_size"][n].detach().cpu().numpy()
                box_center = batch["box_center"][n].detach().cpu().numpy()
                pred_cam = out['pred_cam'][n].detach().cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                cam_t = pred_cam_t_full[n]
                focal_length = scaled_focal_length.detach().cpu().numpy()
                res = int(box_size)
                camera_t = np.array([-pred_cam[1], -pred_cam[2], -2 * focal_length / (res * pred_cam[0] + 1e-9)])
                faces_new = np.array([[92, 38, 234],
                                      [234, 38, 239],
                                      [38, 122, 239],
                                      [239, 122, 279],
                                      [122, 118, 279],
                                      [279, 118, 215],
                                      [118, 117, 215],
                                      [215, 117, 214],
                                      [117, 119, 214],
                                      [214, 119, 121],
                                      [119, 120, 121],
                                      [121, 120, 78],
                                      [120, 108, 78],
                                      [78, 108, 79]])
                faces = np.concatenate([self.model.mano.faces, faces_new], axis=0)
                mesh = Trimesh(vertices=verts, faces=faces)
                h, w = int(box_size), int(box_size)
                rays_o, rays_d = get_rays(w, h, focal_length, focal_length, w / 2, h / 2, camera_t, True)
                if int(box_size) == 0:
                    continue

                coords = np.array(list(np.ndindex(h, w))).reshape(h, w, -1).transpose(1, 0, 2).reshape(-1, 2)
                intersector = RayMeshIntersector(mesh)
                points, index_ray, _ = intersector.intersects_location(rays_o, rays_d, multiple_hits=False)

                tri_index = intersector.intersects_first(rays_o, rays_d)

                tri_index = tri_index[index_ray]

                assert len(index_ray) == len(tri_index)
                if is_right == 0:
                    discriminator = (np.sum(mesh.face_normals[tri_index] * rays_d[index_ray], axis=-1) >= 0)
                else:
                    discriminator = (np.sum(mesh.face_normals[tri_index] * rays_d[index_ray], axis=-1) <= 0)
                points = points[discriminator]  # ray intersects in interior faces, discard them

                if len(points) == 0:
                    print("no hands detected")
                    continue

                depth = (points + camera_t)[:, -1]
                index_ray = index_ray[discriminator]
                pixel_ray = coords[index_ray]

                minval = np.min(depth)
                maxval = np.max(depth)
                depthmap = np.zeros([h, w])
                depthmap[pixel_ray[:, 0], pixel_ray[:, 1]] = 1.0 - (0.8 * (depth - minval) / (maxval - minval))
                depthmap *= 255

                cropped_depthmap = depthmap
                if cropped_depthmap is None:
                    print("Depth reconstruction failed for image")
                    continue

                resized_cropped_depthmap = cv2.resize(cropped_depthmap, (int(box_size), int(box_size)),
                                                      interpolation=cv2.INTER_LINEAR)
                nonzero_y, nonzero_x = (resized_cropped_depthmap != 0).nonzero()
                if len(nonzero_y) == 0 or len(nonzero_x) == 0:
                    print("Depth reconstruction failed for image")
                    continue

                crop_xc = box_center[0]
                crop_yc = box_center[1]
                crop_y_min = int(crop_yc - box_size / 2)
                crop_x_min = int(crop_xc - box_size / 2)

                padded_depthmap[crop_y_min + nonzero_y, crop_x_min + nonzero_x] = resized_cropped_depthmap[
                    nonzero_y, nonzero_x]

                keyp2d = keyp2d + 0.5
                canv = np.zeros(shape=(int(box_size), int(box_size), 3), dtype=np.uint8)
                peaks = []
                peaks.append(keyp2d)
                pose = draw_handpose(canv, peaks)
                pose = cv2.cvtColor(pose, cv2.COLOR_BGR2RGB)

                if is_right == 0:
                    pose = np.flip(pose, 1)
                nonzero_y, nonzero_x, _ = (pose != 0).nonzero()
                crop_xc = box_center[0]
                crop_yc = box_center[1]
                crop_y_min = int(crop_yc - box_size / 2)
                crop_x_min = int(crop_xc - box_size / 2)

                padded_posemap[crop_y_min + nonzero_y, crop_x_min + nonzero_x, :] = pose[nonzero_y, nonzero_x, :]

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_hand_peaks.append(keyp2d)
                all_box_size.append(int(box_size))
                all_box_center.append(box_center)

        depth_condition = Image.fromarray(padded_depthmap[0:int(H), 0:int(W)]).convert('L')
        pose_condition = Image.fromarray(
            cv2.cvtColor(np.uint8(padded_posemap[0:int(H), 0:int(W), :]), cv2.COLOR_BGR2RGB))

        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=COLOR,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = self.renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=img_size[n],
                is_right=all_right,
                **misc_args,
            )

            mesh_condition = cam_view[:, :, :3] * cam_view[:, :, 3:]
            mesh_condition = Image.fromarray(
                cv2.cvtColor(np.uint8(255 * mesh_condition[:, :, ::-1]), cv2.COLOR_BGR2RGB))
        return depth_condition, pose_condition, mesh_condition
