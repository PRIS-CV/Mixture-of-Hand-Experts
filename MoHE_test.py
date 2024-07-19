import json
import os

import argparse
from functools import partial

import numpy as np
from PIL import Image
import cv2
from Hamer.hamer_detector import HamerDetector
from gating_network import Gate
from diffusers import ControlNetModel, AutoencoderKL, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from diffusers.utils import load_image
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
from asdff import AdCnPipeline, AdPipeline, AdCnXLPipeline, yolo_detector
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    # checkpoints
    parser.add_argument('--base_model', type=str, default="")
    parser.add_argument('--vae', type=str, default="")
    parser.add_argument('--controlnet_mesh', type=str, default="")
    parser.add_argument('--controlnet_pose', type=str, default="")
    parser.add_argument('--controlnet_depth', type=str, default="")
    parser.add_argument('--condition_extractor', type=str, default="")
    parser.add_argument('--hand_detector', type=str, default="")
    parser.add_argument('--gate_network', type=str, default="")
    # test images description
    parser.add_argument('--meta_json', type=str, default="")
    # set seed
    parser.add_argument('--seed', type=int, default=42)
    # moe
    parser.add_argument('--moe', type=bool, default=True)
    # ADetailer
    parser.add_argument('--ad', type=bool, default=True)
    # inpainting
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--strength', type=float, default=0.75)
    # output dir
    parser.add_argument('--output', type=str, default="")

    args = parser.parse_args()
    return args


args = parse_args()


class MoE(nn.Module):
    def __init__(self, trained_controlnets):
        super(MoE, self).__init__()
        self.controlnets = trained_controlnets
        self.num_experts = len(trained_controlnets)
        self.vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float16).to("cuda")
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            args.base_model,
            controlnet=trained_controlnets,
            vae=self.vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16,
        ).to("cuda")
        self.gating_network = Gate()
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()

    def forward(self, prompt, a_prompt, n_prompt, init_image, control_image, mask_image, controlnet_conditioning_scale,
                MOE):
        g = torch.Generator()
        g.manual_seed(args.seed)
        images = self.pipe(
            prompt, prompt_2=a_prompt, negative_prompt=n_prompt, image=init_image, control_image=control_image,
            mask_image=mask_image,
            num_inference_steps=args.num_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale, strength=args.strength, generator=g, MOE=MOE,
            gating_network=self.gating_network
        ).images
        return images[0]


# load checkpoints
controlnet_mesh = ControlNetModel.from_pretrained(
    args.controlnet_mesh,
    # variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
controlnet_pose = ControlNetModel.from_pretrained(
    args.controlnet_pose,
    # variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
controlnet_depth = ControlNetModel.from_pretrained(
    args.controlnet_depth,
    # variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
hamer_detector = HamerDetector(
        model_dir=args.condition_extractor,
        body_detector="vitdet",
        rescale_factor=2.0,
        device="cuda:0"
    )
# After Detailer pipeline
pipe = AdCnXLPipeline.from_pretrained(
    args.base_model,
    controlnet=[controlnet_mesh, controlnet_depth, controlnet_pose],
    vae=args.vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# yolo detector
hand_detector = partial(yolo_detector, model_path=args.hand_detector)

trained_controlnets = [controlnet_mesh, controlnet_depth, controlnet_pose]
model = MoE(trained_controlnets)
model.gating_network.load_state_dict(
    torch.load(args.gate_network, map_location='cpu')['model_dict'])
model.cuda()

# start inference
f_prompt = open(args.meta_json)
inputs = f_prompt.readlines()
for file_info in inputs:
    file_info = json.loads(file_info)
    file_name = file_info["img"]
    prompt = file_info["txt"]
    n_prompt = file_info["negtive_prompt"]
    a_prompt = 'perfect hand, realskin, realistic, best quality, extremely detailed'
    if n_prompt is not None:
        n_prompt = n_prompt + " deformed, fake 3D rendered image, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"
    else:
        n_prompt = "deformed, fake 3D rendered image, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"

    init_image = load_image('/home/wangyuxuan/HandRefiner/testset_100_20240620/1/' + file_name).convert(
        "RGB")
    depth_condition, pose_condition, mesh_condition, mask = hamer_detector(init_image, 2.5, 1.1)
    if depth_condition is None:
        print('no valid hands')
        outputs = init_image
    else:
        control_image = [mesh_condition, depth_condition, pose_condition]
        controlnet_conditioning_scale = [0.5, 0.5, 0.5]  # default
        moe = args.moe
        outputs = model(prompt, a_prompt, n_prompt, init_image, control_image, mask, controlnet_conditioning_scale,
                        moe)
        if args.ad:
            common = {
                "image": outputs,
                "control_image": [mesh_condition, depth_condition, pose_condition],
                "num_inference_steps": 30,
            }
            inpaint = {
                "prompt": prompt,
                "control_scale": controlnet_conditioning_scale,
                "negtive_prompt": n_prompt,
            }
            images = pipe(
                common=common, inpaint_only=inpaint, detectors=hand_detector, images=outputs
            ).images
    outputs.save(args.output + file_name)
