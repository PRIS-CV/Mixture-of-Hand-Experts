import json
import os
from Hamer.hamer_detector import HamerDetector
import argparse
import numpy as np
from PIL import Image
import cv2
from torch import optim
from gating_network import Gate
from diffusers import ControlNetModel, AutoencoderKL, UniPCMultistepScheduler
from diffusers.utils import load_image
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
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
    # train images description
    parser.add_argument('--meta_json', type=str, default="")
    # train
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--generate', type=str, default="")
    parser.add_argument('--ground_truth', type=str, default="")
    # set seed
    parser.add_argument('--seed', type=int, default=42)
    # moe
    parser.add_argument('--moe', type=bool, default=True)
    # inpainting
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--strength', type=float, default=0.75)
    # output dir
    parser.add_argument('--output', type=str, default="")

    args = parser.parse_args()
    return args


args = parse_args()


# 定义混合专家模型
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

    def forward(self, prompt, n_prompt, init_image, control_image, mask_image, target_mask,
                controlnet_conditioning_scale, MOE):
        g = torch.Generator()
        g.manual_seed(args.seed)
        images = self.pipe(
            prompt, negative_prompt=n_prompt, image=init_image, control_image=control_image,
            mask_image=mask_image,
            num_inference_steps=args.num_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale, strength=args.strength, generator=g, MOE=MOE,
            gating_network=self.gating_network
        ).images
        image = target_mask * images[0]
        return image

### load checkpoints
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
trained_controlnets = [controlnet_mesh, controlnet_depth, controlnet_pose]
model = MoE(trained_controlnets)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.gating_network.parameters(), lr=args.lr)


def save_model(save_path, iteration, optimizer, model):
    torch.save({'iteration': iteration,
                'optimizer_dict': optimizer.state_dict(),
                'model_dict': model.state_dict()},
               save_path)
    print("model save success")


num_epochs = args.epoch
for epoch in range(num_epochs):
    f_prompt = open(args.meta_json)
    inputs = f_prompt.readlines()
    for file_info in inputs:
        file_info = json.loads(file_info)
        file_name = file_info["img"]
        n_prompt = file_info["negtive_prompt"]
        prompt = 'perfect hand, realskin, realistic, best quality, extremely detailed'
        if n_prompt is not None:
            n_prompt = n_prompt + " deformed, fake 3D rendered image, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"
        else:
            n_prompt = "deformed, fake 3D rendered image, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blue"
        target_image = load_image(args.ground_truth + file_name).convert("RGB")
        init_image = load_image(args.generate + file_name).convert("RGB")
        depth_condition, pose_condition, mesh_condition, mask = hamer_detector(init_image, 2.5, 1.1)
        mask_target = np.array(mask) / 255
        mask_target = 1 * (mask_target > 0.5)
        target_data = mask_target * target_image
        control_image = [mesh_condition, depth_condition, pose_condition]
        mask = np.array(mask) / 255
        mask = 1 * (mask > 0.5)
        controlnet_conditioning_scale = 0.5  # default
        moe = args.moe
        # 前向传播
        outputs = model(prompt, n_prompt, init_image, control_image, mask, mask_target,
                        controlnet_conditioning_scale, moe)
        # 计算损失
        loss = criterion(torch.tensor(outputs, dtype=float), torch.tensor(target_data, dtype=float))
        loss.requires_grad_(True)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
    save_model(args.output, epoch, optimizer, model)
