from __future__ import annotations

from functools import cached_property

from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from .pipeline_controlnet_inpaint_sd_xl import StableDiffusionXLControlNetInpaintPipeline
from .base import AdPipelineBase


class AdPipeline(AdPipelineBase, StableDiffusionPipeline):
    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionPipeline


class AdCnPipeline(AdPipelineBase, StableDiffusionControlNetPipeline):
    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionControlNetInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=self.config.requires_safety_checker,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionControlNetPipeline


class AdCnXLPipeline(AdPipelineBase, StableDiffusionXLControlNetPipeline, StableDiffusionXLControlNetInpaintPipeline):
    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionXLControlNetInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.scheduler,
        )

    @property
    def txt2img_class(self):
        return StableDiffusionXLControlNetPipeline

