"""Real implementation of QwenImageEditPipeline using Diffusers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class LTXQwenEditPipeline:
    """Instruction-based image editing pipeline backed by Qwen-Image-Edit-2511."""

    def __init__(self, pipe: object, device: torch.device) -> None:
        self._pipe = pipe
        self._device = device
        self._active_lora_path: str | None = None
        self._lora_scale: float = 0.8

    @staticmethod
    def create(
        transformer_path: str,
        text_encoder_path: str,
        vae_path: str,
        device: torch.device,
        *,
        quantization: str = "fp8",  # "fp8" | "nf4" | "bf16"
    ) -> "LTXQwenEditPipeline":
        try:
            from diffusers.pipelines.pipeline_utils import DiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "diffusers is required for QwenImageEditPipeline. "
                "Install it with: pip install diffusers"
            ) from exc

        if quantization == "nf4":
            try:
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
                )
                pipe = DiffusionPipeline.from_pretrained(
                    transformer_path,
                    text_encoder=text_encoder_path or None,
                    vae=vae_path or None,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,
                ).to(device)
            except ImportError as exc:
                raise RuntimeError(
                    "bitsandbytes is required for nf4 quantization. "
                    "Install it with: pip install bitsandbytes"
                ) from exc
        elif quantization == "fp8":
            pipe = DiffusionPipeline.from_pretrained(
                transformer_path,
                text_encoder=text_encoder_path or None,
                vae=vae_path or None,
                torch_dtype=torch.bfloat16,
            ).to(device)
        else:
            # bf16 — plain load
            pipe = DiffusionPipeline.from_pretrained(
                transformer_path,
                text_encoder=text_encoder_path or None,
                vae=vae_path or None,
                torch_dtype=torch.bfloat16,
            ).to(device)

        logger.info("Loaded QwenImageEditPipeline with quantization=%s on %s", quantization, device)
        return LTXQwenEditPipeline(pipe=pipe, device=device)

    @torch.inference_mode()
    def edit(
        self,
        *,
        image: Image.Image,
        instruction: str,
        seed: int,
        num_inference_steps: int,
        height: int | None = None,
        width: int | None = None,
        negative_prompt: str = " ",
        cfg_scale: float = 4.0,
    ) -> Image.Image:
        generator = torch.Generator(device=self._device).manual_seed(seed)

        call_kwargs: dict[str, object] = dict(
            image=image,
            prompt=instruction,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=cfg_scale,
            generator=generator,
        )
        if self._active_lora_path is not None:
            call_kwargs["cross_attention_kwargs"] = {"scale": self._lora_scale}
        if height is not None:
            call_kwargs["height"] = height
        if width is not None:
            call_kwargs["width"] = width

        result = self._pipe(**call_kwargs)  # type: ignore[operator]
        images: list[Image.Image] = result.images  # type: ignore[union-attr]
        return images[0]

    def load_lora(self, lora_path: str, *, strength: float = 0.8) -> None:
        self._pipe.load_lora_weights(lora_path)  # type: ignore[union-attr]
        self._active_lora_path = lora_path
        self._lora_scale = strength
        logger.info("Loaded LoRA from %s with strength=%.2f", lora_path, strength)

    def unload_lora(self) -> None:
        self._pipe.unload_lora_weights()  # type: ignore[union-attr]
        self._active_lora_path = None
        logger.info("Unloaded LoRA")

    def to(self, device: str) -> None:
        self._pipe.to(device)  # type: ignore[union-attr]
        self._device = torch.device(device)
