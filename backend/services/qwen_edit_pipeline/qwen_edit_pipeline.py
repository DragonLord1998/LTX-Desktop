"""Qwen-Image-Edit-2511 pipeline Protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch
    from PIL import Image


@runtime_checkable
class QwenImageEditPipeline(Protocol):
    """Protocol for instruction-based image editing using Qwen-Image-Edit-2511."""

    @staticmethod
    def create(
        transformer_path: str,
        text_encoder_path: str,
        vae_path: str,
        device: torch.device,
        *,
        quantization: str = "fp8",  # "fp8" | "nf4" | "bf16"
    ) -> QwenImageEditPipeline: ...

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
    ) -> Image.Image: ...

    def load_lora(self, lora_path: str, *, strength: float = 0.8) -> None: ...

    def unload_lora(self) -> None: ...

    def to(self, device: str) -> None: ...
