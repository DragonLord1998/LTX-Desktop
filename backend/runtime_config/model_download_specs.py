"""Canonical model download specs and required-model policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from state.app_state_types import ModelFileType


@dataclass(frozen=True, slots=True)
class ModelFileDownloadSpec:
    relative_path: Path
    expected_size_bytes: int
    is_folder: bool
    repo_id: str
    description: str

    @property
    def name(self) -> str:
        return self.relative_path.name


MODEL_FILE_ORDER: tuple[ModelFileType, ...] = (
    "checkpoint",
    "upsampler",
    "text_encoder",
    "zit",
    "dev_checkpoint",
    "distilled_lora",
    "qwen_transformer",
    "qwen_text_encoder",
    "qwen_vae",
)


DEFAULT_MODEL_DOWNLOAD_SPECS: dict[ModelFileType, ModelFileDownloadSpec] = {
    "checkpoint": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-22b-distilled.safetensors"),
        expected_size_bytes=44_000_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="Main transformer model (distilled, BF16 — FP8 quantized at runtime)",
    ),
    "upsampler": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
        expected_size_bytes=1_900_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="2x Upscaler",
    ),
    "text_encoder": ModelFileDownloadSpec(
        relative_path=Path("gemma-3-12b-it-qat-q4_0-unquantized"),
        expected_size_bytes=25_000_000_000,
        is_folder=True,
        repo_id="Lightricks/gemma-3-12b-it-qat-q4_0-unquantized",
        description="Gemma text encoder (bfloat16)",
    ),
    "zit": ModelFileDownloadSpec(
        relative_path=Path("Z-Image-Turbo"),
        expected_size_bytes=31_000_000_000,
        is_folder=True,
        repo_id="Tongyi-MAI/Z-Image-Turbo",
        description="Z-Image-Turbo model for text-to-image generation",
    ),
    "dev_checkpoint": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-22b-dev-fp8.safetensors"),
        expected_size_bytes=28_000_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="Dev transformer model (FP8 — full quality, slower inference)",
    ),
    "distilled_lora": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-22b-distilled-lora-384.safetensors"),
        expected_size_bytes=7_600_000_000,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3",
        description="Distilled LoRA for dev model two-stage refinement",
    ),
    "qwen_transformer": ModelFileDownloadSpec(
        relative_path=Path("qwen-image-edit-transformer"),
        expected_size_bytes=10_000_000_000,
        is_folder=True,
        repo_id="Qwen/Qwen-Image-Edit-2511",
        description="Qwen Image Edit transformer (instruction-based editing)",
    ),
    "qwen_text_encoder": ModelFileDownloadSpec(
        relative_path=Path("qwen-image-text-encoder"),
        expected_size_bytes=5_000_000_000,
        is_folder=True,
        repo_id="Qwen/Qwen-Image",
        description="Qwen Image text encoder",
    ),
    "qwen_vae": ModelFileDownloadSpec(
        relative_path=Path("qwen-image-vae"),
        expected_size_bytes=500_000_000,
        is_folder=True,
        repo_id="Qwen/Qwen-Image",
        description="Qwen Image VAE",
    ),
}


DEFAULT_REQUIRED_MODEL_TYPES: frozenset[ModelFileType] = frozenset(
    {"checkpoint", "upsampler", "zit"}
)


def resolve_required_model_types(
    base_required: frozenset[ModelFileType],
    has_api_key: bool,
    use_local_text_encoder: bool = False,
) -> frozenset[ModelFileType]:
    if not base_required:
        return base_required
    if has_api_key and not use_local_text_encoder:
        return base_required
    return cast(frozenset[ModelFileType], base_required | {"text_encoder"})
