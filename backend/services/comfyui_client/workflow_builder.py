"""Build ComfyUI workflow JSON for LTX-2.3 video generation.

These workflows are based on the official Lightricks/ComfyUI-LTXVideo
example workflows (LTX-2.3 T2V/I2V Single Stage and Two Stage).

Reference: https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.3
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class WorkflowParams:
    """Parameters for building a ComfyUI LTX-2.3 workflow.

    All model-name fields are required — the caller must supply them
    from ComfyUISettings so there is a single source of truth.
    """

    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    checkpoint_name: str
    dev_checkpoint_name: str
    text_encoder_name: str
    upscaler_name: str
    distilled_lora_name: str
    lora_strength: float
    seed: int | None = None
    model: Literal["fast", "dev"] = "fast"
    image_filename: str | None = None
    use_two_stage: bool = False


def build_workflow(params: WorkflowParams) -> dict[str, Any]:
    """Build a ComfyUI API-format workflow for LTX-2.3 generation.

    Supports:
    - Text-to-video (T2V) when image_filename is None
    - Image-to-video (I2V) when image_filename is provided
    - Single-stage (distilled fast, or dev full)
    - Two-stage (distilled first pass + upscale + refinement pass)
    """
    if params.use_two_stage:
        return _build_two_stage_workflow(params)
    return _build_single_stage_workflow(params)


# ---------------------------------------------------------------------------
# Shared node builders
# ---------------------------------------------------------------------------


def _add_model_loaders(
    wf: dict[str, Any],
    params: WorkflowParams,
    checkpoint: str,
) -> None:
    """Add checkpoint, audio VAE, and text encoder loaders (nodes 1-3)."""
    wf["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": checkpoint},
    }
    wf["2"] = {
        "class_type": "LTXVAudioVAELoader",
        "inputs": {"ckpt_name": checkpoint},
    }
    wf["3"] = {
        "class_type": "LTXAVTextEncoderLoader",
        "inputs": {
            "clip_name": params.text_encoder_name,
            "ckpt_name": checkpoint,
            "type": "default",
        },
    }


def _add_text_encoding(
    wf: dict[str, Any],
    params: WorkflowParams,
) -> None:
    """Add CLIP text encode + LTX conditioning (nodes 10-12)."""
    wf["10"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["3", 0], "text": params.prompt},
    }
    wf["11"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["3", 0], "text": params.negative_prompt},
    }
    wf["12"] = {
        "class_type": "LTXVConditioning",
        "inputs": {
            "positive": ["10", 0],
            "negative": ["11", 0],
            "frame_rate": params.fps,
        },
    }


def _add_latent_setup(
    wf: dict[str, Any],
    params: WorkflowParams,
    *,
    i2v_strength: float = 0.7,
) -> None:
    """Add empty latent, optional I2V conditioning, audio latent, and AV concat (nodes 20-25)."""
    is_i2v = params.image_filename is not None

    wf["20"] = {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": params.width,
            "height": params.height,
            "length": params.num_frames,
            "batch_size": 1,
        },
    }

    if is_i2v:
        wf["21"] = {
            "class_type": "LoadImage",
            "inputs": {"image": params.image_filename, "upload": "image"},
        }
        wf["22"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": ["21", 0], "target_tokens": 18},
        }

    wf["23"] = {
        "class_type": "LTXVImgToVideoConditionOnly",
        "inputs": {
            "vae": ["1", 2],
            "image": ["22", 0] if is_i2v else ["20", 0],
            "latent": ["20", 0],
            "strength": i2v_strength,
            "bypass": not is_i2v,
        },
    }

    wf["24"] = {
        "class_type": "LTXVEmptyLatentAudio",
        "inputs": {
            "audio_vae": ["2", 0],
            "frames_number": params.num_frames,
            "frame_rate": params.fps,
            "batch_size": 1,
        },
    }

    wf["25"] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {
            "video_latent": ["23", 0],
            "audio_latent": ["24", 0],
        },
    }


def _add_sampling(
    wf: dict[str, Any],
    *,
    prefix: str,
    seed: int,
    model_source: str,
    sampler_name: str,
    sigmas: str,
    cfg: float,
    latent_source: str,
) -> None:
    """Add noise, guider, sampler, sigmas, and SamplerCustomAdvanced nodes."""
    wf[f"{prefix}0"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }
    wf[f"{prefix}1"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": [model_source, 0],
            "positive": ["12", 0],
            "negative": ["12", 1],
            "cfg": cfg,
        },
    }
    wf[f"{prefix}2"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": sampler_name},
    }
    wf[f"{prefix}3"] = {
        "class_type": "ManualSigmas",
        "inputs": {"sigmas_string": sigmas},
    }
    wf[f"{prefix}4"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": [f"{prefix}0", 0],
            "guider": [f"{prefix}1", 0],
            "sampler": [f"{prefix}2", 0],
            "sigmas": [f"{prefix}3", 0],
            "latent_image": [latent_source, 0],
        },
    }


def _add_decode_and_save(
    wf: dict[str, Any],
    *,
    prefix: str,
    sampler_output: str,
    fps: int,
) -> None:
    """Add AV separate, VAE decode, audio decode, create video, and save (5 nodes)."""
    wf[f"{prefix}0"] = {
        "class_type": "LTXVSeparateAVLatent",
        "inputs": {"av_latent": [sampler_output, 0]},
    }
    wf[f"{prefix}1"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": [f"{prefix}0", 0],
            "vae": ["1", 2],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 512,
            "temporal_overlap": 4,
        },
    }
    wf[f"{prefix}2"] = {
        "class_type": "LTXVAudioVAEDecode",
        "inputs": {
            "samples": [f"{prefix}0", 1],
            "audio_vae": ["2", 0],
        },
    }
    wf[f"{prefix}3"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": [f"{prefix}1", 0],
            "audio": [f"{prefix}2", 0],
            "fps": fps,
        },
    }
    wf[f"{prefix}4"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": [f"{prefix}3", 0],
            "filename_prefix": "ltx_output",
        },
    }


# ---------------------------------------------------------------------------
# Distilled sigmas
# ---------------------------------------------------------------------------

_DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_DEV_SIGMAS = "1.0, 0.9688, 0.9375, 0.9063, 0.875, 0.8125, 0.75, 0.6875, 0.625, 0.5625, 0.5, 0.4375, 0.375, 0.3125, 0.25, 0.1875, 0.125, 0.0625, 0.0"
_REFINEMENT_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"


# ---------------------------------------------------------------------------
# Public workflow builders
# ---------------------------------------------------------------------------


def _build_single_stage_workflow(params: WorkflowParams) -> dict[str, Any]:
    """Build a single-stage LTX-2.3 workflow (distilled or dev).

    Based on: LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json
    """
    seed = params.seed if params.seed is not None else random.randint(0, 2_147_483_647)
    is_dev = params.model == "dev"
    checkpoint = params.dev_checkpoint_name if is_dev else params.checkpoint_name

    wf: dict[str, Any] = {}

    _add_model_loaders(wf, params, checkpoint)

    # LoRA only for distilled
    if not is_dev:
        wf["4"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": params.distilled_lora_name,
                "strength_model": params.lora_strength,
            },
        }
        model_source = "4"
    else:
        model_source = "1"

    _add_text_encoding(wf, params)
    _add_latent_setup(wf, params, i2v_strength=1.0 if is_dev else 0.7)

    if is_dev:
        sampler_name, sigmas, cfg = "euler_cfg_pp", _DEV_SIGMAS, 3.5
    else:
        sampler_name, sigmas, cfg = "euler_ancestral_cfg_pp", _DISTILLED_SIGMAS, 1.0

    _add_sampling(
        wf, prefix="3", seed=seed, model_source=model_source,
        sampler_name=sampler_name, sigmas=sigmas, cfg=cfg, latent_source="25",
    )
    _add_decode_and_save(wf, prefix="4", sampler_output="34", fps=params.fps)

    return wf


def _build_two_stage_workflow(params: WorkflowParams) -> dict[str, Any]:
    """Build a two-stage LTX-2.3 workflow (distilled + upscale + refinement).

    Based on: LTX-2.3_T2V_I2V_Two_Stage_Distilled.json
    Stage 1: Distilled model generates at base resolution
    Stage 2: Spatial upscale + second sampling pass for refinement
    """
    seed = params.seed if params.seed is not None else random.randint(0, 2_147_483_647)
    is_i2v = params.image_filename is not None

    wf: dict[str, Any] = {}

    _add_model_loaders(wf, params, params.dev_checkpoint_name)

    # LoRA for distilled first pass
    wf["4"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": params.distilled_lora_name,
            "strength_model": params.lora_strength,
        },
    }

    # Upscaler
    wf["5"] = {
        "class_type": "LatentUpscaleModelLoader",
        "inputs": {"model_name": params.upscaler_name},
    }

    _add_text_encoding(wf, params)
    _add_latent_setup(wf, params, i2v_strength=0.7)

    # Stage 1 sampling (distilled)
    _add_sampling(
        wf, prefix="3", seed=seed, model_source="4",
        sampler_name="euler_ancestral_cfg_pp", sigmas=_DISTILLED_SIGMAS,
        cfg=1.0, latent_source="25",
    )

    # --- Stage 2: Upscale + Refinement ---
    wf["50"] = {
        "class_type": "LTXVSeparateAVLatent",
        "inputs": {"av_latent": ["34", 0]},
    }

    wf["51"] = {
        "class_type": "LTXVLatentUpsampler",
        "inputs": {
            "samples": ["50", 0],
            "upscale_model": ["5", 0],
            "vae": ["1", 2],
        },
    }

    if is_i2v:
        wf["52_resize"] = {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["21", 0],
                "resize_mode": "scale longer dimension",
                "size": 1536,
                "interpolation": "lanczos",
            },
        }

    wf["52"] = {
        "class_type": "LTXVImgToVideoConditionOnly",
        "inputs": {
            "vae": ["1", 2],
            "image": ["52_resize", 0] if is_i2v else ["51", 0],
            "latent": ["51", 0],
            "strength": 1.0,
            "bypass": not is_i2v,
        },
    }

    wf["53"] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {
            "video_latent": ["52", 0],
            "audio_latent": ["50", 1],
        },
    }

    # Stage 2 sampling (refinement)
    _add_sampling(
        wf, prefix="6", seed=seed + 1, model_source="4",
        sampler_name="euler_cfg_pp", sigmas=_REFINEMENT_SIGMAS,
        cfg=1.0, latent_source="53",
    )

    _add_decode_and_save(wf, prefix="7", sampler_output="64", fps=params.fps)

    return wf
