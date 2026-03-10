"""Build ComfyUI workflow JSON for LTX-2.3 video generation.

These workflows are based on the official Lightricks/ComfyUI-LTXVideo
example workflows (LTX-2.3 T2V/I2V Single Stage and Two Stage).

Reference: https://github.com/Lightricks/ComfyUI-LTXVideo/tree/master/example_workflows/2.3
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class WorkflowParams:
    """Parameters for building a ComfyUI LTX-2.3 workflow."""

    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_frames: int
    fps: int
    seed: int | None = None
    model: str = "fast"
    image_filename: str | None = None
    checkpoint_name: str = "ltx-2.3-22b-distilled.safetensors"
    dev_checkpoint_name: str = "ltx-2.3-22b-dev.safetensors"
    text_encoder_name: str = "comfy_gemma_3_12B_it.safetensors"
    upscaler_name: str = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    distilled_lora_name: str = "ltx-2.3-22b-distilled-lora-384.safetensors"
    lora_strength: float = 0.5
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


def _resolve_seed(seed: int | None) -> int:
    if seed is not None:
        return seed
    return random.randint(0, 2_147_483_647)


def _build_single_stage_workflow(params: WorkflowParams) -> dict[str, Any]:
    """Build a single-stage LTX-2.3 workflow (distilled or dev).

    Based on: LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json
    """
    seed = _resolve_seed(params.seed)
    is_dev = params.model == "dev"
    checkpoint = params.dev_checkpoint_name if is_dev else params.checkpoint_name

    is_i2v = params.image_filename is not None
    bypass_i2v = not is_i2v

    if is_dev:
        sampler_name = "euler_cfg_pp"
        sigmas = "1.0, 0.9688, 0.9375, 0.9063, 0.875, 0.8125, 0.75, 0.6875, 0.625, 0.5625, 0.5, 0.4375, 0.375, 0.3125, 0.25, 0.1875, 0.125, 0.0625, 0.0"
        cfg_scale = 3.5
    else:
        sampler_name = "euler_ancestral_cfg_pp"
        sigmas = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
        cfg_scale = 1.0

    workflow: dict[str, Any] = {}

    # --- Model loading ---
    workflow["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": checkpoint},
    }

    # Audio VAE loader (for audio-video models)
    workflow["2"] = {
        "class_type": "LTXVAudioVAELoader",
        "inputs": {"ckpt_name": checkpoint},
    }

    # Text encoder
    workflow["3"] = {
        "class_type": "LTXAVTextEncoderLoader",
        "inputs": {
            "clip_name": params.text_encoder_name,
            "ckpt_name": checkpoint,
            "type": "default",
        },
    }

    # LoRA (for distilled model, connects to checkpoint model output)
    if not is_dev:
        workflow["4"] = {
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

    # --- Text encoding ---
    workflow["10"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["3", 0],
            "text": params.prompt,
        },
    }

    workflow["11"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": ["3", 0],
            "text": params.negative_prompt,
        },
    }

    # --- LTX conditioning (frame rate) ---
    workflow["12"] = {
        "class_type": "LTXVConditioning",
        "inputs": {
            "positive": ["10", 0],
            "negative": ["11", 0],
            "frame_rate": params.fps,
        },
    }

    # --- Latent space ---
    workflow["20"] = {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": params.width,
            "height": params.height,
            "length": params.num_frames,
            "batch_size": 1,
        },
    }

    # Image preprocessing (for I2V)
    if is_i2v:
        workflow["21"] = {
            "class_type": "LoadImage",
            "inputs": {"image": params.image_filename, "upload": "image"},
        }

        workflow["22"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {
                "image": ["21", 0],
                "target_tokens": 18,
            },
        }

    # I2V condition
    workflow["23"] = {
        "class_type": "LTXVImgToVideoConditionOnly",
        "inputs": {
            "vae": ["1", 2],
            "image": ["22", 0] if is_i2v else ["21", 0],
            "latent": ["20", 0],
            "strength": 0.7 if not is_dev else 1.0,
            "bypass": bypass_i2v,
        },
    }

    # Audio latent
    workflow["24"] = {
        "class_type": "LTXVEmptyLatentAudio",
        "inputs": {
            "audio_vae": ["2", 0],
            "frames_number": params.num_frames,
            "frame_rate": params.fps,
            "batch_size": 1,
        },
    }

    # Concat AV latent
    workflow["25"] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {
            "video_latent": ["23", 0],
            "audio_latent": ["24", 0],
        },
    }

    # --- Sampling ---
    workflow["30"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }

    workflow["31"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": [model_source, 0],
            "positive": ["12", 0],
            "negative": ["12", 1],
            "cfg": cfg_scale,
        },
    }

    workflow["32"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": sampler_name},
    }

    workflow["33"] = {
        "class_type": "ManualSigmas",
        "inputs": {"sigmas_string": sigmas},
    }

    workflow["34"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["30", 0],
            "guider": ["31", 0],
            "sampler": ["32", 0],
            "sigmas": ["33", 0],
            "latent_image": ["25", 0],
        },
    }

    # --- Decode ---
    # Separate AV latent
    workflow["40"] = {
        "class_type": "LTXVSeparateAVLatent",
        "inputs": {"av_latent": ["34", 0]},
    }

    # VAE decode video
    workflow["41"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": ["40", 0],
            "vae": ["1", 2],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 512,
            "temporal_overlap": 4,
        },
    }

    # Audio VAE decode
    workflow["42"] = {
        "class_type": "LTXVAudioVAEDecode",
        "inputs": {
            "samples": ["40", 1],
            "audio_vae": ["2", 0],
        },
    }

    # Create video
    workflow["43"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["41", 0],
            "audio": ["42", 0],
            "fps": params.fps,
        },
    }

    # Save video
    workflow["44"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["43", 0],
            "filename_prefix": "ltx_output",
        },
    }

    # Remove placeholder I2V nodes if not needed for T2V
    if not is_i2v:
        # For T2V, the I2V condition node still exists but with bypass=True
        # We need a dummy image input - use an empty latent directly
        workflow["23"]["inputs"]["image"] = ["20", 0]

    return workflow


def _build_two_stage_workflow(params: WorkflowParams) -> dict[str, Any]:
    """Build a two-stage LTX-2.3 workflow (distilled + upscale + refinement).

    Based on: LTX-2.3_T2V_I2V_Two_Stage_Distilled.json
    Stage 1: Distilled model generates at base resolution
    Stage 2: Spatial upscale + second sampling pass for refinement
    """
    seed = _resolve_seed(params.seed)
    seed_stage2 = seed + 1

    is_i2v = params.image_filename is not None
    bypass_i2v = not is_i2v

    workflow: dict[str, Any] = {}

    # --- Model loading ---
    workflow["1"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": params.dev_checkpoint_name},
    }

    workflow["2"] = {
        "class_type": "LTXVAudioVAELoader",
        "inputs": {"ckpt_name": params.dev_checkpoint_name},
    }

    workflow["3"] = {
        "class_type": "LTXAVTextEncoderLoader",
        "inputs": {
            "clip_name": params.text_encoder_name,
            "ckpt_name": params.dev_checkpoint_name,
            "type": "default",
        },
    }

    # LoRA for distilled generation
    workflow["4"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": params.distilled_lora_name,
            "strength_model": params.lora_strength,
        },
    }

    # Upscaler model
    workflow["5"] = {
        "class_type": "LatentUpscaleModelLoader",
        "inputs": {"model_name": params.upscaler_name},
    }

    # --- Text encoding ---
    workflow["10"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["3", 0], "text": params.prompt},
    }

    workflow["11"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["3", 0], "text": params.negative_prompt},
    }

    workflow["12"] = {
        "class_type": "LTXVConditioning",
        "inputs": {
            "positive": ["10", 0],
            "negative": ["11", 0],
            "frame_rate": params.fps,
        },
    }

    # --- Stage 1: Distilled generation ---
    workflow["20"] = {
        "class_type": "EmptyLTXVLatentVideo",
        "inputs": {
            "width": params.width,
            "height": params.height,
            "length": params.num_frames,
            "batch_size": 1,
        },
    }

    if is_i2v:
        workflow["21"] = {
            "class_type": "LoadImage",
            "inputs": {"image": params.image_filename, "upload": "image"},
        }
        workflow["22"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": ["21", 0], "target_tokens": 18},
        }

    workflow["23"] = {
        "class_type": "LTXVImgToVideoConditionOnly",
        "inputs": {
            "vae": ["1", 2],
            "image": ["22", 0] if is_i2v else ["20", 0],
            "latent": ["20", 0],
            "strength": 0.7,
            "bypass": bypass_i2v,
        },
    }

    workflow["24"] = {
        "class_type": "LTXVEmptyLatentAudio",
        "inputs": {
            "audio_vae": ["2", 0],
            "frames_number": params.num_frames,
            "frame_rate": params.fps,
            "batch_size": 1,
        },
    }

    workflow["25"] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {
            "video_latent": ["23", 0],
            "audio_latent": ["24", 0],
        },
    }

    # Stage 1 sampling (distilled: 8 steps)
    workflow["30"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed},
    }

    workflow["31"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": ["4", 0],
            "positive": ["12", 0],
            "negative": ["12", 1],
            "cfg": 1.0,
        },
    }

    workflow["32"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler_ancestral_cfg_pp"},
    }

    workflow["33"] = {
        "class_type": "ManualSigmas",
        "inputs": {"sigmas_string": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"},
    }

    workflow["34"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["30", 0],
            "guider": ["31", 0],
            "sampler": ["32", 0],
            "sigmas": ["33", 0],
            "latent_image": ["25", 0],
        },
    }

    # --- Stage 2: Upscale + Refinement ---
    # Separate AV from stage 1
    workflow["50"] = {
        "class_type": "LTXVSeparateAVLatent",
        "inputs": {"av_latent": ["34", 0]},
    }

    # Upscale video latent
    workflow["51"] = {
        "class_type": "LTXVLatentUpsampler",
        "inputs": {
            "samples": ["50", 0],
            "upscale_model": ["5", 0],
            "vae": ["1", 2],
        },
    }

    # I2V condition for stage 2 (uses upscaled latent)
    if is_i2v:
        workflow["52_resize"] = {
            "class_type": "ResizeImageMaskNode",
            "inputs": {
                "input": ["21", 0],
                "resize_mode": "scale longer dimension",
                "size": 1536,
                "interpolation": "lanczos",
            },
        }

    workflow["52"] = {
        "class_type": "LTXVImgToVideoConditionOnly",
        "inputs": {
            "vae": ["1", 2],
            "image": ["52_resize", 0] if is_i2v else ["51", 0],
            "latent": ["51", 0],
            "strength": 1.0,
            "bypass": bypass_i2v,
        },
    }

    # Concat AV for stage 2
    workflow["53"] = {
        "class_type": "LTXVConcatAVLatent",
        "inputs": {
            "video_latent": ["52", 0],
            "audio_latent": ["50", 1],
        },
    }

    # Stage 2 sampling (refinement: fewer steps, lower sigma range)
    workflow["60"] = {
        "class_type": "RandomNoise",
        "inputs": {"noise_seed": seed_stage2},
    }

    workflow["61"] = {
        "class_type": "CFGGuider",
        "inputs": {
            "model": ["4", 0],
            "positive": ["12", 0],
            "negative": ["12", 1],
            "cfg": 1.0,
        },
    }

    workflow["62"] = {
        "class_type": "KSamplerSelect",
        "inputs": {"sampler_name": "euler_cfg_pp"},
    }

    workflow["63"] = {
        "class_type": "ManualSigmas",
        "inputs": {"sigmas_string": "0.85, 0.7250, 0.4219, 0.0"},
    }

    workflow["64"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["60", 0],
            "guider": ["61", 0],
            "sampler": ["62", 0],
            "sigmas": ["63", 0],
            "latent_image": ["53", 0],
        },
    }

    # --- Final decode ---
    workflow["70"] = {
        "class_type": "LTXVSeparateAVLatent",
        "inputs": {"av_latent": ["64", 0]},
    }

    workflow["71"] = {
        "class_type": "VAEDecodeTiled",
        "inputs": {
            "samples": ["70", 0],
            "vae": ["1", 2],
            "tile_size": 512,
            "overlap": 64,
            "temporal_size": 512,
            "temporal_overlap": 4,
        },
    }

    workflow["72"] = {
        "class_type": "LTXVAudioVAEDecode",
        "inputs": {
            "samples": ["70", 1],
            "audio_vae": ["2", 0],
        },
    }

    workflow["73"] = {
        "class_type": "CreateVideo",
        "inputs": {
            "images": ["71", 0],
            "audio": ["72", 0],
            "fps": params.fps,
        },
    }

    workflow["74"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["73", 0],
            "filename_prefix": "ltx_output",
        },
    }

    return workflow
