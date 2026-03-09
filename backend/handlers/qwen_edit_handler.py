"""Qwen-Image-Edit orchestration handler."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from PIL import Image

from _routes._errors import HTTPError
from api_types import QwenEditRequest, QwenEditResponse, QwenLoraInfo, QwenLoraListResponse
from handlers.base import StateHandlerBase
from handlers.generation_handler import GenerationHandler
from handlers.pipelines_handler import PipelinesHandler
from state.app_state_types import AppState

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class QwenEditHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        generation_handler: GenerationHandler,
        pipelines_handler: PipelinesHandler,
        outputs_dir: Path,
        config: RuntimeConfig,
        loras_dir: Path,
    ) -> None:
        super().__init__(state, lock)
        self._generation = generation_handler
        self._pipelines = pipelines_handler
        self._outputs_dir = outputs_dir
        self._config = config
        self._loras_dir = loras_dir

    def edit(self, req: QwenEditRequest) -> QwenEditResponse:
        if self._generation.is_generation_running():
            raise HTTPError(409, "Generation already in progress")

        generation_id = uuid.uuid4().hex[:8]
        seed = req.seed if req.seed is not None else int(time.time()) % 2147483647

        try:
            # Load pipeline BEFORE start_generation to avoid the pipeline swap bug.
            pipeline_state = self._pipelines.load_qwen_edit_pipeline()

            # LoRA swap: load/unload if the requested lora differs from active.
            if req.lora_id != pipeline_state.active_lora:
                if pipeline_state.active_lora is not None:
                    pipeline_state.pipeline.unload_lora()
                    pipeline_state.active_lora = None

                if req.lora_id is not None:
                    lora_path = self._loras_dir / f"{req.lora_id}.safetensors"
                    if not lora_path.exists():
                        raise HTTPError(404, f"LoRA not found: {req.lora_id}")
                    pipeline_state.pipeline.load_lora(str(lora_path), strength=req.lora_strength)
                    pipeline_state.active_lora = req.lora_id

            self._generation.start_generation(generation_id)

            source_image = Image.open(req.image_path).convert("RGB")

            self._generation.update_progress("inference", 10, 0, req.num_steps)

            result_image = pipeline_state.pipeline.edit(
                image=source_image,
                instruction=req.instruction,
                seed=seed,
                num_inference_steps=req.num_steps,
            )

            self._generation.update_progress("saving", 95, req.num_steps, req.num_steps)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self._outputs_dir / f"qwen_edit_{timestamp}_{uuid.uuid4().hex[:8]}.png"
            result_image.save(str(output_path))

            self._generation.complete_generation(str(output_path))
            return QwenEditResponse(status="complete", image_path=str(output_path))

        except HTTPError:
            self._generation.fail_generation("HTTP error during edit")
            raise
        except Exception as e:
            self._generation.fail_generation(str(e))
            if "cancelled" in str(e).lower():
                logger.info("Qwen edit cancelled by user")
                return QwenEditResponse(status="cancelled")
            raise HTTPError(500, str(e)) from e

    def list_loras(self) -> QwenLoraListResponse:
        loras: list[QwenLoraInfo] = []
        if self._loras_dir.exists():
            for path in sorted(self._loras_dir.glob("*.safetensors")):
                stem = path.stem
                # Convert underscores/hyphens to spaces for a human-readable name.
                name = stem.replace("_", " ").replace("-", " ").title()
                loras.append(QwenLoraInfo(id=stem, name=name))
        return QwenLoraListResponse(loras=loras)
