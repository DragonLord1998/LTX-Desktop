"""Integration tests for ComfyUI server video generation."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app_handler import AppHandler
from tests.fakes.services import FakeComfyUIClient, FakeServices


def _enable_comfyui(test_state: AppHandler) -> None:
    """Enable ComfyUI mode in app settings."""
    test_state.settings.update_settings({"comfyui": {"enabled": True}})


class TestComfyUIGeneration:
    def test_t2v_via_comfyui(self, client, test_state: AppHandler, fake_services: FakeServices) -> None:
        _enable_comfyui(test_state)

        resp = client.post("/api/generate", json={
            "prompt": "A cat walking on a beach",
            "resolution": "540p",
            "model": "fast",
            "duration": "2",
            "fps": "24",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"
        assert data["video_path"] is not None

        comfyui = fake_services.comfyui_client
        assert len(comfyui.queue_prompt_calls) == 1
        workflow = comfyui.queue_prompt_calls[0]["workflow"]

        # Verify workflow structure: has checkpoint loader, text encode, sampler
        class_types = {v["class_type"] for v in workflow.values()}
        assert "CheckpointLoaderSimple" in class_types
        assert "CLIPTextEncode" in class_types
        assert "SamplerCustomAdvanced" in class_types
        assert "SaveVideo" in class_types

        # Output file was downloaded
        assert len(comfyui.download_output_calls) == 1
        assert comfyui.download_output_calls[0]["filename"] == "ltx_output_00001.mp4"

    def test_i2v_via_comfyui(self, client, test_state: AppHandler, fake_services: FakeServices, tmp_path: Path) -> None:
        _enable_comfyui(test_state)

        img = Image.new("RGB", (64, 64), "blue")
        img_path = tmp_path / "input.png"
        img.save(img_path)

        resp = client.post("/api/generate", json={
            "prompt": "A cat walking on a beach",
            "resolution": "540p",
            "model": "fast",
            "duration": "2",
            "fps": "24",
            "imagePath": str(img_path),
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"

        comfyui = fake_services.comfyui_client
        assert len(comfyui.upload_image_calls) == 1
        assert len(comfyui.queue_prompt_calls) == 1

        workflow = comfyui.queue_prompt_calls[0]["workflow"]
        class_types = {v["class_type"] for v in workflow.values()}
        assert "LoadImage" in class_types
        assert "LTXVImgToVideoConditionOnly" in class_types

    def test_comfyui_disabled_uses_local_pipeline(self, client, test_state: AppHandler, fake_services: FakeServices, create_fake_model_files) -> None:
        # ComfyUI disabled by default - should use local pipeline
        create_fake_model_files()

        resp = client.post("/api/generate", json={
            "prompt": "A cat walking on a beach",
            "resolution": "540p",
            "model": "fast",
            "duration": "2",
            "fps": "24",
        })

        assert resp.status_code == 200
        comfyui = fake_services.comfyui_client
        assert len(comfyui.queue_prompt_calls) == 0
        assert len(fake_services.fast_video_pipeline.generate_calls) == 1

    def test_comfyui_error_returns_500(self, client, test_state: AppHandler, fake_services: FakeServices) -> None:
        _enable_comfyui(test_state)
        fake_services.comfyui_client.raise_on_queue_prompt = RuntimeError("ComfyUI server crashed")

        resp = client.post("/api/generate", json={
            "prompt": "A cat walking on a beach",
            "resolution": "540p",
            "model": "fast",
        })

        assert resp.status_code == 500

    def test_dev_model_via_comfyui(self, client, test_state: AppHandler, fake_services: FakeServices) -> None:
        _enable_comfyui(test_state)

        resp = client.post("/api/generate", json={
            "prompt": "A scenic landscape",
            "resolution": "720p",
            "model": "dev",
            "duration": "2",
            "fps": "24",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "complete"

        workflow = fake_services.comfyui_client.queue_prompt_calls[0]["workflow"]
        # Dev model should use dev checkpoint, no LoRA node
        checkpoint_node = workflow["1"]
        assert "dev" in checkpoint_node["inputs"]["ckpt_name"]
        assert "4" not in workflow  # No LoRA for dev

    def test_comfyui_status_available(self, client, fake_services: FakeServices) -> None:
        fake_services.comfyui_client.available = True
        resp = client.get("/api/comfyui/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "available"

    def test_comfyui_status_unavailable(self, client, fake_services: FakeServices) -> None:
        fake_services.comfyui_client.available = False
        resp = client.get("/api/comfyui/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "unavailable"


class TestComfyUIWorkflowBuilder:
    def test_t2v_fast_workflow_structure(self) -> None:
        from services.comfyui_client.workflow_builder import WorkflowParams, build_workflow

        params = WorkflowParams(
            prompt="A cat on a beach",
            negative_prompt="ugly, blurry",
            width=960,
            height=544,
            num_frames=49,
            fps=24,
            seed=42,
            model="fast",
        )
        workflow = build_workflow(params)

        class_types = {v["class_type"] for v in workflow.values()}
        assert "CheckpointLoaderSimple" in class_types
        assert "LoraLoaderModelOnly" in class_types  # Fast model uses LoRA
        assert "CLIPTextEncode" in class_types
        assert "EmptyLTXVLatentVideo" in class_types
        assert "SamplerCustomAdvanced" in class_types
        assert "SaveVideo" in class_types

        # Check text encoding has our prompt
        text_nodes = [v for v in workflow.values() if v["class_type"] == "CLIPTextEncode"]
        prompts = [n["inputs"]["text"] for n in text_nodes]
        assert "A cat on a beach" in prompts
        assert "ugly, blurry" in prompts

    def test_t2v_dev_workflow_no_lora(self) -> None:
        from services.comfyui_client.workflow_builder import WorkflowParams, build_workflow

        params = WorkflowParams(
            prompt="test",
            negative_prompt="bad",
            width=960,
            height=544,
            num_frames=49,
            fps=24,
            seed=42,
            model="dev",
        )
        workflow = build_workflow(params)

        class_types = {v["class_type"] for v in workflow.values()}
        assert "LoraLoaderModelOnly" not in class_types

    def test_i2v_workflow_has_image_nodes(self) -> None:
        from services.comfyui_client.workflow_builder import WorkflowParams, build_workflow

        params = WorkflowParams(
            prompt="test",
            negative_prompt="bad",
            width=960,
            height=544,
            num_frames=49,
            fps=24,
            seed=42,
            image_filename="uploaded_image.png",
        )
        workflow = build_workflow(params)

        class_types = {v["class_type"] for v in workflow.values()}
        assert "LoadImage" in class_types
        assert "LTXVPreprocess" in class_types

        # I2V condition should not be bypassed
        i2v_node = workflow["23"]
        assert i2v_node["inputs"]["bypass"] is False

    def test_two_stage_workflow(self) -> None:
        from services.comfyui_client.workflow_builder import WorkflowParams, build_workflow

        params = WorkflowParams(
            prompt="test",
            negative_prompt="bad",
            width=960,
            height=544,
            num_frames=49,
            fps=24,
            seed=42,
            use_two_stage=True,
        )
        workflow = build_workflow(params)

        class_types = {v["class_type"] for v in workflow.values()}
        # Two-stage should have upscaler
        assert "LatentUpscaleModelLoader" in class_types
        assert "LTXVLatentUpsampler" in class_types

        # Should have two SamplerCustomAdvanced (stage 1 + stage 2)
        sampler_nodes = [v for v in workflow.values() if v["class_type"] == "SamplerCustomAdvanced"]
        assert len(sampler_nodes) == 2
