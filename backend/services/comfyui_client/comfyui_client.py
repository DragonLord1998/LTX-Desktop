"""ComfyUI server client protocol."""

from __future__ import annotations

from typing import Any, Protocol

from services.comfyui_client.comfyui_types import ComfyUIGenerationResult, ComfyUIProgressCallback


class ComfyUIClient(Protocol):
    """Protocol for communicating with a ComfyUI server."""

    def is_available(self) -> bool:
        """Check if the ComfyUI server is reachable."""
        ...

    def queue_prompt(
        self,
        workflow: dict[str, Any],
        *,
        on_progress: ComfyUIProgressCallback | None = None,
    ) -> ComfyUIGenerationResult:
        """Queue a workflow prompt on the ComfyUI server and wait for completion.

        Args:
            workflow: The workflow JSON in ComfyUI API format (node-id keyed dict).
            on_progress: Optional callback for progress updates.

        Returns:
            A ComfyUIGenerationResult with output file paths.
        """
        ...

    def upload_image(self, image_path: str, *, subfolder: str = "") -> str:
        """Upload an image to the ComfyUI server.

        Returns the filename as stored on the server.
        """
        ...

    def download_output(self, filename: str, *, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download an output file from the ComfyUI server."""
        ...
