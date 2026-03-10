"""Types for ComfyUI client service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ComfyUIProgressUpdate:
    """Progress update from ComfyUI websocket."""

    node_id: str
    value: int
    max_value: int


ComfyUIProgressCallback = Callable[[ComfyUIProgressUpdate], None]


@dataclass
class ComfyUIOutputFile:
    """Reference to an output file on the ComfyUI server."""

    filename: str
    subfolder: str = ""
    folder_type: str = "output"


@dataclass
class ComfyUIGenerationResult:
    """Result of a ComfyUI generation job."""

    prompt_id: str
    output_files: list[ComfyUIOutputFile] = field(default_factory=list)
