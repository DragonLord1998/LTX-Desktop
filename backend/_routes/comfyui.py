"""Route handlers for /api/comfyui/* endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import StatusResponse
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api/comfyui", tags=["comfyui"])


@router.get("/status", response_model=StatusResponse)
def route_comfyui_status(
    handler: AppHandler = Depends(get_state_service),
) -> StatusResponse:
    """GET /api/comfyui/status — check if ComfyUI server is reachable."""
    return StatusResponse(status=handler.check_comfyui_status())
