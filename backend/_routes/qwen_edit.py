"""Route handlers for /api/qwen-edit endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api_types import (
    QwenEditRequest,
    QwenEditResponse,
    QwenLoraListResponse,
)
from state import get_state_service
from app_handler import AppHandler

router = APIRouter(prefix="/api/qwen-edit", tags=["qwen-edit"])


@router.post("/edit", response_model=QwenEditResponse)
def route_qwen_edit(
    req: QwenEditRequest,
    handler: AppHandler = Depends(get_state_service),
) -> QwenEditResponse:
    """POST /api/qwen-edit/edit — instruction-based image editing."""
    return handler.qwen_edit.edit(req)


@router.get("/list-loras", response_model=QwenLoraListResponse)
def route_list_loras(
    handler: AppHandler = Depends(get_state_service),
) -> QwenLoraListResponse:
    """GET /api/qwen-edit/list-loras — list available LoRAs."""
    return handler.qwen_edit.list_loras()
