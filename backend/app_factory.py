"""FastAPI app factory decoupled from runtime bootstrap side effects."""

from __future__ import annotations

import os as _os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from _routes._errors import HTTPError
from _routes.generation import router as generation_router
from _routes.health import router as health_router
from _routes.ic_lora import router as ic_lora_router
from _routes.image_gen import router as image_gen_router
from _routes.models import router as models_router
from _routes.suggest_gap_prompt import router as suggest_gap_prompt_router
from _routes.qwen_edit import router as qwen_edit_router
from _routes.retake import router as retake_router
from _routes.runtime_policy import router as runtime_policy_router
from _routes.settings import router as settings_router
from logging_policy import log_http_error, log_unhandled_exception
from state import init_state_service

if TYPE_CHECKING:
    from app_handler import AppHandler

# Default static dir: backend/static/ (populated by frontend build)
_DEFAULT_STATIC_DIR = Path(__file__).parent / "static"


def _build_default_origins() -> list[str]:
    env_origins = _os.environ.get("LTX_CORS_ORIGINS", "")
    if env_origins:
        return [o.strip() for o in env_origins.split(",") if o.strip()]
    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

DEFAULT_ALLOWED_ORIGINS: list[str] = _build_default_origins()

# Only enable the RunPod CORS regex when running on RunPod (RUNPOD_POD_ID is set).
_RUNPOD_ORIGIN_REGEX: str | None = (
    r"https://[a-zA-Z0-9-]+\.proxy\.runpod\.net"
    if _os.environ.get("RUNPOD_POD_ID")
    else None
)


def create_app(
    *,
    handler: "AppHandler",
    allowed_origins: list[str] | None = None,
    title: str = "LTX-2 Video Generation Server",
    static_dir: Path | None = None,
) -> FastAPI:
    """Create a configured FastAPI app bound to the provided handler."""
    init_state_service(handler)

    app = FastAPI(title=title)
    origins = allowed_origins if allowed_origins is not None else DEFAULT_ALLOWED_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=_RUNPOD_ORIGIN_REGEX,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    async def _route_http_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPError):
            log_http_error(request, exc)
            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
        return JSONResponse(status_code=500, content={"error": str(exc)})

    async def _validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, RequestValidationError):
            return JSONResponse(status_code=422, content={"error": str(exc)})
        return JSONResponse(status_code=422, content={"error": str(exc)})

    async def _route_generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        log_unhandled_exception(request, exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(HTTPError, _route_http_error_handler)
    app.add_exception_handler(Exception, _route_generic_error_handler)

    app.include_router(health_router)
    app.include_router(generation_router)
    app.include_router(models_router)
    app.include_router(settings_router)
    app.include_router(image_gen_router)
    app.include_router(suggest_gap_prompt_router)
    app.include_router(retake_router)
    app.include_router(ic_lora_router)
    app.include_router(qwen_edit_router)
    app.include_router(runtime_policy_router)

    # Serve generated outputs (videos/images) so the browser can access them
    outputs_dir = handler.config.outputs_dir
    if outputs_dir.is_dir():
        app.mount("/outputs", StaticFiles(directory=outputs_dir), name="outputs")

    # Static frontend serving — mount only if the build output exists
    resolved_static = static_dir if static_dir is not None else _DEFAULT_STATIC_DIR
    if resolved_static.is_dir():
        app.mount("/assets", StaticFiles(directory=resolved_static / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa(full_path: str) -> FileResponse:  # noqa: F811  # pyright: ignore[reportUnusedFunction]
            """Catch-all: serve index.html for client-side routing."""
            return FileResponse(resolved_static / "index.html")

    return app
