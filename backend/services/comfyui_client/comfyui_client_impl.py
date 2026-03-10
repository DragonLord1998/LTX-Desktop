"""Real ComfyUI server client implementation."""

from __future__ import annotations

import io
import json
import logging
import urllib.parse
import uuid
from pathlib import Path
from typing import Any

# websocket-client does not ship type stubs

import websocket  # type: ignore[import-untyped]

from services.http_client.http_client import HTTPClient
from services.comfyui_client.comfyui_types import (
    ComfyUIGenerationResult,
    ComfyUIOutputFile,
    ComfyUIProgressCallback,
    ComfyUIProgressUpdate,
)

logger = logging.getLogger(__name__)

_POLL_INTERVAL_SECONDS = 0.5
_WS_TIMEOUT_SECONDS = 600


class ComfyUIClientImpl:
    """Client that communicates with a running ComfyUI server."""

    def __init__(self, *, http: HTTPClient, server_url: str = "http://127.0.0.1:8188") -> None:
        self._http = http
        self.server_url = server_url
        self._client_id = uuid.uuid4().hex

    @property
    def server_url(self) -> str:
        return self._server_url

    @server_url.setter
    def server_url(self, value: str) -> None:
        self._server_url = value.rstrip("/")

    @property
    def _ws_url(self) -> str:
        http_url = self._server_url
        if http_url.startswith("https://"):
            ws_url = "wss://" + http_url[len("https://"):]
        elif http_url.startswith("http://"):
            ws_url = "ws://" + http_url[len("http://"):]
        else:
            ws_url = "ws://" + http_url
        return f"{ws_url}/ws?clientId={self._client_id}"

    def is_available(self) -> bool:
        try:
            resp = self._http.get(f"{self._server_url}/system_stats", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def queue_prompt(
        self,
        workflow: dict[str, Any],
        *,
        on_progress: ComfyUIProgressCallback | None = None,
    ) -> ComfyUIGenerationResult:
        ws = websocket.WebSocket()
        ws.settimeout(_WS_TIMEOUT_SECONDS)

        try:
            ws.connect(self._ws_url)

            payload = {"prompt": workflow, "client_id": self._client_id}
            resp = self._http.post(
                f"{self._server_url}/prompt",
                json_payload=payload,
                timeout=30,
            )

            if resp.status_code != 200:
                raise RuntimeError(f"ComfyUI /prompt returned {resp.status_code}: {resp.text}")

            result_data = resp.json()
            if "error" in result_data:
                node_errors = result_data.get("node_errors", {})
                raise RuntimeError(f"ComfyUI workflow validation error: {result_data['error']}, node_errors={node_errors}")

            prompt_id: str = result_data["prompt_id"]
            logger.info("ComfyUI prompt queued: %s", prompt_id)

            self._wait_for_completion(ws, prompt_id, on_progress)

        finally:
            ws.close()

        return self._collect_outputs(prompt_id)

    def _wait_for_completion(
        self,
        ws: websocket.WebSocket,
        prompt_id: str,
        on_progress: ComfyUIProgressCallback | None,
    ) -> None:
        while True:
            try:
                out = ws.recv()
            except websocket.WebSocketTimeoutException:
                raise RuntimeError("ComfyUI generation timed out") from None

            if not isinstance(out, str):
                continue

            message = json.loads(out)
            msg_type = message.get("type", "")
            data = message.get("data", {})

            if msg_type == "progress" and on_progress is not None:
                on_progress(ComfyUIProgressUpdate(
                    node_id=data.get("node", ""),
                    value=data.get("value", 0),
                    max_value=data.get("max", 1),
                ))

            if msg_type == "executing":
                if data.get("prompt_id") == prompt_id and data.get("node") is None:
                    logger.info("ComfyUI generation complete: %s", prompt_id)
                    return

            if msg_type == "execution_error":
                if data.get("prompt_id") == prompt_id:
                    node_type = data.get("node_type", "unknown")
                    exception_msg = data.get("exception_message", "Unknown error")
                    raise RuntimeError(f"ComfyUI execution error in {node_type}: {exception_msg}")

    def _collect_outputs(self, prompt_id: str) -> ComfyUIGenerationResult:
        resp = self._http.get(f"{self._server_url}/history/{prompt_id}", timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI /history returned {resp.status_code}")

        history = resp.json()
        prompt_history = history.get(prompt_id, {})
        outputs = prompt_history.get("outputs", {})

        output_files: list[ComfyUIOutputFile] = []
        for _node_id, node_output in outputs.items():
            for key in ("videos", "images", "gifs"):
                for file_info in node_output.get(key, []):
                    output_files.append(ComfyUIOutputFile(
                        filename=file_info["filename"],
                        subfolder=file_info.get("subfolder", ""),
                        folder_type=file_info.get("type", "output"),
                    ))

        return ComfyUIGenerationResult(prompt_id=prompt_id, output_files=output_files)

    def upload_image(self, image_path: str, *, subfolder: str = "") -> str:
        path = Path(image_path)
        # Sanitise filename for Content-Disposition header
        safe_name = path.name.replace('"', "_")
        boundary = uuid.uuid4().hex

        body = io.BytesIO()
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="image"; filename="{safe_name}"\r\n'.encode())
        body.write(b"Content-Type: application/octet-stream\r\n\r\n")
        # Stream file directly into body to avoid an extra copy
        with open(path, "rb") as f:
            body.write(f.read())
        body.write(b"\r\n")

        if subfolder:
            body.write(f"--{boundary}\r\n".encode())
            body.write(b'Content-Disposition: form-data; name="subfolder"\r\n\r\n')
            body.write(subfolder.encode())
            body.write(b"\r\n")

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="overwrite"\r\n\r\n')
        body.write(b"true\r\n")
        body.write(f"--{boundary}--\r\n".encode())

        resp = self._http.post(
            f"{self._server_url}/upload/image",
            data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=60,
        )

        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI /upload/image returned {resp.status_code}: {resp.text}")

        result = resp.json()
        return result.get("name", path.name)

    def download_output(self, filename: str, *, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        resp = self._http.get(f"{self._server_url}/view?{params}", timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"ComfyUI /view returned {resp.status_code}")
        return resp.content
