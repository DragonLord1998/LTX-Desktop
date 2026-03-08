"""
Idle watchdog: auto-stops the RunPod pod after N minutes of inactivity.
Prevents runaway GPU charges if the app crashes or user forgets to close.

Monitors the LTX-2 backend /health endpoint — if the server has been idle
(no active generations) for longer than the timeout, the pod is stopped
via RunPod GraphQL API.
"""

import logging
import os
import sys
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("ltx-watchdog")

LTX_URL = "http://127.0.0.1:8000"
POLL_INTERVAL = 30  # seconds
STARTUP_TIMEOUT = 600  # 10 minutes max wait for backend to start


def check_ltx_active() -> bool:
    """Returns True if the LTX backend is busy (active generation in progress)."""
    try:
        resp = requests.get(f"{LTX_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Health endpoint may report status; treat any non-idle state as active
            status = data.get("status", "idle")
            return status not in ("idle", "ready", "ok")
    except Exception:
        pass
    return False


def stop_pod(pod_id: str, api_key: str) -> None:
    """Stop the current pod via RunPod GraphQL API."""
    query = """
    mutation stopPod($podId: String!) {
        podStop(input: { podId: $podId }) {
            id
            desiredStatus
        }
    }
    """

    try:
        response = requests.post(
            "https://api.runpod.io/graphql",
            json={"query": query, "variables": {"podId": pod_id}},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                logger.error(f"GraphQL error stopping pod: {data['errors']}")
            else:
                logger.info(f"Pod {pod_id} stop requested successfully")
        else:
            logger.error(
                f"Failed to stop pod: HTTP {response.status_code} "
                f"{response.text}"
            )
    except Exception as e:
        logger.error(f"Exception stopping pod: {e}")


def main() -> None:
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    api_key = os.environ.get("RUNPOD_API_KEY", "")
    idle_timeout = int(os.environ.get("IDLE_TIMEOUT_MINUTES", "30")) * 60

    if not pod_id or not api_key:
        logger.warning(
            "Missing RUNPOD_POD_ID or RUNPOD_API_KEY — watchdog disabled. "
            "Set these environment variables to enable auto-stop."
        )
        return

    logger.info(f"Watchdog started: auto-stop after {idle_timeout // 60} min idle")

    # Wait for the LTX backend to start (with timeout)
    logger.info("Waiting for LTX-2 backend to start...")
    startup_start = time.time()
    while True:
        try:
            requests.get(f"{LTX_URL}/health", timeout=3)
            break
        except Exception:
            if time.time() - startup_start > STARTUP_TIMEOUT:
                logger.error(
                    f"LTX-2 backend did not start within {STARTUP_TIMEOUT}s — "
                    "watchdog exiting. Check backend logs."
                )
                return
            time.sleep(5)

    logger.info("LTX-2 backend is up — monitoring activity")
    last_activity = time.time()

    while True:
        time.sleep(POLL_INTERVAL)

        if check_ltx_active():
            last_activity = time.time()

        elapsed = time.time() - last_activity
        remaining = idle_timeout - elapsed

        if 0 < remaining <= 300 and remaining % POLL_INTERVAL < POLL_INTERVAL:
            logger.warning(f"Idle warning: pod will stop in ~{remaining:.0f}s")

        if elapsed >= idle_timeout:
            logger.warning(
                f"Idle for {elapsed:.0f}s (limit: {idle_timeout}s). Stopping pod..."
            )
            stop_pod(pod_id, api_key)
            sys.exit(0)


if __name__ == "__main__":
    main()
