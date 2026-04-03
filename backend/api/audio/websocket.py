# backend/api/audio/websocket.py
"""
READ – Audio Deepfake Detection WebSocket

Endpoint: WS /ws/audio/{task_id}

Flow:
    1. Client connects immediately after calling POST /audio/analyze
    2. Server subscribes to two Redis channels:
          task_audio_detection:{task_id}   → verdict + waveform
          task_audio_xai:{task_id}         → IG + SHAP heatmaps
    3. Messages are forwarded to the client as JSON
    4. Server sends {"type": "complete"} and closes once both results arrive

This mirrors the pattern in api/video/websocket.py.
"""

import asyncio
import json
import logging

import redis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum wall-clock seconds to wait for detection + XAI before timing out
WS_TIMEOUT_SECONDS = 300
WS_POLL_INTERVAL   = 0.5   # seconds between Redis polls


@router.websocket("/ws/audio/{task_id}")
async def audio_websocket(ws: WebSocket, task_id: str):
    """
    Stream audio deepfake detection results to the client in real time.

    The client connects after triggering POST /audio/analyze and will
    receive the following message types (as JSON):

    - ``audio_detection_ready``  — verdict, probabilities, and waveform PNG
    - ``audio_xai_ready``        — IG and SHAP heatmap PNGs
    - ``complete``               — both results have arrived; connection closes
    - ``error``                  — something went wrong
    - ``timeout``                — results didn't arrive within WS_TIMEOUT_SECONDS

    Args:
        task_id: The pipeline identifier returned by POST /audio/analyze.
    """
    await ws.accept()
    logger.info(f"[AudioWS] Client connected for task_id={task_id}")

    detection_received = False
    xai_received       = False

    try:
        # ------------------------------------------------------------------
        # Set up synchronous Redis pub/sub in an executor so the async event
        # loop isn't blocked.
        # ------------------------------------------------------------------
        r = redis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = r.pubsub()
        detection_channel = f"task_audio_detection:{task_id}"
        xai_channel       = f"task_audio_xai:{task_id}"
        pubsub.subscribe(detection_channel, xai_channel)

        # ------------------------------------------------------------------
        # Also check Redis keys in case results arrived before WS connected
        # (avoids hanging forever on a completed task)
        # ------------------------------------------------------------------
        detection_key = f"audio_detection_result:{task_id}"
        xai_key       = f"audio_xai_result:{task_id}"

        # Check for already-stored detection result
        cached_detection = r.get(detection_key)
        if cached_detection:
            try:
                payload = json.loads(cached_detection)
                payload["type"] = "audio_detection_ready"
                await ws.send_json(payload)
                detection_received = True
                logger.info(f"[AudioWS] Sent cached detection for task_id={task_id}")
            except Exception as e:
                logger.warning(f"[AudioWS] Cached detection parse error: {e}")

        # Check for already-stored XAI result
        cached_xai = r.get(xai_key)
        if cached_xai:
            try:
                payload = json.loads(cached_xai)
                payload["type"] = "audio_xai_ready"
                await ws.send_json(payload)
                xai_received = True
                logger.info(f"[AudioWS] Sent cached XAI for task_id={task_id}")
            except Exception as e:
                logger.warning(f"[AudioWS] Cached XAI parse error: {e}")

        # If both already available, send complete and close
        if detection_received and xai_received:
            await ws.send_json({"type": "complete", "task_id": task_id})
            await ws.close()
            return

        # ------------------------------------------------------------------
        # Real-time message forwarding loop
        # ------------------------------------------------------------------
        elapsed = 0.0
        while elapsed < WS_TIMEOUT_SECONDS:
            # Run blocking pubsub.get_message in a thread pool
            message = await asyncio.get_event_loop().run_in_executor(
                None, lambda: pubsub.get_message(timeout=WS_POLL_INTERVAL)
            )

            if message and message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    msg_type = data.get("type", "")

                    if msg_type == "audio_detection_ready":
                        await ws.send_json(data)
                        detection_received = True
                        logger.info(f"[AudioWS] Forwarded detection for task_id={task_id}")

                    elif msg_type == "audio_xai_ready" and "ig_heatmap_b64" in data:
                        await ws.send_json(data)
                        xai_received = True
                        logger.info(f"[AudioWS] Forwarded XAI for task_id={task_id}")

                    # Both results delivered → close gracefully
                    if detection_received and xai_received:
                        await ws.send_json({"type": "complete", "task_id": task_id})
                        logger.info(f"[AudioWS] Pipeline complete for task_id={task_id}")
                        break

                except Exception as parse_err:
                    logger.warning(f"[AudioWS] Message parse error: {parse_err}")

            elapsed += WS_POLL_INTERVAL
            # Small async yield so other coroutines can run
            await asyncio.sleep(0)

        else:
            # Loop exhausted — timed out
            await ws.send_json({
                "type":    "timeout",
                "task_id": task_id,
                "message": "Timed out waiting for results. Poll GET /audio/result/{task_id}.",
            })
            logger.warning(f"[AudioWS] Timed out for task_id={task_id}")

    except WebSocketDisconnect:
        logger.info(f"[AudioWS] Client disconnected for task_id={task_id}")

    except Exception as e:
        logger.error(f"[AudioWS] Unexpected error for task_id={task_id}: {e}", exc_info=True)
        try:
            await ws.send_json({
                "type":    "error",
                "task_id": task_id,
                "message": str(e),
            })
        except Exception:
            pass

    finally:
        try:
            pubsub.unsubscribe(detection_channel, xai_channel)
            pubsub.close()
            r.close()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        logger.info(f"[AudioWS] Connection closed for task_id={task_id}")