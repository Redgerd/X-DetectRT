import os
import uuid
import asyncio
import io
import logging
import cv2
import numpy as np
import base64
import time
from fastapi import WebSocket, APIRouter, WebSocketDisconnect
from core.celery.frame_selection import extract_faces_with_optical_flow
from core.celery.detection_tasks import run_gend_inference
import json
from json import JSONDecodeError
import redis
from config import settings
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from typing import Optional, Callable, Any
from models import VideoAnalysisTask
from models.tasks import TaskStatus
from core.database import SessionLocal
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

# Small delay between sending frames to prevent batching
FRAME_SEND_DELAY = 0.05  # 50ms between frames


async def watch_disconnect(ws: WebSocket, event: asyncio.Event) -> None:
    """Sets event when client disconnects or sends connection_closed."""
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                event.set()
                return
            try:
                if json.loads(msg.get("text", "{}")).get("type") == "connection_closed":
                    event.set()
                    return
            except Exception:
                pass
    except (WebSocketDisconnect, RuntimeError):
        event.set()


def revoke_task(task: Any, task_id: str, label: str) -> None:
    """Revoke a Celery task cleanly."""
    try:
        task.revoke(terminate=True, signal="SIGTERM")
        logger.info(f"Revoked {label} task: {task_id}")
    except Exception as e:
        logger.warning(f"Failed to revoke {label} task {task_id}: {e}")


async def wait_for_image_results(
    ws: WebSocket,
    task_id: str,
    detection_task: Any = None,
    detection_timeout: int = 300,
    xai_timeout: int = 300,
    wait_for_xai: bool = True
) -> dict:
    """
    Wait for both detection and XAI results for image processing.
    Subscribes to both channels and keeps the socket open until both arrive
    (or their respective timeouts expire).
    """
    redis_client = None
    results = {"detection": None, "xai": None}

    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = redis_client.pubsub()
        pubsub.subscribe(f"task_detection:{task_id}")
        if wait_for_xai:
            pubsub.subscribe(f"task_xai:{task_id}")

        detection_received = False
        xai_received = False

        start_time = time.time()

        while not (detection_received and (xai_received if wait_for_xai else True)):
            elapsed = time.time() - start_time

            # Enforce individual timeouts
            if not detection_received and elapsed >= detection_timeout:
                logger.warning(f"Detection result timed out for image task: {task_id}")
                break
            if wait_for_xai and not xai_received and elapsed >= xai_timeout:
                logger.warning(f"XAI result timed out for image task: {task_id}")
                break

            try:
                message = pubsub.get_message(timeout=1.0)
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        msg_type = data.get("type")

                        if msg_type == "detection_ready" and not detection_received:
                            detection_received = True
                            results["detection"] = data
                            logger.info(f"Detection result received for image task: {task_id}")

                        elif msg_type == "xai_ready" and not xai_received:
                            xai_received = True
                            results["xai"] = data
                            logger.info(f"XAI result received for image task: {task_id}")

                    except Exception as e:
                        logger.error(f"Error parsing message in wait_for_image_results: {e}")

                # Check if Celery task is complete (detection side only)
                if detection_task and detection_task.ready() and not detection_received:
                    detection_received = True

            except Exception as e:
                logger.error(f"Error in Redis message loop (wait_for_image_results): {e}")
                break

        pubsub.unsubscribe(f"task_detection:{task_id}")
        if wait_for_xai:
            pubsub.unsubscribe(f"task_xai:{task_id}")
        pubsub.close()

        # Fallback: try Redis direct fetch for detection if still missing
        if not detection_received:
            detection_result_json = redis_client.get(f"detection_result:{task_id}")
            if detection_result_json:
                try:
                    detection_result = json.loads(detection_result_json)
                    results["detection"] = detection_result
                    logger.info(f"Retrieved detection result from Redis for task: {task_id}")
                except Exception as e:
                    logger.error(f"Error parsing fallback detection result: {e}")

        return results

    except Exception as e:
        logger.error(f"Error in wait_for_image_results: {e}")
        return results
    finally:
        if redis_client:
            redis_client.close()


async def receive_file_data(
    ws: WebSocket,
    file_name: str,
    file_extension: str,
    send_signal: str
) -> Optional[str]:
    """
    Receive file data from WebSocket and save to disk.
    """
    try:
        await ws.send_text(send_signal)

        file_path = f"{UPLOAD_DIR}/{file_name}.{file_extension}"

        with open(file_path, "wb") as f:
            while True:
                try:
                    data = await ws.receive()
                except (WebSocketDisconnect, RuntimeError) as e:
                    logger.info(f"WebSocket disconnected while receiving {file_extension} data: {e}")
                    return None

                if "bytes" in data and data["bytes"]:
                    f.write(data["bytes"])
                elif "text" in data and data["text"] == "END":
                    break

        logger.info(f"File saved to: {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error receiving file data: {e}")
        return None


async def safe_send_json(ws: WebSocket, data: dict) -> bool:
    """Safely send JSON with disconnect handling."""
    try:
        await ws.send_json(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket disconnected while sending JSON")
        return False
    except Exception as e:
        logger.error(f"Error sending JSON: {e}")
        return False


async def safe_send_text(ws: WebSocket, text: str) -> bool:
    """Safely send text with disconnect handling."""
    try:
        await ws.send_text(text)
        return True
    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket disconnected while sending text")
        return False
    except Exception as e:
        logger.error(f"Error sending text: {e}")
        return False


@router.websocket("/ws/task")
async def websocket_task(ws: WebSocket):
    await ws.accept()

    try:
        # ------------------------------------------------------------------ #
        # 1. Receive initial handshake                                         #
        # ------------------------------------------------------------------ #
        msg = await ws.receive_text()
        try:
            data = json.loads(msg)
            task_id = data.get("task_id", "").strip()
            video_duration = data.get("video_duration", None)
            file_type = data.get("file_type", "video")
            file_name = data.get("file_name", task_id)
            user_id = data.get("user_id")
        except JSONDecodeError:
            task_id = msg.strip()
            video_duration = None
            file_type = "video"
            file_name = task_id

        logger.info(f"Subscribed to task: {task_id}, file_name: {file_name}, file_type: {file_type}, video_duration: {video_duration}")

        is_image = file_type == "image"

        # ------------------------------------------------------------------ #
        # 2a. IMAGE path                                                        #
        # ------------------------------------------------------------------ #
        if is_image:
            file_path = await receive_file_data(ws, file_name, "jpg", "SEND_IMAGE")
            if not file_path:
                return

            try:
                if user_id:
                    db = SessionLocal()
                    try:
                        task = VideoAnalysisTask(
                            task_id=task_id,
                            user_id=user_id,
                            video_path=file_path,
                            status=TaskStatus.processing,
                        )
                        db.add(task)
                        db.commit()
                        db.refresh(task)
                    except Exception as e:
                        logger.error(f"Error creating task: {e}")
                        db.rollback()
                        await safe_send_json(ws, {"type": "error", "message": "Failed to create task"})
                        return
                    finally:
                        db.close()

                if not await safe_send_text(ws, "Processing image..."):
                    return

                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError(f"Could not load image from {file_path}")

                success, buffer = cv2.imencode(".jpg", img)
                if not success:
                    raise ValueError("Failed to encode image")

                frame_data_b64 = base64.b64encode(buffer).decode("utf-8")

                detection_task = run_gend_inference.delay(
                    task_id=task_id,
                    frame_data=frame_data_b64,
                    frame_index=0,
                    timestamp="00:00:00.000",
                    user_id=user_id
                )

                logger.info(f"Sent image {file_name} to detection worker for task: {task_id}")

                redis_client = None

                # ---------------------------------------------------------- #
                # Tracking state for image processing                          #
                # ---------------------------------------------------------- #
                detections_received = 0      # detection_ready messages
                xai_received = 0             # xai_ready messages

                # For image, we consider it as 1 frame
                frames_received = 1
                anomaly_frame_indices: set = set()

                # Results storage
                detection_result = None
                xai_result = None

                # Has the pipeline finished?
                pipeline_done = False

                try:
                    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                    pubsub = redis_client.pubsub()
                    pubsub.subscribe(f"task_detection:{task_id}")
                    pubsub.subscribe(f"task_xai:{task_id}")

                    last_message_time = time.time()
                    # Hard outer timeout: 10 minutes
                    HARD_TIMEOUT = 600
                    start_time = time.time()

                    # ------------------------------------------------------ #
                    # Disconnect watcher                                       #
                    # ------------------------------------------------------ #
                    disconnected = asyncio.Event()
                    watcher = asyncio.create_task(watch_disconnect(ws, disconnected))

                    while not pipeline_done:

                        # -------------------------------------------------- #
                        # Hard timeout guard                                   #
                        # -------------------------------------------------- #
                        if time.time() - start_time > HARD_TIMEOUT:
                            logger.error(f"Hard timeout reached for image task: {task_id}")
                            break

                        # -------------------------------------------------- #
                        # Disconnect check                                     #
                        # -------------------------------------------------- #
                        if disconnected.is_set():
                            logger.info(f"Client disconnected, revoking image task: {task_id}")
                            revoke_task(detection_task, task_id, "image")
                            pipeline_done = True
                            break

                        # -------------------------------------------------- #
                        # Completion check                                      #
                        # -------------------------------------------------- #
                        if (
                            detections_received >= frames_received
                            and xai_received >= len(anomaly_frame_indices)
                            and frames_received > 0
                        ):
                            logger.info(
                                f"Image pipeline complete for task {task_id}: "
                                f"detections={detections_received}, "
                                f"xai={xai_received}/{len(anomaly_frame_indices)}"
                            )
                            pipeline_done = True
                            break

                        try:
                            message = pubsub.get_message(timeout=1.0)
                        except Exception as e:
                            logger.error(f"Error polling Redis pubsub: {e}", exc_info=True)
                            break

                        if not message or message["type"] != "message":
                            continue

                        # -------------------------------------------------- #
                        # Route the message                                     #
                        # -------------------------------------------------- #
                        last_message_time = time.time()

                        try:
                            data = json.loads(message["data"])
                        except Exception as e:
                            logger.error(f"Failed to parse Redis message: {e}")
                            continue

                        data_type = data.get("type", "")

                        # -- detection_ready -------------------------------- #
                        if data_type == "detection_ready":
                            detections_received += 1
                            detection_result = data

                            if data.get("is_anomaly"):
                                anomaly_frame_indices.add(0)  # Image is frame 0

                            # Send detection result to frontend
                            await safe_send_json(ws, data)
                            logger.info(f"Detection result sent for image task: {task_id}")

                        # -- xai_ready ------------------------------------- #
                        elif data_type == "xai_ready":
                            xai_received += 1
                            xai_result = data

                            # Send XAI result to frontend
                            await safe_send_json(ws, data)
                            logger.info(f"XAI result sent for image task: {task_id}")

                    pubsub.unsubscribe(f"task_detection:{task_id}")
                    pubsub.unsubscribe(f"task_xai:{task_id}")
                    pubsub.close()

                    # ------------------------------------------------------ #
                    # Cancel disconnect watcher                               #
                    # ------------------------------------------------------ #
                    watcher.cancel()

                except Exception as e:
                    logger.error(f"Error in image processing loop: {e}")
                finally:
                    if redis_client:
                        redis_client.close()

                await safe_send_json(ws, {
                    "type": "processing_complete",
                    "message": "Image processing completed",
                    "total_frames": 1,
                    "anomaly_count": (
                        detection_result.get("anomaly_count", 0)
                        if detection_result and isinstance(detection_result, dict)
                        else 0
                    ),
                    "task_id": task_id,
                    "is_image": True,
                    "detection_result": detection_result,
                    "xai_result": xai_result,
                })

                if user_id:
                    db = SessionLocal()
                    try:
                        task = db.query(VideoAnalysisTask).filter_by(task_id=task_id).first()
                        if task:
                            task.status = TaskStatus.completed
                            task.completed_at = datetime.utcnow()
                            db.commit()
                    except Exception as e:
                        logger.error(f"Error updating task: {e}")
                        db.rollback()
                    finally:
                        db.close()

            except Exception as e:
                logger.error(f"Error in image processing: {e}", exc_info=True)
                await safe_send_json(ws, {
                    "type": "error",
                    "message": f"Image processing failed: {str(e)}",
                    "task_id": task_id,
                })

        # ------------------------------------------------------------------ #
        # 2b. VIDEO path                                                        #
        # ------------------------------------------------------------------ #
        else:
            file_path = await receive_file_data(ws, file_name, "mp4", "SEND_VIDEO")
            if not file_path:
                return

            logger.info(f"Starting frame extraction for task: {task_id}, file: {file_name}")

            if user_id:
                db = SessionLocal()
                try:
                    task = VideoAnalysisTask(
                        task_id=task_id,
                        user_id=user_id,
                        video_path=file_path,
                        status=TaskStatus.processing,
                    )
                    db.add(task)
                    db.commit()
                    db.refresh(task)
                except Exception as e:
                    logger.error(f"Error creating task: {e}")
                    db.rollback()
                    await safe_send_json(ws, {"type": "error", "message": "Failed to create task"})
                    return
                finally:
                    db.close()

            try:
                celery_task = extract_faces_with_optical_flow.delay(
                    file_path,
                    task_id=task_id,
                    video_duration=video_duration,
                    user_id=user_id
                )

                if not await safe_send_text(ws, "Processing..."):
                    return

                redis_client = None

                # ---------------------------------------------------------- #
                # Tracking state                                               #
                # ---------------------------------------------------------- #
                processed_frames: dict = {}

                # Counts driven purely by messages received
                frames_received = 0          # frame_ready messages
                detections_received = 0      # detection_ready messages
                xai_received = 0             # xai_ready messages

                # We only know the expected XAI count once all detections are
                # in, so we track anomalies as they arrive.
                anomaly_frame_indices: set = set()

                # Has the frame-extraction Celery task finished?
                extraction_complete = False

                # A sentinel published by the worker when it is truly done
                # with ALL work (detection + xai).  If your workers publish
                # such a message use it here; otherwise we rely on counts.
                pipeline_done = False

                try:
                    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                    pubsub = redis_client.pubsub()
                    pubsub.subscribe(f"task_frames:{task_id}")
                    pubsub.subscribe(f"task_detection:{task_id}")
                    pubsub.subscribe(f"task_xai:{task_id}")

                    last_message_time = time.time()
                    # How long to wait with no new messages once extraction
                    # is complete and counts look satisfied before giving up.
                    IDLE_GRACE = 5.0   # seconds
                    # Hard outer timeout: 10 minutes
                    HARD_TIMEOUT = 600
                    start_time = time.time()

                    # ------------------------------------------------------ #
                    # Disconnect watcher                                       #
                    # ------------------------------------------------------ #
                    disconnected = asyncio.Event()
                    watcher = asyncio.create_task(watch_disconnect(ws, disconnected))
                    cancelled = False  # track if we cancelled due to disconnect

                    while not pipeline_done:

                        # -------------------------------------------------- #
                        # Hard timeout guard                                   #
                        # -------------------------------------------------- #
                        if time.time() - start_time > HARD_TIMEOUT:
                            logger.error(f"Hard timeout reached for task: {task_id}")
                            break

                        # -------------------------------------------------- #
                        # Disconnect check                                     #
                        # -------------------------------------------------- #
                        if disconnected.is_set():
                            logger.info(f"Client disconnected, revoking video task: {task_id}")
                            revoke_task(celery_task, task_id, "video")
                            cancelled = True
                            pipeline_done = True
                            break

                        # -------------------------------------------------- #
                        # Completion check                                      #
                        # -------------------------------------------------- #
                        if (
                            extraction_complete
                            and detections_received >= frames_received
                            and xai_received >= len(anomaly_frame_indices)
                            and frames_received > 0
                            and (time.time() - last_message_time) > IDLE_GRACE
                        ):
                            logger.info(
                                f"Pipeline complete for task {task_id}: "
                                f"frames={frames_received}, detections={detections_received}, "
                                f"xai={xai_received}/{len(anomaly_frame_indices)}"
                            )
                            pipeline_done = True
                            break

                        try:
                            message = pubsub.get_message(timeout=1.0)
                        except Exception as e:
                            logger.error(f"Error polling Redis pubsub: {e}", exc_info=True)
                            break

                        if not message or message["type"] != "message":
                            # No new message; check if extraction task finished
                            if not extraction_complete and celery_task.ready():
                                extraction_complete = True
                                logger.info(f"Frame extraction Celery task finished for {task_id}")
                            continue

                        # -------------------------------------------------- #
                        # Route the message                                     #
                        # -------------------------------------------------- #
                        last_message_time = time.time()

                        try:
                            data = json.loads(message["data"])
                        except Exception as e:
                            logger.error(f"Failed to parse Redis message: {e}")
                            continue

                        data_type = data.get("type", "")

                        # -- frame_ready ------------------------------------ #
                        if data_type == "frame_ready":
                            frame_index = data.get("frame_index", 0)
                            frame_data = data.get("frame_data")
                            timestamp = data.get("timestamp", "")
                            timestamp_seconds = data.get("timestamp_seconds", 0.0)
                            fps = data.get("fps", 30.0)

                            # Store for later combination with detection
                            processed_frames[frame_index] = {"frame": data, "detection": None}
                            frames_received += 1

                            # Forward raw frame to frontend immediately
                            frame_message = {
                                "type": "frame_ready",
                                "frame_index": frame_index,
                                "frame_data": frame_data,
                                "timestamp": timestamp,
                                "timestamp_seconds": timestamp_seconds,
                                "fps": fps,
                                "video_duration": video_duration or 0.0,
                                "task_id": task_id,
                                "is_image": False,
                            }
                            if not await safe_send_json(ws, frame_message):
                                break   # client disconnected
                            await asyncio.sleep(FRAME_SEND_DELAY)

                        # -- detection_ready -------------------------------- #
                        elif data_type == "detection_ready":
                            frame_index = data.get("frame_index", 0)
                            data["is_processed"] = True
                            detections_received += 1

                            if data.get("is_anomaly"):
                                anomaly_frame_indices.add(frame_index)

                            # Combine with stored frame if available
                            if frame_index in processed_frames:
                                processed_frames[frame_index]["detection"] = data
                                combined_message = {
                                    "type": "frame_with_detection",
                                    "frame_index": frame_index,
                                    "frame_data": processed_frames[frame_index]["frame"].get("frame_data"),
                                    "detection": data,
                                    "timestamp": processed_frames[frame_index]["frame"].get("timestamp", ""),
                                    "timestamp_seconds": processed_frames[frame_index]["frame"].get("timestamp_seconds", 0.0),
                                    "task_id": task_id,
                                }
                                if not await safe_send_json(ws, combined_message):
                                    break
                            else:
                                # Detection arrived before frame (edge case)
                                if not await safe_send_json(ws, data):
                                    break

                            logger.debug(f"Forwarded detection {detections_received} for frame {frame_index}")
                            await asyncio.sleep(FRAME_SEND_DELAY)

                        # -- xai_ready ------------------------------------- #
                        elif data_type == "xai_ready":
                            xai_received += 1
                            if not await safe_send_json(ws, data):
                                break
                            logger.debug(f"Forwarded XAI result for frame {data.get('frame_index')} ({xai_received}/{len(anomaly_frame_indices)})")
                            await asyncio.sleep(FRAME_SEND_DELAY)

                        # -- timeshap_ready -------------------------------- #
                        elif data_type == "timeshap_ready":
                            if not await safe_send_json(ws, data):
                                break
                            logger.debug(f"Forwarded TimeSHAP result for task {data.get('task_id')}")
                            await asyncio.sleep(FRAME_SEND_DELAY)

                        # -- pipeline_done sentinel (optional) ------------- #
                        elif data_type == "pipeline_done":
                            # Workers can publish this to signal all work is finished
                            pipeline_done = True
                            logger.info(f"Received pipeline_done sentinel for task {task_id}")

                        # -- anything else --------------------------------- #
                        else:
                            if not await safe_send_json(ws, data):
                                break

                        # Re-check extraction flag on every iteration
                        if not extraction_complete and celery_task.ready():
                            extraction_complete = True
                            logger.info(f"Frame extraction Celery task finished for {task_id}")

                    # Cleanup pubsub
                    pubsub.unsubscribe(f"task_frames:{task_id}")
                    pubsub.unsubscribe(f"task_detection:{task_id}")
                    pubsub.unsubscribe(f"task_xai:{task_id}")
                    pubsub.close()

                    # ------------------------------------------------------ #
                    # Cancel disconnect watcher                               #
                    # ------------------------------------------------------ #
                    watcher.cancel()

                except Exception as e:
                    logger.error(f"Error in Redis subscription loop: {e}", exc_info=True)

                # ---------------------------------------------------------- #
                # Retrieve final Celery result and XAI from Redis             #
                # ---------------------------------------------------------- #
                result = {}
                if cancelled:
                    logger.info(f"Skipping result fetch for cancelled task: {task_id}")
                else:
                    try:
                        result = celery_task.get(timeout=30)
                        logger.info(f"Celery task result: {result}")
                    except Exception as e:
                        logger.error(f"Error getting Celery task result: {e}", exc_info=True)

                if user_id:
                    db = SessionLocal()
                    try:
                        task = db.query(VideoAnalysisTask).filter_by(task_id=task_id).first()
                        if task:
                            task.status = TaskStatus.cancelled if cancelled else TaskStatus.completed
                            task.completed_at = datetime.utcnow()
                            if not cancelled:
                                task.faces_detected_frames = result.get("faces_detected_frames", 0)
                                task.frames_skipped = result.get("frames_skipped", 0)
                            db.commit()
                    except Exception as e:
                        logger.error(f"Error updating task: {e}")
                        db.rollback()
                    finally:
                        db.close()

                xai_result = None
                if redis_client:
                    try:
                        xai_result_json = redis_client.get(f"xai_result:{task_id}")
                        if xai_result_json:
                            xai_result = json.loads(xai_result_json)
                            result["xai"] = xai_result
                    except Exception as e:
                        logger.error(f"Error parsing XAI result: {e}", exc_info=True)
                    finally:
                        redis_client.close()

                # ---------------------------------------------------------- #
                # Send completion message                                       #
                # ---------------------------------------------------------- #
                completion_message = {
                    "type": "processing_complete",
                    "message": "Frame extraction and spatial detection completed",
                    "total_frames": result.get("total_frames", frames_received),
                    "processed_frames": detections_received,
                    "anomaly_count": len(anomaly_frame_indices),
                    "task_id": task_id,
                }

                if xai_result:
                    completion_message["xai_results"] = xai_result.get("xai_results", [])
                    completion_message["timeshap_result"] = xai_result.get("timeshap_result", None)
                    completion_message["total_frames_explained"] = xai_result.get("total_frames_explained", 0)

                await safe_send_json(ws, completion_message)

            except Exception as e:
                logger.error(f"Error in video frame extraction: {e}", exc_info=True)
                await safe_send_json(ws, {
                    "type": "error",
                    "message": f"Frame extraction failed: {str(e)}",
                    "task_id": task_id,
                })

        # ------------------------------------------------------------------ #
        # 3. Close WebSocket only after all work is done                       #
        # ------------------------------------------------------------------ #
        try:
            await ws.close()
        except Exception:
            pass

    except (WebSocketDisconnect, RuntimeError) as e:
        logger.info(f"WebSocket disconnected during task handling: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_task: {e}", exc_info=True)