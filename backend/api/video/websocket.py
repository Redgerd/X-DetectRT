import os
import asyncio
import io
import logging
import cv2
import base64
import time
from fastapi import WebSocket, APIRouter, WebSocketDisconnect
from core.celery.frame_selection import extract_faces_with_optical_flow
from core.celery.detection_tasks import run_gend_inference
import json
from json import JSONDecodeError
import redis
from config import settings
from typing import Optional
from models import VideoAnalysisTask
from models.tasks import TaskStatus
from core.database import SessionLocal
from core.storage import storage
from datetime import datetime

logger = logging.getLogger(__name__)

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MINIO_UPLOAD_BUCKET = os.environ.get("MINIO_BUCKET_UPLOADS", "uploads")

router = APIRouter()
FRAME_SEND_DELAY = 0.05


async def receive_file_data(
    ws: WebSocket,
    file_name: str,
    file_extension: str,
    send_signal: str,
    task_id: str,
) -> Optional[str]:
    """
    Receive file bytes from WebSocket, upload directly to MinIO,
    and also write to local disk so Celery workers (which expect a
    file path) keep working unchanged.
    """
    try:
        await ws.send_text(send_signal)

        buffer = io.BytesIO()
        while True:
            try:
                data = await ws.receive()
            except (WebSocketDisconnect, RuntimeError) as e:
                logger.info(f"WebSocket disconnected while receiving {file_extension}: {e}")
                return None

            if "bytes" in data and data["bytes"]:
                buffer.write(data["bytes"])
            elif "text" in data and data["text"] == "END":
                break

        raw = buffer.getvalue()
        logger.info(f"Received {len(raw)} bytes for {file_name}.{file_extension}")

        # ── Upload to MinIO ───────────────────────────────────────────── #
        content_type = "video/mp4" if file_extension == "mp4" else "image/jpeg"
        object_name  = f"tasks/{task_id}/{file_name}.{file_extension}"
        storage.upload_bytes(MINIO_UPLOAD_BUCKET, raw, object_name, content_type)
        logger.info(f"Uploaded to MinIO: {MINIO_UPLOAD_BUCKET}/{object_name}")

        # ── Write to local disk for Celery workers ────────────────────── #
        file_path = f"{UPLOAD_DIR}/{file_name}.{file_extension}"
        with open(file_path, "wb") as f:
            f.write(raw)

        return file_path

    except Exception as e:
        logger.error(f"Error receiving file data: {e}")
        return None


async def safe_send_json(ws: WebSocket, data: dict) -> bool:
    try:
        await ws.send_json(data)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as e:
        logger.error(f"Error sending JSON: {e}")
        return False


async def safe_send_text(ws: WebSocket, text: str) -> bool:
    try:
        await ws.send_text(text)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as e:
        logger.error(f"Error sending text: {e}")
        return False


@router.websocket("/ws/task")
async def websocket_task(ws: WebSocket):
    await ws.accept()

    try:
        msg = await ws.receive_text()
        try:
            data           = json.loads(msg)
            task_id        = data.get("task_id", "").strip()
            video_duration = data.get("video_duration", None)
            file_type      = data.get("file_type", "video")
            file_name      = data.get("file_name", task_id)
            user_id        = data.get("user_id")
            if not user_id:
                await safe_send_json(ws, {"type": "error", "message": "user_id is required"})
                return
        except JSONDecodeError:
            task_id        = msg.strip()
            video_duration = None
            file_type      = "video"
            file_name      = task_id

        logger.info(f"Task {task_id} | type={file_type}")
        is_image = file_type == "image"

        # ── IMAGE ──────────────────────────────────────────────────────── #
        if is_image:
            file_path = await receive_file_data(ws, file_name, "jpg", "SEND_IMAGE", task_id)
            if not file_path:
                return

            try:
                db = SessionLocal()
                try:
                    task = VideoAnalysisTask(
                        task_id=task_id,
                        user_id=user_id,
                        video_path=f"minio://{MINIO_UPLOAD_BUCKET}/tasks/{task_id}/{file_name}.jpg",
                        status=TaskStatus.processing,
                    )
                    db.add(task)
                    db.commit()
                    db.refresh(task)
                except Exception as e:
                    logger.error(f"DB error: {e}")
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
                success, buf = cv2.imencode(".jpg", img)
                if not success:
                    raise ValueError("Failed to encode image")
                frame_data_b64 = base64.b64encode(buf).decode("utf-8")

                detection_task = run_gend_inference.delay(
                    task_id=task_id, frame_data=frame_data_b64,
                    frame_index=0, timestamp="00:00:00.000",
                )

                redis_client         = None
                detections_received  = 0
                xai_received         = 0
                anomaly_frame_indices: set = set()
                detection_result     = None
                xai_result           = None
                pipeline_done        = False

                try:
                    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                    pubsub = redis_client.pubsub()
                    pubsub.subscribe(f"task_detection:{task_id}", f"task_xai:{task_id}")

                    start_time = time.time()
                    while not pipeline_done:
                        if time.time() - start_time > 600:
                            break
                        if detections_received >= 1 and xai_received >= len(anomaly_frame_indices):
                            pipeline_done = True
                            break

                        message = pubsub.get_message(timeout=1.0)
                        if not message or message["type"] != "message":
                            continue

                        try:
                            d = json.loads(message["data"])
                        except Exception:
                            continue

                        if d.get("type") == "detection_ready":
                            detections_received += 1
                            detection_result = d
                            if d.get("is_anomaly"):
                                anomaly_frame_indices.add(0)
                            await safe_send_json(ws, d)
                        elif d.get("type") == "xai_ready":
                            xai_received += 1
                            xai_result = d
                            await safe_send_json(ws, d)

                    pubsub.unsubscribe()
                    pubsub.close()
                finally:
                    if redis_client:
                        redis_client.close()

                await safe_send_json(ws, {
                    "type": "processing_complete", "message": "Image processing completed",
                    "total_frames": 1, "task_id": task_id, "is_image": True,
                    "anomaly_count": detection_result.get("anomaly_count", 0) if detection_result else 0,
                    "detection_result": detection_result, "xai_result": xai_result,
                })

                db = SessionLocal()
                try:
                    t = db.query(VideoAnalysisTask).filter_by(task_id=task_id).first()
                    if t:
                        t.status = TaskStatus.completed
                        t.completed_at = datetime.utcnow()
                        db.commit()
                except Exception as e:
                    db.rollback()
                finally:
                    db.close()

            except Exception as e:
                logger.error(f"Image processing error: {e}", exc_info=True)
                await safe_send_json(ws, {"type": "error", "message": str(e), "task_id": task_id})

        # ── VIDEO ──────────────────────────────────────────────────────── #
        else:
            file_path = await receive_file_data(ws, file_name, "mp4", "SEND_VIDEO", task_id)
            if not file_path:
                return

            db = SessionLocal()
            try:
                task = VideoAnalysisTask(
                    task_id=task_id, user_id=user_id,
                    video_path=f"minio://{MINIO_UPLOAD_BUCKET}/tasks/{task_id}/{file_name}.mp4",
                    status=TaskStatus.processing,
                )
                db.add(task)
                db.commit()
                db.refresh(task)
            except Exception as e:
                logger.error(f"DB error: {e}")
                db.rollback()
                await safe_send_json(ws, {"type": "error", "message": "Failed to create task"})
                return
            finally:
                db.close()

            try:
                celery_task = extract_faces_with_optical_flow.delay(
                    file_path, task_id=task_id, video_duration=video_duration,
                )
                if not await safe_send_text(ws, "Processing..."):
                    return

                redis_client           = None
                processed_frames: dict = {}
                frames_received        = 0
                detections_received    = 0
                xai_received           = 0
                anomaly_frame_indices: set = set()
                extraction_complete    = False
                pipeline_done          = False

                try:
                    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                    pubsub = redis_client.pubsub()
                    pubsub.subscribe(
                        f"task_frames:{task_id}",
                        f"task_detection:{task_id}",
                        f"task_xai:{task_id}",
                    )

                    last_message_time = time.time()
                    IDLE_GRACE        = 5.0
                    start_time        = time.time()

                    while not pipeline_done:
                        if time.time() - start_time > 600:
                            break

                        if (
                            extraction_complete
                            and detections_received >= frames_received
                            and xai_received >= len(anomaly_frame_indices)
                            and frames_received > 0
                            and (time.time() - last_message_time) > IDLE_GRACE
                        ):
                            pipeline_done = True
                            break

                        message = pubsub.get_message(timeout=1.0)
                        if not message or message["type"] != "message":
                            if not extraction_complete and celery_task.ready():
                                extraction_complete = True
                            continue

                        last_message_time = time.time()
                        try:
                            d = json.loads(message["data"])
                        except Exception:
                            continue

                        dtype = d.get("type", "")

                        if dtype == "frame_ready":
                            fi = d.get("frame_index", 0)
                            processed_frames[fi] = {"frame": d, "detection": None}
                            frames_received += 1
                            if not await safe_send_json(ws, {
                                "type": "frame_ready", "frame_index": fi,
                                "frame_data": d.get("frame_data"),
                                "timestamp": d.get("timestamp", ""),
                                "timestamp_seconds": d.get("timestamp_seconds", 0.0),
                                "fps": d.get("fps", 30.0),
                                "video_duration": video_duration or 0.0,
                                "task_id": task_id, "is_image": False,
                            }):
                                break
                            await asyncio.sleep(FRAME_SEND_DELAY)

                        elif dtype == "detection_ready":
                            fi = d.get("frame_index", 0)
                            d["is_processed"] = True
                            detections_received += 1
                            if d.get("is_anomaly"):
                                anomaly_frame_indices.add(fi)
                            if fi in processed_frames:
                                processed_frames[fi]["detection"] = d
                                if not await safe_send_json(ws, {
                                    "type": "frame_with_detection", "frame_index": fi,
                                    "frame_data": processed_frames[fi]["frame"].get("frame_data"),
                                    "detection": d,
                                    "timestamp": processed_frames[fi]["frame"].get("timestamp", ""),
                                    "timestamp_seconds": processed_frames[fi]["frame"].get("timestamp_seconds", 0.0),
                                    "task_id": task_id,
                                }):
                                    break
                            else:
                                if not await safe_send_json(ws, d):
                                    break
                            await asyncio.sleep(FRAME_SEND_DELAY)

                        elif dtype == "xai_ready":
                            xai_received += 1
                            if not await safe_send_json(ws, d):
                                break
                            await asyncio.sleep(FRAME_SEND_DELAY)

                        elif dtype == "timeshap_ready":
                            if not await safe_send_json(ws, d):
                                break
                            await asyncio.sleep(FRAME_SEND_DELAY)

                        elif dtype == "pipeline_done":
                            pipeline_done = True

                        else:
                            if not await safe_send_json(ws, d):
                                break

                        if not extraction_complete and celery_task.ready():
                            extraction_complete = True

                    pubsub.unsubscribe()
                    pubsub.close()

                except Exception as e:
                    logger.error(f"Redis loop error: {e}", exc_info=True)

                result = {}
                try:
                    result = celery_task.get(timeout=30)
                except Exception as e:
                    logger.error(f"Celery result error: {e}", exc_info=True)

                db = SessionLocal()
                try:
                    t = db.query(VideoAnalysisTask).filter_by(task_id=task_id).first()
                    if t:
                        t.status                = TaskStatus.completed
                        t.completed_at          = datetime.utcnow()
                        t.faces_detected_frames = result.get("faces_detected_frames", 0)
                        t.frames_skipped        = result.get("frames_skipped", 0)
                        db.commit()
                except Exception as e:
                    db.rollback()
                finally:
                    db.close()

                xai_result = None
                if redis_client:
                    try:
                        xr = redis_client.get(f"xai_result:{task_id}")
                        if xr:
                            xai_result = json.loads(xr)
                            result["xai"] = xai_result
                    except Exception as e:
                        logger.error(f"XAI result error: {e}", exc_info=True)
                    finally:
                        redis_client.close()

                msg_out = {
                    "type": "processing_complete",
                    "message": "Frame extraction and spatial detection completed",
                    "total_frames": result.get("total_frames", frames_received),
                    "processed_frames": detections_received,
                    "anomaly_count": len(anomaly_frame_indices),
                    "task_id": task_id,
                }
                if xai_result:
                    msg_out["xai_results"]            = xai_result.get("xai_results", [])
                    msg_out["timeshap_result"]        = xai_result.get("timeshap_result")
                    msg_out["total_frames_explained"] = xai_result.get("total_frames_explained", 0)

                await safe_send_json(ws, msg_out)

            except Exception as e:
                logger.error(f"Video processing error: {e}", exc_info=True)
                await safe_send_json(ws, {"type": "error", "message": str(e), "task_id": task_id})

        try:
            await ws.close()
        except Exception:
            pass

    except (WebSocketDisconnect, RuntimeError) as e:
        logger.info(f"WebSocket disconnected: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)