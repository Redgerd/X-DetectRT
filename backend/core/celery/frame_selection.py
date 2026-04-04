# backend/core/celery/frame_tasks.py
import cv2
import numpy as np
import base64
import json # Potentially useful for logging or debugging JSON data
import time # For time.sleep
import logging # For logging
from datetime import datetime # For timestamp

import redis # Import synchronous redis client for Celery tasks

from celery import current_app, shared_task
from core.celery.celery_app import celery_app
from config import settings # For REDIS_URL
from mtcnn import MTCNN  # Face detection
from PIL import Image     # For resizing and image conversion
from tensorflow.keras.applications.inception_v3 import preprocess_input
import os
from models import ProcessedFrame, VideoAnalysisTask
from models.tasks import TaskStatus
from core.database import SessionLocal
from datetime import datetime
# Import the GenD detection task from detection_tasks
from .detection_tasks import run_gend_inference

# Import the deepfake detection task from its new module
# Ensure this import path is correct based on your project structure
# from backend.core.celery.detection_tasks import perform_deepfake_detection
# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HEIGHT = 300
WIDTH = 400
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)


def seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


MIN_CROP_SIZE = 40
PADDING = 20
FLOW_THRESHOLD = 3.0

@shared_task(name="frame_selection_pipeline.run")
def extract_faces_with_optical_flow(video_path, task_id=None, max_frames=15, video_duration=None):

    try:

        if not task_id:
            task_id = os.path.basename(video_path).replace(".mp4", "")

        db = SessionLocal()

        if not os.path.exists(video_path):
            return {"error": "Video not found", "task_id": task_id}

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {"error": "Failed to open video", "task_id": task_id}

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or np.isnan(fps):
            fps = 30.0

        frame_count_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if frame_count_total > 0:
            total_duration = frame_count_total / fps
        else:
            total_duration = video_duration if video_duration else 0

        logger.info(f"[{task_id}] FPS={fps:.2f}, total_frames={frame_count_total}")

        detector = MTCNN()

        prev_gray = None
        processed_faces = []

        processed_count = 0
        skipped_count = 0
        frame_index = 0

        while processed_count < max_frames:

            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            try:

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                face_for_model = None

                # ---------- optical flow ----------
                if prev_gray is not None:

                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray,
                        gray,
                        None,
                        0.5,
                        3,
                        15,
                        3,
                        5,
                        1.2,
                        0
                    )

                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                    motion_mask = mag > FLOW_THRESHOLD

                    if motion_mask.any():

                        ys, xs = np.where(motion_mask)

                        x1, x2 = xs.min(), xs.max()
                        y1, y2 = ys.min(), ys.max()

                        # padding
                        x1 = max(0, x1 - PADDING)
                        y1 = max(0, y1 - PADDING)
                        x2 = min(frame_rgb.shape[1], x2 + PADDING)
                        y2 = min(frame_rgb.shape[0], y2 + PADDING)

                        width = x2 - x1
                        height = y2 - y1

                        if width > MIN_CROP_SIZE and height > MIN_CROP_SIZE:

                            motion_crop = frame_rgb[y1:y2, x1:x2]

                            # Skip redundant None check - NumPy slicing always returns array
                            detections = detector.detect_faces(motion_crop)

                            if detections:

                                x, y, w, h = detections[0]["box"]

                                x = max(0, x)
                                y = max(0, y)
                                w = max(1, w)
                                h = max(1, h)

                                face_candidate = motion_crop[y:y+h, x:x+w]

                                # Skip redundant None check - NumPy slicing always returns array
                                face_for_model = face_candidate

                prev_gray = gray

                # ---------- skip if no face ----------
                if face_for_model is None:
                    skipped_count += 1
                    continue

                # ---------- preprocess ----------
                face_img = Image.fromarray(face_for_model).resize((WIDTH, HEIGHT))

                face_arr = preprocess_input(np.array(face_img))

                processed_faces.append(face_arr)

                # ---------- encode ----------
                frame_bgr = cv2.cvtColor(face_for_model, cv2.COLOR_RGB2BGR)

                success, buffer = cv2.imencode(".jpg", frame_bgr)

                if success:

                    frame_data_b64 = base64.b64encode(buffer).decode("utf-8")

                    timestamp_sec = frame_index / fps

                    timestamp_str = seconds_to_hhmmss(timestamp_sec)

                    # Insert frame into database
                    try:
                        frame_db = ProcessedFrame(
                            task_id=task_id,
                            frame_index=processed_count,
                            timestamp=timestamp_str,
                            timestamp_seconds=round(timestamp_sec, 3),
                            frame_data=frame_data_b64,
                            fps=round(fps, 2),
                            video_duration=round(total_duration, 2)
                        )
                        db.add(frame_db)
                        db.commit()
                        db.refresh(frame_db)
                    except Exception as e:
                        logger.warning(f"[{task_id}] Failed to insert frame {processed_count}: {e}")
                        db.rollback()

                    # ---------- redis stream ----------
                    redis_client.publish(
                        f"task_frames:{task_id}",
                        json.dumps(
                            {
                                "type": "frame_ready",
                                "frame_index": processed_count,
                                "frame_data": frame_data_b64,
                                "timestamp": timestamp_str,
                                "timestamp_seconds": round(timestamp_sec, 3),
                                "fps": round(fps, 2),
                                "video_duration": round(total_duration, 2),
                                "task_id": task_id
                            }
                        )
                    )

                    # ---------- GenD ----------
                    run_gend_inference.delay(
                        task_id=task_id,
                        frame_data=frame_data_b64,
                        frame_index=processed_count,
                        timestamp=timestamp_str
                    )

                processed_count += 1

            except Exception as e:

                logger.warning(f"[{task_id}] frame skipped error: {e}")

                skipped_count += 1

        cap.release()

        # ---------- preview ----------
        preview_frames = []

        for arr in processed_faces:

            img = (arr * 255).astype(np.uint8)

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode(".jpg", img_bgr)

            preview_frames.append(
                base64.b64encode(buffer).decode("utf-8")
            )

        result = {
            "message": "Only frames with detected faces were processed",
            "task_id": task_id,
            "video_path": video_path,
            "faces_detected_frames": processed_count,
            "frames_skipped": skipped_count,
            "preview_frames": preview_frames
        }

        redis_client.set(
            f"task_result:{task_id}",
            json.dumps(result)
        )

        # Update task status
        try:
            task = db.query(VideoAnalysisTask).filter_by(task_id=task_id).first()
            if task:
                task.status = TaskStatus.completed
                task.completed_at = datetime.utcnow()
                task.faces_detected_frames = processed_count
                task.frames_skipped = skipped_count
                db.commit()
        except Exception as e:
            logger.warning(f"[{task_id}] Failed to update task: {e}")
            db.rollback()
        finally:
            db.close()

        logger.info(
            f"[{task_id}] done | faces={processed_count} skipped={skipped_count}"
        )

        return result

    except Exception as e:

        logger.exception(f"[{task_id}] pipeline crashed: {e}")

        # Update task status to failed
        try:
            task = db.query(VideoAnalysisTask).filter_by(task_id=task_id).first()
            if task:
                task.status = TaskStatus.failed
                db.commit()
        except Exception as update_e:
            logger.warning(f"[{task_id}] Failed to update task to failed: {update_e}")
            db.rollback()
        finally:
            db.close()

        return {
            "error": str(e),
            "task_id": task_id
        }