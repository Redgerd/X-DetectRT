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
from .spatialDetection import run_chained_detection

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


@shared_task(name="frame_selection_pipeline.run")
def extract_faces_with_optical_flow(video_path, task_id=None, max_frames=60, video_duration=None):

    if not task_id:
        task_id = os.path.basename(video_path).replace(".mp4", "")

    if not os.path.exists(video_path):
        return {"error": "Video not found", "task_id": task_id}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Failed to open video", "task_id": task_id}

    # Get actual FPS and duration from video
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30.0  # Fallback to 30 FPS if not detected
    
    # Get video duration in seconds
    total_duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position
    frame_count_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_count_total > 0 and fps > 0:
        total_duration = frame_count_total / fps
    elif video_duration:
        total_duration = video_duration
    
    logger.info(f"Video FPS: {fps}, Total Duration: {total_duration:.2f}s, Total Frames: {frame_count_total}")
    
    detector = MTCNN()
    prev_gray = None
    processed_faces = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_for_model = None

        if prev_gray is not None:
            # ---------------- Optical Flow ----------------
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = mag > 1.2

            if motion_mask.sum() > 0:
                ys, xs = np.where(motion_mask)
                x1, x2 = xs.min(), xs.max()
                y1, y2 = ys.min(), ys.max()
                motion_crop = frame_rgb[y1:y2, x1:x2]

                # ---------------- MTCNN ----------------
                detections = detector.detect_faces(motion_crop)
                if detections:
                    x, y, w, h = detections[0]["box"]
                    x, y = max(0, x), max(0, y)
                    face_for_model = motion_crop[y:y+h, x:x+w]
                else:
                    face_for_model = motion_crop

        prev_gray = gray

        if face_for_model is None:
            face_for_model = frame_rgb

        # ---------------- Preprocess ----------------
        face_img = Image.fromarray(face_for_model).resize((WIDTH, HEIGHT))
        face_arr = preprocess_input(np.array(face_img))
        processed_faces.append(face_arr)

        # ---------------- Redis streaming ----------------
        try:
            frame_bgr = cv2.cvtColor(face_for_model, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode(".jpg", frame_bgr)
            
            # Get actual frame position from video for accurate timestamp
            current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            timestamp_sec = current_frame_pos / fps if fps > 0 else frame_count / fps

            if success:
                redis_client.publish(
                    f"task_frames:{task_id}",
                    json.dumps({
                        "type": "frame_ready",
                        "frame_index": frame_count,
                        "frame_data": base64.b64encode(buffer).decode("utf-8"),
                        "timestamp": seconds_to_hhmmss(timestamp_sec),
                        "timestamp_seconds": round(timestamp_sec, 3),
                        "fps": round(fps, 2),
                        "video_duration": round(total_duration, 2),
                        "task_id": task_id
                    })
                )
        except Exception as e:
            print("[REDIS ERROR]", e)

        frame_count += 1

    cap.release()

    # ---------------- Final result ----------------
    preview_frames = []
    for i in range(min(3, len(processed_faces))):
        img = (processed_faces[i] * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr)
        preview_frames.append(base64.b64encode(buffer).decode("utf-8"))

    while len(preview_frames) < 3:
        preview_frames.append("")

    result = {
        "message": "Optical flow → face extraction → preprocessing complete",
        "task_id": task_id,
        "video_path": video_path,
        "total_frames": frame_count,
        "preview_frames": preview_frames
    }

    redis_client.set(f"task_result:{task_id}", json.dumps(result))
    
    # Trigger spatial detection as a chained task
    try:
        detection_task = run_chained_detection.delay(task_id, result)
        result["detection_task_id"] = detection_task.id
        result["message"] = "Frame extraction complete. Spatial detection started."
        logger.info(f"Spatial detection task dispatched: {detection_task.id}")
    except Exception as chain_err:
        logger.warning(f"Failed to chain spatial detection task: {chain_err}")
    
    return result

