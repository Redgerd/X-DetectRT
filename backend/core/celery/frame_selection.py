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
# Import the deepfake detection task from its new module
# Ensure this import path is correct based on your project structure
# from backend.core.celery.detection_tasks import perform_deepfake_detection
# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TIME_STEPS = 60
HEIGHT = 300
WIDTH = 400
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

# @shared_task(name="frame_selection_pipeline.run")
# def extract_faces_with_optical_flow(video_path, task_id=None):
#     """
#     Extract frames from the video and publish them to Redis for real-time updates.
#     """
#     # Debug: Check if file exists
#     if not os.path.exists(video_path):
#         print(f"[ERROR] Video file not found: {video_path}")
#         result = {
#             "message": f"Video file not found: {video_path}",
#             "video_path": str(video_path),
#             "preview_frames": ["", "", ""]
#         }
#         return result

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"[ERROR] Failed to open video: {video_path}")
#         result = {
#             "message": f"Failed to open video: {video_path}",
#             "video_path": str(video_path),
#             "preview_frames": ["", "", ""]
#         }
#         return result

#     frames = []
#     count = 0
#     max_frames = 60

#     while count < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             print(f"[DEBUG] Frame read failed at count={count}")
#             break
        
#         # Process frame immediately and publish to Redis
#         try:
#             # Convert frame to base64
#             success, buffer = cv2.imencode(".jpg", frame)
#             if success:
#                 frame_base64 = base64.b64encode(buffer).decode("utf-8")
#                 # Publish frame to Redis channel
#                 frame_data = {
#                     "type": "frame_ready",
#                     "frame_index": count,
#                     "frame_data": frame_base64,
#                     "timestamp": count / 30.0,  # Assuming 30fps, adjust as needed
#                     "task_id": task_id
#                 }
#                 redis_client.publish(f"task_frames:{task_id}", json.dumps(frame_data))
#                 print(f"[DEBUG] Published frame {count} to Redis")
#         except Exception as e:
#             print(f"[ERROR] Failed to publish frame {count}: {e}")
        
#         frames.append(frame)
#         count += 1

#     cap.release()

#     # Store all frames in Redis for final result
#     preview_frames = []
#     for frame_bgr in frames:
#         success, buffer = cv2.imencode(".jpg", frame_bgr)
#         if success:
#             preview_frames.append(base64.b64encode(buffer).decode("utf-8"))
#         else:
#             preview_frames.append("")

#     while len(preview_frames) < 3:
#         preview_frames.append("")

#     if not task_id:
#         task_id = os.path.basename(video_path).replace(".mp4", "")
    
#     result = {
#         "message": f"First {count} frames extracted successfully",
#         "video_path": str(video_path),
#         "preview_frames": preview_frames,
#         "total_frames": count
#     }

#     redis_client.set(f"task_result:{task_id}", json.dumps(result))
#     return result

# Test
@shared_task(name="frame_selection_pipeline.run")
def extract_faces_with_optical_flow(video_path, task_id=None, max_frames=60):

    if not task_id:
        task_id = os.path.basename(video_path).replace(".mp4", "")

    if not os.path.exists(video_path):
        return {"error": "Video not found", "task_id": task_id}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Failed to open video", "task_id": task_id}

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
            if success:
                redis_client.publish(
                    f"task_frames:{task_id}",
                    json.dumps({
                        "type": "frame_ready",
                        "frame_index": frame_count,
                        "frame_data": base64.b64encode(buffer).decode("utf-8"),
                        "timestamp": frame_count / 30.0,
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
    return result


class ResizeLongestSide:
    """
    Resizes an image such that its longest side matches a target size,
    preserving the aspect ratio.
    """
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize an image to the target longest side.
        Args:
            image (np.ndarray): HWC image.
        Returns:
            np.ndarray: Resized HWC image.
        """
        oldh, oldw = image.shape[0], image.shape[1]
        scale = self.target_length * 1.0 / max(oldh, oldw)
        newh, neww = int(oldh * scale), int(oldw * scale)
        # Ensure dimensions are divisible by 32, which is common for many CNN architectures
        newh = int(round(newh / 32.0)) * 32
        neww = int(round(neww / 32.0)) * 32

        # After rounding, ensure that the longest side is *exactly* target_length.
        # This prevents slight deviations due to rounding that could affect model input.
        if max(newh, neww) != self.target_length:
            if newh > neww:
                newh = self.target_length
            else: # neww is greater or equal to newh
                neww = self.target_length

        resized_image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)
        return resized_image

    def get_transform_info(self, image: np.ndarray) -> dict:
        """
        Get information about the transformation applied.
        """
        oldh, oldw = image.shape[0], image.shape[1]
        scale = self.target_length * 1.0 / max(oldh, oldw)
        newh, neww = int(oldh * scale), int(oldw * scale)
        newh = int(round(newh / 32.0)) * 32
        neww = int(round(neww / 32.0)) * 32
        if max(newh, neww) != self.target_length:
            if newh > neww:
                newh = self.target_length
            else:
                neww = self.target_length
        return {"original_height": oldh, "original_width": oldw, "resized_height": newh, "resized_width": neww, "scale": scale}


# --- Helper functions for in-memory frame handling ---
def deserialize_frame(frame_bytes: bytes) -> np.ndarray:
    """
    Deserializes frame bytes (e.g., JPEG encoded) back into an OpenCV numpy array.
    """
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image bytes into a valid frame.")
    return frame

def serialize_frame(frame: np.ndarray) -> bytes:
    """
    Serializes an OpenCV numpy array (image) into bytes (e.g., JPEG format).
    """
    # Use JPEG compression for efficiency and compatibility
    # Quality can be adjusted (0-100), higher quality = larger size
    _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return buffer.tobytes()

def open_video_capture(source_path:str, stream_id:str, max_tries:int=10, timeout:int=6):
    """
    Helper function to open video capture, generalized for any video source.
    Retries multiple times in case of temporary issues opening the stream.
    """
    for attempt in range(0, max_tries):
        cap_obj = cv2.VideoCapture(source_path)
        if cap_obj.isOpened():
            logger.info(f"Video Capture object for Stream {stream_id} successfully created (Attempt {attempt+1}).")
            return cap_obj
        else:
            logger.error(f"Attempt {attempt+1} of starting capture for Stream {stream_id} failed from source: {source_path}. Retrying in {timeout} seconds.")
            cap_obj.release() # Release any partial resources
            time.sleep(timeout)
    logger.error(f"Failed to create Capture object for Stream {stream_id} after {max_tries} attempts from source: {source_path}")
    raise Exception(f"Failed to create Capture object for Stream {stream_id}")

# --- Main Task: Frame Selection Pipeline ---

# @celery_app.task(bind=True, name="frame_selection_pipeline.run")
# def run_frame_selection_pipeline(self, stream_id: str, video_source_path: str):
#     """
#     This is the core task for the Frame Selection Pipeline worker.
#     It continuously ingests video frames, applies frame selection heuristics
#     (Optical Flow, Scene Change, Background Subtraction) to reduce frame count,
#     and then if a frame is selected, it resizes it and dispatches it
#     for deepfake analysis.

#     Goal: Reduce frame count by applying intelligent filtering, typically aiming for
#     a significant reduction (e.g., 90%) while retaining semantically important frames.

#     Args:
#         self: The Celery task instance.
#         stream_id (str): A unique identifier for the current video processing session.
#         video_source_path (str): The path or URL to the video file/stream (e.g., local file, RTSP, RTMP).
#     """
#     logger.info(f"[Stream {stream_id}] Starting Frame Selection Pipeline for source: {video_source_path} (Task ID: {self.request.id})")

#     # Initialize Redis client for inter-worker communication (e.g., sending messages back to FastAPI)
#     # Using synchronous redis client as Celery tasks are typically synchronous.
#     redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=False)
#     # Initialize the image resizer for the target model input size (e.g., 1024 for a high-res deepfake model)
#     resize_transform = ResizeLongestSide(1024)

#     cap = None
#     frame_count = 0
#     selected_frame_count = 0 # Track frames actually sent for deepfake analysis

#     try:
#         cap = open_video_capture(video_source_path, stream_id)
#         # Attempt to reduce buffer size for lower latency in live streams if supported by backend
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

#         prev_frame_gray = None # To store the previous grayscale frame for optical flow
#         # Initialize MOG2 background subtractor. `history` and `varThreshold` are important for tuning.
#         # `detectShadows=True` can sometimes help, sometimes adds noise depending on use case.
#         fgbg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
#         prev_hist = None # To store previous histogram for scene change detection

#         while True:
#             ret, original_frame = cap.read() # Read the original frame from the video source
#             if not ret:
#                 logger.info(f"[Stream {stream_id}] End of video stream or failed to read frame from {video_source_path}. Stopping pipeline.")
#                 break # Break loop if no more frames or an error occurred

#             frame_count += 1
#             # Basic frame information to pass along with the frame data
#             frame_info = {
#                 "frame_number": frame_count,
#                 "stream_id": stream_id,
#                 "video_source_path": video_source_path,
#                 "timestamp": datetime.now().isoformat(),
#                 "original_width": original_frame.shape[1],
#                 "original_height": original_frame.shape[0]
#             }

#             current_gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

#             # --- Heuristic 1: Optical Flow for Motion Detection ---
#             # Detects significant pixel movement between frames.
#             motion_detected = False
#             motion_score = 0.0
#             if prev_frame_gray is not None:
#                 # Farneback is a dense optical flow algorithm. Adjust params as needed.
#                 flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, current_gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#                 # Calculate magnitude of flow vectors
#                 magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#                 motion_score = float(np.mean(magnitude)) # Average magnitude of motion
#                 # A threshold to consider motion "significant" (tune based on video content)
#                 motion_detected = motion_score > 1.5
#                 logger.debug(f"[Stream {stream_id}] Frame {frame_count}: Optical Flow - Motion detected={motion_detected}, Score={motion_score:.2f}")
#             else:
#                 logger.debug(f"[Stream {stream_id}] Frame {frame_count}: Optical Flow - No previous frame for calculation.")
#             prev_frame_gray = current_gray_frame.copy() # Update previous frame for next iteration


#             # --- Heuristic 2: Scene Change Detection (Histogram Comparison) ---
#             # Detects abrupt changes in overall scene composition.
#             current_hist = cv2.calcHist([current_gray_frame], [0], None, [256], [0, 256])
#             current_hist = cv2.normalize(current_hist, current_hist).flatten()

#             is_scene_change = False
#             if prev_hist is not None:
#                 # Bhattacharyya distance is common for histogram comparison (lower is more similar)
#                 hist_diff = cv2.compareHist(current_hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
#                 # Threshold for considering it a scene change (tune this)
#                 is_scene_change = hist_diff > 0.4
#                 logger.debug(f"[Stream {stream_id}] Frame {frame_count}: Scene Change - Detected={is_scene_change}, Diff={hist_diff:.2f}")
#             else:
#                 logger.debug(f"[Stream {stream_id}] Frame {frame_count}: Scene Change - No previous histogram for comparison.")
#             prev_hist = current_hist.copy() # Update previous histogram for next iteration


#             # --- Heuristic 3: Background Subtraction for Foreground Activity ---
#             # Identifies pixels belonging to moving objects or foreground, excluding static background.
#             fg_mask = fgbg_subtractor.apply(original_frame, learningRate=0.005) # Adjust learningRate
#             # Calculate the ratio of foreground pixels (non-zero in mask)
#             foreground_activity_ratio = np.sum(fg_mask > 0) / (fg_mask.shape[0] * fg_mask.shape[1])
#             # Threshold for significant foreground activity (tune this)
#             is_significant_foreground = foreground_activity_ratio > 0.05
#             logger.debug(f"[Stream {stream_id}] Frame {frame_count}: Background Subtraction - Significant foreground={is_significant_foreground}, Ratio={foreground_activity_ratio:.4f}")


#             # --- Overall Frame Selection Logic ---
#             # A frame is selected if any of the heuristics indicate it's "interesting".
#             # You can combine these with AND/OR as appropriate for your reduction goal.
#             # For 90% reduction, a strong filter is needed. This 'OR' approach is more inclusive.
#             should_send_for_deepfake = (
#                 motion_detected or
#                 is_scene_change or
#                 is_significant_foreground
#             )

#             # --- Optional: Add a fallback for frames if no heuristics trigger for a long time ---
#             # This ensures that even in static scenes, some frames are analyzed.
#             # Example: Always process every 100th frame even if no changes.
#             if frame_count % 100 == 0:
#                 logger.debug(f"[Stream {stream_id}] Frame {frame_count}: Forcing selection (every 100th frame fallback).")
#                 should_send_for_deepfake = True


#             if should_send_for_deepfake:
#                 selected_frame_count += 1
#                 logger.info(f"[Stream {stream_id}] Frame {frame_count} SELECTED ({selected_frame_count} total selected). Preparing for Deepfake Analysis.")

#                 # --- Apply ResizeLongestSide transformation ---
#                 # Convert to RGB as your ResizeLongestSide class expects RGB
#                 original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
#                 resized_frame_rgb = resize_transform.apply_image(original_frame_rgb)
#                 # Convert back to BGR for serialization with OpenCV's imencode (which often uses BGR)
#                 resized_frame_bgr = cv2.cvtColor(resized_frame_rgb, cv2.COLOR_RGB2BGR)

#                 # Serialize the RESIZED frame data for efficient transport to the next worker
#                 final_frame_bytes = serialize_frame(resized_frame_bgr)

#                 # Update frame_info with resized dimensions for completeness
#                 frame_info["resized_width"] = resized_frame_bgr.shape[1]
#                 frame_info["resized_height"] = resized_frame_bgr.shape[0]

#                 # --- Dispatch to Deepfake Detection Queue ---
#                 # The 'perform_deepfake_detection' task will be picked up by 'celery-detection-worker'
#                 # perform_deepfake_detection.apply_async(
#                 #     args=[
#                 #         base64.b64encode(final_frame_bytes).decode('utf-8'), # Encode bytes to base64 string for JSON
#                 #         stream_id,
#                 #         frame_count,
#                 #         frame_info["original_width"], # Pass original dimensions as context
#                 #         frame_info["original_height"]
#                 #     ],
#                 #     queue='deepfake_detection_queue' # Explicitly route this task
#                 # )
#                 logger.debug(f"[Stream {stream_id}] Dispatched resized frame {frame_count} to deepfake_detection_queue.")

#             #     # --- Send Status Update to Client via WebSocket ---
#             #     current_app.send_task(
#             #         'backend.core.celery.tasks.send_websocket_message',
#             #         args=[
#             #             stream_id,
#             #             {"frame_number": frame_count, "status": "selected_for_analysis",
#             #              "resized_width": resized_frame_bgr.shape[1],
#             #              "resized_height": resized_frame_bgr.shape[0]}
#             #         ],
#             #         queue='websocket_messages_queue' # Route this message via the notification worker
#             #     )
#             #     logger.debug(f"[Stream {stream_id}] Sent 'selected' status for frame {frame_count} to client.")
#             # else:
#             #     logger.debug(f"[Stream {stream_id}] Frame {frame_count} SKIPPED (no significant change detected).")
#             #     # --- Send Status Update to Client for Skipped Frames (Optional, but good for feedback) ---
#             #     current_app.send_task(
#             #         'backend.core.celery.tasks.send_websocket_message',
#             #         args=[
#             #             stream_id,
#             #             {"frame_number": frame_count, "status": "skipped", "reason": "frame_reduction"}
#             #         ],
#             #         queue='websocket_messages_queue'
#             #     )
#             #     logger.debug(f"[Stream {stream_id}] Sent 'skipped' status for frame {frame_count} to client.")

#             # Add a small sleep to prevent the worker from hogging CPU if processing is too fast,
#             # allowing other tasks or the system to breathe. Adjust as needed.
#             time.sleep(0.001)

#     except Exception as e:
#         logger.error(f"[Stream {stream_id}] Critical error in Frame Selection Pipeline: {e}", exc_info=True)
#         # Attempt to retry the task if there's a transient issue (e.g., stream temporarily unavailable)
#         # max_retries should be set carefully, avoiding infinite loops.
#         self.retry(exc=e, countdown=10, max_retries=3) # Retry after 10s, up to 3 times
#     finally:
#         # Ensure video capture object and Redis client are released/closed
#         if cap:
#             cap.release()
#             logger.info(f"[Stream {stream_id}] VideoCapture released.")
#         if redis_client:
#             redis_client.close()
#             logger.info(f"[Stream {stream_id}] Redis client closed.")

#         # Log final statistics for the pipeline run
#         final_reduction_percentage = ((frame_count - selected_frame_count) / frame_count * 100) if frame_count > 0 else 0
#         logger.info(f"[Stream {stream_id}] Frame Selection Pipeline finished for source: {video_source_path}. "
#                     f"Total frames processed: {frame_count}, Total frames selected for deepfake analysis: {selected_frame_count}, "
#                     f"Achieved Reduction: {final_reduction_percentage:.2f}%.")