# another_module.py
import base64
import json
from celery import shared_task
import redis
from config import settings 
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

def seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

@shared_task(name="frame_processing_pipeline.run")
def process_faces_task(frames_b64, task_id, video_duration=None):
    """
    Receives base64 frames, optionally does further processing,
    and returns frames with timestamps for frontend.
    """

    processed_frames = []

    total_frames = len(frames_b64)
    fps = total_frames / video_duration if video_duration and video_duration > 0 else 30.0

    for idx, frame_b64 in enumerate(frames_b64):
        # ---------------- Timestamp calculation ----------------
        timestamp_sec = idx / fps
        timestamp_hms = seconds_to_hhmmss(timestamp_sec)

        processed_frames.append({
            "frame_index": idx,
            "frame_data": frame_b64,
            "timestamp": timestamp_hms,
            "timestamp_seconds": round(timestamp_sec, 3),
        })

        # Optional: publish to Redis for live frontend updates
        try:
            redis_client.publish(
                f"task_frames_processed:{task_id}",
                json.dumps({
                    "type": "frame_processed",
                    "frame_index": idx,
                    "frame_data": frame_b64,
                    "timestamp": timestamp_hms,
                    "timestamp_seconds": round(timestamp_sec, 3),
                    "task_id": task_id
                })
            )
        except Exception as e:
            print("[REDIS ERROR]", e)

    # ---------------- Final result ----------------
    result = {
        "message": "Frames processed successfully",
        "task_id": task_id,
        "total_frames": total_frames,
        "video_duration": round(video_duration, 2) if video_duration else None,
        "frames": processed_frames
    }

    # Store final result in Redis for retrieval
    redis_client.set(f"task_result_processed:{task_id}", json.dumps(result))

    return result
