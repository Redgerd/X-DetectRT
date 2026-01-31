import os
import uuid
from fastapi import WebSocket, APIRouter
from celery import chain
from core.celery.process_face import process_faces_task
from core.celery.frame_selection import extract_faces_with_optical_flow
import asyncio
import json
from json import JSONDecodeError
import redis
from config import settings
UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

@router.websocket("/ws/task")
async def websocket_task(ws: WebSocket):
    await ws.accept()

    # Receive task_id and video_duration
    msg = await ws.receive_text()
    try:
        data = json.loads(msg)
        task_id = data.get("task_id", "").strip()
        video_duration = data.get("video_duration", None)
    except json.JSONDecodeError:
        task_id = msg.strip()
        video_duration = None
    
    print(f"Subscribed to task: {task_id}, video_duration: {video_duration}")

    # Ask frontend to start sending video
    await ws.send_text("SEND_VIDEO")

    # Save incoming video
    file_path = f"{UPLOAD_DIR}/{task_id}.mp4"
    with open(file_path, "wb") as f:
        while True:
            data = await ws.receive()
            if "bytes" in data and data["bytes"]:
                f.write(data["bytes"])
            elif "text" in data and data["text"] == "END":
                break

    print(f"Video received for task: {task_id}")

    try:
        # ---------------- Create Celery chain ----------------
        task_chain = chain(
            extract_faces_with_optical_flow.s(
                file_path,
                task_id=task_id,
                video_duration=video_duration
            ),
            process_faces_task.s(video_duration=video_duration)
        )

        celery_result = task_chain.apply_async()
        await ws.send_text("Processing started...")

        # ---------------- Setup Redis subscription ----------------
        try:
            redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            pubsub = redis_client.pubsub()
            pubsub.subscribe(f"task_frames:{task_id}")
            pubsub.subscribe(f"task_frames_processed:{task_id}")

            while True:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        frame_data = json.loads(message['data'])
                        await ws.send_json(frame_data)
                    except Exception as e:
                        print(f"Error forwarding frame: {e}")

                if celery_result.ready():
                    break

            pubsub.unsubscribe(f"task_frames:{task_id}")
            pubsub.unsubscribe(f"task_frames_processed:{task_id}")
            pubsub.close()
            redis_client.close()
        
        except Exception as e:
            print(f"Redis subscription error: {e}")
        
        # ---------------- Final result ----------------
        result = celery_result.get()
        await ws.send_json({
            "type": "processing_complete",
            "message": "Frame extraction and processing completed",
            "total_frames": result.get("total_frames", 0),
            "task_id": task_id,
            "frames": result.get("frames", [])
        })

    except Exception as e:
        print(f"Error in task chain: {e}")
        await ws.send_json({
            "type": "error",
            "message": f"Processing failed: {str(e)}",
            "task_id": task_id
        })

    await ws.close()