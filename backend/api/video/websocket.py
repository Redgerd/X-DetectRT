import os
import uuid
from fastapi import WebSocket, APIRouter
from core.celery.frame_selection import extract_faces_with_optical_flow
import asyncio
import json
import redis
from config import settings

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

@router.websocket("/ws/task")
async def websocket_task(ws: WebSocket):
    await ws.accept()

    # Receive task_id
    msg = await ws.receive_text()
    task_id = msg.strip()
    print(f"Subscribed to task: {task_id}")

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

    # Start Celery task
    print(f"Starting frame extraction for task: {task_id}")
    
    try:
        # Start the task
        celery_task = extract_faces_with_optical_flow.delay(file_path, task_id=task_id)
        
        # Send processing status
        await ws.send_text("Processing...")
        
        # Subscribe to Redis channel for real-time frame updates
        try:
            redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            pubsub = redis_client.pubsub()
            pubsub.subscribe(f"task_frames:{task_id}")
            
            # Listen for frames and forward them to the frontend
            frame_count = 0
            while True:
                try:
                    message = pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        try:
                            frame_data = json.loads(message['data'])
                            await ws.send_json(frame_data)
                            frame_count += 1
                            print(f"Forwarded frame {frame_count} to frontend")
                        except Exception as e:
                            print(f"Error forwarding frame: {e}")
                    
                    # Check if Celery task is complete
                    if celery_task.ready():
                        break
                except Exception as e:
                    print(f"Error in Redis message loop: {e}")
                    break
            
            # Unsubscribe from Redis channel
            pubsub.unsubscribe(f"task_frames:{task_id}")
            pubsub.close()
            redis_client.close()
            
        except Exception as e:
            print(f"Error setting up Redis subscription: {e}")
            # Continue without real-time updates
        
        # Get final result
        result = celery_task.get()
        print(f"Task completed: {result}")
        
        # Send completion message
        await ws.send_json({
            "type": "processing_complete",
            "message": "Frame extraction completed",
            "total_frames": result.get("total_frames", 0),
            "task_id": task_id
        })
        
    except Exception as e:
        print(f"Error in frame extraction: {e}")
        await ws.send_json({
            "type": "error",
            "message": f"Frame extraction failed: {str(e)}",
            "task_id": task_id
        })
    
    await ws.close()
