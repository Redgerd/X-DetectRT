import os
import uuid
from fastapi import WebSocket, APIRouter
from core.celery.frame_selection import extract_faces_with_optical_flow
import asyncio

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
    celery_task = extract_faces_with_optical_flow.delay(file_path)

    # Poll Celery task until done (could also use Redis pub/sub)
    while not celery_task.ready():
        await ws.send_text("Processing...")
        await asyncio.sleep(2)

    result = celery_task.get()  # Returns dict
    print(f"Task result: {result}")
    print(f"Preview frames count: {len(result.get('preview_frames', []))}")
    await ws.send_json(result)
    await ws.close()
