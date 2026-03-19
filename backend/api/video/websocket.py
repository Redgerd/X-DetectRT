import os
import uuid
import asyncio
from fastapi import WebSocket, APIRouter
from core.celery.frame_selection import extract_faces_with_optical_flow
import json
from json import JSONDecodeError
import redis
from config import settings

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

FRAME_SEND_DELAY = 0.05  # 50ms between frames

@router.websocket("/ws/task")
async def websocket_task(ws: WebSocket):
    await ws.accept()

    # Receive task_id and video_duration
    msg = await ws.receive_text()
    try:
        data = json.loads(msg)
        task_id = data.get("task_id", "").strip()
        video_duration = data.get("video_duration", None)
    except JSONDecodeError:
        task_id = msg.strip()
        video_duration = None

    print(f"Subscribed to task: {task_id}, video_duration: {video_duration}")

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

    print(f"Starting frame extraction for task: {task_id}")

    try:
        celery_task = extract_faces_with_optical_flow.delay(
            file_path,
            task_id=task_id,
            video_duration=video_duration
        )

        await ws.send_text("Processing...")

        try:
            redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
            pubsub = redis_client.pubsub()

            # FIX: Subscribe to all three channels
            pubsub.subscribe(f"task_frames:{task_id}")
            pubsub.subscribe(f"task_detection:{task_id}")
            pubsub.subscribe(f"task_xai:{task_id}")

            frame_count = 0
            detection_count = 0
            xai_count = 0

            while True:
                try:
                    message = pubsub.get_message(timeout=1.0)
                    if message and message['type'] == 'message':
                        try:
                            data = json.loads(message['data'])
                            data_type = data.get('type', '')

                            if data_type == 'frame_ready':
                                await ws.send_json(data)
                                frame_count += 1
                                print(f"Forwarded frame {frame_count} to frontend")
                                await asyncio.sleep(FRAME_SEND_DELAY)

                            elif data_type == 'detection_ready':
                                data['is_processed'] = True
                                await ws.send_json(data)
                                detection_count += 1
                                print(f"Forwarded detection result {detection_count} to frontend")
                                await asyncio.sleep(FRAME_SEND_DELAY)

                            # FIX: Forward XAI results to frontend
                            elif data_type == 'xai_ready':
                                await ws.send_json(data)
                                xai_count += 1
                                print(f"Forwarded XAI result {xai_count} to frontend")
                                await asyncio.sleep(FRAME_SEND_DELAY)

                            else:
                                await ws.send_json(data)

                        except Exception as e:
                            print(f"Error forwarding message: {e}")

                    # FIX: Wait for all three stages to complete
                    frame_task_ready = celery_task.ready()
                    detection_result = redis_client.get(f"detection_result:{task_id}")
                    xai_result = redis_client.get(f"xai_result:{task_id}")

                    detection_complete = detection_result is not None
                    xai_complete = xai_result is not None

                    if frame_task_ready and detection_complete and xai_complete:
                        print(f"All stages complete for task: {task_id}")
                        break

                except Exception as e:
                    print(f"Error in Redis message loop: {e}")
                    break

            pubsub.unsubscribe(f"task_frames:{task_id}")
            pubsub.unsubscribe(f"task_detection:{task_id}")
            pubsub.unsubscribe(f"task_xai:{task_id}")
            pubsub.close()

        except Exception as e:
            print(f"Error setting up Redis subscription: {e}")

        # Get final results
        result = celery_task.get()
        print(f"Frame task completed: {result}")

        detection_result_json = redis_client.get(f"detection_result:{task_id}")
        if detection_result_json:
            try:
                result['detection'] = json.loads(detection_result_json)
            except:
                pass

        xai_result_json = redis_client.get(f"xai_result:{task_id}")
        if xai_result_json:
            try:
                result['xai'] = json.loads(xai_result_json)
            except:
                pass

        await ws.send_json({
            "type": "processing_complete",
            "message": "Detection and XAI analysis completed",
            "total_frames": result.get("total_frames", 0),
            "anomaly_count": result.get("detection", {}).get("anomaly_count", 0),
            "xai_frames_explained": result.get("xai", {}).get("total_frames_explained", 0),
            "task_id": task_id
        })

        redis_client.close()

    except Exception as e:
        print(f"Error in pipeline: {e}")
        await ws.send_json({
            "type": "error",
            "message": f"Pipeline failed: {str(e)}",
            "task_id": task_id
        })

    await ws.close()