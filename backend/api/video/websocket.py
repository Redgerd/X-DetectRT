import os
import uuid
import asyncio
import io
import logging
import cv2
import numpy as np
import base64
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

# Configure logger
logger = logging.getLogger(__name__)

UPLOAD_DIR = "/app/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter()

# Small delay between sending frames to prevent batching
FRAME_SEND_DELAY = 0.05  # 50ms between frames

async def wait_for_detection_result(
    ws: WebSocket,
    task_id: str,
    frame_index: int = None,
    detection_task: Any = None,
    timeout: int = 60
) -> Optional[dict]:
    """
    Wait for detection result from Redis and forward to frontend
    
    Args:
        ws: WebSocket connection
        task_id: Task ID to listen for
        frame_index: Optional frame index to match specific frame
        detection_task: Optional Celery task to check for completion
        timeout: Timeout in seconds
    
    Returns:
        Detection result dict or None if failed
    """
    redis_client = None
    detection_result = None
    
    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = redis_client.pubsub()
        pubsub.subscribe(f"task_detection:{task_id}")
        
        detection_received = False
        timeout_counter = 0
        
        while not detection_received and timeout_counter < timeout:
            try:
                # Try to get message from Redis
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    try:
                        result_data = json.loads(message['data'])
                        if result_data.get('type') == 'detection_ready':
                            # Check if this is for the specific frame we're waiting for
                            if frame_index is not None:
                                result_frame_index = result_data.get('frame_index')
                                if result_frame_index != frame_index:
                                    # Not the frame we're waiting for, continue
                                    continue
                            
                            # Forward detection result to frontend
                            await ws.send_json(result_data)
                            detection_received = True
                            detection_result = result_data
                            logger.info(f"Detection result received for task: {task_id}, frame: {frame_index}")
                            break
                    except Exception as e:
                        logger.error(f"Error parsing detection message: {e}")
                
                # Check if Celery task is complete (if provided)
                if detection_task and detection_task.ready():
                    detection_received = True
                    break
                    
                timeout_counter += 1
                
            except Exception as e:
                logger.error(f"Error in Redis message loop: {e}")
                break
        
        # Unsubscribe and close
        pubsub.unsubscribe(f"task_detection:{task_id}")
        pubsub.close()
        
        # If no result received via pubsub, try to get from Redis directly
        if not detection_received:
            detection_result_json = redis_client.get(f"detection_result:{task_id}")
            if detection_result_json:
                try:
                    detection_result = json.loads(detection_result_json)
                    await ws.send_json(detection_result)
                    detection_received = True
                    logger.info(f"Retrieved detection result from Redis for task: {task_id}")
                except Exception as e:
                    logger.error(f"Error parsing detection result: {e}")
        
        return detection_result
        
    except Exception as e:
        logger.error(f"Error in detection result subscription: {e}")
        return None
    finally:
        if redis_client:
            redis_client.close()


async def receive_file_data(
    ws: WebSocket,
    task_id: str,
    file_extension: str,
    send_signal: str
) -> Optional[str]:
    """
    Receive file data from WebSocket and save to disk
    
    Args:
        ws: WebSocket connection
        task_id: Task ID for filename
        file_extension: File extension (e.g., 'jpg', 'mp4')
        send_signal: Signal to send to client before receiving data
    
    Returns:
        File path if successful, None if failed
    """
    try:
        # Send signal to client
        await ws.send_text(send_signal)
        
        # Save incoming file
        file_path = f"{UPLOAD_DIR}/{task_id}.{file_extension}"
        
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
    """Safely send JSON with disconnect handling"""
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
    """Safely send text with disconnect handling"""
    try:
        await ws.send_text(text)
        return True
    except (WebSocketDisconnect, RuntimeError):
        logger.info("WebSocket disconnected while sending text")
        return False
    except Exception as e:
        logger.error(f"Error sending text: {e}")
        return False


async def send_frame_with_detection(
    ws: WebSocket,
    task_id: str,
    frame_data: str,
    frame_index: int,
    timestamp: str,
    timestamp_seconds: float,
    fps: float,
    video_duration: float,
    is_image: bool = False,
    wait_for_detection: bool = True
) -> Optional[dict]:
    """
    Send frame and wait for its detection result
    
    Args:
        ws: WebSocket connection
        task_id: Task ID
        frame_data: Base64 encoded frame data
        frame_index: Frame index
        timestamp: Timestamp string
        timestamp_seconds: Timestamp in seconds
        fps: Frames per second
        video_duration: Total video duration
        is_image: Whether this is an image
        wait_for_detection: Whether to wait for detection result
    
    Returns:
        Detection result dict if wait_for_detection is True, None otherwise
    """
    # Send frame to frontend
    frame_message = {
        "type": "frame_ready",
        "frame_index": frame_index,
        "frame_data": frame_data,
        "timestamp": timestamp,
        "timestamp_seconds": timestamp_seconds,
        "fps": fps,
        "video_duration": video_duration,
        "task_id": task_id,
        "is_image": is_image
    }
    
    if not await safe_send_json(ws, frame_message):
        return None
    
    logger.debug(f"Sent frame {frame_index} to frontend")
    
    # Wait for detection result if requested
    if wait_for_detection:
        detection_result = await wait_for_detection_result(
            ws=ws,
            task_id=task_id,
            frame_index=frame_index,
            timeout=30
        )
        
        # Send combined frame with detection result
        if detection_result:
            combined_message = {
                "type": "frame_with_detection",
                "frame_index": frame_index,
                "frame_data": frame_data,
                "detection": detection_result,
                "timestamp": timestamp,
                "timestamp_seconds": timestamp_seconds,
                "task_id": task_id
            }
            await safe_send_json(ws, combined_message)
        
        return detection_result
    
    return None


@router.websocket("/ws/task")
async def websocket_task(ws: WebSocket):
    await ws.accept()

    try:
        # Receive task_id and video_duration
        msg = await ws.receive_text()
        try:
            # Try parsing as JSON (new format with video_duration and file_type)
            data = json.loads(msg)
            task_id = data.get("task_id", "").strip()
            video_duration = data.get("video_duration", None)
            file_type = data.get("file_type", "video")  # 'video' or 'image'
        except JSONDecodeError:
            # Fall back to old format (just task_id)
            task_id = msg.strip()
            video_duration = None
            file_type = "video"
        
        logger.info(f"Subscribed to task: {task_id}, file_type: {file_type}, video_duration: {video_duration}")

        # Determine if this is an image or video upload
        is_image = file_type == "image"
        
        if is_image:
            # Handle image upload
            file_path = await receive_file_data(ws, task_id, "jpg", "SEND_IMAGE")
            if not file_path:
                return
            
            try:
                # Send processing status
                if not await safe_send_text(ws, "Processing image..."):
                    return
                
                # Load and process the image directly
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError(f"Could not load image from {file_path}")
                
                # Encode the image to base64
                success, buffer = cv2.imencode(".jpg", img)
                if not success:
                    raise ValueError("Failed to encode image")
                
                frame_data_b64 = base64.b64encode(buffer).decode("utf-8")
                
                # Send frame with detection result
                detection_result = await send_frame_with_detection(
                    ws=ws,
                    task_id=task_id,
                    frame_data=frame_data_b64,
                    frame_index=0,
                    timestamp="00:00:00.000",
                    timestamp_seconds=0.0,
                    fps=1.0,
                    video_duration=0.0,
                    is_image=True,
                    wait_for_detection=True
                )
                
                if detection_result is None:
                    # If waiting for detection failed, start detection task separately
                    detection_task = run_gend_inference.delay(
                        task_id=task_id,
                        frame_data=frame_data_b64,
                        frame_index=0,
                        timestamp="00:00:00.000",
                    )
                    
                    logger.info(f"Sent image to detection worker for task: {task_id}")
                    
                    # Wait for detection result using shared function
                    detection_result = await wait_for_detection_result(
                        ws=ws,
                        task_id=task_id,
                        detection_task=detection_task,
                        timeout=60
                    )
                
                # Send completion message
                await safe_send_json(ws, {
                    "type": "processing_complete",
                    "message": "Image processing completed",
                    "total_frames": 1,
                    "anomaly_count": detection_result.get("anomaly_count", 0) if detection_result else 0,
                    "task_id": task_id,
                    "is_image": True,
                    "detection_result": detection_result
                })
                
            except Exception as e:
                logger.error(f"Error in image processing: {e}", exc_info=True)
                await safe_send_json(ws, {
                    "type": "error",
                    "message": f"Image processing failed: {str(e)}",
                    "task_id": task_id
                })
        
        else:
            # Handle video upload
            file_path = await receive_file_data(ws, task_id, "mp4", "SEND_VIDEO")
            if not file_path:
                return
            
            logger.info(f"Starting frame extraction for task: {task_id}")
            
            try:
                # Start the Celery task
                celery_task = extract_faces_with_optical_flow.delay(
                    file_path, 
                    task_id=task_id,
                    video_duration=video_duration
                )
                
                # Send processing status
                if not await safe_send_text(ws, "Processing..."):
                    return
                
                # Subscribe to Redis channels
                redis_client = None
                processed_frames = {}  # Store frames and their detection results
                
                try:
                    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                    pubsub = redis_client.pubsub()
                    pubsub.subscribe(f"task_frames:{task_id}")
                    pubsub.subscribe(f"task_detection:{task_id}")
                    # Subscribe to XAI results for Grad-CAM
                    pubsub.subscribe(f"task_xai:{task_id}")
                    
                    # Listen for frames and detection results
                    frame_count = 0
                    detection_count = 0
                    pending_detections = {}  # Store frames waiting for detection results
                    
                    while True:
                        try:
                            message = pubsub.get_message(timeout=1.0)
                            if message and message['type'] == 'message':
                                try:
                                    data = json.loads(message['data'])
                                    data_type = data.get('type', '')
                                    
                                    if data_type == 'frame_ready':
                                        # Frame from frame_selection task
                                        frame_index = data.get('frame_index', 0)
                                        frame_data = data.get('frame_data')
                                        timestamp = data.get('timestamp', '')
                                        timestamp_seconds = data.get('timestamp_seconds', 0.0)
                                        fps = data.get('fps', 30.0)
                                        
                                        # Send frame and wait for detection result
                                        detection_result = await send_frame_with_detection(
                                            ws=ws,
                                            task_id=task_id,
                                            frame_data=frame_data,
                                            frame_index=frame_index,
                                            timestamp=timestamp,
                                            timestamp_seconds=timestamp_seconds,
                                            fps=fps,
                                            video_duration=video_duration or 0.0,
                                            is_image=False,
                                            wait_for_detection=True
                                        )
                                        
                                        if detection_result:
                                            processed_frames[frame_index] = {
                                                'frame': data,
                                                'detection': detection_result
                                            }
                                            detection_count += 1
                                        
                                        frame_count += 1
                                        logger.debug(f"Processed frame {frame_count} with detection")
                                        await asyncio.sleep(FRAME_SEND_DELAY)
                                        
                                    elif data_type == 'detection_ready':
                                        # Detection result from spatial detection task
                                        frame_index = data.get('frame_index', 0)
                                        data['is_processed'] = True
                                        
                                        # If we have the frame stored, combine and send
                                        if frame_index in processed_frames:
                                            combined_message = {
                                                "type": "frame_with_detection",
                                                "frame_index": frame_index,
                                                "frame_data": processed_frames[frame_index]['frame'].get('frame_data'),
                                                "detection": data,
                                                "timestamp": processed_frames[frame_index]['frame'].get('timestamp', ''),
                                                "timestamp_seconds": processed_frames[frame_index]['frame'].get('timestamp_seconds', 0.0),
                                                "task_id": task_id
                                            }
                                            await safe_send_json(ws, combined_message)
                                        else:
                                            # Just send detection result
                                            await safe_send_json(ws, data)
                                        
                                        detection_count += 1
                                        logger.debug(f"Forwarded detection result {detection_count} for frame {frame_index}")
                                        await asyncio.sleep(FRAME_SEND_DELAY)
                                        
                                    elif data_type == 'xai_ready':
                                        # XAI/Grad-CAM result from explainable_ai task
                                        # Forward directly to frontend
                                        await safe_send_json(ws, data)
                                        logger.debug(f"Forwarded XAI result for frame {data.get('frame_index')}")
                                        await asyncio.sleep(FRAME_SEND_DELAY)
                                        
                                    else:
                                        # Forward other message types
                                        await safe_send_json(ws, data)
                                        
                                except Exception as e:
                                    logger.error(f"Error forwarding message: {e}", exc_info=True)
                            
                            # Check if processing is complete
                            frame_task_ready = celery_task.ready()
                            detection_result = redis_client.get(f"detection_result:{task_id}")
                            detection_complete = detection_result is not None
                            
                            if frame_task_ready and detection_complete:
                                break
                                
                        except Exception as e:
                            logger.error(f"Error in Redis message loop: {e}", exc_info=True)
                            break
                    
                    # Unsubscribe from Redis channels
                    pubsub.unsubscribe(f"task_frames:{task_id}")
                    pubsub.unsubscribe(f"task_detection:{task_id}")
                    pubsub.unsubscribe(f"task_xai:{task_id}")
                    pubsub.close()
                    
                except Exception as e:
                    logger.error(f"Error setting up Redis subscription: {e}", exc_info=True)
                
                # Get final result
                result = celery_task.get()
                logger.info(f"Task completed: {result}")
                
                # Get detection result
                detection_result = None
                if redis_client:
                    detection_result_json = redis_client.get(f"detection_result:{task_id}")
                    if detection_result_json:
                        try:
                            detection_result = json.loads(detection_result_json)
                            result['detection'] = detection_result
                        except Exception as e:
                            logger.error(f"Error parsing detection result: {e}", exc_info=True)
                    
                    redis_client.close()
                
                # Send completion message
                await safe_send_json(ws, {
                    "type": "processing_complete",
                    "message": "Frame extraction and spatial detection completed",
                    "total_frames": result.get("total_frames", 0),
                    "processed_frames": len(processed_frames),
                    "anomaly_count": detection_result.get("anomaly_count", 0) if detection_result else 0,
                    "task_id": task_id,
                    "frames_with_detection": processed_frames
                })
                
            except Exception as e:
                logger.error(f"Error in frame extraction: {e}", exc_info=True)
                await safe_send_json(ws, {
                    "type": "error",
                    "message": f"Frame extraction failed: {str(e)}",
                    "task_id": task_id
                })
        
        try:
            await ws.close()
        except:
            pass
            
    except (WebSocketDisconnect, RuntimeError) as e:
        logger.info(f"WebSocket disconnected during task handling: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_task: {e}", exc_info=True)