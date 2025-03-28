from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import redis
import asyncio
from collections import defaultdict
import json

# Redis client setup
redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)

# WebSocket router
router = APIRouter()

# Dictionary to manage WebSocket connections per camera
camera_connections = defaultdict(set)

# Set to manage WebSocket connections for all cameras
all_connections = set()

# Working 
async def redis_listener():
    """Listens for messages from Redis and broadcasts them to WebSockets."""
    pubsub = redis_client.pubsub()
    pubsub.psubscribe("camera_alerts:*")

    while True:
            message = pubsub.get_message(ignore_subscribe_messages=True)
            if message:
                print(f"📨 Redis Message Received: {message}")  # Debug log
                channel = message["channel"]
                alert_data = message["data"]
                camera_id = channel.split(":")[-1]

                print(f"Received alert for camera {camera_id}: {alert_data}")
                await broadcast_alert(camera_id, alert_data)

            await asyncio.sleep(0.1)  # Prevents busy-waiting
            
async def broadcast_alert(camera_id: str, alert_data: str):
    """Sends alert messages to all WebSocket clients subscribed to a specific camera and all cameras."""
    to_remove = set()
    message = json.dumps({"camera_id": camera_id, "alert": alert_data})

     # Broadcast to specific camera connections
    if camera_id in camera_connections:
        for connection in camera_connections[camera_id]:
            try:
                await connection.send_text(message)
            except Exception as e:
                to_remove.add(connection)

    # Broadcast to clients subscribed to all cameras
    for connection in all_connections:
        try:
            await connection.send_text(message)
        except Exception as e:
            to_remove.add(connection)

    # Remove disconnected clients
    for conn in to_remove:
        for camera in camera_connections.keys():
            camera_connections[camera].discard(conn)
        if not camera_connections[camera]:
            del camera_connections[camera]

        all_connections.discard(conn)      

@router.websocket("/ws/alerts/{camera_id}")
async def websocket_camera(websocket: WebSocket, camera_id: str):
    """Handles WebSocket connections for specific cameras."""
    await websocket.accept()
    camera_connections[camera_id].add(websocket)
    print(f"Client connected to camera {camera_id}")

    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        print(f"Client disconnected from camera {camera_id}")
    finally:
        camera_connections[camera_id].discard(websocket)
        if not camera_connections[camera_id]:
            del camera_connections[camera_id]

@router.websocket("/ws/alerts")
async def websocket_all_cameras(websocket: WebSocket):
    """Handles WebSocket connections for all cameras."""
    await websocket.accept()
    all_connections.add(websocket)
    print("Client connected to all camera alerts")

    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        print("Client disconnected from all cameras")
    finally:
        all_connections.discard(websocket)

# Function to start Redis listener on startup
async def start_redis_listener():
    asyncio.create_task(redis_listener())