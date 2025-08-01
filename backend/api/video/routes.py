from fastapi import APIRouter, Depends, Cookie
from fastapi.responses import JSONResponse
import uuid
from typing import Optional

router = APIRouter(prefix="/video",tags=["Video Processing"])

# Dummy user auth dependency using JWT in cookies
async def get_current_user(token: Optional[str] = Cookie(None)):
    # Normally verify JWT here
    if not token:
        return None
    return {"user_id": "123", "name": "Haris"}

@router.post("/start-task")
async def start_task():
    task_id = str(uuid.uuid4())
    return JSONResponse({"task_id": task_id})
