from typing import List
from core.database import get_db
from models.cameras import Camera
from sqlalchemy.orm import Session
from api.auth.security import is_admin, get_current_user
from models.intrusion import Intrusion
from api.auth.schemas import UserResponseSchema
from fastapi import APIRouter, Depends, HTTPException
from api.intrusion.schemas import IntrusionCreate, IntrusionResponse
from core.celery.feed_worker import start_all_feed_workers, stop_all_feed_workers, stop_feed_worker, start_feed_worker

router = APIRouter(prefix="/intrusions", tags=["Intrusions"])

# admin only routes
@router.get("/start_all_feed_workers")
async def start_all_feed_workers_route(current_user: UserResponseSchema = Depends(is_admin)):  
    start_all_feed_workers.apply_async(queue='feed_tasks', priority=10)
    return {"status": "Starting all feed workers..."}


@router.get("/stop_all_feed_workers")
async def stop_all_feed_workers_route(current_user: UserResponseSchema = Depends(is_admin)):
    stop_all_feed_workers.apply_async(queue='feed_tasks', priority=0)
    return {"status": "Stopping all feed workers..."}


@router.get("/start_feed_worker/{worker_id}")
async def start_feed_worker_route(worker_id: int, current_user: UserResponseSchema = Depends(is_admin)):
    start_feed_worker.apply_async(queue='feed_tasks', args=[worker_id], priority=10)
    return {"status": f"Starting feed worker {worker_id}..."}


@router.get("/stop_feed_worker/{worker_id}")
async def stop_feed_worker_route(worker_id: int, current_user: UserResponseSchema = Depends(is_admin)):
    stop_feed_worker.apply_async(queue='feed_tasks', args=[worker_id], priority=0)
    return {"status": f"Stopping feed worker {worker_id}..."}


# any user routes

# Create an intrusion
@router.post("/", response_model=IntrusionResponse)
def create_intrusion(data: IntrusionCreate, db: Session = Depends(get_db)):
    # Check if camera exists
    camera = db.query(Camera).filter(Camera.id == data.camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    new_intrusion = Intrusion(**data.model_dump())
    db.add(new_intrusion)
    db.commit()
    db.refresh(new_intrusion)
    return new_intrusion

# Get all intrusions
@router.get("/", response_model=List[IntrusionResponse])
def get_intrusions(db: Session = Depends(get_db), current_user: UserResponseSchema = Depends(is_admin)):
    return db.query(Intrusion).all()

# Get intrusions by camera ID
@router.get("/camera/{camera_id}", response_model=List[IntrusionResponse])
def get_intrusions_by_camera(camera_id: int, db: Session = Depends(get_db), current_user: UserResponseSchema = Depends(get_current_user)):
    intrusions = db.query(Intrusion).filter(Intrusion.camera_id == camera_id).all()
    if not intrusions:
        raise HTTPException(status_code=404, detail="No intrusions found for this camera")
    return intrusions

# Delete an intrusion by ID
@router.delete("/{intrusion_id}")
def delete_intrusion(intrusion_id: int, db: Session = Depends(get_db), current_user: UserResponseSchema = Depends(is_admin)):
    intrusion = db.query(Intrusion).filter(Intrusion.id == intrusion_id).first()
    if not intrusion:
        raise HTTPException(status_code=404, detail="Intrusion not found")

    db.delete(intrusion)
    db.commit()
    return {"message": "Intrusion deleted successfully"}