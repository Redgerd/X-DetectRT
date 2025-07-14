from typing import List
from core.database import get_db
from models.cameras import Camera
from models.license_detection import License
from models import Users
from sqlalchemy.orm import Session
from api.auth.security import is_admin, get_current_user
from api.auth.schemas import UserResponseSchema
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from core.celery.license_plate_worker import start_all_license_plate_workers, stop_all_license_plate_workers, stop_license_plate_worker, start_license_plate_worker
from api.license_plate.schemas import LicensePlateCreate, LicensePlateResponse
import cv2
import numpy as np
import os
from ultralytics import YOLO

router = APIRouter(prefix="/license-plates", tags=["License Plates"])
# admin only routes
@router.get("/start_all_workers")
async def start_all_license_plate_workers_route(current_user: UserResponseSchema = Depends(is_admin), db: Session = Depends(get_db)):  
    # get all cameras from the database
    cameras = db.query(Camera).all()
    camera_ids = [camera.id for camera in cameras]
    start_all_license_plate_workers.apply_async(queue='license_plate_tasks', args=[camera_ids], priority=10)
    return {"status": "Starting all license plate detection workers..."}


@router.get("/stop_all_workers")
async def stop_all_license_plate_workers_route(current_user: UserResponseSchema = Depends(is_admin)):
    stop_all_license_plate_workers.apply_async(queue='license_plate_tasks', priority=0)
    return {"status": "Stopping all license plate detection workers..."}


@router.get("/start_worker/{camera_id}")
async def start_license_plate_worker_route(camera_id: int, current_user: UserResponseSchema = Depends(is_admin)):
    start_license_plate_worker.apply_async(queue='license_plate_tasks', args=[camera_id], priority=10)
    return {"status": f"Starting license plate detection for Camera {camera_id}..."}


@router.get("/stop_worker/{camera_id}")
async def stop_license_plate_worker_route(camera_id: int, current_user: UserResponseSchema = Depends(is_admin)):
    stop_license_plate_worker.apply_async(queue='license_plate_tasks', args=[camera_id], priority=0)
    return {"status": f"Stopping license plate detection for Camera {camera_id}..."}

# Create a license plate detection record
@router.post("/", response_model=LicensePlateResponse)
def create_license_plate_detection(data: LicensePlateCreate, db: Session = Depends(get_db)):
    # Check if camera exists
    camera = db.query(Camera).filter(Camera.id == data.camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    new_license = License(**data.model_dump())
    db.add(new_license)
    db.commit()
    print("license added in db!!!!!!")
    db.refresh(new_license)
    return new_license

#Get License Plate Image
@router.get("/{license_id}/image")
def get_license_image(
    license_id: int,
    db: Session = Depends(get_db),
    current_user: UserResponseSchema = Depends(get_current_user)
):
    """Fetch the image associated with a license detection."""
    license_record = db.query(License).filter(License.id == license_id).first()
    if not license_record:
        raise HTTPException(status_code=404, detail="License record not found")

    # check if the user has access to the camera
    if not current_user.is_admin:
        user = db.query(Users).filter(Users.id == current_user.id).first()
        if license_record.camera_id not in [camera_id for camera_id in user.cameras]:
            raise HTTPException(status_code=403, detail="Access denied to this camera")

    try:
        return FileResponse(
            path=license_record.file_path,
            media_type="image/jpeg",
            filename=license_record.file_path.split("/")[-1]
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image file not found")

# Get all license plate detections
@router.get("/", response_model=List[LicensePlateResponse])
def get_license_plates(db: Session = Depends(get_db), current_user: UserResponseSchema = Depends(is_admin)):
    return db.query(License).all()

# Get license plate detections by camera ID
@router.get("/camera/{camera_id}", response_model=List[LicensePlateResponse])
def get_license_plates_by_camera(camera_id: int, db: Session = Depends(get_db), current_user: UserResponseSchema = Depends(get_current_user)):
    license_plates = db.query(License).filter(License.camera_id == camera_id).all()
    if not license_plates:
        raise HTTPException(status_code=404, detail="No license plate detections found for this camera")
    return license_plates

# Delete all license plates detected
@router.delete("/delete_all")
def delete_all_license_plates(db: Session = Depends(get_db), current_user: UserResponseSchema = Depends(is_admin)):
    deleted_count = db.query(License).delete()
    db.commit()
    return {"message": f"Deleted {deleted_count} license plate detection(s) successfully"}

# Delete a license plate detection by ID
@router.delete("/{license_id}")
def delete_license_plate(license_id: int, db: Session = Depends(get_db), current_user: UserResponseSchema = Depends(is_admin)):
    license_plate = db.query(License).filter(License.id == license_id).first()
    if not license_plate:
        raise HTTPException(status_code=404, detail="License plate detection not found")

    db.delete(license_plate)
    db.commit()
    return {"message": "License plate detection deleted successfully"} 
