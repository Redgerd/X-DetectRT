from pydantic import BaseModel
from typing import List, Optional

class CameraBase(BaseModel):
    url: str
    location: Optional[str] = None
    detection_threshold: int
    resize_dims: Optional[str] = None
    crop_region: Optional[str] = None
    lines: Optional[str] = None
    detect_intrusions: Optional[bool] = True

class CameraCreate(CameraBase):
    id: Optional[int] = None  # Allow specifying the ID manually

class CameraUpdate(CameraBase):
    url: Optional[str] = None
    location: Optional[str] = None
    detection_threshold: Optional[int] = None
    resize_dims: Optional[str] = None
    crop_region: Optional[str] = None
    lines: Optional[str] = None
    detect_intrusions: Optional[bool] = None

class Camera(CameraBase):
    id: int

    class Config:
        from_attributes = True

class CameraListResponse(BaseModel):
    cameras: List[Camera]
