from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class IntrusionBase(BaseModel):
    camera_id: int
    confidence: float
    bounding_box: Optional[str] = None
    meta_info: Optional[str] = None  # Renamed from 'metadata'

class IntrusionCreate(IntrusionBase):
    pass

class IntrusionResponse(IntrusionBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True
