from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class LicensePlateBase(BaseModel):
    camera_id: int
    license_number: str
    timestamp: datetime = datetime.utcnow()

class LicensePlateCreate(LicensePlateBase):
    pass

class LicensePlateResponse(LicensePlateBase):
    id: int

    class Config:
        from_attributes = True 