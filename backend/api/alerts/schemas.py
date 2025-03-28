from pydantic import BaseModel
from typing import Optional

class AlertBase(BaseModel):
    camera_id: int
    timestamp: str
    is_acknowledged: bool = False
    file_path: Optional[str] = None

class AlertCreate(BaseModel):
    message: str
    severity: str

class AlertUpdateAcknowledgment(BaseModel):
    is_acknowledged: bool

class AlertResponse(AlertBase):
    id: int
    class Config:
        from_attributes = True  