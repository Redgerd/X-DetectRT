from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base

class License(Base):
    __tablename__ = 'license_detection'

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    license_number = Column(String, nullable=True)  # JSON string or None
    timestamp = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String)

    # Relationship with the Camera model
    camera = relationship("Camera", back_populates="license_detection")
