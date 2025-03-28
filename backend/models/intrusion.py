from sqlalchemy import Column, Integer, Float, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import Base

class Intrusion(Base):
    __tablename__ = 'intrusions'

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    confidence = Column(Float, nullable=False)
    bounding_box = Column(String, nullable=True)  # JSON string or None
    meta_info = Column(String, nullable=True)  # Rename from 'metadata' to 'meta_info'
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship with the Camera model
    camera = relationship("Camera", back_populates="intrusions")
