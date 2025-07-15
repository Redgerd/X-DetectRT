# app/cameras/models.py
from models.base import Base
from config import settings
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean

class Camera(Base):
    __tablename__ = 'cameras'

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, index=True)
    location = Column(String)
    detection_threshold = Column(Integer)
    resize_dims = Column(String, nullable=True, default=settings.FEED_DIMS) # format: "(width, height)"
    crop_region = Column(String, nullable=True) # format: "((x1, y1), (x2, y2))"
    lines = Column(String)
    detect_intrusions = Column(Boolean, default=True) 


    intrusions = relationship("Intrusion", back_populates="camera", cascade="all, delete-orphan")