from sqlalchemy import Column, Integer, Boolean, Float, String, ForeignKey
from models.base import Base

class DetectionResult(Base):
    __tablename__ = 'detection_results'

    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, ForeignKey('processed_frames.id'), nullable=False)
    is_anomaly = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    real_prob = Column(Float, nullable=False)
    fake_prob = Column(Float, nullable=False)
    anomaly_type = Column(String, nullable=True)