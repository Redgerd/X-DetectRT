import json
from sqlalchemy import Column, Integer, Text, ForeignKey
from models.base import Base

class XAIResult(Base):
    __tablename__ = 'xai_results'

    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, ForeignKey('processed_frames.id'), nullable=False)
    gradcam_b64 = Column(Text, nullable=True)
    ela_b64 = Column(Text, nullable=True)
    fft_data = Column(Text, nullable=True)  # JSON string
    lime_data = Column(Text, nullable=True)  # JSON string
    error = Column(Text, nullable=True)