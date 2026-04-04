from sqlalchemy import Column, Integer, Text, ForeignKey
from models.base import Base

class XAIResult(Base):
    __tablename__ = 'xai_results'

    id = Column(Integer, primary_key=True, index=True)
    frame_id = Column(Integer, ForeignKey('processed_frames.id'), nullable=False)
    gradcam_b64 = Column(Text, nullable=True)
    lime_b64 = Column(Text, nullable=True)
    error = Column(Text, nullable=True)