from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey
from models.base import Base

class ProcessedFrame(Base):
    __tablename__ = 'processed_frames'

    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String, ForeignKey('video_analysis_tasks.task_id'), nullable=False)
    frame_index = Column(Integer, nullable=False)
    timestamp = Column(String, nullable=False)
    timestamp_seconds = Column(Float, nullable=False)
    frame_data = Column(Text, nullable=False)
    fps = Column(Float, nullable=True)
    video_duration = Column(Float, nullable=True)