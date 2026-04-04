import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, Enum, ForeignKey
from models.base import Base

class TaskStatus(enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"

class VideoAnalysisTask(Base):
    __tablename__ = 'video_analysis_tasks'

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    task_id = Column(String, unique=True, index=True, nullable=False)
    video_path = Column(String, nullable=False)
    status = Column(Enum(TaskStatus), default=TaskStatus.pending, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    faces_detected_frames = Column(Integer, default=0)
    frames_skipped = Column(Integer, default=0)