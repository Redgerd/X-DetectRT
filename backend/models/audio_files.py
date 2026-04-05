from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Sequence
from models.base import Base

class AudioFile(Base):
    __tablename__ = 'audio_files'

    id = Column(Integer, Sequence('audio_files_seq'), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    filename = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False) 
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    file_path = Column(String, nullable=True)  