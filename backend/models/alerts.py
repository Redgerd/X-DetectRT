from sqlalchemy import Column, Integer, ForeignKey, Boolean, String
from sqlalchemy.orm import relationship
from models.base import Base

class Alert(Base):
    __tablename__ = 'alerts'

    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'))
    timestamp = Column(String)  # Store as ISO string
    is_acknowledged = Column(Boolean, default=False)
    file_path = Column(String)

    camera = relationship("Camera")
