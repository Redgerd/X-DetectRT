# app/auth/user_cameras.py
from sqlalchemy import Table, Column, Integer, ForeignKey
from sqlalchemy.orm import relationship
from models.base import Base

user_cameras = Table(
    "user_cameras",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("camera_id", Integer, ForeignKey("cameras.id"), primary_key=True)
)

