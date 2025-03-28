from pydantic import BaseModel, EmailStr
from typing import Optional, List
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")    

class UserBase(BaseModel):
    id: int  # Add ID field here
    username: str
    email: EmailStr
    is_admin: Optional[bool] = False
    ip_address: Optional[str] = None
    cameras: List[int] = []

    class Config:
        from_attributes = True

    @classmethod
    def from_orm(cls, obj):
        """Convert ORM model to schema, extracting camera IDs"""
        return cls(
            id=obj.id,  # Ensure ID is included in the response
            username=obj.username,
            email=obj.email,
            is_admin=obj.is_admin,
            ip_address=obj.ip_address,
            cameras=[camera.id for camera in obj.cameras] if hasattr(obj, "cameras") else []
        )


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    is_admin: Optional[bool] = False
    ip_address: Optional[str] = None
    cameras: List[int] = []

    def hash_password(self):
        """Hash the password before storing"""
        self.password = pwd_context.hash(self.password)


class UserUpdate(BaseModel):  # No need to inherit from UserBase to keep fields optional
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_admin: Optional[bool] = None
    ip_address: Optional[str] = None
    cameras: Optional[List[int]] = None  # Make cameras optional

    class Config:
        from_attributes = True
