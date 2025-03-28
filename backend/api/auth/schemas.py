from pydantic import BaseModel, EmailStr
from typing import Optional

class UserLoginSchema(BaseModel):
    """Schema for user login request"""
    username: str
    password: str

class UserResponseSchema(BaseModel):
    """Schema for user response"""
    id: int
    username: str
    email: Optional[EmailStr]
    is_admin: bool

    class Config:
        from_attributes = True

class UserCreateSchema(BaseModel):
    username: str
    email: EmailStr
    password: str

    class Config:
        from_attributes = True

class TokenSchema(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str
