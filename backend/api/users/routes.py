from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

# Import models and database
from models import Users
from models.users import UserRole
from core.database import SessionLocal

# Import auth dependency
from api.video.routes import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])

class UpdateRoleRequest(BaseModel):
    role: str  # Should be 'user' or 'admin'

@router.get("/count")
async def get_users_count(current_user = Depends(get_current_user)):
    """
    Get total number of users (admin only).
    """
    if not current_user or current_user['role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    db = SessionLocal()
    try:
        count = db.query(Users).count()
        return {"total_users": count}
    finally:
        db.close()

@router.get("/test/count")
async def get_users_count_test():
    """
    Test route: Get total number of users (no auth).
    """
    db = SessionLocal()
    try:
        count = db.query(Users).count()
        return {"total_users": count}
    finally:
        db.close()

@router.get("/test/all")
async def get_all_users_test():
    """
    Test route: Get all user data (no auth).
    """
    db = SessionLocal()
    try:
        users = db.query(Users).all()
        result = []
        for user in users:
            result.append({
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "created_at": user.created_at.isoformat() if user.created_at else None,
            })
        return {"users": result}
    finally:
        db.close()

@router.put("/{user_id}/role")
async def update_user_role(user_id: int, request: UpdateRoleRequest):


    if request.role not in ['user', 'admin']:
        raise HTTPException(status_code=400, detail="Invalid role. Must be 'user' or 'admin'")

    db = SessionLocal()
    try:
        user = db.query(Users).filter(Users.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        user.role = UserRole(request.role)
        db.commit()
        return {"message": f"User {user.username} role updated to {request.role}"}
    finally:
        db.close()