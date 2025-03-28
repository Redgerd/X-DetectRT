from jose import jwt
from config import settings
from datetime import datetime, timedelta
from api.auth.schemas import UserResponseSchema
from fastapi import HTTPException, status, Request, Depends

from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def create_access_token(user_id: int, username: str, is_admin: bool):
    expire_minutes = settings.ADMIN_TOKEN_EXPIRE_MINUTES if is_admin else settings.ACCESS_TOKEN_EXPIRE_MINUTES
    expire = datetime.utcnow() + timedelta(minutes=expire_minutes)

    payload = {
        "sub": username,
        "id": user_id,
        "role": "admin" if is_admin else "user",
        "exp": expire
    }

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Extract and validate the token using oauth2_scheme.
    """
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("id")
        role: str = payload.get("role")

        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")

        return UserResponseSchema(id=user_id, username=username, email=None, is_admin=(role == "admin"))

    except jwt.JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    
# def get_current_user(request: Request) -> UserResponseSchema:
#     """
#     Dependency to get the current user from the request.
#     """
#     token = request.headers.get("Authorization")
#     if not token:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

#     token = token.replace("Bearer ", "")  # Remove "Bearer " prefix if present

#     try:
#         payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
#         username: str = payload.get("sub")
#         user_id: int = payload.get("id")
#         role: str = payload.get("role")
#         expiration: datetime = payload.get("exp")

#         if username is None:
#             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
        

#         if expiration < datetime.utcnow().timestamp():
#             raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired")

#         return UserResponseSchema(id=user_id, username=username, email=None, is_admin=(role == "admin"))

#     except jwt.JWTError:
#         raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    

def is_admin(current_user: UserResponseSchema = Depends(get_current_user)):
    """
    Dependency to check if the current user is an admin.
    """
    if current_user.is_admin is False:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this resource. Only admins can access.",
        )
    return current_user