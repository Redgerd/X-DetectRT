from fastapi import Request
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from config import settings
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class JWTAuthenticationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """Middleware to enforce JWT authentication except for /auth routes."""
        
        # Allow public access to authentication endpoints and other specific routes
        allowed_paths = ["/auth", "/docs", "/openapi.json", "/"]
        if any(request.url.path.startswith(path) for path in allowed_paths):
            return await call_next(request)
        
        # Allow public access to authentication endpoints
        if request.url.path.startswith("/docs"):
            return await call_next(request)
        
        if request.url.path.startswith("/docs") or request.url.path.startswith("/redoc") or request.url.path.startswith("/openapi.json"):
            return await call_next(request)

        # Check for Authorization header
        authorization: str = request.headers.get("Authorization")
        if not authorization or not authorization.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid authorization header"})

        # Extract & Verify Token
        token = authorization.split(" ")[1]
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            request.state.user = payload  # Store user data in request state
            logger.info(f"Authenticated user: {payload}")
        except JWTError:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})

        return await call_next(request)