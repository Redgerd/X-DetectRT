from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Allow only these IPs
ALLOWED_IPS = ["*"]  # Add your trusted IPs

class IPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """Allow only specific IPs and block all others."""
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")

        # Prioritize `X-Forwarded-For` if it's present
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        # Log incoming requests
        logger.info(f"🔍 Incoming request from IP: {client_ip}")

        # Block if IP is NOT in the allowed list
        if "*" in ALLOWED_IPS or client_ip in ALLOWED_IPS:
            logger.info(f"✅ Allowed IP: {client_ip}")
            return await call_next(request)

        logger.info(f"✅ Allowed IP: {client_ip}")
        return await call_next(request)
