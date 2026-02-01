import os
from dotenv import load_dotenv

load_dotenv(override=True)

class Settings:
    PROJECT_NAME = "X-Detect-RT"

    # Database settings
    DATABASE_NAME = os.getenv("DATABASE_NAME", "postgres-db")
    DATABASE_USER = os.getenv("DATABASE_USER", "postgres")
    DATABASE_PASSWORD = os.getenv("DATABASE_PASSWORD", "yourpassword")
    DATABASE_HOST = os.getenv("DATABASE_HOST", "postgres-db")
    DATABASE_PORT = int(os.getenv("DATABASE_PORT", 5432)) 
    DATABASE_DOCKER_NAME = os.getenv("DATABASE_DOCKER_NAME", "postgres-db")
    DATABASE_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"
    DATABASE_DOCKER_URL = f"postgresql://{DATABASE_USER}:{DATABASE_PASSWORD}@{DATABASE_DOCKER_NAME}:{DATABASE_PORT}/{DATABASE_NAME}"

    # Celery settings
    REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    CELERY_MODULE = os.getenv("CELERY_MODULE", "core.celery")
    FRAME_SELECTION_WORKER = int(os.getenv("FRAME_SELECTION_WORKER", 2))
    DEEPFAKE_DETECTION_WORKER = int(os.getenv("DEEPFAKE_DETECTION_WORKER", 1))

    # JWT settings
    SECRET_KEY = os.getenv("SECRET_KEY", "use_random_secret_key") 
    ALGORITHM = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 3000))  
    ADMIN_TOKEN_EXPIRE_MINUTES  = int(os.getenv("ADMIN_TOKEN_EXPIRE_MINUTES", 2000))  

    # OAuth settings
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "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")

    # SMTP settings for email notifications
    SMTP_SERVER = os.getenv("SMTP_SERVER")
    SMTP_PORT = int(os.getenv("SMTP_PORT") or 465)
    SMTP_EMAIL = os.getenv("SMTP_EMAIL")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    
    # Receiver emails list
    RECEIVER_EMAILS = os.getenv("RECEIVER_EMAILS", "")

settings = Settings()