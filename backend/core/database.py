import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
from config import settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# use DATABASE_URL for local development and DATABASE_DOCKER_URL for if postgres is running in a docker container
# DATABASE_URL = settings.DATABASE_URL
try:
    DATABASE_URL = settings.DATABASE_NAME
    engine = create_engine(DATABASE_URL)
except Exception as e:
    DATABASE_URL = settings.DATABASE_URL
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def test_db_connection():
    """Tests the database connection."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        logger.info("Database connected successfully.")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# Import models to ensure tables are created
from models import Base

logger.info(f"Tables detected by SQLAlchemy: {Base.metadata.tables.keys()}")
Base.metadata.create_all(bind=engine)
logger.info("✅ Database tables are ready.")


# on startup, test conn
test_db_connection()

def get_db():
    """Dependency to get the database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
