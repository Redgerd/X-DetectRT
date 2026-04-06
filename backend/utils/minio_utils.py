import io
import logging
from minio import Minio
from minio.error import S3Error
from config import settings

logger = logging.getLogger(__name__)

class MinIOClient:
    def __init__(self):
        self.client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self.bucket_name = settings.MINIO_BUCKET_NAME
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")
            else:
                logger.info(f"MinIO bucket already exists: {self.bucket_name}")
        except S3Error as e:
            logger.error(f"Error ensuring MinIO bucket exists: {e}")
            raise

    def upload_file(self, file_path: str, object_name: str, content_type: str = "application/octet-stream") -> str:
        """
        Upload a file to MinIO.

        Args:
            file_path: Local file path
            object_name: Object name in MinIO (key)
            content_type: MIME type

        Returns:
            Object name (key) in MinIO
        """
        try:
            self.client.fput_object(
                self.bucket_name,
                object_name,
                file_path,
                content_type=content_type,
            )
            logger.info(f"Uploaded file to MinIO: {object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"Error uploading file to MinIO: {e}")
            raise

    def upload_bytes(self, data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        """
        Upload bytes to MinIO.

        Args:
            data: File data as bytes
            object_name: Object name in MinIO (key)
            content_type: MIME type

        Returns:
            Object name (key) in MinIO
        """
        try:
            data_stream = io.BytesIO(data)
            self.client.put_object(
                self.bucket_name,
                object_name,
                data_stream,
                length=len(data),
                content_type=content_type,
            )
            logger.info(f"Uploaded bytes to MinIO: {object_name}")
            return object_name
        except S3Error as e:
            logger.error(f"Error uploading bytes to MinIO: {e}")
            raise

    def get_file_url(self, object_name: str) -> str:
        """
        Get the URL for a file in MinIO.

        Args:
            object_name: Object name in MinIO

        Returns:
            Presigned URL
        """
        try:
            url = self.client.presigned_get_object(self.bucket_name, object_name)
            return url
        except S3Error as e:
            logger.error(f"Error getting file URL from MinIO: {e}")
            raise

# Global instance
minio_client = MinIOClient()