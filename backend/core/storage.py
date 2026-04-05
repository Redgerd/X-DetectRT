"""
core/storage.py
---------------
    storage.upload_file("uploads", "/app/uploads/foo.mp4", "tasks/abc/foo.mp4")
    storage.upload_bytes("xai", png_bytes, "tasks/abc/heatmap.png", "image/png")
    url = storage.presigned_url("results", "tasks/abc/result.json", expires_seconds=3600)
    storage.download_file("uploads", "tasks/abc/foo.mp4", "/tmp/foo.mp4")
"""

from __future__ import annotations

import io
import logging
import os
from datetime import timedelta
from pathlib import Path

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


class MinIOStorage:
    def __init__(self) -> None:
        endpoint = os.environ.get("MINIO_ENDPOINT", "minio:9000")
        access   = os.environ.get("MINIO_ROOT_USER", "minioadmin")
        secret   = os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin")
        secure   = os.environ.get("MINIO_SECURE", "false").lower() == "true"

        self._client = Minio(endpoint, access_key=access, secret_key=secret, secure=secure)
        logger.info(f"[storage] MinIO client → {endpoint}")

    def _ensure_bucket(self, bucket: str) -> None:
        if not self._client.bucket_exists(bucket):
            self._client.make_bucket(bucket)
            logger.info(f"[storage] Created bucket: {bucket}")

    def upload_file(
        self,
        bucket: str,
        local_path: str | Path,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload a local file. Returns object_name."""
        self._ensure_bucket(bucket)
        self._client.fput_object(bucket, object_name, str(local_path), content_type=content_type)
        logger.info(f"[storage] ↑ {local_path} → {bucket}/{object_name}")
        return object_name

    def upload_bytes(
        self,
        bucket: str,
        data: bytes,
        object_name: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload raw bytes (e.g. in-memory PNG). Returns object_name."""
        self._ensure_bucket(bucket)
        self._client.put_object(
            bucket, object_name, io.BytesIO(data), length=len(data), content_type=content_type
        )
        logger.info(f"[storage] ↑ bytes({len(data)}) → {bucket}/{object_name}")
        return object_name

    def download_file(self, bucket: str, object_name: str, local_path: str | Path) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        self._client.fget_object(bucket, object_name, str(local_path))
        logger.info(f"[storage] ↓ {bucket}/{object_name} → {local_path}")

    def download_bytes(self, bucket: str, object_name: str) -> bytes:
        r = self._client.get_object(bucket, object_name)
        try:
            return r.read()
        finally:
            r.close()
            r.release_conn()

    def presigned_url(self, bucket: str, object_name: str, expires_seconds: int = 3600) -> str:
        """Presigned GET URL — safe to hand directly to the frontend."""
        return self._client.presigned_get_object(
            bucket, object_name, expires=timedelta(seconds=expires_seconds)
        )

    def exists(self, bucket: str, object_name: str) -> bool:
        try:
            self._client.stat_object(bucket, object_name)
            return True
        except S3Error:
            return False

    def delete(self, bucket: str, object_name: str) -> None:
        try:
            self._client.remove_object(bucket, object_name)
        except S3Error as e:
            logger.warning(f"[storage] delete failed {bucket}/{object_name}: {e}")


# module-level singleton — import this everywhere
storage = MinIOStorage()