"""GCSStorage — Google Cloud Storage adapter for audio files and KB ZIPs."""
from __future__ import annotations

import asyncio
from pathlib import Path

import structlog
from google.cloud import storage

from healthbridgeai.config.settings import settings

log = structlog.get_logger(__name__)


class GCSStorage:
    """Thin async wrapper around GCS for audio upload/download."""

    def __init__(self) -> None:
        self._client = storage.Client(project=settings.GCP_PROJECT_ID)
        self._bucket = self._client.bucket(settings.GCS_BUCKET_NAME)

    async def upload_file(self, local_path: str, gcs_path: str) -> str:
        """Upload a local file and return its GCS URI (gs://bucket/path)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._upload_sync, local_path, gcs_path)
        return f"gs://{settings.GCS_BUCKET_NAME}/{gcs_path}"

    def _upload_sync(self, local_path: str, gcs_path: str) -> None:
        blob = self._bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        log.info("gcs.uploaded", gcs_path=gcs_path, local=local_path)

    async def download_bytes(self, gcs_path: str) -> bytes:
        """Download a GCS file as bytes."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._download_sync, gcs_path)

    def _download_sync(self, gcs_path: str) -> bytes:
        blob = self._bucket.blob(gcs_path)
        return blob.download_as_bytes()

    async def get_signed_url(self, gcs_path: str, expiry_seconds: int = 300) -> str:
        """Return a time-limited public URL for WhatsApp to download audio."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._signed_url_sync, gcs_path, expiry_seconds
        )

    def _signed_url_sync(self, gcs_path: str, expiry_seconds: int) -> str:
        import datetime
        blob = self._bucket.blob(gcs_path)
        return blob.generate_signed_url(
            expiration=datetime.timedelta(seconds=expiry_seconds),
            method="GET",
            version="v4",
        )

    def gcs_uri_to_path(self, gcs_uri: str) -> str:
        """Strip gs://bucket/ prefix from a GCS URI."""
        prefix = f"gs://{settings.GCS_BUCKET_NAME}/"
        return gcs_uri.removeprefix(prefix)
