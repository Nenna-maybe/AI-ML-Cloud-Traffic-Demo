"""
cloud_storage.py - Optional AWS S3 storage with a local fallback.

* S3 is used ONLY when USE_S3=true and S3_BUCKET is set.
* Credentials are never stored in code. boto3 reads them from the standard
  places: environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY),
  ~/.aws/credentials, or an IAM role.
* If S3 is disabled, misconfigured or unreachable, files stay on local disk
  and the app keeps working.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class StoredFile:
    backend: str  # "local" or "s3"
    location: str  # local path or s3://bucket/key
    warning: Optional[str] = None


class StorageManager:
    def __init__(self, *, use_s3: bool, bucket: str, prefix: str, region: str, client=None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self.enabled = bool(use_s3 and bucket)
        self._client = client

    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            import boto3  # imported lazily so the app runs without boto3 configured

            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def _key(self, name: str) -> str:
        return f"{self.prefix}/{name}" if self.prefix else name

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        """Short error text that never contains credentials."""
        code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
        return code or exc.__class__.__name__

    # ------------------------------------------------------------------
    def status(self) -> tuple[bool, str]:
        """Check whether the bucket is reachable. Returns (ok, message)."""
        if not self.enabled:
            return False, "S3 disabled - files are stored locally (set USE_S3=true and S3_BUCKET to enable)."
        try:
            self._get_client().head_bucket(Bucket=self.bucket)
            return True, f"Connected to S3 bucket '{self.bucket}'."
        except Exception as exc:
            return False, f"S3 not reachable ({self._safe_error(exc)}). Using local storage."

    def save(self, local_path: Path, object_name: str) -> StoredFile:
        """Upload a local file to S3 if enabled; otherwise keep it local."""
        local_path = Path(local_path)
        if not self.enabled:
            return StoredFile("local", str(local_path))
        key = self._key(object_name)
        try:
            self._get_client().upload_file(str(local_path), self.bucket, key)
            return StoredFile("s3", f"s3://{self.bucket}/{key}")
        except Exception as exc:
            reason = self._safe_error(exc)
            logger.warning("S3 upload failed (%s); keeping local copy", reason)
            return StoredFile("local", str(local_path), warning=f"S3 upload failed ({reason}); saved locally.")

    def download(self, location: str, destination: Path) -> Path:
        """Fetch a stored file to `destination` (works for local and s3:// locations)."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if location.startswith("s3://"):
            bucket, _, key = location[5:].partition("/")
            self._get_client().download_file(bucket, key, str(destination))
            return destination
        source = Path(location)
        if not source.exists():
            raise FileNotFoundError(f"Local file not found: {source}")
        return source
