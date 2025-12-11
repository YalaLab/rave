"""
S3 helpers for vision-engine.

We keep the core processors filesystem-based (SimpleITK/pydicom), and optionally
materialize `s3://...` inputs into a local cache directory before processing.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class S3Url:
    bucket: str
    key: str  # may be empty; never starts with "/"

    @property
    def url(self) -> str:
        if self.key:
            return f"s3://{self.bucket}/{self.key}"
        return f"s3://{self.bucket}"


def is_s3_url(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def parse_s3_url(url: str) -> S3Url:
    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URL: {url}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return S3Url(bucket=bucket, key=key)


def _require_boto3():
    try:
        import boto3  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "S3 input support requires boto3. Install with `pip install boto3` "
            "(or add it to your environment) and ensure AWS credentials are configured."
        ) from e


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def default_s3_cache_dir() -> Path:
    # Prefer a deterministic, user-writable cache location.
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "vision-engine" / "s3"


def _cache_path_for_s3(url: str, cache_dir: Path) -> Path:
    # Avoid extremely long paths and ensure uniqueness across different prefixes/files.
    s3 = parse_s3_url(url)
    key_sanitized = s3.key.strip("/").replace("..", "__")
    # Keep some human readability but cap length.
    tail = key_sanitized[-120:] if key_sanitized else ""
    return cache_dir / s3.bucket / f"{tail}.{_sha1(url)[:12]}".strip(".")


def _iter_s3_keys(client, bucket: str, prefix: str) -> Iterable[str]:
    continuation_token: Optional[str] = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            key = obj.get("Key")
            if key and not key.endswith("/"):
                yield key
        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break


def _looks_like_file_key(key: str) -> bool:
    # Heuristic: treat common medical inputs as files.
    lowered = key.lower()
    return lowered.endswith(".nii") or lowered.endswith(".nii.gz") or lowered.endswith(".dcm")


def materialize_s3_to_local(
    s3_url: str,
    cache_dir: Optional[str] = None,
    *,
    endpoint_url: Optional[str] = None,
    region_name: Optional[str] = None,
    force: bool = False,
) -> Tuple[str, str]:
    """
    Download an S3 "file" or "directory prefix" into a local cache.

    Returns:
        (local_path, kind) where kind is "file" or "prefix".
    """
    _require_boto3()
    import boto3

    cache_root = Path(cache_dir) if cache_dir else default_s3_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)

    s3 = parse_s3_url(s3_url)
    client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region_name)

    target_base = _cache_path_for_s3(s3_url, cache_root)

    # If it looks like a file, try HEAD first (fast).
    if s3.key and _looks_like_file_key(s3.key):
        local_file = target_base
        local_file.parent.mkdir(parents=True, exist_ok=True)
        if local_file.exists() and not force:
            return str(local_file), "file"
        try:
            client.head_object(Bucket=s3.bucket, Key=s3.key)
            logger.info(f"Downloading S3 file {s3_url} -> {local_file}")
            client.download_file(s3.bucket, s3.key, str(local_file))
            return str(local_file), "file"
        except Exception:
            # Fall through to prefix download.
            pass

    # Prefix download (directory-like)
    prefix = s3.key.strip("/")
    if prefix:
        prefix = prefix + "/"
    local_dir = target_base
    local_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already populated (best-effort)
    if not force:
        try:
            if any(local_dir.iterdir()):
                return str(local_dir), "prefix"
        except Exception:
            pass

    logger.info(f"Downloading S3 prefix s3://{s3.bucket}/{prefix} -> {local_dir}")
    found_any = False
    for key in _iter_s3_keys(client, s3.bucket, prefix):
        found_any = True
        rel = key[len(prefix) :] if key.startswith(prefix) else key
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            continue
        client.download_file(s3.bucket, key, str(dest))

    if not found_any:
        raise FileNotFoundError(f"No objects found for S3 URL/prefix: {s3_url}")

    return str(local_dir), "prefix"

