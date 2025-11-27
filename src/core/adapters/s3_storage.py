from __future__ import annotations

from typing import Optional

import boto3
from botocore.exceptions import ClientError

from src.core.abstractions.storage import StorageBackend
from src.utils.logger_loader import LoggerLoader


class S3Storage(StorageBackend):
    """Storage backend that persists files to AWS S3."""

    def __init__(self, bucket_name: str, prefix: str = "subtitles/") -> None:
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client = boto3.client("s3")
        self.logger = LoggerLoader.get_logger()

    def _build_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def save_file(self, content: str, key: str) -> str:
        s3_key = self._build_key(key)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=content.encode("utf-8"),
        )
        url = f"s3://{self.bucket_name}/{s3_key}"
        self.logger.info("Saved file to S3: %s", url)
        return url

    def load_file(self, key: str) -> Optional[str]:
        s3_key = self._build_key(key)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response["Body"].read().decode("utf-8")
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "NoSuchKey":
                return None
            self.logger.error("Error loading %s from S3: %s", s3_key, exc)
            raise

    def file_exists(self, key: str) -> bool:
        s3_key = self._build_key(key)
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError:
            return False

    def delete_file(self, key: str) -> bool:
        s3_key = self._build_key(key)
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as exc:
            self.logger.error("Error deleting %s from S3: %s", s3_key, exc)
            return False

