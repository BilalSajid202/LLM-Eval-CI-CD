"""S3 raw output storage with local fallback."""

from __future__ import annotations

import json
from uuid import UUID

from llm_eval.config.loader import AppSettings
from llm_eval.models.types import QuestionResult
from llm_eval.storage.local import LocalStorage


class S3Storage:
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key: str = "",
        secret_key: str = "",
    ):
        self.bucket = bucket
        self._client = None
        self._region = region
        self._access_key = access_key
        self._secret_key = secret_key

    def _get_client(self):
        if self._client is None:
            import boto3

            kwargs = {"region_name": self._region}
            if self._access_key and self._secret_key:
                kwargs["aws_access_key_id"] = self._access_key
                kwargs["aws_secret_access_key"] = self._secret_key
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def save_raw_output(self, run_id: UUID, result: QuestionResult) -> str:
        key = f"{run_id}/{result.question_id}/response.json"
        body = result.model_dump_json(indent=2)
        self._get_client().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return key


class OutputStore:
    """Unified output store — S3 when configured, else local filesystem."""

    def __init__(self, settings: AppSettings):
        self.local = LocalStorage(settings.local_storage_path)
        self.s3: S3Storage | None = None
        if settings.s3_bucket:
            self.s3 = S3Storage(
                bucket=settings.s3_bucket,
                region=settings.aws_region,
                access_key=settings.aws_access_key_id,
                secret_key=settings.aws_secret_access_key,
            )

    def save_raw_output(self, run_id: UUID, result: QuestionResult) -> str:
        if self.s3:
            try:
                return self.s3.save_raw_output(run_id, result)
            except Exception:
                pass
        return self.local.save_raw_output(run_id, result)


def create_output_store(settings: AppSettings) -> OutputStore:
    return OutputStore(settings)
