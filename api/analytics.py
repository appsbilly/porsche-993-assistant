"""
Usage analytics for the 993 Repair Assistant.

Logs each query/response interaction to S3 as daily JSONL files.
Path: analytics/YYYY-MM-DD.jsonl

Best-effort — never breaks the chat flow if logging fails.
"""

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)


def _get_s3():
    """Get S3 client (same pattern as chat_store)."""
    import boto3
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def _bucket():
    return os.getenv("AWS_S3_BUCKET", "porsche-993-rag")


def log_query(
    user_type: str,
    query: str,
    response: str,
    conv_id: str | None = None,
    sources_count: int = 0,
    car_profile: dict | None = None,
    has_images: bool = False,
):
    """
    Log a single query/response interaction to S3.

    Appends one JSON line to analytics/YYYY-MM-DD.jsonl.
    Silently fails — never interrupts the chat experience.
    """
    try:
        now = datetime.now(timezone.utc)
        date_key = now.strftime("%Y-%m-%d")
        s3_key = f"analytics/{date_key}.jsonl"

        # Build log entry
        profile_summary = {}
        if car_profile:
            for field in ("year", "model", "transmission", "mileage"):
                val = car_profile.get(field, "")
                if val:
                    profile_summary[field] = val

        entry = {
            "ts": now.isoformat(),
            "user_type": user_type,
            "query": query,
            "response_preview": response[:300] if response else "",
            "response_length": len(response) if response else 0,
            "conv_id": conv_id,
            "sources_count": sources_count,
            "car_profile": profile_summary,
            "has_images": has_images,
        }

        new_line = json.dumps(entry, ensure_ascii=False) + "\n"

        # Read existing file (if any), append new line, re-upload
        s3 = _get_s3()
        bucket = _bucket()

        existing = ""
        try:
            resp = s3.get_object(Bucket=bucket, Key=s3_key)
            existing = resp["Body"].read().decode("utf-8")
        except s3.exceptions.NoSuchKey:
            pass
        except Exception:
            # File doesn't exist yet — that's fine
            pass

        updated = existing + new_line

        s3.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=updated.encode("utf-8"),
            ContentType="application/jsonl",
        )

    except Exception:
        # Never break the chat flow
        pass
