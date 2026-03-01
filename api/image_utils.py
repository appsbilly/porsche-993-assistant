"""
Image processing utilities for the 993 Repair Assistant.

Handles:
- Compression and resizing for Claude Vision API (max 1568px, <5MB)
- Base64 encoding for API content blocks
- S3 upload/download for persistent image storage

S3 path: users/{user_id}/images/{uuid}.jpg
"""

import io
import os
import uuid
import base64
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

# Claude Vision optimal constraints
MAX_DIMENSION = 1568  # Claude's optimal max dimension
MAX_FILE_SIZE = 4_500_000  # Stay under 5MB limit with margin
JPEG_QUALITY = 85
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


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


def process_uploaded_image(uploaded_file) -> tuple[bytes, str, str]:
    """
    Resize and compress an uploaded image for Claude Vision.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        (processed_bytes, media_type, filename)
    """
    from PIL import Image

    img = Image.open(uploaded_file)

    # Convert RGBA/palette to RGB for JPEG output
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")

    # Resize if either dimension exceeds the max
    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        ratio = MAX_DIMENSION / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Compress as JPEG
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)

    # If still too large, progressively reduce quality
    quality = JPEG_QUALITY
    while buffer.tell() > MAX_FILE_SIZE and quality > 30:
        quality -= 10
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)

    processed = buffer.getvalue()
    filename = getattr(uploaded_file, "name", "image.jpg")
    return processed, "image/jpeg", filename


def image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string for Claude API."""
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def upload_image_to_s3(image_bytes: bytes, user_id: str, filename: str) -> str:
    """
    Upload processed image to S3.

    Args:
        image_bytes: Processed image data
        user_id: User identifier for namespacing
        filename: Original filename (used for extension)

    Returns:
        S3 key string (e.g., "users/abc123/images/f3a1b2c4d5e6.jpg")
    """
    s3 = _get_s3()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    if ext not in ("jpg", "jpeg", "png", "webp", "gif"):
        ext = "jpg"
    s3_key = f"users/{user_id}/images/{uuid.uuid4().hex[:12]}.{ext}"
    s3.put_object(
        Bucket=_bucket(),
        Key=s3_key,
        Body=image_bytes,
        ContentType="image/jpeg",
    )
    return s3_key


def load_image_from_s3(s3_key: str) -> bytes | None:
    """
    Download image from S3.

    Returns:
        Image bytes, or None if not found / error.
    """
    try:
        s3 = _get_s3()
        resp = s3.get_object(Bucket=_bucket(), Key=s3_key)
        return resp["Body"].read()
    except Exception:
        return None


def delete_images_from_s3(s3_keys: list[str]):
    """Delete multiple images from S3 (best-effort, ignores errors)."""
    if not s3_keys:
        return
    try:
        s3 = _get_s3()
        bucket = _bucket()
        for key in s3_keys:
            try:
                s3.delete_object(Bucket=bucket, Key=key)
            except Exception:
                pass
    except Exception:
        pass
