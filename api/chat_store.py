"""
Chat conversation storage using AWS S3.

Stores conversation history with auto-generated titles.
Each conversation is a JSON file in s3://bucket/chats/.
An index.json file tracks all conversations for fast sidebar loading.
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)


def _get_s3():
    """Get S3 client (lazy import)."""
    import boto3
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def _bucket():
    return os.getenv("AWS_S3_BUCKET", "porsche-993-rag")


def load_index() -> list[dict]:
    """Load conversation index from S3.

    Returns list of: {id, title, created_at, updated_at}
    """
    try:
        s3 = _get_s3()
        resp = s3.get_object(Bucket=_bucket(), Key="chats/index.json")
        return json.loads(resp["Body"].read().decode())
    except Exception:
        return []


def save_index(index: list[dict]):
    """Save conversation index to S3."""
    s3 = _get_s3()
    s3.put_object(
        Bucket=_bucket(),
        Key="chats/index.json",
        Body=json.dumps(index, indent=2),
        ContentType="application/json",
    )


def load_conversation(conv_id: str) -> list[dict] | None:
    """Load messages for a conversation from S3."""
    try:
        s3 = _get_s3()
        resp = s3.get_object(Bucket=_bucket(), Key=f"chats/{conv_id}.json")
        data = json.loads(resp["Body"].read().decode())
        return data.get("messages", [])
    except Exception:
        return None


def save_conversation(conv_id: str, messages: list[dict]):
    """Save conversation messages to S3."""
    s3 = _get_s3()
    s3.put_object(
        Bucket=_bucket(),
        Key=f"chats/{conv_id}.json",
        Body=json.dumps({"id": conv_id, "messages": messages}, indent=2),
        ContentType="application/json",
    )


def delete_conversation(conv_id: str, index: list[dict]) -> list[dict]:
    """Delete a conversation from S3 and return updated index."""
    try:
        s3 = _get_s3()
        s3.delete_object(Bucket=_bucket(), Key=f"chats/{conv_id}.json")
    except Exception:
        pass
    index = [c for c in index if c["id"] != conv_id]
    save_index(index)
    return index


def new_conversation_id() -> str:
    """Generate a short unique conversation ID."""
    return uuid.uuid4().hex[:8]


def generate_title(message: str) -> str:
    """Generate a short conversation title using Claude Haiku."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return message[:40].strip()

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": (
                    "Generate a 3-6 word title for this Porsche 993 question. "
                    "Just the title, no quotes or extra punctuation.\n\n"
                    f"Question: {message}"
                ),
            }],
        )
        title = resp.content[0].text.strip().strip("\"'.")
        return title[:50]
    except Exception:
        # Fallback: truncate the message
        return message[:40].strip() + ("..." if len(message) > 40 else "")
