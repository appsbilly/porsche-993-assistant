"""
Authentication and user profile management backed by AWS S3.

S3 layout:
  auth/credentials.json         — all user credentials (hashed passwords)
  users/{username}/profile.json — car profile per user
"""

import os
import json
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


# ---------------------------------------------------------------------------
# Credentials (used by streamlit-authenticator)
# ---------------------------------------------------------------------------

def load_credentials() -> dict:
    """Load the authenticator config dict from S3.

    Returns a dict compatible with streamlit-authenticator:
    {
      "credentials": {
        "usernames": {
          "jsmith": {
            "email": "js@example.com",
            "name": "John Smith",
            "password": "$2b$12$..."   # bcrypt hash
          }
        }
      },
      "cookie": {
        "name": "porsche_993_auth",
        "key": "<random>",
        "expiry_days": 30
      }
    }
    """
    try:
        s3 = _get_s3()
        resp = s3.get_object(Bucket=_bucket(), Key="auth/credentials.json")
        return json.loads(resp["Body"].read().decode())
    except Exception:
        # First run — return empty skeleton
        return {
            "credentials": {"usernames": {}},
            "cookie": {
                "name": "porsche_993_auth",
                "key": os.getenv("AUTH_COOKIE_KEY", "porsche993secretkey"),
                "expiry_days": 30,
            },
        }


def save_credentials(config: dict):
    """Save the authenticator config dict to S3."""
    s3 = _get_s3()
    s3.put_object(
        Bucket=_bucket(),
        Key="auth/credentials.json",
        Body=json.dumps(config, indent=2),
        ContentType="application/json",
    )


# ---------------------------------------------------------------------------
# Car profiles
# ---------------------------------------------------------------------------

def load_user_profile(username: str) -> dict | None:
    """Load a user's car profile from S3, or None if not set."""
    try:
        s3 = _get_s3()
        resp = s3.get_object(
            Bucket=_bucket(),
            Key=f"users/{username}/profile.json",
        )
        return json.loads(resp["Body"].read().decode())
    except Exception:
        return None


def save_user_profile(username: str, profile: dict):
    """Save a user's car profile to S3."""
    s3 = _get_s3()
    s3.put_object(
        Bucket=_bucket(),
        Key=f"users/{username}/profile.json",
        Body=json.dumps(profile, indent=2),
        ContentType="application/json",
    )


def decode_vin(vin: str) -> dict | None:
    """Decode a VIN using the free NHTSA vPIC API.

    Returns a dict with keys: year, make, model, engine, transmission, body_class
    or None on failure.
    """
    import requests as _requests

    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"
    try:
        resp = _requests.get(url, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("Results", [{}])[0]

        # Only return if we got meaningful data
        year = results.get("ModelYear", "")
        if not year or year == "0":
            return None

        return {
            "year": year,
            "make": results.get("Make", ""),
            "model": results.get("Model", ""),
            "engine": results.get("DisplacementL", "") + "L" if results.get("DisplacementL") else "",
            "transmission": results.get("TransmissionStyle", ""),
            "body_class": results.get("BodyClass", ""),
        }
    except Exception:
        return None
