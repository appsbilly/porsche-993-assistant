#!/usr/bin/env python3
"""
View usage analytics for the 993 Repair Assistant.

Pulls daily JSONL logs from S3 and displays a summary of queries.

Usage:
    python scripts/view_analytics.py              # today
    python scripts/view_analytics.py --days 7     # last 7 days
    python scripts/view_analytics.py --days 30    # last 30 days
    python scripts/view_analytics.py --full       # show full responses (not just previews)
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Load .env from project root
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)


def get_s3():
    import boto3
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def fetch_day(s3, bucket, date_str):
    """Fetch and parse a single day's JSONL log. Returns list of dicts."""
    key = f"analytics/{date_str}.jsonl"
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        content = resp["Body"].read().decode("utf-8")
        entries = []
        for line in content.strip().split("\n"):
            if line.strip():
                entries.append(json.loads(line))
        return entries
    except Exception:
        return []


def format_entry(entry, show_full=False):
    """Format a single log entry for display."""
    ts = entry.get("ts", "")
    try:
        dt = datetime.fromisoformat(ts)
        time_str = dt.strftime("%H:%M")
    except Exception:
        time_str = ts[:5] if len(ts) >= 5 else ts

    user_type = entry.get("user_type", "unknown")
    tag = f"[{user_type}]"

    query = entry.get("query", "")
    if len(query) > 80 and not show_full:
        query = query[:77] + "..."

    # Car profile summary
    profile = entry.get("car_profile", {})
    profile_parts = []
    if profile.get("year"):
        profile_parts.append(profile["year"])
    if profile.get("model"):
        profile_parts.append(profile["model"])
    profile_str = f"  ({' '.join(profile_parts)})" if profile_parts else ""

    images = " [img]" if entry.get("has_images") else ""
    sources = entry.get("sources_count", 0)

    line = f"  {time_str}  {tag:<12} \"{query}\"{profile_str}{images}  ({sources} sources)"

    if show_full:
        preview = entry.get("response_preview", "")
        if preview:
            line += f"\n{'':>18}Response: {preview}"

    return line


def main():
    parser = argparse.ArgumentParser(description="View 993 Assistant usage analytics")
    parser.add_argument("--days", type=int, default=1, help="Number of days to show (default: 1 = today)")
    parser.add_argument("--full", action="store_true", help="Show response previews")
    args = parser.parse_args()

    s3 = get_s3()
    bucket = os.getenv("AWS_S3_BUCKET", "porsche-993-rag")

    today = datetime.now(timezone.utc).date()
    total_queries = 0
    total_guests = 0
    total_signed_in = 0

    for days_ago in range(args.days - 1, -1, -1):
        date = today - timedelta(days=days_ago)
        date_str = date.strftime("%Y-%m-%d")
        entries = fetch_day(s3, bucket, date_str)

        if not entries:
            if args.days == 1:
                print(f"\n  No queries logged for {date_str}.\n")
            continue

        guests = sum(1 for e in entries if e.get("user_type") == "guest")
        signed_in = sum(1 for e in entries if e.get("user_type") == "signed_in")
        total_queries += len(entries)
        total_guests += guests
        total_signed_in += signed_in

        print(f"\n{'='*60}")
        print(f"  {date_str}  ({len(entries)} queries — {guests} guest, {signed_in} signed-in)")
        print(f"{'='*60}")

        for entry in entries:
            print(format_entry(entry, show_full=args.full))

    if args.days > 1:
        print(f"\n{'─'*60}")
        print(f"  Total: {total_queries} queries over {args.days} days")
        print(f"  Guests: {total_guests}  |  Signed-in: {total_signed_in}")
        print(f"{'─'*60}")

    if total_queries == 0 and args.days > 1:
        print(f"\n  No queries logged in the last {args.days} days.\n")


if __name__ == "__main__":
    main()
