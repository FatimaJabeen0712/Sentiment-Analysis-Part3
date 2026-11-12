"""Link Part 1 to Part 3: turn Part 1's comment CSV (and optional YouTube video statistics) into
Table A (posts) and Table B (comments).

Real data only: nothing here invents numbers. Without video statistics, a post's ``comments_count`` is just
the number of comments Part 1 collected (a lower bound), and ``likes`` / ``views`` stay empty, so
content and anomaly analysis is limited to what exists.
"""
from __future__ import annotations

import re

import pandas as pd

_RULES = [  # content type guessed from the video title
    (r"#shorts?\b|\bshorts?\b|\breel", "Short video"),
    (r"unbox|haul|first impression", "Unboxing / haul"),
    (r"review|rated|honest|opinion|rating|worth", "Review"),
    (r"tutorial|how to|grwm|styling|style|outfit|lookbook", "Styling video"),
    (r"collection|campaign|launch|new arrival|eid|festive", "Collection showcase"),
    (r"vs\b|fake|real|compar", "Comparison"),
]


def guess_content_type(title: str) -> str:
    raw = str(title or "").casefold()
    if re.search(r"#shorts?\b|\breels?\b", raw):
        return "Short video"
    t = re.sub(r"#\w+", " ", raw)        # hashtags are often stuffed with unrelated words (#style #fashion)
    for rx, label in _RULES[1:]:
        if re.search(rx, t):
            return label
    return "Other video"


def to_tables(raw: pd.DataFrame, stats: dict[str, dict] | None = None, brand: str = "",
              platform: str = "YouTube") -> tuple[pd.DataFrame, pd.DataFrame]:
    """``raw``: Part 1's raw_comments.csv (needs text, video_id; uses comment_id, video_title, created_at).
    ``stats``: optional {video_id: {views, likes, comment_count, published_at, title}} from the API."""
    if "text" not in raw or "video_id" not in raw:
        raise ValueError("Part 1 comment CSV needs 'text' and 'video_id' columns (raw_comments.csv from the agent).")
    stats = stats or {}
    df = raw.copy()
    df["video_id"] = df["video_id"].astype(str)
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for vid, g in df.groupby("video_id"):
        s = stats.get(vid, {})
        title = s.get("title") or (g["video_title"].dropna().iat[0] if "video_title" in g and g["video_title"].notna().any() else "")
        published = pd.to_datetime(s.get("published_at"), utc=True, errors="coerce") if s.get("published_at") else pd.NaT
        days = int((now - published).days) if pd.notna(published) else float("nan")
        rows.append({
            "post_id": vid, "brand": brand or (g["brand"].dropna().iat[0] if "brand" in g and g["brand"].notna().any() else ""),
            "platform": platform, "campaign": "", "content_type": guess_content_type(title),
            "product_category": "", "likes": s.get("likes", float("nan")),
            "comments_count": s.get("comment_count", len(g)), "shares": float("nan"),
            "views": s.get("views", float("nan")), "days_since_post": days,
            "data_origin": "observed_youtube", "source_url": f"https://www.youtube.com/watch?v={vid}",
            "collected_at": str(now.date())})
    posts = pd.DataFrame(rows)
    comments = pd.DataFrame({
        "comment_id": df["comment_id"].astype(str) if "comment_id" in df else [f"c{i + 1}" for i in range(len(df))],
        "post_id": df["video_id"], "comment_text": df["text"],
        "data_origin": "observed_youtube", "collected_at": str(now.date())})
    return posts, comments
