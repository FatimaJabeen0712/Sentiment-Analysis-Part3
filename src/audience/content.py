"""Which content attracts attention? Engagement rate by content type / platform / campaign.

engagement rate = (likes + comments + shares) / views, per post. Medians (not means) are used because
a few viral posts would otherwise dominate; a bootstrap 95% interval shows the uncertainty and a
Kruskal-Wallis test says whether the groups are distinguishable at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

MIN_POSTS = 8        # groups smaller than this are flagged "low sample"
MIN_AGE_DAYS = 2     # posts younger than this have not collected their engagement yet
BOOT = 300


def add_engagement(posts: pd.DataFrame) -> pd.DataFrame:
    out = posts.copy()
    for c in ("likes", "comments_count", "shares", "views", "days_since_post"):
        out[c] = pd.to_numeric(out[c], errors="coerce") if c in out else np.nan
    out["interactions"] = out["likes"].fillna(0) + out["comments_count"].fillna(0) + out["shares"].fillna(0)
    views = out["views"].where(out["views"] > 0)
    out["engagement_rate"] = out["interactions"] / views
    out["comment_rate"] = out["comments_count"] / views
    out["share_rate"] = out["shares"] / views
    return out


def _median_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]),) * 3
    boots = np.median(rng.choice(x, size=(BOOT, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize_groups(posts: pd.DataFrame, by: str | list[str], seed: int = 0) -> pd.DataFrame:
    """One row per group: posts, median engagement rate with 95% interval, relative to all posts."""
    rng = np.random.default_rng(seed)
    p = posts[(posts["days_since_post"].isna()) | (posts["days_since_post"] >= MIN_AGE_DAYS)]
    overall = float(np.nanmedian(p["engagement_rate"])) if p["engagement_rate"].notna().any() else np.nan
    rows = []
    for key, g in p.groupby(by, observed=True):
        er = g["engagement_rate"].to_numpy(float)
        med, lo, hi = _median_ci(er, rng)
        row = dict(zip([by] if isinstance(by, str) else by, key if isinstance(key, tuple) else (key,)))
        row.update({"posts": len(g), "posts_with_views": int(np.isfinite(er).sum()),
                    "median_engagement_rate": med, "ci_low": lo, "ci_high": hi,
                    "vs_all_posts": med / overall if overall and overall > 0 else np.nan,
                    "median_likes": float(g["likes"].median()), "median_comments": float(g["comments_count"].median()),
                    "median_shares": float(g["shares"].median()) if g["shares"].notna().any() else np.nan,
                    "low_sample": len(g) < MIN_POSTS or int(np.isfinite(er).sum()) < MIN_POSTS})
        rows.append(row)
    cols = [by] if isinstance(by, str) else list(by)
    return (pd.DataFrame(rows).sort_values("median_engagement_rate", ascending=False).reset_index(drop=True)
            if rows else pd.DataFrame(columns=cols))


def groups_differ(posts: pd.DataFrame, by: str = "content_type") -> dict:
    """Kruskal-Wallis across groups: are engagement rates distinguishable? Returns p-value and n."""
    p = posts[(posts["days_since_post"].isna()) | (posts["days_since_post"] >= MIN_AGE_DAYS)]
    samples = [g["engagement_rate"].dropna().to_numpy() for _, g in p.groupby(by, observed=True)]
    samples = [s for s in samples if len(s) >= MIN_POSTS]
    if len(samples) < 2:
        return {"p_value": np.nan, "groups": len(samples), "distinguishable": False}
    h, pv = stats.kruskal(*samples)
    return {"p_value": float(pv), "groups": len(samples), "distinguishable": bool(pv < 0.01)}
