"""Which posts are unusual? Compared with *comparable* posts (same platform and content type).

Signals, all for investigation rather than conclusions:
1. robust z-score of log engagement rate against the peer group (median / MAD, |z| >= 3.5);
2. an Isolation Forest over reach, engagement, comment share, share rate and complaint share (top ~3%);
3. complaint-heavy: a post whose comments are at least 40% complaints / order-status issues.
Posts without view counts are compared on raw interactions instead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.ensemble import IsolationForest

from .content import MIN_AGE_DAYS

Z_CUT = 3.5
MIN_PEERS = 8
COMPLAINT_SHARE = 0.40
COMPLAINT_MIN_COMMENTS = 8


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if not mad or np.isnan(mad):
        sd = x.std()
        return (x - med) / sd if sd else x * 0
    return 0.6745 * (x - med) / mad


def _peer_z(p: pd.DataFrame, col: str, kind: pd.Series, kinds=("engagement rate", "raw interactions")):
    """Robust z of ``col`` within peers (platform + content type, else platform, else everyone)."""
    z = pd.Series(np.nan, index=p.index)
    med = pd.Series(np.nan, index=p.index)
    n = pd.Series(0, index=p.index)
    for k in kinds:
        sub = p[kind == k]
        for keys in (["platform", "content_type"], ["platform"], None):
            groups = [sub] if keys is None else [g for _, g in sub.groupby(keys)]
            for g in groups:
                todo = z.loc[g.index].isna()
                if len(g) >= MIN_PEERS and todo.any():
                    zz = _robust_z(g[col])
                    z.loc[g.index[todo]] = zz[todo]
                    m = g[col].median()
                    med.loc[g.index[todo]] = np.exp(m) if k == "engagement rate" else (np.expm1(m) if k == "raw interactions" else m)
                    n.loc[g.index[todo]] = len(g)
    return z, med, n


def detect(posts: pd.DataFrame, comments: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    p = posts.copy()
    p = p[(p["days_since_post"].isna()) | (p["days_since_post"] >= MIN_AGE_DAYS)].copy()
    has_views = p["engagement_rate"].notna()
    p["metric"] = np.where(has_views, np.log(p["engagement_rate"].where(has_views).clip(lower=1e-6)),
                           np.log1p(p["interactions"]))
    p["metric_kind"] = np.where(has_views, "engagement rate", "raw interactions")
    z, peer_med, peer_n = _peer_z(p, "metric", p["metric_kind"])
    p["robust_z"], p["peer_median"], p["peer_posts"] = z, peer_med, peer_n
    p["log_views"] = np.log1p(p["views"])
    zr, rmed, _ = _peer_z(p, "log_views", pd.Series(np.where(p["views"].notna(), "views", "none"), index=p.index), kinds=("views",))
    p["reach_z"], p["reach_peer_median"] = zr, np.expm1(rmed)

    stat = comments.groupby("post_id").agg(
        analyzed_comments=("comment_id", "size"),
        negative_share=("sentiment", lambda s: (s == "negative").mean()),
        complaint_share=("topic", lambda s: s.isin(["order_status", "price_complaint"]).mean()),
        order_issue_share=("topic", lambda s: (s == "order_status").mean()))
    p = p.drop(columns=[c for c in stat.columns if c in p.columns]).merge(stat, left_on="post_id", right_index=True, how="left")
    for c in ("analyzed_comments",):
        p[c] = p[c].fillna(0).astype(int)

    # Isolation Forest on standardized features
    feat = pd.DataFrame({
        "log_reach": np.log1p(p["views"].fillna(p["interactions"] * 20)),
        "metric": p["metric"], "comment_rate": np.log1p(1000 * p["comment_rate"].fillna(p["comments_count"] / p["interactions"].clip(lower=1))),
        "share_rate": np.log1p(1000 * p["share_rate"].fillna(0)), "complaint_share": p["complaint_share"].fillna(0)})
    feat = (feat - feat.mean()) / feat.std().replace(0, 1)
    feat = feat.fillna(0)
    if len(feat) >= 50:
        iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=seed).fit(feat)
        p["iforest_flag"] = iso.predict(feat) == -1
        p["iforest_score"] = -iso.score_samples(feat)
    else:
        p["iforest_flag"], p["iforest_score"] = False, 0.0

    p["z_flag"] = (p["robust_z"].abs() >= Z_CUT) | (p["reach_z"].abs() >= Z_CUT)
    base = float(comments["topic"].isin(["order_status", "price_complaint"]).mean())
    k_c = (p["complaint_share"].fillna(0) * p["analyzed_comments"]).round()
    p["complaint_p"] = [binom.sf(k - 1, n, base) if n > 0 else 1.0 for k, n in zip(k_c, p["analyzed_comments"])]
    p["complaint_heavy"] = ((p["analyzed_comments"] >= COMPLAINT_MIN_COMMENTS) & (p["complaint_share"] >= COMPLAINT_SHARE)
                            & (p["complaint_p"] < 0.001))
    # the Isolation Forest only supports another signal; on its own it is mostly noise
    p["flagged"] = p["z_flag"] | p["complaint_heavy"] | (p["iforest_flag"] & (p["robust_z"].abs() >= 2.5))
    out = p[p["flagged"]].copy()
    if out.empty:
        return pd.DataFrame(columns=["post_id", "platform", "content_type", "finding", "why_flagged"])

    def kind(r) -> str:
        if r["complaint_heavy"]:
            return "Complaint-heavy comments"
        zs = [v for v in (r["robust_z"], r["reach_z"]) if pd.notna(v) and abs(v) >= Z_CUT] or [r["robust_z"]]
        zmax = max(zs, key=abs)
        if pd.notna(zmax) and zmax > 0:
            return "Unusually high engagement"
        if pd.notna(zmax) and zmax < 0:
            return "Unusually low engagement"
        return "Unusual overall pattern"

    def why(r) -> str:
        bits = []
        if pd.notna(r["robust_z"]) and r["z_flag"] and pd.notna(r["peer_median"]) and r["peer_median"] > 0:
            val = np.exp(r["metric"]) if r["metric_kind"] == "engagement rate" else np.expm1(r["metric"])
            bits.append(f"{r['metric_kind']} {val / r['peer_median']:.1f}x the median of {int(r['peer_posts'])} comparable posts (z={r['robust_z']:+.1f})")
        if pd.notna(r["reach_z"]) and abs(r["reach_z"]) >= Z_CUT and pd.notna(r["reach_peer_median"]) and r["reach_peer_median"] > 0:
            bits.append(f"reach {r['views'] / r['reach_peer_median']:.1f}x the peer median (z={r['reach_z']:+.1f})")
        if r["complaint_heavy"]:
            bits.append(f"{100 * r['complaint_share']:.0f}% of {int(r['analyzed_comments'])} comments are complaints" +
                        (f", {100 * r['order_issue_share']:.0f}% order-status" if r["order_issue_share"] > 0 else ""))
        if r["iforest_flag"] and not bits:
            bits.append("combination of reach, engagement and comment mix is rare in this dataset")
        return "; ".join(bits)

    out["finding"] = out.apply(kind, axis=1)
    out["why_flagged"] = out.apply(why, axis=1)
    cols = ["post_id", "brand", "platform", "content_type", "campaign", "views", "likes", "comments_count", "shares",
            "days_since_post", "engagement_rate", "finding", "why_flagged", "robust_z", "reach_z", "iforest_score",
            "analyzed_comments", "complaint_share", "data_origin"]
    out = out[[c for c in cols if c in out.columns]]
    return out.sort_values("robust_z", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
