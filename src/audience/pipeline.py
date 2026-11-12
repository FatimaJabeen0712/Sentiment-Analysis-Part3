"""Run the whole Step 3 analysis: validate -> clean -> classify -> content -> segments -> anomalies -> actions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.preprocessing import clean_text, spam_reason
from src.schemas import MissingColumnError, PipelineError

from . import anomalies as anom
from . import content, recommend, segments
from .topics import TOPIC_LABELS, classify, normalize

POST_REQUIRED = ["post_id", "platform", "content_type", "likes", "comments_count"]
POST_OPTIONAL = ["brand", "campaign", "product_category", "shares", "views", "days_since_post",
                 "data_origin", "source_url", "collected_at"]
COMMENT_REQUIRED = ["post_id", "comment_text"]


@dataclass
class AudienceResult:
    posts: pd.DataFrame
    comments: pd.DataFrame
    funnel: dict
    by_content: pd.DataFrame
    by_platform_content: pd.DataFrame
    by_campaign: pd.DataFrame
    differ: dict
    needs: pd.DataFrame
    segments: pd.DataFrame
    segment_info: dict
    archetypes: pd.DataFrame
    anomalies: pd.DataFrame
    recommendations: pd.DataFrame
    provenance: dict
    evaluation: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def contains_simulated(self) -> bool:
        return self.provenance.get("simulated_posts", 0) > 0 or self.provenance.get("simulated_comments", 0) > 0


def _need(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MissingColumnError(f"{name} is missing required column(s): {', '.join(missing)}")


def _provenance(posts: pd.DataFrame, comments: pd.DataFrame) -> dict:
    def split(df):
        origin = df["data_origin"].astype(str) if "data_origin" in df else pd.Series("unknown", index=df.index)
        sim = int(origin.str.startswith("simulated").sum())
        return sim, len(df) - sim
    sp, op = split(posts)
    sc, oc = split(comments)
    return {"simulated_posts": sp, "observed_posts": op, "simulated_comments": sc, "observed_comments": oc}


def run_audience_analysis(posts_raw: pd.DataFrame, comments_raw: pd.DataFrame, seed: int = 0) -> AudienceResult:
    _need(posts_raw, POST_REQUIRED, "Posts table (Table A)")
    _need(comments_raw, COMMENT_REQUIRED, "Comments table (Table B)")
    if posts_raw["post_id"].duplicated().any():
        raise PipelineError("Posts table has duplicate post_id values; each post must appear once.")
    warnings: list[str] = []

    posts = content.add_engagement(posts_raw)
    comments = comments_raw.copy()
    if "comment_id" not in comments:
        comments["comment_id"] = [f"c{i + 1}" for i in range(len(comments))]
    comments["comment_id"] = comments["comment_id"].astype(str)
    comments["comment_text"] = comments["comment_text"].map(clean_text)
    funnel = {"comments_in": len(comments)}

    comments = comments[comments["comment_text"].str.len() > 0]
    funnel["empty_removed"] = funnel["comments_in"] - len(comments)
    spam = comments["comment_text"].map(spam_reason).notna()
    funnel["spam_removed"] = int(spam.sum())
    comments = comments[~spam]
    known = comments["post_id"].isin(posts["post_id"])
    funnel["orphan_removed"] = int((~known).sum())
    comments = comments[known]
    key = comments["post_id"].astype(str) + "||" + comments["comment_text"].map(normalize)
    dup = key.duplicated()
    funnel["duplicate_removed"] = int(dup.sum())   # same text on the SAME post only; "Price?" on different posts is kept
    comments = comments[~dup].reset_index(drop=True)
    funnel["comments_analyzed"] = len(comments)
    if comments.empty:
        raise PipelineError("No usable comments remain after cleaning.")

    cls = pd.DataFrame([classify(t) for t in comments["comment_text"]])
    comments = pd.concat([comments, cls], axis=1)

    posts["analyzed_comments"] = posts["post_id"].map(comments["post_id"].value_counts()).fillna(0).astype(int)
    by_content = content.summarize_groups(posts, "content_type")
    by_pc = content.summarize_groups(posts, ["platform", "content_type"])
    by_campaign = content.summarize_groups(posts, "campaign") if "campaign" in posts and posts["campaign"].notna().any() else pd.DataFrame()
    differ = content.groups_differ(posts, "content_type")
    needs = segments.audience_needs(comments)

    seg_df, seg_info, arche = pd.DataFrame(), {}, pd.DataFrame()
    try:
        labels, seg_info, vec, km = segments.cluster_comments(comments, seed=seed)
        comments["segment_id"] = labels
        seg_df = segments.describe_segments(comments, labels, vec, km, posts)
        name = dict(zip(seg_df["segment_id"], seg_df["segment"]))
        comments["segment"] = comments["segment_id"].map(name)
        arche = segments.post_archetypes(posts, comments, seed=seed)
        if len(arche):
            mapping = arche.attrs["post_archetype"]
            posts["archetype_id"] = posts["post_id"].map(mapping)
    except ValueError as exc:
        warnings.append(str(exc))

    flagged = anom.detect(posts, comments, seed=seed)
    recs = recommend.build(comments, by_content, differ, flagged, seg_df, posts)
    prov = _provenance(posts_raw, comments_raw)
    if prov["simulated_posts"] or prov["simulated_comments"]:
        warnings.append("This run includes SIMULATED data. Results illustrate the method, not a real brand.")
    if len(posts) < 200:
        warnings.append(f"Only {len(posts)} posts: content and anomaly findings are weak below a few hundred posts.")
    if len(comments) < 2000:
        warnings.append(f"{len(comments):,} comments analyzed (the project target is about 2,000 or more).")

    result = AudienceResult(posts=posts, comments=comments, funnel=funnel, by_content=by_content,
                            by_platform_content=by_pc, by_campaign=by_campaign, differ=differ, needs=needs,
                            segments=seg_df, segment_info=seg_info, archetypes=arche, anomalies=flagged,
                            recommendations=recs, provenance=prov, warnings=warnings)
    result.evaluation = evaluate_against_truth(result, comments_raw, posts_raw)
    return result


def evaluate_against_truth(res: AudienceResult, comments_raw: pd.DataFrame, posts_raw: pd.DataFrame) -> dict:
    """Only for simulated data that carries hidden ground truth (_true_topic, _planted_anomaly)."""
    out: dict = {}
    if "_true_topic" in comments_raw and "comment_id" in comments_raw:
        truth = comments_raw.set_index(comments_raw["comment_id"].astype(str))["_true_topic"]
        c = res.comments[res.comments["comment_id"].isin(truth.index)]
        t = c["comment_id"].map(truth)
        out["topic_agreement"] = round(float((c["topic"] == t).mean()), 4)
        out["topic_agreement_by_topic"] = {k: round(float((c.loc[t == k, "topic"] == k).mean()), 3)
                                           for k in sorted(set(t))}
        if "segment_id" in c:
            out["segment_vs_true_topic_ARI"] = round(float(adjusted_rand_score(t, c["segment_id"])), 4)
        if "_true_language" in comments_raw:
            tl = comments_raw.set_index(comments_raw["comment_id"].astype(str))["_true_language"]
            out["language_agreement"] = round(float((c["language_type"] == c["comment_id"].map(tl)).mean()), 4)
    if "_planted_anomaly" in posts_raw:
        planted = set(posts_raw.loc[posts_raw["_planted_anomaly"].astype(str) != "", "post_id"])
        eligible = set(res.posts.loc[(res.posts["days_since_post"].isna()) | (res.posts["days_since_post"] >= content.MIN_AGE_DAYS), "post_id"])
        planted &= eligible
        found = set(res.anomalies["post_id"])
        tp = len(planted & found)
        out["planted_anomalies"] = len(planted)
        out["anomaly_recall"] = round(tp / len(planted), 3) if planted else float("nan")
        out["anomaly_precision"] = round(tp / len(found), 3) if found else float("nan")
        out["flagged_posts"] = len(found)
    return out


def exports(res: AudienceResult) -> dict[str, pd.DataFrame]:
    """Frames for CSV export (no hidden ground-truth columns, no commenter identities)."""
    drop = [c for c in res.posts.columns if c.startswith("_")]
    cdrop = [c for c in res.comments.columns if c.startswith("_")]
    return {
        "audience_posts.csv": res.posts.drop(columns=drop),
        "audience_comments.csv": res.comments.drop(columns=cdrop),
        "audience_segments.csv": res.segments,
        "audience_needs.csv": res.needs,
        "content_performance.csv": res.by_content,
        "unusual_posts.csv": res.anomalies,
        "marketing_actions.csv": res.recommendations,
    }
