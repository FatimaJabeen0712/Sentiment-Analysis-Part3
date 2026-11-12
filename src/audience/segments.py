"""Discover audience segments: cluster comments by what they say, then name each cluster by the
topic that dominates it. Also groups *posts* by the mix of topics in their comments.

Method: each comment is described by (1) TF-IDF over words and word pairs (filler words removed, reduced to
60 dimensions with SVD) and (2) which topic keywords it contains (from topics.py, weighted at 1.2). k-means
then groups them, k picked by silhouette between 6 and 14. Comments in
different languages that say the same thing can land in different clusters, so a topic may appear as
more than one segment (the language is shown in the name). A cluster's name comes from its dominant rule topic
only if that topic covers at least 40% of the cluster ("purity"); otherwise it is called "Mixed".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize as l2_normalize

from .topics import normalize

SEGMENT_NAMES = {
    "price_inquiry": "Price askers", "price_complaint": "Price-resistant shoppers",
    "discount": "Discount-focused commenters", "delivery": "Delivery-concerned shoppers",
    "order_status": "Customers waiting on an order", "availability": "Availability & location seekers",
    "sizing": "Size & fit seekers", "fabric": "Fabric & quality evaluators",
    "purchase_interest": "Ready-to-buy (inbox / details)", "praise": "Admirers", "other": "General chatter",
}
STOP = sorted(set(ENGLISH_STOP_WORDS) | {"hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "ye", "yeh", "wo", "woh", "hy", "kar",
                                          "dein", "den", "plz", "pls", "please", "bhi", "tha", "thi", "aap", "ji"})
TOPIC_WEIGHT = 1.2
MIN_COMMENTS = 60
PURITY_MIN = 0.40


def cluster_comments(comments: pd.DataFrame, k_range=range(6, 15), seed: int = 0):
    """Returns (labels Series aligned to comments.index, info dict, vectorizer, model)."""
    if len(comments) < MIN_COMMENTS:
        raise ValueError(f"Need at least {MIN_COMMENTS} comments to find segments (have {len(comments)}).")
    text = comments["comment_text"].map(normalize)
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words=STOP,
                          token_pattern=r"(?u)[\w\u0600-\u06FF]{2,}", max_features=8000)
    Xs = vec.fit_transform(text)
    if Xs.shape[1] < 10:
        raise ValueError("Comments are too repetitive or too short to find segments.")
    svd = TruncatedSVD(n_components=min(60, Xs.shape[1] - 1), random_state=seed)
    from .topics import PRIORITY
    multi = np.array([[t in str(v).split("|") for t in PRIORITY] for v in comments["topics_all"]], dtype=float)
    X = np.hstack([l2_normalize(svd.fit_transform(Xs)), TOPIC_WEIGHT * l2_normalize(multi)])
    rng = np.random.default_rng(seed)
    sample = rng.choice(X.shape[0], size=min(3000, X.shape[0]), replace=False)
    best = None
    scores = {}
    for k in k_range:
        if k >= X.shape[0]:
            break
        km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(X)
        sil = float(silhouette_score(X[sample], km.labels_[sample]))
        scores[k] = round(sil, 4)
        if best is None or sil > best[0]:
            best = (sil, k, km)
    sil, k, km = best
    model = type("Model", (), {})()
    model.labels_ = km.labels_
    # centroids back in word space (mean tf-idf of members) so top terms are readable words
    model.cluster_centers_ = np.vstack([np.asarray(Xs[km.labels_ == c].mean(axis=0)).ravel() for c in range(k)])
    return (pd.Series(km.labels_, index=comments.index, name="segment_id"),
            {"k": int(k), "silhouette": round(sil, 4), "silhouette_by_k": scores}, vec, model)


def describe_segments(comments: pd.DataFrame, labels: pd.Series, vec, km, posts: pd.DataFrame | None = None) -> pd.DataFrame:
    terms = np.array(vec.get_feature_names_out())
    rows = []
    total = len(comments)
    for sid in sorted(labels.unique()):
        g = comments[labels == sid]
        share = g["topic"].value_counts(normalize=True)
        dom, purity = share.index[0], float(share.iloc[0])
        lang = g["language_type"].value_counts(normalize=True)
        name = SEGMENT_NAMES[dom] if purity >= PURITY_MIN else "Mixed needs"
        name += f" ({lang.index[0]})" if lang.index[0] != "English" else ""
        top_terms = [t.strip() for t in terms[np.argsort(km.cluster_centers_[sid])[::-1][:6]]]
        row = {"segment_id": int(sid), "segment": name, "comments": len(g), "share_pct": round(100 * len(g) / total, 1),
               "dominant_topic": dom, "purity_pct": round(100 * purity, 1),
               "avg_sentiment": round(float(g["sentiment_score"].mean()), 3),
               "negative_pct": round(100 * float((g["sentiment"] == "negative").mean()), 1),
               "main_language": lang.index[0], "language_share_pct": round(100 * float(lang.iloc[0]), 1),
               "ready_or_strong_pct": round(100 * float(g["purchase_intent"].isin(["Ready to buy", "Strong interest"]).mean()), 1),
               "top_terms": ", ".join(top_terms),
               "example_comments": " | ".join(g["comment_text"].sample(min(3, len(g)), random_state=0).str.slice(0, 80)),
               "example_comment_ids": ",".join(g["comment_id"].head(3).astype(str))}
        if posts is not None and "post_id" in g:
            pt = g.merge(posts[["post_id", "platform", "content_type"]], on="post_id", how="left")
            row["top_platform"] = pt["platform"].mode().iat[0] if pt["platform"].notna().any() else ""
            row["top_content_type"] = pt["content_type"].mode().iat[0] if pt["content_type"].notna().any() else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values("comments", ascending=False).reset_index(drop=True)


def audience_needs(comments: pd.DataFrame) -> pd.DataFrame:
    """The rule-based view: share of comments per topic, with sentiment and purchase signal."""
    from .topics import TOPIC_LABELS
    n = len(comments)
    g = comments.groupby("topic")
    out = pd.DataFrame({
        "comments": g.size(), "share_pct": (100 * g.size() / n).round(1),
        "negative_pct": (100 * g["sentiment"].apply(lambda s: (s == "negative").mean())).round(1),
        "ready_or_strong_pct": (100 * g["purchase_intent"].apply(lambda s: s.isin(["Ready to buy", "Strong interest"]).mean())).round(1),
    }).reset_index()
    out.insert(1, "need", out["topic"].map(TOPIC_LABELS))
    return out.sort_values("comments", ascending=False).reset_index(drop=True)


def post_archetypes(posts: pd.DataFrame, comments: pd.DataFrame, min_comments: int = 5, k: int = 4, seed: int = 0) -> pd.DataFrame:
    """Group posts by the topic mix of their comments (e.g. price-question magnets, complaint-heavy posts)."""
    from .topics import PRIORITY
    counts = comments.groupby(["post_id", "topic"]).size().unstack(fill_value=0)
    counts = counts[counts.sum(axis=1) >= min_comments]
    if len(counts) < max(k * 5, 20):
        return pd.DataFrame()
    share = counts.div(counts.sum(axis=1), axis=0)
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(share.to_numpy())
    share["archetype_id"] = km.labels_
    rows = []
    for a, g in share.groupby("archetype_id"):
        mix = g.drop(columns="archetype_id").mean().sort_values(ascending=False)
        top = [t for t in mix.index if t in PRIORITY][:3]
        from .topics import TOPIC_LABELS
        rows.append({"archetype_id": int(a), "posts": len(g),
                     "pattern": " + ".join(f"{TOPIC_LABELS[t]} {100 * mix[t]:.0f}%" for t in top)})
    out = pd.DataFrame(rows).sort_values("posts", ascending=False).reset_index(drop=True)
    out.attrs["post_archetype"] = share["archetype_id"].to_dict()
    return out
