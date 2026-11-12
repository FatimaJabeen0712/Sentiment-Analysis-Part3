"""Evidence-based marketing recommendations. Every row says how many posts/comments support it, how
confident the engine is, and what could make it wrong. A rule that does not meet its minimum evidence
produces no recommendation (so an empty or short list is a normal, honest result)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .topics import cities_in

COLUMNS = ["area", "recommendation", "evidence", "confidence", "type", "example_comment_ids", "limitation"]
TEST_HINT = {
    "price_inquiry": "show the price in the first line", "discount": "lead with the offer and end date",
    "delivery": "lead with delivery time and charges", "order_status": "add a tracking / order-help line",
    "availability": "list the cities and stock status", "sizing": "add the size chart and shirt length",
    "fabric": "lead with the fabric and a close-up", "purchase_interest": "add a one-tap WhatsApp or order link",
    "price_complaint": "explain what the price includes",
}
HYP = "Hypothesis to test"
FIND = "Finding"


def _conf(n: int, high: int, mid: int) -> str:
    return "high" if n >= high else "medium" if n >= mid else "low"


def _ids(df: pd.DataFrame, n: int = 3) -> str:
    return ",".join(df["comment_id"].astype(str).head(n))


def _row(area, rec, evidence, conf, typ, ids="", limit=""):
    return dict(area=area, recommendation=rec, evidence=evidence, confidence=conf, type=typ,
                example_comment_ids=ids, limitation=limit)


def build(comments: pd.DataFrame, by_content: pd.DataFrame, differ: dict, anomalies: pd.DataFrame,
          segments: pd.DataFrame, posts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    n = len(comments)
    if n == 0:
        return pd.DataFrame(columns=COLUMNS)

    def of(topic: str) -> pd.DataFrame:
        return comments[comments["topic"] == topic]

    # ---- content
    ok = by_content[~by_content["low_sample"]] if len(by_content) else by_content
    if len(ok) >= 2 and differ.get("distinguishable"):
        best, worst = ok.iloc[0], ok.iloc[-1]
        sep = best["ci_low"] > worst["ci_high"]
        rows.append(_row("Content", f"Make more {best['content_type']} content; it has the highest median engagement rate "
                         f"({100 * best['median_engagement_rate']:.1f}%, {best['vs_all_posts']:.2f}x all posts).",
                         f"{int(best['posts'])} posts vs {int(worst['posts'])} for the weakest type; groups differ (Kruskal-Wallis p={differ['p_value']:.3g})",
                         "high" if sep else "medium", FIND, "",
                         "Engagement rate is not sales. Content types may also differ in audience, timing and budget."))
        rows.append(_row("Content", f"Review or re-test {worst['content_type']}: its median engagement rate is the lowest "
                         f"({100 * worst['median_engagement_rate']:.1f}%, {worst['vs_all_posts']:.2f}x all posts). "
                         "Try a new hook or format before cutting it.",
                         f"{int(worst['posts'])} posts", "medium" if sep else "low", HYP, "",
                         "Low engagement can still be useful for awareness or product listing."))
    elif len(ok) >= 2:
        rows.append(_row("Content", "Content types are not clearly different in engagement here, so do not shift format "
                         "budget on this data alone; run a controlled test.", f"{int(ok['posts'].sum())} posts compared, p={differ.get('p_value', float('nan')):.2g}",
                         "medium", FIND, "", "A larger or longer sample could still reveal a difference."))

    # content types whose comments are dominated by price / discount questions
    pc = comments.merge(posts[["post_id", "content_type"]], on="post_id", how="left")
    share = pc.assign(pq=pc["topic"].isin(["price_inquiry", "discount"])).groupby("content_type")["pq"].agg(["mean", "size"])
    share = share[(share["size"] >= 100) & (share["mean"] >= 0.30)].sort_values("mean", ascending=False)
    for ct, r in share.head(2).iterrows():
        ex = pc[(pc["content_type"] == ct) & pc["topic"].isin(["price_inquiry", "discount"])]
        rows.append(_row("Content", f"Put the price (or 'from PKR ...') and any offer in the caption of {ct} posts: "
                         f"{100 * r['mean']:.0f}% of their comments ask about price or discounts.",
                         f"{int(r['size'])} comments on {ct} posts, {int(r['mean'] * r['size'])} price/discount questions",
                         _conf(int(r["size"]), 400, 100), HYP, _ids(ex), "Test whether it reduces repeat questions and lifts clicks."))

    # ---- promotions
    disc = of("discount")
    if len(disc) >= 30 and len(disc) / n >= 0.08:
        rows.append(_row("Promotion", "Test a time-limited offer or 2-suit bundle for the next 11.11 / 12.12 push, with a unique promo "
                         f"code, since {100 * len(disc) / n:.0f}% of comments ask for discounts.",
                         f"{len(disc)} of {n} comments", _conf(len(disc), 200, 60), HYP, _ids(disc),
                         "Asking for a discount is not proof they would buy at that price; the code measures it."))
    pcm = of("price_complaint")
    if len(pcm) >= 15 and len(pcm) / n >= 0.03:
        rows.append(_row("Promotion", "Show what the price includes (3-piece, fabric weight, embroidery) before discounting; "
                         f"{len(pcm)} comments say the price is too high.", f"{len(pcm)} of {n} comments, {100 * (pcm['sentiment'] == 'negative').mean():.0f}% negative",
                         _conf(len(pcm), 150, 40), HYP, _ids(pcm), "Price complaints are usually a minority of all buyers."))

    # ---- customer service
    status, deliv = of("order_status"), of("delivery")
    if len(status) >= 20:
        rows.append(_row("Customer service", "Reply to order-status complaints within 24 hours and send tracking proactively; "
                         "these are existing customers at risk of churning publicly.",
                         f"{len(status)} order-status comments ({100 * len(status) / n:.1f}% of all)", _conf(len(status), 100, 40), FIND, _ids(status),
                         "Comments show perceived delay, not verified order data."))
    if len(deliv) >= 20:
        rows.append(_row("Customer service", "State delivery time and charges in captions and highlights; "
                         f"{len(deliv)} comments ask how long delivery takes or costs.", f"{len(deliv)} of {n} comments",
                         _conf(len(deliv), 120, 40), HYP, _ids(deliv), "Check real courier times before publishing a promise."))
    size = of("sizing")
    if len(size) >= 20 and len(size) / n >= 0.04:
        rows.append(_row("Customer service", "Pin a size chart and shirt length for every suit post.", f"{len(size)} sizing questions",
                         _conf(len(size), 100, 40), HYP, _ids(size), "Sizing questions may come from a small group of repeat askers."))
    av = of("availability")
    if len(av) >= 20 and len(av) / n >= 0.05:
        cities = pd.Series([c for t in av["comment_text"] for c in cities_in(t)]).value_counts().head(3)
        city_txt = ", ".join(f"{c.title()} ({v})" for c, v in cities.items()) or "no specific city"
        rows.append(_row("Customer service", f"Add stock and city availability to captions or a pinned comment; most-named places: {city_txt}.",
                         f"{len(av)} availability questions", _conf(len(av), 120, 40), HYP, _ids(av), "Check real stock levels per city first."))
    hot = comments[comments["purchase_intent"].isin(["Ready to buy", "Strong interest"])]
    if len(hot) >= 30:
        rows.append(_row("Customer service", "Set up an auto-reply with a WhatsApp/order link so 'inbox me' and 'I want this' comments convert "
                         f"({len(hot)} such comments).", f"{len(hot)} of {n} comments ({100 * len(hot) / n:.0f}%)", _conf(len(hot), 150, 50), HYP, _ids(hot),
                         "Intent shown in a comment does not guarantee a purchase."))

    # ---- anomalies
    if len(anomalies):
        vc = anomalies["finding"].value_counts()
        hi = anomalies[anomalies["finding"] == "Unusually high engagement"]
        cx = anomalies[anomalies["finding"] == "Complaint-heavy comments"]
        if len(hi):
            rows.append(_row("Campaign testing", f"Study the {len(hi)} unusually high-engagement posts (top: {', '.join(hi['post_id'].head(3))}) "
                             "for what they share (hook, influencer, offer) and replicate it in a test.",
                             f"{len(hi)} posts flagged against comparable posts", "medium", HYP, "",
                             "A viral post can be luck or paid reach; look before copying."))
        if len(cx):
            rows.append(_row("Customer service", f"Open the {len(cx)} complaint-heavy posts (top: {', '.join(cx['post_id'].head(3))}) and respond publicly; "
                             "they may point to a delivery or product problem.", f"{len(cx)} posts where at least 40% of comments are complaints",
                             "medium", FIND, "", "Flagged for investigation, not proof of a systematic issue."))
        low = vc.get("Unusually low engagement", 0)
        if low:
            rows.append(_row("Campaign testing", f"Check the {low} unusually low-engagement posts for posting time, creative or targeting errors.",
                             f"{low} posts", "low", HYP, "", "Low engagement may be intentional (e.g. catalogue posts)."))

    # ---- campaign tests built from the segments (only actionable needs, not praise or chatter)
    if len(segments):
        act = segments[~segments["dominant_topic"].isin(["praise", "other"]) & (segments["segment"] != "Mixed needs")]
        for _, s in act.head(2).iterrows():
            hint = TEST_HINT.get(s["dominant_topic"], "the main concern in the comments")
            rows.append(_row("Campaign testing", f"A/B test a caption for '{s['segment']}' ({s['share_pct']}% of comments): {hint}. "
                             "Compare reply rate and link clicks with the current caption.",
                             f"{int(s['comments'])} comments in the segment, {s['purity_pct']}% on one topic", _conf(int(s["comments"]), 800, 200), HYP,
                             s["example_comment_ids"], "Segments describe comments, not individual people; test before scaling."))
    out = pd.DataFrame(rows, columns=COLUMNS)
    order = {"high": 0, "medium": 1, "low": 2}
    return out.sort_values("confidence", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
