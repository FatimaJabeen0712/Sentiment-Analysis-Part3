"""Project Hub - links Part 1 (customer voice), Part 3 (audience & campaigns) and Part 2 (budget) into one action brief."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ads.optimizer import recommend as recommend_budget  # noqa: E402
from src.ads.simulate import simulate as simulate_ads  # noqa: E402
from src.audience import pipeline as aud_pipeline  # noqa: E402
from src.audience.simulate import simulate_posts_and_comments  # noqa: E402
from src.reporting import to_csv_bytes  # noqa: E402
from src.ui import apply_theme, banner, plot_chart, sidebar_nav, style_fig  # noqa: E402

st.set_page_config(page_title="Project Hub", layout="wide")
sidebar_nav()
apply_theme("Project Hub: one action brief from all three parts",
            "Customer voice (Part 1) + audience & content (Part 3) + budget (Part 2) -> what the brand should do next.",
            tag="COMPLETE PROJECT")
S = st.session_state


@st.cache_data(show_spinner="Running the Part 3 demo...", max_entries=2)
def demo_audience():
    p, c = simulate_posts_and_comments(1200, 42)
    return aud_pipeline.run_audience_analysis(p, c)


@st.cache_data(show_spinner="Running the Part 2 optimizer...", max_entries=4)
def budget(budget_rs: float, incremental: bool):
    return recommend_budget(simulate_ads(100_000), budget_rs, incremental=incremental)


st.markdown("""
<div class="pj-flow">
<div class="pj-step"><b>1 · Collect &amp; understand comments</b><br><span>Part 1 finds YouTube videos, saves comments to CSV, scores sentiment and 12 aspects.</span></div>
<div class="pj-step"><b>2 · Segment the audience</b><br><span>Part 3 reuses the same cleaned comments, groups needs, compares content and flags unusual posts.</span></div>
<div class="pj-step"><b>3 · Decide the budget</b><br><span>Part 2 turns platform performance into a split of the next Rs 1 million.</span></div>
<div class="pj-step"><b>4 · Act</b><br><span>This page merges the findings into one list of actions with evidence and confidence.</span></div>
</div>
""", unsafe_allow_html=True)

p1 = S.get("bundle")
p3 = S.get("aud_result")
using_demo3 = p3 is None
if using_demo3:
    p3 = demo_audience()

c1, c2, c3 = st.columns(3)
c1.metric("Part 1 · Customer voice", f"{p1.n:,} comments" if p1 is not None else "not run")
c2.metric("Part 3 · Audience", f"{len(p3.comments):,} comments" + (" (demo)" if using_demo3 else ""))
c3.metric("Part 2 · Budget", "simulated spend data")
if using_demo3:
    banner("Part 3 shows its SIMULATED demo because you have not run it on your data yet. Run Part 3 (page in the sidebar) to use real comments.")
if p1 is None:
    st.info("Part 1 has not been run in this session. Run it in Part 1 (Collect Data tab) and come back to add customer-voice findings.")
elif p1.contains_demo:
    banner("Part 1 result contains DEMO (synthetic) comments.")

with st.sidebar:
    budget_rs = st.number_input("Budget to allocate (Rs)", 100_000, 100_000_000, 1_000_000, step=100_000)
    extra = st.radio("This budget is...", ["Extra money on top of current spend", "The whole monthly budget"]).startswith("Extra")
rec2 = budget(float(budget_rs), extra)

tabs = st.tabs(["Action brief", "How the parts connect", "Platform view", "Data lineage"])

with tabs[0]:
    rows = []
    if p1 is not None and len(p1.recommendations):
        for _, r in p1.recommendations.head(6).iterrows():
            rows.append({"part": "1 · Customer voice", "area": "Product / service", "action": r.get("action", r.get("recommendation", "")),
                         "evidence": f"{r.get('evidence_count', '')} supporting comments", "confidence": r.get("confidence", "")})
    for _, r in p3.recommendations.iterrows():
        rows.append({"part": "3 · Audience" + (" (demo)" if using_demo3 else ""), "area": r["area"], "action": r["recommendation"],
                     "evidence": r["evidence"], "confidence": r["confidence"]})
    top = rec2.table.head(3)
    split = ", ".join(f"{r.platform} {r.share_pct:.0f}%" for r in rec2.table.itertuples())
    rows.append({"part": "2 · Budget (simulated)", "area": "Budget", "action": f"Allocate Rs {budget_rs:,.0f}: {split}. "
                 f"Put most behind {', '.join(top['platform'])}, then test the shift with a small slice before committing.",
                 "evidence": "fitted response curves on 100,000 simulated ad rows",
                 "confidence": "illustrative"})
    brief = pd.DataFrame(rows)
    st.subheader("Combined action brief")
    st.caption("Every row keeps its source part, evidence and confidence. Part 2 and the Part 3 demo use simulated data.")
    area_filter = st.multiselect("Areas", sorted(brief["area"].unique()), default=sorted(brief["area"].unique()))
    st.dataframe(brief[brief["area"].isin(area_filter)], hide_index=True, width="stretch")
    st.download_button("Download action brief (CSV)", to_csv_bytes(brief), "action_brief.csv", "text/csv")

with tabs[1]:
    st.subheader("Where the parts meet")
    needs = p3.needs.head(5)[["need", "comments", "share_pct", "negative_pct"]]
    st.markdown("**Part 3 audience needs** (what people ask for most)")
    st.dataframe(needs, hide_index=True, width="stretch")
    if p1 is not None and len(p1.aspect_summary):
        st.markdown("**Part 1 aspects** (what they say about the product, same comments pipeline)")
        a = p1.aspect_summary.sort_values("mentions", ascending=False).head(6)[["aspect", "mentions", "pos_pct", "neg_pct"]]
        st.dataframe(a, hide_index=True, width="stretch")
        st.caption("Part 3 'price' needs and Part 1 'price' / 'value for money' aspects describe the same conversation from two angles: "
                   "Part 1 measures how people feel, Part 3 measures what they want to know or do next.")
    else:
        st.info("Run Part 1 to compare its aspect sentiment with the Part 3 needs.")
    st.markdown("**Connection rules used in this project**")
    st.markdown("""
- Part 3 uses Part 1's text cleaner, spam filter and sentiment analyzer, so the same comment is scored the same way in both.
- Part 1's agent output (`raw_comments.csv`) is Part 3's input: each YouTube video becomes a post.
- Part 3's findings on **content** and **audience needs** tell the team *what* to publish and answer; Part 2 tells it *where* to spend.
- Part 2 and the Part 3 demo run on separately simulated data, so their numbers are not linked to each other. With real data, join them on campaign and platform.
""")

with tabs[2]:
    st.subheader("Platform view: attention (Part 3) next to spend return (Part 2)")
    pc = p3.by_platform_content
    if len(pc):
        att = (p3.posts.assign(er=p3.posts["engagement_rate"]).groupby("platform")["er"].median().rename("median_engagement_rate").reset_index())
    else:
        att = pd.DataFrame(columns=["platform", "median_engagement_rate"])
    b = rec2.table[["platform", "share_pct", "current_roas", "marginal_roas_after"]]
    joined = att.merge(b, on="platform", how="inner")
    if joined.empty:
        st.info("No platform appears in both Part 3 and Part 2 results.")
    else:
        st.dataframe(joined.assign(median_engagement_rate=joined["median_engagement_rate"].map(lambda x: f"{100 * x:.1f}%"),
                                   current_roas=joined["current_roas"].round(2), marginal_roas_after=joined["marginal_roas_after"].round(2)),
                     hide_index=True, width="stretch")
        plot_chart((px.bar(b, x="platform", y="share_pct", title="Part 2: recommended share of the budget (%)")), width="stretch")
        st.caption("Illustrative only: the two tables come from separately simulated data. Part 2 also covers Google and YouTube, "
                   "which Part 3 does not collect (no comments or posts from them in this project).")

with tabs[3]:
    st.subheader("Data lineage")
    st.dataframe(pd.DataFrame([
        {"part": "1", "input": "YouTube Data API v3 (or CSV / demo)", "main steps": "clean, sentiment, 12 aspects, product mentions",
         "output": "raw_comments.csv, analyzed_comments.csv, recommendations", "feeds": "Part 3 comments table"},
        {"part": "3", "input": "Table A posts + Table B comments (Part 1 CSV, upload or simulated)", "main steps": "topics, intent, segments, content, unusual posts",
         "output": "segments, content ranking, actions", "feeds": "Hub brief"},
        {"part": "2", "input": "simulated ad records (10k to 10M)", "main steps": "KPIs, Pandas/MapReduce/Spark, response curves",
         "output": "budget split", "feeds": "Hub brief"}]), hide_index=True, width="stretch")
