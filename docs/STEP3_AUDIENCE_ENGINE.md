# Part 3 - Audience Segmentation & Campaign Opportunity Engine

Business question: a Lahore-based fashion brand runs Instagram, TikTok and Facebook campaigns for its
11.11 / 12.12 collection and receives thousands of comments ("Price please?", "Delivery kitne din mein
hogi?"). Which content works, what does the audience want, which posts are unusual, which audience groups
matter, and what should marketing do next? This is not positive/negative classification (that is Part 1).

## The five questions and where they are answered
| Question | Method | Page tab |
|---|---|---|
| Which content attracts attention? | Engagement rate per post (likes + comments + shares) / views, compared by content type, platform and campaign with bootstrap confidence intervals and a Kruskal-Wallis test | Content performance |
| What does the audience want? | Rule-based topic tagging (price inquiry, price complaint, discount, delivery, order status, availability, sizing, fabric, purchase interest, praise) in English, Roman Urdu and Urdu script, plus purchase-intent level | Audience needs & segments |
| Which patterns are unusual? | Robust z-scores (median / MAD) against comparable posts (same platform and content type, at least 8 peers) plus complaint-share rule; posts under 2 days old are skipped | Unusual posts |
| Which audience needs should the brand address? | TF-IDF + k-means clustering of comments (k chosen by silhouette, 6-14), segments named from their dominant topic, with purity, sentiment and language mix; post archetypes from the topic mix of each post | Audience needs & segments |
| What should marketing do next? | Rules turn the findings into content, promotion, customer-service and campaign-test actions; each has evidence, a confidence level, example comment IDs and a "could be wrong because" line | Marketing actions |

## Data (Tables A and B from the brief)
* **Posts (Table A):** post_id, brand, platform, campaign, content_type, product_category, likes,
  comments_count, shares, views, days_since_post, plus data_origin, source_url, collected_at.
* **Comments (Table B):** post_id, comment_text, then language_type, topic, sentiment, purchase_intent
  (added by the engine).
* Three sources: **simulated** demo (2,400 posts, about 16,900 comments, brands "A/B/C", always labeled
  SIMULATED), **Part 1 comments** (real YouTube data through the bridge), or your **own two CSVs**.
* Provenance is kept per row (`data_origin`), so observed and simulated records are never mixed silently.

## Linking the parts (complete project)
1. **Part 1** finds YouTube videos for a brand, collects comments and saves `raw_comments.csv`.
2. **Bridge** (`src/audience/bridge.py`) turns that CSV into Table A and Table B. Only real numbers are
   used: without video statistics, likes and views stay empty and nothing is invented, so the content and
   anomaly analyses simply have less to work with.
3. **Part 3** segments those comments and builds the marketing actions.
4. **Part 2** turns platform performance into a recommended split of the next Rs 1 million.
5. **Project Hub** merges the three into one action brief with the source part, evidence and confidence
   of each row, a platform view and a data-lineage tab.

## Checks (simulated data with planted ground truth)
Run on 2,400 posts / 16,857 comments (looking at the output is not a validation, so the
simulator hides the true topic, language and 69 planted unusual posts, and the engine is scored against them):

| Check | Result |
|---|---|
| Topic agreement with the hidden true topic | 93.9% |
| Language-type agreement | 83.9% |
| Segments vs true topic (adjusted Rand index) | 0.84 |
| Planted unusual posts recovered | recall 88.4%, precision 88.4% |
| Content types distinguishable (Kruskal-Wallis) | yes, p < 0.001 |

**Read these honestly:** the simulator and the topic rules were written by the same author, so agreement
is optimistic. It shows the engine behaves as designed, not that it will score 94% on real comments.

## Run on real Chanel comments (Part 1 data, through the bridge)
2,094 comments from 15 videos -> 1,862 after removing spam and same-post duplicates. Segments found:
General chatter 67%, Admirers 13%, Mixed needs 7%, Fabric & quality 6%, Price askers 5%,
Ready-to-buy 3%. With only 15 videos and no like or view counts the engine reports no content-performance
ranking and flags only 2 unusual posts, and says so, rather than guessing.

## Limits
* Rule-based topics: slang, sarcasm and unseen spellings are missed; Roman Urdu spelling varies a lot.
* Engagement rate is not sales; the actions are hypotheses to A/B test, not forecasts.
* Public Instagram, TikTok and Facebook comments are not downloadable without official access, so use
  permitted exports; the demo data is the stand-in. Reuse of any third-party dataset (for example the
  10,602-row fashion sentiment paper or the Open Data Pakistan order data) needs its own permission check.
* Few posts (under about 200) make content and anomaly findings weak; the page warns about this.

## Run it
```
streamlit run app.py          # sidebar: Part 3 - Audience & Campaigns, then Project Hub
```
