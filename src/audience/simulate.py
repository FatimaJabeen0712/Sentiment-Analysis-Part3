"""Simulated Pakistani women's-lawn social-media data (Table A posts + Table B comments).

ALL OF THIS IS SIMULATED. Brands are "Brand A/B/C" (not the real Sapphire, Khaadi or Gul Ahmed) and no
number is an observation. Built-in ground truth, used only to *test* the engine:

* content type changes reach and engagement (reels and influencer posts do better, plain product photos worse);
* sale posts attract discount/price comments, Eid-campaign posts attract delivery/availability comments,
  styling videos attract fabric/sizing questions;
* about 3% of posts are planted anomalies: ``viral`` (about 8x engagement), ``complaint_storm`` (many
  comments, mostly order-status complaints) and ``dead`` (almost no engagement);
* about 12% of comments get typos/abbreviations so the rule-based classifier is not trivially perfect.

Hidden columns start with ``_`` (``_true_topic``, ``_planted_anomaly``); the analysis never reads them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BRANDS = ["Brand A", "Brand B", "Brand C"]
PLATFORMS = ["Instagram", "TikTok", "Facebook"]
PLATFORM_P = [0.45, 0.30, 0.25]
CONTENT = {  # type: (probability, reach multiplier, like rate, comment rate, share rate)
    "Reel": (0.22, 1.7, 0.050, 0.0040, 0.0060),
    "Product photo": (0.24, 0.6, 0.032, 0.0030, 0.0015),
    "Influencer post": (0.16, 1.3, 0.055, 0.0050, 0.0040),
    "Styling video": (0.14, 1.0, 0.040, 0.0035, 0.0090),
    "Sale announcement": (0.14, 1.1, 0.028, 0.0120, 0.0030),
    "Eid campaign": (0.10, 1.4, 0.045, 0.0080, 0.0045),
}
CAMPAIGNS = ["11.11 Collection", "12.12 Collection", "Eid Collection", "Everyday Lawn"]
PLATFORM_REACH = {"Instagram": 24_000, "TikTok": 38_000, "Facebook": 15_000}
BRAND_SCALE = {"Brand A": 1.4, "Brand B": 1.0, "Brand C": 0.7}
TOPIC_MIX = {  # content type -> topic weights
    "Reel": {"praise": 30, "purchase_interest": 14, "price_inquiry": 18, "availability": 8, "discount": 6, "fabric": 8, "sizing": 4, "delivery": 4, "price_complaint": 3, "other": 5},
    "Product photo": {"price_inquiry": 28, "fabric": 14, "sizing": 8, "availability": 14, "praise": 16, "discount": 6, "purchase_interest": 8, "other": 6},
    "Influencer post": {"praise": 34, "price_inquiry": 16, "discount": 12, "purchase_interest": 10, "availability": 7, "price_complaint": 5, "fabric": 5, "other": 11},
    "Styling video": {"fabric": 24, "sizing": 18, "praise": 20, "price_inquiry": 12, "availability": 8, "purchase_interest": 8, "other": 10},
    "Sale announcement": {"discount": 30, "price_inquiry": 16, "price_complaint": 9, "delivery": 10, "availability": 10, "purchase_interest": 9, "order_status": 7, "praise": 5, "other": 4},
    "Eid campaign": {"delivery": 22, "availability": 18, "price_inquiry": 12, "purchase_interest": 14, "order_status": 8, "praise": 12, "discount": 6, "sizing": 4, "other": 4},
}
CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Peshawar", "Quetta", "Multan", "Faisalabad"]
CITY_P = [0.26, 0.24, 0.12, 0.08, 0.07, 0.04, 0.10, 0.09]
LANG_P = {"English": 0.38, "Roman Urdu": 0.38, "Mixed": 0.14, "Urdu": 0.10}

T = {  # topic -> language -> templates ({city} filled in)
    "price_inquiry": {
        "English": ["Price please?", "How much is this?", "What is the price of this suit?", "Price?", "Can you tell me the price", "price plz", "What's the cost of this 3 piece?"],
        "Roman Urdu": ["Price kya hai?", "Kitne ka hai ye?", "Qeemat bata dein", "Rate kya hai is suit ka", "Kitna price hai", "Price btayen please", "Is ka rate kitna hai?"],
        "Mixed": ["Price bata dein plz", "How much hai ye suit?", "Rate kya hai, any idea?", "Price kitna hai 3 piece ka"],
        "Urdu": ["قیمت کیا ہے؟", "اس کا ریٹ بتائیں", "قیمت بتا دیں"]},
    "price_complaint": {
        "English": ["Way too expensive for lawn", "Very expensive, not worth it", "Overpriced honestly", "Price is too high this year"],
        "Roman Urdu": ["Price bohat zyada hai.", "Bohat mehnga hai ye lawn", "Itna mehnga suit kon lega", "Rate zyada hai bohat", "Mehngai ki had hai"],
        "Mixed": ["Price too high hai yaar", "Itna expensive lawn, no way", "Bohat costly hai"],
        "Urdu": ["بہت مہنگا ہے", "قیمت بہت زیادہ ہے"]},
    "discount": {
        "English": ["Any discount on this suit?", "When is the sale?", "Is there any offer or promo code?", "Please give discount", "Any bundle deal for 2 suits?", "Will this go on sale for 12.12?"],
        "Roman Urdu": ["Discount milega kya?", "Sale kab lag rahi hai?", "Price kam karo please", "Koi offer hai is pe?", "11.11 pe sasta milega?", "Discount code do na"],
        "Mixed": ["Any discount hai is pe?", "Sale kab ayegi, please tell", "Offer hai koi aj?"],
        "Urdu": ["ڈسکاؤنٹ ملے گا؟", "سیل کب لگے گی"]},
    "fabric": {
        "English": ["What fabric is this?", "Is the quality good? Does the color fade?", "Is it pure cotton lawn?", "Is the fabric soft or does it shrink?", "How is the stitching quality?"],
        "Roman Urdu": ["Kapra kaisa hai?", "Lawn ka kapra soft hai kya?", "Quality kaisi hai is ki?", "Rang nikalta hai kya?", "Kapray ki quality achi hai?"],
        "Mixed": ["Fabric kaisa hai, lawn hai ya cotton?", "Quality theek hai ya fade hota hai?"],
        "Urdu": ["کپڑا کیسا ہے؟", "کوالٹی کیسی ہے"]},
    "sizing": {
        "English": ["What sizes do you have?", "Is there a size chart?", "I am XL, will it fit?", "What is the length of the shirt?", "Is it available in small?"],
        "Roman Urdu": ["Size kya hai is ka?", "Shirt ki lambai kitni hai?", "XL size milega?", "Size chart bhej dein"],
        "Mixed": ["Size chart please, fitting kaisi hai?", "Medium size available hai?"],
        "Urdu": ["سائز کتنا ہے؟", "سائز چارٹ بھیجیں"]},
    "delivery": {
        "English": ["How many days for delivery?", "Do you deliver to {city}?", "What are the delivery charges?", "Is cash on delivery available?", "When will it be delivered?"],
        "Roman Urdu": ["Delivery kitne din mein hogi?", "{city} mein delivery kab tak hogi?", "Delivery charges kitne hain?", "COD milta hai kya?", "Kab tak mil jayega parcel?"],
        "Mixed": ["Delivery time kitna hai?", "Delivery charges kya hain, COD hai?"],
        "Urdu": ["ڈیلیوری کتنے دن میں ہوگی؟", "ڈیلیوری چارجز کتنے ہیں"]},
    "order_status": {
        "English": ["Ordered last week, still waiting.", "I ordered 10 days ago and have not received anything", "Tracking shows no update, where is my order?", "Received the wrong size, need an exchange", "Order not delivered yet, please reply", "Haven't received my parcel, this is bad service"],
        "Roman Urdu": ["Order kiya tha abhi tak nahi aya", "Parcel abhi tak nahi mila, koi reply nahi", "Mera order late hai, tracking update nahi", "Wrong size bhej diya, refund karo", "Do hafte se order ka wait hai"],
        "Mixed": ["Order kiya last week, still waiting", "Tracking pe update nahi, order not received"],
        "Urdu": ["آرڈر ابھی تک نہیں ملا", "پارسل نہیں پہنچا"]},
    "availability": {
        "English": ["Is this available in {city}?", "Is it in stock?", "Where can I buy this in {city}?", "Do you ship to UK?", "Is this available in your {city} store?", "Sold out already? Will you restock?"],
        "Roman Urdu": ["{city} mein available hai?", "Ye stock mein hai kya?", "{city} mein kahan milega?", "Store kahan hai aap ka {city} mein?", "Abhi available hai ye suit?"],
        "Mixed": ["Available hai in {city}?", "Stock hai ya sold out?"],
        "Urdu": ["کیا یہ کراچی میں دستیاب ہے؟", "کہاں سے ملے گا"]},
    "purchase_interest": {
        "English": ["Inbox me details please", "I want to order this", "DM sent, please check", "Please send details on WhatsApp", "How can I order this suit?", "I'll take 2 of these"],
        "Roman Urdu": ["Inbox me details send kar dein.", "Mujhe ye chahiye, order kaise karun", "Order karna hai, number bhej dein", "Ye suit lena hai", "WhatsApp number do na"],
        "Mixed": ["Inbox me price aur details bhej do", "Order karna hai, how to?"],
        "Urdu": ["ان باکس میں تفصیل بھیجیں", "مجھے یہ چاہیے"]},
    "praise": {
        "English": ["Very beautiful dress! ❤", "Gorgeous colors 😍", "Love this print", "Stunning!", "So pretty, perfect for Eid", "Amazing collection"],
        "Roman Urdu": ["Bohat khubsurat ❤", "Zabardast design", "Mashallah bohat piyari", "Kamaal ka collection hai", "Achi lag rahi hai"],
        "Mixed": ["Beautiful hai ye, mashallah", "Wow bohat acha design"],
        "Urdu": ["ماشاء اللہ بہت خوبصورت", "بہت پیارا ڈیزائن"]},
    "other": {
        "English": ["First!", "Which model is she wearing?", "What is the background music?", "Follow back please", "Who is this influencer?", "🔥🔥"],
        "Roman Urdu": ["Ye model kon hai?", "Song ka naam kya hai?", "Follow back kar do"],
        "Mixed": ["Which song hai ye?", "Model ka naam batao"],
        "Urdu": ["یہ کون ہے؟", "گانا کون سا ہے"]},
}
SPAM = ["Check my page for free followers http://spam.example/promo", "WIN iPhone click here http://spam.example/win"]


def _typo(text: str, rng: np.random.Generator) -> str:
    words = text.split()
    if not words:
        return text
    i = int(rng.integers(0, len(words)))
    w = words[i]
    mode = int(rng.integers(0, 3))
    if len(w) > 4 and mode == 0:       # drop vowels
        w = "".join(c for c in w if c.lower() not in "aeiou") or w
    elif len(w) > 3 and mode == 1:     # swap two letters
        j = int(rng.integers(0, len(w) - 1))
        w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    else:                              # stretch
        w = w + w[-1] * 2
    words[i] = w
    return " ".join(words)


def _comment(topic: str, rng: np.random.Generator) -> tuple[str, str]:
    lang = rng.choice(list(LANG_P), p=list(LANG_P.values()))
    bank = T[topic][lang]
    text = str(bank[int(rng.integers(0, len(bank)))])
    if "{city}" in text:
        text = text.replace("{city}", str(rng.choice(CITIES, p=CITY_P)))
    if rng.random() < 0.12:
        text = _typo(text, rng)
    if rng.random() < 0.15 and lang != "Urdu":
        text += str(rng.choice([" 🙏", " ❤", "!!", " plz", " ???"]))
    return text, lang


def simulate_posts_and_comments(n_posts: int = 2400, seed: int = 42,
                                collected_at: str = "2026-11-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_posts < 10:
        raise ValueError("n_posts must be at least 10")
    rng = np.random.default_rng(seed)
    ctypes = list(CONTENT)
    ct = rng.choice(ctypes, size=n_posts, p=[CONTENT[c][0] for c in ctypes])
    plat = rng.choice(PLATFORMS, size=n_posts, p=PLATFORM_P)
    brand = rng.choice(BRANDS, size=n_posts, p=[0.4, 0.35, 0.25])
    camp = rng.choice(CAMPAIGNS, size=n_posts, p=[0.28, 0.27, 0.15, 0.30])
    days = rng.integers(1, 61, size=n_posts)

    reach = (np.array([PLATFORM_REACH[p] for p in plat]) * np.array([BRAND_SCALE[b] for b in brand])
             * np.array([CONTENT[c][1] for c in ct]) * rng.lognormal(0, 0.55, n_posts))
    reach *= 1 - np.exp(-days / 4.0)                      # reach accumulates then saturates
    anomaly = np.full(n_posts, "", dtype=object)
    pick = rng.permutation(n_posts)[: int(0.03 * n_posts)]
    kinds = np.array(["viral", "complaint_storm", "dead"])[np.arange(len(pick)) % 3]
    anomaly[pick] = kinds
    boost = np.ones(n_posts)
    boost[pick[kinds == "viral"]] = 8.0
    boost[pick[kinds == "dead"]] = 0.05
    views = np.maximum(reach * np.where(anomaly == "viral", 3.0, 1.0) * np.where(anomaly == "dead", 0.1, 1.0), 50).round()

    like_r = np.array([CONTENT[c][2] for c in ct]) * rng.lognormal(0, 0.18, n_posts) * np.where(anomaly == "viral", 2.0, 1.0)
    com_r = np.array([CONTENT[c][3] for c in ct]) * rng.lognormal(0, 0.25, n_posts)
    com_r = com_r * np.where(anomaly == "complaint_storm", 6.0, 1.0) * np.where(anomaly == "viral", 2.0, 1.0) * np.where(anomaly == "dead", 0.1, 1.0)
    shr_r = np.array([CONTENT[c][4] for c in ct]) * rng.lognormal(0, 0.3, n_posts) * np.where(anomaly == "viral", 2.0, 1.0)
    likes = rng.poisson(views * like_r * np.where(anomaly == "dead", 0.1, 1.0))
    comments = rng.poisson(views * com_r)
    shares = rng.poisson(views * shr_r)
    views_f = views.astype(float)
    hide_views = (plat == "Facebook") & (ct == "Product photo") | (rng.random(n_posts) < 0.04)   # "if available"
    views_f[hide_views] = np.nan
    shares_f = shares.astype(float)
    shares_f[(plat == "TikTok") & (rng.random(n_posts) < 0.10)] = np.nan

    post_id = [f"P{i + 1:05d}" for i in range(n_posts)]
    posts = pd.DataFrame({
        "post_id": post_id, "brand": brand, "platform": plat, "campaign": camp, "content_type": ct,
        "product_category": "Women's lawn suit", "likes": likes, "comments_count": comments,
        "shares": shares_f, "views": views_f, "days_since_post": days,
        "data_origin": "simulated", "source_url": "", "collected_at": collected_at,
        "_planted_anomaly": anomaly})

    rows = []
    cid = 0
    for i in range(n_posts):
        k = int(min(comments[i], rng.integers(4, 11) if anomaly[i] != "complaint_storm" else rng.integers(12, 25)))
        if k == 0:
            continue
        mix = dict(TOPIC_MIX[ct[i]])
        if anomaly[i] == "complaint_storm":
            mix = {"order_status": 60, "delivery": 15, "price_complaint": 5, "other": 5, "price_inquiry": 5}
        names = list(mix)
        p = np.array([mix[n] for n in names], dtype=float)
        for topic in rng.choice(names, size=k, p=p / p.sum()):
            cid += 1
            if rng.random() < 0.012:
                text, lang, topic = str(rng.choice(SPAM)), "English", "other"
            else:
                text, lang = _comment(str(topic), rng)
            rows.append((f"C{cid:06d}", post_id[i], text, lang, topic))
    comm = pd.DataFrame(rows, columns=["comment_id", "post_id", "comment_text", "_true_language", "_true_topic"])
    comm["data_origin"] = "simulated"
    comm["collected_at"] = collected_at
    return posts, comm
