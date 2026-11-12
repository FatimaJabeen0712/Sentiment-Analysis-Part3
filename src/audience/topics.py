"""What is the comment about? Rule-based topics (English + Roman Urdu + a little Urdu script),
purchase intent, and a sentiment score that adds Roman Urdu cues to Part 1's analyzer.

Rules are transparent regular expressions, not a trained model: a comment gets its *primary* topic
from the first rule that matches (priority order below) and keeps every matched topic in ``topics_all``.
Known limits: negation ("not expensive"), sarcasm, misspellings and Urdu script are only partly handled.
"""
from __future__ import annotations

import re
from functools import lru_cache

from src.sentiment import SentimentAnalyzer

from .language import language_type

TOPIC_LABELS = {
    "order_status": "Order status / complaint",
    "delivery": "Delivery time",
    "price_complaint": "Price too high",
    "discount": "Discount / offer",
    "price_inquiry": "Price inquiry",
    "sizing": "Sizing & fit",
    "fabric": "Fabric & quality",
    "availability": "Availability & location",
    "purchase_interest": "Purchase interest",
    "praise": "Praise",
    "other": "Other",
}

_RULES: list[tuple[str, str]] = [
    ("order_status",
     r"(order(ed)?|order (kiya|kia|diya|place))\b.{0,50}(still|abhi tak|nahi (aya|aaya|mila|pohcha|pahuncha)|not (yet )?(received|delivered|arrived)|waiting|wait|late)"
     r"|not received|didn'?t receive|never (got|received|arrived)|haven'?t received|tracking|wrong (size|item|color|colour|suit)|received (wrong|damaged)"
     r"|parcel .{0,20}nahi|abhi tak (nahi|koi)|order .{0,20}(cancel|refund)|refund|آرڈر.{0,30}(نہیں|ابھی تک)|پارسل.{0,20}نہیں"),
    ("delivery",
     r"deliver|shipping|courier|kitne din|kitnay din|kab tak|kab milega|kab milegi|kab aye|kab aaye|\bcod\b|cash on delivery|dispatch|ship karo|delivery charges|ڈیلیوری"),
    ("price_complaint",
     r"(too|very|so|bohat|bahut|bohot|boht|bht|itna|kitna)\s+(expensive|costly|mehng\w*|mahng\w*|zyada|zyadah|high)"
     r"|overpriced|\bmehng\w*|\bmahng\w*|price\W+(is\W+)?(too\W+)?high|(price|rate|qeemat)\W+\w*\W*(bohat|bahut|bht)?\W*(zyada|zyadah|high)"
     r"|\bloot\b|not worth|waste of money|مہنگا|مہنگی"),
    ("discount",
     r"discount|\bsale\b|\boff\b|% ?off|\boffer\b|promo|coupon|\bcode\b|any deal|bargain|\bsasta\b|\bsasti\b|cheap|bundle|price (kam|less)|rate (kam|less)|kam (karo|kar do|krdo)|reduce|ڈسکاؤنٹ|سیل"),
    ("price_inquiry",
     r"\bprice\b|\bprc\b|\brate\b|\bqeemat\b|\bkimat\b|\bkitne ka\b|\bkitnay ka\b|\bkitna\b|\bkitni\b|how much|\bcost\b|\bpkr\b|\brs\.? ?\d|\bprice plz\b|\bpls price\b|قیمت|ریٹ"),
    ("sizing",
     r"\bsize\b|sizing|\bxl\b|\bxs\b|\bsmall\b|\bmedium\b|\blarge\b|length|\bfit\b|fitting|chest|measure|inches|lambai|chhota|chota|\bbara\b|\bbada\b|tight|loose|سائز"),
    ("fabric",
     r"fabric|kapra|kapray|kapda|quality|\blawn\b|cotton|chiffon|\bsilk\b|\bnet\b|organza|material|\bstuff\b|thread|fade|shrink|stitching|texture|\bsoft\b|sheer|transparent|pilling|کپڑا|کوالٹی|کپڑے"),
    ("availability",
     r"availab|in stock|\bstock\b|\bkarachi\b|\blahore\b|islamabad|peshawar|quetta|multan|faisalabad|rawalpindi|sialkot|\bcity\b|\bstore\b|outlet|branch|\bshop\b|where (can|to|do)|\bkahan\b|\bkahaan\b|mil(ta|ti|ega|egi|e) hai|ship(ping)? to|overseas|international|\buk\b|\busa\b|dubai|sold out|restock|دستیاب|کراچی|لاہور|اسلام آباد|کہاں"),
    ("purchase_interest",
     r"\binbox\b|\bdm\b|\bdms\b|details|whatsapp|\bcontact\b|\bnumber\b|order (this|kar|karna|karni|karo|chahiye|krna)|want (to )?(buy|order|this)|i('| wi)ll (take|buy|order)|need this|chahiye|lena hai|le lun|booking|book (kar|this)|interested|how (to|can i) (order|buy)|\blink\b|buy now|add to cart|send (me )?details|ان باکس|چاہیے|تفصیل"),
    ("praise",
     r"beautiful|gorgeous|lovely|love (it|this)|stunning|pretty|\bnice\b|amazing|khubsurat|khoobsurat|piyari|pyari|zabardast|mashallah|masha allah|maa shaa allah|\bwow\b|\bbest\b|perfect|\bacha\b|\bachi\b|\bachha\b|behtareen|shandar|lajawab|ماشاء|خوبصورت|پیارا|پیاری|❤|😍|🔥"),
]
_COMPILED = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in _RULES]
INQUIRY_TOPICS = frozenset({"price_inquiry", "discount", "delivery", "availability", "sizing", "fabric"})
PRIORITY = [name for name, _ in _COMPILED] + ["other"]

_READY = re.compile(r"order (this|kar|karna|karni|karo|krna)|want (to )?(buy|order)|i('| wi)ll (take|buy|order)|chahiye|lena hai|le lun|book(ing)? (kar|this)|buy now|add to cart|how (to|can i) (order|buy)", re.I)
_STRONG = re.compile(r"\binbox\b|\bdms?\b|details|whatsapp|\bcontact\b|\bnumber\b|\blink\b|interested", re.I)

_POS_RU = re.compile(r"khubsurat|khoobsurat|piyari|pyari|zabardast|behtareen|\bacha\b|\bachi\b|\bachha\b|kamaal|shandar|lajawab|mashallah|masha allah|خوبصورت|ماشاء", re.I)
_NEG_RU = re.compile(r"bekar|bakwas|ghatia|\bmehng\w*|\bmahng\w*|afsos|\bbura\b|\bburi\b|kharab|dhoka|fraud|scam|nahi (aya|aaya|mila)|abhi tak nahi|waste", re.I)
_CITIES = ("karachi", "lahore", "islamabad", "rawalpindi", "peshawar", "quetta", "multan", "faisalabad", "sialkot",
           "hyderabad", "gujranwala", "dubai", "uk", "usa", "canada")
_CITY_RE = re.compile(r"\b(" + "|".join(_CITIES) + r")\b", re.I)


def normalize(text: str) -> str:
    t = re.sub(r"https?://\S+", " ", str(text).casefold())
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    t = re.sub(r"[^\w\s%؀-ۿ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def _analyzer() -> SentimentAnalyzer:
    return SentimentAnalyzer()


def topics_for(text: str) -> list[str]:
    s = str(text)
    hits = [name for name, rx in _COMPILED if rx.search(s)]
    return hits or ["other"]


def purchase_intent(text: str, topics: list[str]) -> str:
    if "order_status" in topics:
        return "Existing customer issue"
    if _READY.search(text):
        return "Ready to buy"
    if _STRONG.search(text):
        return "Strong interest"
    if "price_complaint" in topics:
        return "Price barrier"
    if any(t in topics for t in ("price_inquiry", "discount", "delivery", "availability", "sizing", "fabric")):
        return "Possible inquiry"
    return "None"


def sentiment_for(text: str, lang: str, topic: str) -> tuple[str, float]:
    """Part 1's VADER + fashion lexicon, plus Roman Urdu cues and two topic rules."""
    a = _analyzer()
    score = a.score(text)
    if lang in ("Roman Urdu", "Mixed", "Urdu"):
        score += 0.4 * len(_POS_RU.findall(text)) - 0.4 * len(_NEG_RU.findall(text))
        score = max(-1.0, min(1.0, score))
    if topic in ("price_complaint", "order_status"):
        score = min(score, -0.3)      # a complaint is negative even when polite
    elif topic in INQUIRY_TOPICS and 0 < score < 0.5:
        score = 0.0                   # "price please?" is a question, not praise
    return a.label(score), round(score, 4)


def cities_in(text: str) -> list[str]:
    return sorted({m.lower() for m in _CITY_RE.findall(str(text))})


def classify(text: str) -> dict:
    lang = language_type(text)
    topics = topics_for(text)
    primary = topics[0]
    sentiment, score = sentiment_for(text, lang, primary)
    return {"language_type": lang, "topic": primary, "topics_all": "|".join(topics),
            "sentiment": sentiment, "sentiment_score": score,
            "purchase_intent": purchase_intent(text, topics)}
