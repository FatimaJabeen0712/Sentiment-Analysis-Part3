"""Language type of a comment: English / Roman Urdu / Urdu / Mixed / Emoji or other.

Heuristic and word-list based (no model download). Roman Urdu is Urdu written in Latin letters with
inconsistent spelling, so detection is approximate; it is good enough for grouping, not for linguistics.
"""
from __future__ import annotations

import re

ROMAN_URDU = frozenset("""
hai hain hy hn kya kia kitne kitna kitni kitnay bohat bohot bahut boht bht zyada zyadah zada mein mai
nahi nahin nahe aur yeh ye woh wo acha achi achha achay kro karo karen krein kar krdo dein den dena
hogi hoga hoge hogaya kab kahan kahin mujhe mjhe muje aap apka aapka apki aapki ji jee lekin magar abhi
tak din milega milegi mil milta milti chahiye chahye chahiyay lena lein lun kijiye bhej bhejein bhejain
batao bata btao btayen batayen ka ki ke ko se par pe bhi sirf sab koi kuch kapra kapray kapda qeemat
kimat mehnga mehngi mahnga mahngi khubsurat khoobsurat piyari pyari shukriya salam jazakallah mashallah
inshaallah wali wala walay hum tum hamare mera meri mere tha thi thay kaisa kaisi kese kaise rakhein
rakho ab phir jaldi dobara wapis wapas kyun kyu liye lye lia diya diye hua hui huay gaya gayi gaye
ayega aaye aya aaya aayi mila mili paisay paise rupay rupee karna karni karne krna krni lagta lagti
lagay sasta sasti bekar bakwas ghatia dhoka
""".split())

ENGLISH = frozenset("""
the is are was were this that these those very beautiful available please any in of to for it you i my
can do will how what when where send details delivery discount last week still waiting ordered dress suit
price much have has with and or not but so too love lovely nice good great amazing want need would
size color colour fabric quality sale off stock store city shipping order received tracking wrong your
me we us they them if on at be been get got buy bought from just really looks look pretty stunning
""".split())

_TOKEN = re.compile(r"[A-Za-z']+")
_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-z]")

LANGUAGE_TYPES = ("English", "Roman Urdu", "Urdu", "Mixed", "Emoji / other")


def language_type(text: str) -> str:
    if not isinstance(text, str):
        return "Emoji / other"
    arabic = len(_ARABIC.findall(text))
    latin = len(_LATIN.findall(text))
    letters = arabic + latin
    if letters == 0:
        return "Emoji / other"
    arabic_share = arabic / letters
    if arabic_share >= 0.6:
        return "Urdu"
    if arabic_share >= 0.15:
        return "Mixed"
    tokens = [t.lower().strip("'") for t in _TOKEN.findall(text)]
    n = len(tokens)
    if n == 0:
        return "Emoji / other"
    ru = sum(t in ROMAN_URDU for t in tokens)
    en = sum(t in ENGLISH for t in tokens)
    ru_share, en_share = ru / n, en / n
    if ru_share >= 0.45:
        return "Roman Urdu"
    if ru_share >= 0.15 and en_share >= 0.15:
        return "Mixed"
    if ru_share >= 0.15:
        return "Roman Urdu"
    return "English"
