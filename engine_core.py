"""
engine_core.py — SIP Retention AI Discovery Engine
====================================================

Standalone ingestion, sanitisation, enrichment and export module.

Pipeline
--------
1. INGEST     Public, unauthenticated sources only:
              • Google Play Store reviews (google-play-scraper)
              • Reddit public JSON search endpoints (no OAuth, no tokens)
2. SANITISE   Strip handles, emails, phone numbers, PAN/Aadhaar/UPI IDs and
              account-like digit strings. Author names are never collected.
3. ENRICH     Deterministic, explainable NLP layer: weighted lexicon scoring
              assigns each record to one of the 8 SIP-churn drivers, then
              extracts tenure, SIP amount, behaviour, emotion and
              preventability. No external LLM / API key required.
4. SYNTHESISE Summary metrics + a formatted 3-sheet Excel workbook.

The module has no Streamlit dependency, so it can be imported from notebooks,
cron jobs or tests:

    >>> from engine_core import load_real_benchmark, build_discovery_workbook
    >>> df, meta = load_real_benchmark()
    >>> xlsx_bytes = build_discovery_workbook(df)
"""

from __future__ import annotations

import io
import random
import re
import time
from collections import Counter
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# 0. CONSTANTS & SCHEMA
# ---------------------------------------------------------------------------

COLUMNS = [
    "Product",
    "Source Platform",
    "Source URL",
    "Original Comment",
    "User Type",
    "SIP Tenure (Months)",
    "Monthly SIP Amount (INR)",
    "SIP Stage",
    "Problem Theme",
    "Problem Behaviour",
    "Problem Root Cause",
    "Investor Emotional Trigger",
    "Churn Preventability",
    "Recommended Groww Intervention",
    "Target North-Star Metric Impact",
]

THEMES = [
    "Disappointment Valley",
    "Bank & Auto-Debit Friction",
    "Same-Category Overlap",
    "Directionless Onboarding",
    "Liquidity Shortage",
    "Market Correction Panic",
    "Social Influence",
    "Platform Bugs",
]

PREVENT_UX = "High – Product/UX Fix"
PREVENT_NUDGE = "High – FinTech Nudge"
PREVENT_EDU = "Medium – Investor Education"
PREVENT_LOW = "Low – External / Life Event"
PREVENTABILITY_LEVELS = [PREVENT_UX, PREVENT_NUDGE, PREVENT_EDU, PREVENT_LOW]
ADDRESSABLE_LEVELS = {PREVENT_UX, PREVENT_NUDGE}

STAGES = [
    "Activation (M0–3)",
    "Habit Formation (M4–9)",
    "Retention Cliff (M10–14)",
    "Post-Cliff Maturity (M15+)",
]

TENURE_BANDS = [  # (label, lower, upper) — inclusive
    ("0–3 months", 0, 3),
    ("4–6 months", 4, 6),
    ("7–9 months", 7, 9),
    ("10–14 months (Cliff)", 10, 14),
    ("15–24 months", 15, 24),
    ("25–36 months", 25, 36),
    ("37+ months", 37, 600),
]

GROWW_APP_ID = "com.nextbillion.groww"
GROWW_PLAY_URL = f"https://play.google.com/store/apps/details?id={GROWW_APP_ID}"

# Defaults for live ingestion. App IDs are editable in the UI. Every ID is
# validated against the Play Store before scraping; if an ID is wrong, the
# app is re-resolved by name search (see resolve_play_app_id).
DEFAULT_PLAY_APPS = {
    "Groww": "com.nextbillion.groww",
    "Zerodha Coin": "com.zerodha.coin",
    "Kuvera": "com.kuvera.android",
}
# name -> (search query, substring expected in the developer name)
PLAY_SEARCH_HINTS = {
    "Groww": ("Groww mutual funds stocks", "groww"),
    "Zerodha Coin": ("Coin by Zerodha mutual funds", "zerodha"),
    "Kuvera": ("Kuvera mutual funds SIP", "arevuk"),
    "INDmoney": ("INDmoney", "indmoney"),
    "Paytm Money": ("Paytm Money", "paytm"),
    "ET Money": ("ET Money mutual funds", "times"),
}
DEFAULT_SUBREDDITS = ["IndiaInvestments", "personalfinanceindia"]
DEFAULT_REDDIT_QUERIES = [
    "stopped SIP",
    "cancel SIP",
    "pause SIP",
    "SIP mandate failed",
    "SIP one year returns",
]

# Real, harvested benchmark (produced locally by harvest_real_data.py)
REAL_BENCHMARK_FILE = "real_benchmark_220.json"

# Post-level URL patterns — anything else is rejected from the real benchmark
REDDIT_PERMALINK_RX = re.compile(r"^https://www\.reddit\.com/r/[\w-]+/comments/[a-z0-9]+/", re.I)
PLAY_REVIEW_URL_RX = re.compile(r"^https://play\.google\.com/store/apps/details\?id=[\w.]+&reviewId=[\w:.-]+$")

# Signals that a comment describes stopping / breaking / pausing an SIP
CHURN_SIGNAL_PATTERN = re.compile(
    r"\b(stop(ped|ping)?|cancel(l?ed|l?ing)?|paus(e|ed|ing)|discontinu\w*|redeem(ed)?|"
    r"withdr[ae]w\w*|quit|broke|break|lapse[ds]?|fail(ed|s|ure)?|bounce[ds]?|skip(ped)?|"
    r"exit(ed)?|sold|switch(ed)?|moved to)\b",
    re.I,
)

USER_AGENT = (
    "SIPRetentionDiscoveryEngine/1.0 (public-data product research; "
    "no PII collected)"
)

# ---------------------------------------------------------------------------
# 1. THEME KNOWLEDGE BASE  (drives both enrichment and the benchmark corpus)
# ---------------------------------------------------------------------------

THEME_KB: dict[str, dict] = {
    "Disappointment Valley": {
        "lexicon": [
            (r"\b(no|zero|negative|flat|poor|low|bad)\s+returns?\b", 3),
            (r"\breturns?\b.{0,30}\b(fd|fixed deposit|savings account|rd)\b", 3),
            (r"\b(fd|fixed deposit)\b.{0,40}\bbetter\b", 3),
            (r"\b(red|in loss|losing money|lost money|xirr)\b", 2),
            (r"\bnot worth\b|\bwaste of (time|money)\b|\bwhat'?s the point\b", 2),
            (r"\b(a|one|1)\s*(year|yr)\b.{0,40}\b(nothing|same|flat|loss)\b", 3),
            (r"\bdisappoint", 2),
        ],
        "behaviours": ["Cancelled SIP", "Cancelled SIP", "Redeemed full corpus", "Paused SIP"],
        "root_causes": [
            "Return expectations anchored to FD/RD; no context on how equity returns vary over short holding periods",
            "First 9–12 months of a SIP show near-flat XIRR, read as 'the product does not work'",
            "Portfolio screen headlines absolute P&L, making a small drawdown feel like failure",
        ],
        "emotions": ["Disappointment", "Regret", "Impatience"],
        "prevent_mix": {PREVENT_NUDGE: 0.60, PREVENT_UX: 0.20, PREVENT_EDU: 0.20},
        "intervention": "‘Your SIP in Context’ card: historical rolling-return ranges by holding period + units accumulated (factual, no performance promises, no fund picks)",
        "nsm": "↑ Month-12 SIP retention; ↓ cancellations in M10–14",
        "tenure_center": 12,
    },
    "Bank & Auto-Debit Friction": {
        "lexicon": [
            (r"\bmandate\b", 3),
            (r"\b(auto[\s-]?pay|auto[\s-]?debit|e-?nach|nach)\b", 3),
            (r"\bbounce(d)?\b|\bdebit failed\b|\bpayment failed\b|\binsufficient (funds|balance)\b", 3),
            (r"\b(penalty|bounce charge|charges? by (my )?bank)\b", 2),
            (r"\bchang(e|ed|ing) (my )?bank\b|\bnew bank account\b", 2),
            (r"\bupi\b.{0,20}\b(fail|limit|expired?)\b", 2),
        ],
        "behaviours": ["Let mandate lapse", "Cancelled SIP", "Cancelled SIP", "Switched platform"],
        "root_causes": [
            "Debit date collides with salary credit timing; bounce fee from bank creates a negative association with the SIP",
            "Mandate expired or broke after a bank-account change with no proactive repair flow",
            "Repeated silent instalment failures — investor only discovers the gap months later",
        ],
        "emotions": ["Frustration", "Embarrassment", "Distrust"],
        "prevent_mix": {PREVENT_UX: 0.70, PREVENT_NUDGE: 0.20, PREVENT_EDU: 0.05, PREVENT_LOW: 0.05},
        "intervention": "Pre-Debit Balance Predictor (T-3 day heads-up + one-tap date shift) and Mandate Health Check with guided re-registration",
        "nsm": "↑ SIP instalment success rate; ↓ involuntary SIP failures",
        "tenure_center": 7,
    },
    "Same-Category Overlap": {
        "lexicon": [
            (r"\boverlap(ping)?\b", 4),
            (r"\b(too many|multiple|\d+|five|four|six|seven) (mutual )?funds\b", 2),
            (r"\b(same|similar) (stocks?|holdings?|portfolio)\b", 3),
            (r"\b(flexi ?cap|large ?cap|mid ?cap|small ?cap|index fund).{0,40}\b(another|also|again)\b", 2),
            (r"\bconsolidat", 3),
            (r"\b(confus|clutter|messy portfolio)", 2),
        ],
        "behaviours": ["Cancelled SIP", "Cancelled SIP", "Reduced number of SIPs", "Paused SIP"],
        "root_causes": [
            "Multiple SIPs started in the same category at different times; investor discovers duplicated holdings and stops several at once",
            "No portfolio-level view of category overlap, so ‘cleaning up’ defaults to cancelling rather than consolidating",
        ],
        "emotions": ["Confusion", "Overwhelm", "Self-doubt"],
        "prevent_mix": {PREVENT_UX: 0.70, PREVENT_EDU: 0.25, PREVENT_NUDGE: 0.05},
        "intervention": "Same-Category Overlap Consolidator: category-level overlap insight + guided ‘merge into one SIP’ flow (shows the investor’s own holdings; no scheme recommendation)",
        "nsm": "↑ Consolidated (not cancelled) SIP amount; ↑ M12 SIP-book retention",
        "tenure_center": 13,
    },
    "Directionless Onboarding": {
        "lexicon": [
            (r"\b(no|without a|didn'?t have a|never had a) (goal|plan|reason)\b", 4),
            (r"\bjust started\b|\bstarted because\b|\bsaw (a|on) (youtube|instagram|reel)\b", 2),
            (r"\bdon'?t (know|understand) (why|what)\b", 2),
            (r"\b(random(ly)?|blindly|impulse)\b", 2),
            (r"\bwhat (am i|was i) (investing|saving) for\b", 4),
            (r"\bgoal\b", 1),
        ],
        "behaviours": ["Cancelled SIP", "Let mandate lapse", "Paused SIP"],
        "root_causes": [
            "SIP started without a goal or horizon, so there is no anchor when motivation fades",
            "Onboarding optimised for ‘first SIP in 2 minutes’ but never connects the SIP to a life outcome",
        ],
        "emotions": ["Aimlessness", "Indifference", "Self-doubt"],
        "prevent_mix": {PREVENT_UX: 0.60, PREVENT_NUDGE: 0.20, PREVENT_EDU: 0.20},
        "intervention": "Goal-anchored SIP setup (goal, horizon, amount) with neutral milestone check-ins at M3, M6, M9 and M12",
        "nsm": "↑ % of SIPs linked to a goal; ↑ M12 SIP retention",
        "tenure_center": 11,
    },
    "Liquidity Shortage": {
        "lexicon": [
            (r"\b(lost|losing) (my )?job\b|\blaid off\b|\blayoff\b|\bjob change\b|\bnotice period\b", 4),
            (r"\b(rent|emi|loan|wedding|medical|hospital|emergency|fees)\b", 2),
            (r"\b(tight|short|no) (on )?(money|cash|funds)\b|\bcan'?t afford\b|\bcash ?crunch\b", 4),
            (r"\bneeded (the )?money\b|\bneed(ed)? cash\b", 3),
            (r"\bsalary (delayed|cut|late)\b", 3),
        ],
        "behaviours": ["Cancelled SIP", "Redeemed full corpus", "Paused SIP", "Reduced SIP amount"],
        "root_causes": [
            "Temporary cash-flow shock; the only visible options are ‘continue’ or ‘cancel’",
            "SIP amount sized on peak-month salary with no buffer for variable income",
            "Emergency need met by redeeming the SIP corpus because no emergency fund exists",
        ],
        "emotions": ["Anxiety", "Stress", "Guilt"],
        "prevent_mix": {PREVENT_UX: 0.50, PREVENT_NUDGE: 0.25, PREVENT_LOW: 0.25},
        "intervention": "SIP Sabbatical: pause 1–3 instalments or step down the amount instead of cancelling; auto-resume with a reminder",
        "nsm": "↑ Pause-instead-of-cancel ratio; ↑ SIP reactivation rate",
        "tenure_center": 10,
    },
    "Market Correction Panic": {
        "lexicon": [
            (r"\b(crash(ed|ing)?|correction|fall(ing)?|bloodbath|dip|down \d+%|sensex|nifty)\b", 3),
            (r"\b(scared|panic(ked)?|fear|afraid|worried)\b", 3),
            (r"\b(war|election|recession|tariff|fii selling)\b", 2),
            (r"\bwill go (lower|down)\b|\bmarket (is|was) (down|falling)\b", 3),
        ],
        "behaviours": ["Paused SIP", "Cancelled SIP", "Redeemed full corpus"],
        "root_causes": [
            "First market drawdown experienced as a first-time investor; loss feels permanent without historical context",
            "News and social feeds amplify the fall while the app shows only a red number",
        ],
        "emotions": ["Fear", "Loss aversion", "Anxiety"],
        "prevent_mix": {PREVENT_NUDGE: 0.55, PREVENT_EDU: 0.25, PREVENT_UX: 0.15, PREVENT_LOW: 0.05},
        "intervention": "Calm-Market Context: factual historical drawdown-and-recovery ranges and how SIP units accumulate in falls — neutral tone, no urgency or ‘buy the dip’ prompts",
        "nsm": "↓ SIP stops during >5% drawdown weeks; ↑ M12 retention",
        "tenure_center": 11,
    },
    "Social Influence": {
        "lexicon": [
            (r"\b(friend|friends|colleague|cousin|brother|uncle|everyone)\b.{0,40}\b(said|told|making|earning|switched|moved)\b", 3),
            (r"\b(f&o|futures|options trading|crypto|intraday|direct stocks?)\b", 3),
            (r"\b(finfluencer|youtuber|telegram|tip)\b", 2),
            (r"\bmutual funds? (are|is) (for|too) (slow|boring|old)\b", 3),
        ],
        "behaviours": ["Cancelled SIP", "Redeemed full corpus", "Switched to direct stocks / F&O"],
        "root_causes": [
            "Peer narratives of fast returns in stocks/F&O/crypto make a steady SIP feel ‘too slow’",
            "Unverified advice from social circles or creators overrides the investor’s original plan",
        ],
        "emotions": ["Peer comparison", "Impatience", "Self-doubt"],
        "prevent_mix": {PREVENT_EDU: 0.50, PREVENT_NUDGE: 0.30, PREVENT_UX: 0.10, PREVENT_LOW: 0.10},
        "intervention": "Neutral education module on holding-period outcomes and the cost of switching strategies — no counter-FOMO messaging",
        "nsm": "↓ SIP-to-trading cannibalisation; ↑ M12 retention",
        "tenure_center": 12,
    },
    "Platform Bugs": {
        "lexicon": [
            (r"\b(bug|glitch|crash(es|ed)? (the )?app|app (is )?(not working|crashing|hangs))\b", 4),
            (r"\b(wrong|incorrect|not updated|mismatch)\b.{0,30}\b(nav|units|amount|value|statement|status)\b", 3),
            (r"\b(showing|shows|showed) (cancelled|failed|pending)\b", 3),
            (r"\b(support|customer care|ticket)\b.{0,30}\b(no response|useless|not respond)\b", 2),
            (r"\b(otp|login|kyc)\b.{0,20}\b(issue|problem|fail)\b", 2),
        ],
        "behaviours": ["Switched platform", "Cancelled SIP", "Let mandate lapse"],
        "root_causes": [
            "SIP status or units displayed incorrectly, eroding trust that the money is safe",
            "Instalment marked pending/failed with no explanation and slow support resolution",
        ],
        "emotions": ["Distrust", "Frustration", "Anxiety"],
        "prevent_mix": {PREVENT_UX: 0.85, PREVENT_NUDGE: 0.05, PREVENT_EDU: 0.05, PREVENT_LOW: 0.05},
        "intervention": "SIP-status reliability fixes + proactive in-app instalment receipts and a transparent ‘what happened to my SIP’ timeline",
        "nsm": "↓ SIP-related support tickets; ↑ CSAT; ↑ M12 retention",
        "tenure_center": 6,
    },
}

EMOTION_LEXICON = {
    "Disappointment": r"\bdisappoint|\blet ?down\b|\bexpected more\b",
    "Regret": r"\bregret|\bshould have\b|\bwish i\b",
    "Impatience": r"\bwaited\b|\btoo slow\b|\bhow long\b|\bstill nothing\b",
    "Frustration": r"\bfrustrat|\bannoy|\bfed up\b|\birritat",
    "Embarrassment": r"\bembarrass|\bpenalty\b|\bbounce charge",
    "Distrust": r"\b(don'?t|do not|can'?t) trust\b|\bscam\b|\bsafe\?",
    "Confusion": r"\bconfus|\bdon'?t understand\b|\bno idea\b",
    "Overwhelm": r"\boverwhelm|\btoo many\b",
    "Anxiety": r"\banxious|\banxiety\b|\bworried\b|\btension\b|\bstress",
    "Fear": r"\bscared\b|\bfear\b|\bafraid\b|\bpanic",
    "Peer comparison": r"\bfriends?\b|\bcolleague|\beveryone\b",
    "Guilt": r"\bguilt|\bfeel bad\b",
}

LIFE_EVENT_PATTERN = re.compile(
    r"\b(lost (my )?job|laid off|medical|hospital|wedding|death in|surgery)\b", re.I
)

RELEVANCE_PATTERN = re.compile(
    r"\b(sip|sips|mutual funds?|\bmf\b|mandate|auto[\s-]?pay|nach|instal+ments?|"
    r"flexi ?cap|index fund|elss|nav)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# 2. SANITISATION  (zero-PII guarantee)
# ---------------------------------------------------------------------------

_PII_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),                       # emails
    (re.compile(r"\b[\w.-]{2,}@(ok\w+|ybl|ibl|axl|paytm|upi|apl|axisbank|icici|sbi)\b", re.I), "[upi-id]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[pan]"),                          # PAN
    (re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b"), "[aadhaar]"),               # Aadhaar
    (re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{9}\b"), "[phone]"),                # Indian mobile
    (re.compile(r"\b\d{9,18}\b"), "[account-no]"),                             # account-like digits
    (re.compile(r"(?<![\w/])/?u/[\w-]{3,20}", re.I), "[user]"),               # reddit handles
    (re.compile(r"(?<![\w.])@[A-Za-z_][\w.]{2,30}"), "[handle]"),              # @handles
    (re.compile(r"\b(folio|client id|ucc)\s*(no\.?|number|:)?\s*[\w/-]{4,}", re.I), r"\1 [redacted]"),
]


def sanitize_text(text: str | None, max_len: int = 1200) -> str:
    """Remove PII-like tokens and normalise whitespace."""
    if not text:
        return ""
    out = str(text)
    for pattern, repl in _PII_RULES:
        out = pattern.sub(repl, out)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "…"
    return out


# ---------------------------------------------------------------------------
# 3. ENRICHMENT
# ---------------------------------------------------------------------------

_COMPILED_LEXICON = {
    theme: [(re.compile(p, re.I), w) for p, w in kb["lexicon"]] for theme, kb in THEME_KB.items()
}
_COMPILED_EMOTIONS = {e: re.compile(p, re.I) for e, p in EMOTION_LEXICON.items()}


def score_themes(text: str) -> dict[str, int]:
    """Weighted lexicon score per theme (explainable classification)."""
    return {
        theme: sum(w for rx, w in rules if rx.search(text))
        for theme, rules in _COMPILED_LEXICON.items()
    }


def classify_theme(text: str, min_score: int = 2) -> str | None:
    scores = score_themes(text)
    theme, best = max(scores.items(), key=lambda kv: (kv[1], -THEMES.index(kv[0])))
    return theme if best >= min_score else None


def explain_theme(text: str) -> dict:
    """Audit trail for a classification: winning theme, its score, the
    runner-up margin and the exact phrases that matched."""
    scores = score_themes(text)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], THEMES.index(kv[0])))
    theme, best = ranked[0]
    matched = []
    for rx, _w in _COMPILED_LEXICON[theme]:
        m = rx.search(text)
        if m:
            matched.append(m.group(0))
    return {"theme": theme, "score": best, "margin": best - ranked[1][1], "matched": matched}


_WORD_NUM = {"one": 1, "a": 1, "an": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def extract_tenure_months(text: str) -> int | None:
    t = text.lower()
    m = re.search(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*(months?|mths?|mos?)\b", t)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _WORD_NUM[m.group(1)]
        return n if 0 < n <= 120 else None
    m = re.search(r"\b(\d(?:\.\d)?|one|two|three|four|five|a|an)\s*(?:and a half\s*)?(years?|yrs?)\b", t)
    if m:
        raw = m.group(1)
        n = float(raw) if raw[0].isdigit() else _WORD_NUM[raw]
        if "and a half" in m.group(0):
            n += 0.5
        months = int(round(n * 12))
        return months if 0 < months <= 120 else None
    return None


def extract_sip_amount(text: str) -> int | None:
    t = text.lower().replace("₹", " rs ")
    patterns = [
        r"\b(?:rs\.?|inr)\s*([\d,]{3,7})(?:\s*k)?",
        r"\b([\d,]{3,7})\s*(?:rs|rupees|inr)\b",
        r"\b([\d,]{3,7})\s*(?:/\s*month|per month|a month|pm|monthly)\b",
        r"\b(\d{1,2})\s*k\s*(?:sip|per month|/month|monthly|a month)\b",
    ]
    for i, p in enumerate(patterns):
        m = re.search(p, t)
        if m:
            try:
                val = int(m.group(1).replace(",", ""))
            except ValueError:
                continue
            if i == 3:
                val *= 1000
            if 100 <= val <= 500_000:
                return val
    return None


def stage_for_tenure(months: int | None) -> str:
    if months is None or pd.isna(months):
        return "Unknown Stage"
    months = int(months)
    if months <= 3:
        return STAGES[0]
    if months <= 9:
        return STAGES[1]
    if months <= 14:
        return STAGES[2]
    return STAGES[3]


def detect_behaviour(text: str, theme: str) -> str:
    t = text.lower()
    if re.search(r"\b(redeem|withdr[ae]w|sold|exit(ed)?|took out)\b", t):
        return "Redeemed full corpus"
    if re.search(r"\bpaus", t):
        return "Paused SIP"
    if re.search(r"\b(reduc|lower(ed)? (the )?amount|step(ped)? down)\b", t):
        return "Reduced SIP amount"
    if re.search(r"\b(switch|moved to|shift(ed)? to)\b", t):
        return "Switched platform"
    if re.search(r"\b(stop|cancel|discontinu|quit|closed)\w*", t):
        return "Cancelled SIP"
    return THEME_KB[theme]["behaviours"][0]


def detect_emotion(text: str, theme: str) -> str:
    theme_emotions = THEME_KB[theme]["emotions"]
    hits = [e for e, rx in _COMPILED_EMOTIONS.items() if rx.search(text)]
    for e in hits:  # prefer an emotion consistent with the theme
        if e in theme_emotions:
            return e
    return hits[0] if hits else theme_emotions[0]


def detect_preventability(text: str, theme: str) -> str:
    if LIFE_EVENT_PATTERN.search(text):
        return PREVENT_LOW
    mix = THEME_KB[theme]["prevent_mix"]
    return max(mix.items(), key=lambda kv: kv[1])[0]


def pick_root_cause(text: str, theme: str) -> str:
    causes = THEME_KB[theme]["root_causes"]
    # Deterministic choice: highest word-overlap with the comment, ties -> first.
    words = set(re.findall(r"[a-z]{4,}", text.lower()))
    return max(causes, key=lambda c: len(words & set(re.findall(r"[a-z]{4,}", c.lower()))))


def detect_user_type(text: str, platform: str) -> str:
    t = text.lower()
    if re.search(r"\b(first (ever |time )?(sip|investment|mutual fund)|new to (investing|mutual funds)|beginner)\b", t):
        return "First-time MF investor"
    if re.search(r"\b(freelanc|gig|business|commission|variable income)\b", t):
        return "Gig / variable-income earner"
    if re.search(r"\b(first job|fresher|college|student|internship)\b", t):
        return "Salaried first-jobber (22–27)"
    return "Retail investor (self-reported, public post)" if "Reddit" in platform else "Retail investor (app reviewer)"


def enrich_record(product: str, platform: str, url: str, comment: str,
                  require_churn_signal: bool = False) -> dict | None:
    """Turn one raw public comment into a 15-column discovery record.
    With require_churn_signal=True, the comment must also describe stopping,
    pausing, failing or breaking (used for the curated real benchmark)."""
    clean = sanitize_text(comment)
    if not clean or not RELEVANCE_PATTERN.search(clean):
        return None
    if require_churn_signal and not CHURN_SIGNAL_PATTERN.search(clean):
        return None
    theme = classify_theme(clean)
    if theme is None:
        return None
    tenure = extract_tenure_months(clean)
    kb = THEME_KB[theme]
    return {
        "Product": product,
        "Source Platform": platform,
        "Source URL": url,
        "Original Comment": clean,
        "User Type": detect_user_type(clean, platform),
        "SIP Tenure (Months)": tenure,
        "Monthly SIP Amount (INR)": extract_sip_amount(clean),
        "SIP Stage": stage_for_tenure(tenure),
        "Problem Theme": theme,
        "Problem Behaviour": detect_behaviour(clean, theme),
        "Problem Root Cause": pick_root_cause(clean, theme),
        "Investor Emotional Trigger": detect_emotion(clean, theme),
        "Churn Preventability": detect_preventability(clean, theme),
        "Recommended Groww Intervention": kb["intervention"],
        "Target North-Star Metric Impact": kb["nsm"],
    }


def enrich_records(raw: Iterable[dict]) -> pd.DataFrame:
    rows = []
    for r in raw:
        rec = enrich_record(r["product"], r["platform"], r["url"], r["text"])
        if rec:
            rows.append(rec)
    df = pd.DataFrame(rows, columns=COLUMNS)
    if not df.empty:
        df = df.drop_duplicates(subset=["Original Comment"]).reset_index(drop=True)
    return _coerce_types(df)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["SIP Tenure (Months)", "Monthly SIP Amount (INR)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# 4. LIVE PUBLIC-WEB INGESTION
# ---------------------------------------------------------------------------
# Both the Streamlit live tab and harvest_real_data.py use these functions.
# They never raise on network/platform problems: failures are returned in
# a log so the UI can explain them and fall back.


def _iso(ts) -> str | None:
    """datetime or unix seconds -> ISO-8601 (UTC) string."""
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            from datetime import datetime, timezone
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")
        return ts.isoformat(timespec="seconds")
    except Exception:
        return None


def play_review_url(app_id: str, review_id: str) -> str:
    """Public, review-level Play Store link."""
    return f"https://play.google.com/store/apps/details?id={app_id}&reviewId={review_id}"


def resolve_play_app_id(name: str, app_id: str | None, lang: str = "en",
                        country: str = "in") -> tuple[str | None, str]:
    """Confirm a package ID exists on the Play Store; if it doesn't, search by
    app name and pick the first hit from the expected developer.
    Returns (resolved_id or None, human-readable note)."""
    try:
        from google_play_scraper import app as gp_app, search as gp_search
    except ImportError:
        return None, "google-play-scraper is not installed — run `pip install google-play-scraper`."

    if app_id:
        try:
            info = gp_app(app_id, lang=lang, country=country)
            return app_id, f"{name}: verified {app_id} (“{info.get('title')}” by {info.get('developer')})"
        except Exception:
            pass
    query, dev_hint = PLAY_SEARCH_HINTS.get(name, (name, name.split()[0].lower()))
    try:
        hits = gp_search(query, lang=lang, country=country, n_hits=8)
    except Exception as exc:
        return None, f"{name}: could not verify {app_id!r} and search failed ({type(exc).__name__})"
    for h in hits:
        if dev_hint in (h.get("developer") or "").lower() and h.get("appId"):
            return h["appId"], (f"{name}: {app_id!r} not found — resolved by search to {h['appId']} "
                                f"(“{h.get('title')}” by {h.get('developer')})")
    return None, f"{name}: package {app_id!r} not found and no search hit from developer ~“{dev_hint}”"


def fetch_play_store_reviews(app_ids: dict[str, str], count_per_app: int = 200,
                             country: str = "in", lang: str = "en",
                             sorts: tuple[str, ...] = ("NEWEST",)) -> tuple[list[dict], list[str]]:
    """Fetch public reviews (paginated). Only review text, review id, rating and
    date are kept — reviewer names and avatars are discarded at source."""
    log: list[str] = []
    out: list[dict] = []
    try:
        from google_play_scraper import Sort, reviews
    except ImportError:
        return out, ["⚠️ google-play-scraper is not installed — run `pip install google-play-scraper`."]

    for product, given_id in app_ids.items():
        app_id, note = resolve_play_app_id(product, given_id, lang, country)
        log.append(("✅ " if app_id else "⚠️ ") + "Play Store · " + note)
        if not app_id:
            continue
        seen: set[str] = set()
        for sort_name in sorts:
            token, fetched = None, 0
            try:
                while fetched < count_per_app:
                    batch = min(200, count_per_app - fetched)
                    result, token = reviews(app_id, lang=lang, country=country,
                                            sort=getattr(Sort, sort_name), count=batch,
                                            continuation_token=token)
                    if not result:
                        break
                    fetched += len(result)
                    for r in result:
                        rid = r.get("reviewId")
                        if not rid or rid in seen:
                            continue
                        seen.add(rid)
                        out.append({
                            "product": product, "platform": "Google Play Store",
                            "url": play_review_url(app_id, rid), "text": r.get("content") or "",
                            "source_id": rid, "posted_at": _iso(r.get("at")),
                            "rating": r.get("score"),
                        })
                    if token is None or getattr(token, "token", "x") is None:
                        break
                    time.sleep(0.3)
            except Exception as exc:
                log.append(f"⚠️ Play Store · {product} ({sort_name}): {type(exc).__name__} — {exc}")
        log.append(f"✅ Play Store · {product}: {len(seen)} unique reviews fetched")
    return out, log


class RedditBlocked(Exception):
    """Reddit refused or throttled us (403/429/network). Stop calling it."""


def _reddit_get(session, url: str, params: dict | None = None,
                max_wait_s: float = 5.0, retries: int = 1):
    import requests

    for attempt in range(retries + 1):
        try:
            resp = session.get(url, params=params, timeout=15)
        except requests.exceptions.RequestException as exc:
            raise RedditBlocked(f"network error ({type(exc).__name__})") from exc
        if resp.status_code == 429:
            wait = resp.headers.get("Retry-After", "")
            wait_s = float(wait) if wait.replace(".", "", 1).isdigit() else 2.0
            if attempt < retries and wait_s <= max_wait_s:
                time.sleep(wait_s)
                continue
            raise RedditBlocked("HTTP 429 (rate limited)")
        if resp.status_code in (401, 403):
            raise RedditBlocked(f"HTTP {resp.status_code} (blocked)")
        if resp.status_code != 200:
            return None
        if "json" not in resp.headers.get("Content-Type", ""):
            raise RedditBlocked("non-JSON response (likely a block page)")
        return resp.json()
    raise RedditBlocked("HTTP 429 (rate limited)")


def _walk_comments(children: list, depth: int = 0):
    for ch in children or []:
        if ch.get("kind") != "t1":
            continue
        d = ch.get("data", {})
        yield d
        replies = d.get("replies")
        if isinstance(replies, dict) and depth < 4:
            yield from _walk_comments(replies.get("data", {}).get("children", []), depth + 1)


def _removed(text: str) -> bool:
    return text.strip().lower() in {"", "[removed]", "[deleted]"}


def fetch_reddit_posts(subreddits: list[str], queries: list[str], limit: int = 25,
                       timeframe: str = "year", pause_s: float = 1.2,
                       include_comments: bool = False, max_wait_s: float = 5.0,
                       comment_filter=None) -> tuple[list[dict], list[str], str]:
    """Search public subreddit JSON endpoints (no OAuth; author fields are never
    read). Returns (records, log, status) where status is "ok", "skipped" or
    "blocked: <reason>". On the first 403/429/network block, all further Reddit
    calls stop immediately — we never retry around a block."""
    import requests

    log: list[str] = []
    out: list[dict] = []
    if not subreddits or not queries:
        return out, log, "skipped"
    seen: set[str] = set()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    try:
        for sub in subreddits:
            got = 0
            for q in queries:
                data = _reddit_get(session, f"https://www.reddit.com/r/{sub}/search.json",
                                   {"q": q, "restrict_sr": 1, "sort": "relevance", "limit": limit,
                                    "t": timeframe}, max_wait_s)
                time.sleep(pause_s)
                if not data:
                    continue
                for child in data.get("data", {}).get("children", []):
                    d = child.get("data", {})
                    pid, permalink = d.get("id"), d.get("permalink", "")
                    if not pid or not permalink or pid in seen:
                        continue
                    seen.add(pid)
                    body = d.get("selftext", "")
                    text = f"{d.get('title', '')}. {'' if _removed(body) else body}"
                    out.append({
                        "product": _infer_product(text), "platform": f"Reddit · r/{sub}",
                        "url": f"https://www.reddit.com{permalink}", "text": text,
                        "source_id": f"t3_{pid}", "posted_at": _iso(d.get("created_utc")),
                        "rating": d.get("score"),
                    })
                    got += 1
                    if include_comments and d.get("num_comments", 0) > 0:
                        thread = _reddit_get(session, f"https://www.reddit.com{permalink}.json",
                                             {"limit": 200, "sort": "top", "raw_json": 1}, max_wait_s)
                        time.sleep(pause_s)
                        if isinstance(thread, list) and len(thread) > 1:
                            for c in _walk_comments(thread[1].get("data", {}).get("children", [])):
                                cid, cbody, cperma = c.get("id"), c.get("body", ""), c.get("permalink", "")
                                if not cid or not cperma or _removed(cbody) or f"t1_{cid}" in seen:
                                    continue
                                if comment_filter and not comment_filter(cbody):
                                    continue
                                seen.add(f"t1_{cid}")
                                out.append({
                                    "product": _infer_product(cbody + " " + text),
                                    "platform": f"Reddit · r/{sub} (comment)",
                                    "url": f"https://www.reddit.com{cperma}", "text": cbody,
                                    "source_id": f"t1_{cid}", "posted_at": _iso(c.get("created_utc")),
                                    "rating": c.get("score"),
                                })
                                got += 1
            log.append(f"✅ Reddit · r/{sub}: {got} unique posts/comments fetched")
    except RedditBlocked as exc:
        log.append(f"⛔ Reddit stopped responding to this server: {exc}. No further Reddit requests were made.")
        return out, log, f"blocked: {exc}"
    return out, log, "ok"


def _infer_product(text: str) -> str:
    t = text.lower()
    if "groww" in t:
        return "Groww"
    if "kuvera" in t:
        return "Kuvera"
    if "zerodha" in t or re.search(r"\bcoin\b", t):
        return "Zerodha Coin"
    if any(k in t for k in ["paytm money", "et money", "indmoney"]):
        return "Peer MF Platform"
    return "Unspecified MF Platform"


def run_live_ingestion(app_ids: dict[str, str] | None = None, reviews_per_app: int = 200,
                       subreddits: list[str] | None = None, queries: list[str] | None = None,
                       include_play: bool = True, include_reddit: bool = True) -> tuple[pd.DataFrame, list[str], dict]:
    """End-to-end live run: ingest → sanitise → enrich. Never raises.

    Reddit runs first. If Reddit is blocked (403/429/network) the run falls
    back to Play Store ingestion automatically — even when Play Store was not
    selected — so a live run always attempts at least one working source."""
    raw: list[dict] = []
    log: list[str] = []
    stats = {"raw": 0, "retained": 0, "dropped": 0, "reddit_status": "skipped",
             "fallback_used": False}

    if include_reddit:
        try:
            r, l, status = fetch_reddit_posts(subreddits or DEFAULT_SUBREDDITS,
                                              queries or DEFAULT_REDDIT_QUERIES)
        except Exception as exc:  # belt and braces
            r, l, status = [], [f"⛔ Reddit ingestion error: {type(exc).__name__} — {exc}"], f"blocked: {exc}"
        raw += r
        log += l
        stats["reddit_status"] = status

    reddit_failed = stats["reddit_status"].startswith("blocked")
    if include_play or (reddit_failed and not raw):
        if reddit_failed and not include_play:
            stats["fallback_used"] = True
            log.append("↪️ Falling back to live Google Play Store ingestion.")
        try:
            r, l = fetch_play_store_reviews(app_ids or DEFAULT_PLAY_APPS, reviews_per_app)
        except Exception as exc:
            r, l = [], [f"⚠️ Play Store ingestion error: {type(exc).__name__} — {exc}"]
        raw += r
        log += l
    elif reddit_failed:
        stats["fallback_used"] = True

    try:
        df = enrich_records(raw)
    except Exception as exc:
        log.append(f"⚠️ Enrichment error: {type(exc).__name__} — {exc}")
        df = pd.DataFrame(columns=COLUMNS)
    stats.update(raw=len(raw), retained=len(df), dropped=len(raw) - len(df))
    log.append(f"🧠 Enrichment: {stats['raw']} raw → {stats['retained']} SIP-churn records "
               f"({stats['dropped']} off-topic / unclassifiable dropped)")
    return df, log, stats


# ---------------------------------------------------------------------------
# 4b. REAL BENCHMARK LOADER
# ---------------------------------------------------------------------------

def is_post_level_url(url: str) -> bool:
    return bool(REDDIT_PERMALINK_RX.match(url or "") or PLAY_REVIEW_URL_RX.match(url or ""))


def load_real_benchmark(path: str | None = None) -> tuple[pd.DataFrame | None, dict]:
    """Load the harvested real benchmark. Any record whose Source URL is not a
    post-/review-level link is rejected, so every link in the table opens a
    specific public comment. Returns (df or None, meta incl. 'error' if any)."""
    import json
    import os

    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), REAL_BENCHMARK_FILE)
    if not os.path.exists(path):
        return None, {"error": f"{os.path.basename(path)} not found — run harvest_real_data.py and commit the file."}
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as exc:
        return None, {"error": f"Could not read {os.path.basename(path)}: {exc}"}

    records = payload.get("records", payload if isinstance(payload, list) else [])
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    valid = [r for r in records if is_post_level_url(r.get("Source URL", ""))]
    meta["rejected_non_post_urls"] = len(records) - len(valid)
    if not valid:
        return None, {**meta, "error": "No records with post-level URLs in the benchmark file."}
    df = pd.DataFrame(valid)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None
    extra = [c for c in df.columns if c.startswith("_")]
    df = _coerce_types(df[COLUMNS + extra].reset_index(drop=True))
    return df, meta


# ---------------------------------------------------------------------------
# 5. DEMO CORPUS (synthetic fallback — used only if the real benchmark is missing)
# ---------------------------------------------------------------------------
# IMPORTANT PROVENANCE NOTE
# The demo comments are SYNTHETIC COMPOSITES written to mirror recurring
# patterns in public Indian personal-finance discussions. They are not quotes
# from identifiable people. Source URLs point to the public community or app
# listing where such conversations happen — not to specific posts. Use Mode A
# (live ingestion) for verbatim, individually-linked evidence.

DEMO_PROVENANCE = (
    "Demo corpus = 220 synthetic composite comments modelled on recurring public "
    "discussion patterns (Play Store, Reddit, YouTube, TradingQ&A). Not verbatim quotes; "
    "URLs link to the source community/listing, not individual posts. Use Live mode for "
    "verbatim, post-level evidence."
)

_SOURCES = [
    ("Groww", "Google Play Store", GROWW_PLAY_URL, 0.38),
    ("Groww", "Reddit · r/IndiaInvestments", "https://www.reddit.com/r/IndiaInvestments/", 0.18),
    ("Peer MF Platform", "Reddit · r/IndiaInvestments", "https://www.reddit.com/r/IndiaInvestments/", 0.08),
    ("Groww", "Reddit · r/personalfinanceindia", "https://www.reddit.com/r/personalfinanceindia/", 0.14),
    ("Peer MF Platform", "Reddit · r/personalfinanceindia", "https://www.reddit.com/r/personalfinanceindia/", 0.06),
    ("Groww", "YouTube · personal-finance comments", "https://www.youtube.com/results?search_query=stopped+SIP+after+1+year", 0.10),
    ("Unspecified MF Platform", "TradingQ&A", "https://tradingqna.com/search?q=stop%20SIP", 0.06),
]

_CATEGORIES = ["flexi cap", "large cap index", "mid cap", "small cap", "ELSS",
               "Nifty 50 index", "multi cap", "large & mid cap"]

_TEMPLATES: dict[str, list[str]] = {
    "Disappointment Valley": [
        "Started my first SIP of Rs {amt} in a {cat} fund {m} months ago. Portfolio is basically flat, my FD would have done better. Stopped it last week, not sure what the point was.",
        "{m} months of SIP and I am in the red. Everyone says SIP is magic but I see nothing. Cancelled and moved the money to an RD for now.",
        "Honestly disappointed. Rs {amt} every month for {m} months and the gain is less than my savings account interest. Thinking of redeeming everything.",
        "The app shows -2% after {m} months of a {cat} SIP. I know it is long term but it does not feel like it is working. Paused it for now.",
        "My XIRR after {m} months is near zero. I was expecting at least 12% like the videos said. Stopped the SIP, will maybe restart later.",
        "Did a Rs {amt} SIP for {m} months. Returns are the same as what I put in. Feels like a waste of time, cancelled it.",
    ],
    "Bank & Auto-Debit Friction": [
        "My mandate failed twice because salary comes on the 7th and SIP date was the 5th. Bank charged a bounce penalty both times. Cancelled the SIP out of irritation after {m} months.",
        "Changed my bank account and the auto-debit mandate just stopped working. Nobody told me, found out {m} months later that instalments were not going through.",
        "Autopay keeps failing for my Rs {amt} SIP. UPI mandate shows expired. Tried re-registering, did not work, so I just let it lapse.",
        "Insufficient balance on SIP date = bounce charge from my bank every time. After {m} months I stopped the SIP, the penalty is more than the returns.",
        "eNACH registration took weeks and then debit failed in month {m}. Too much hassle, I stopped the SIP.",
        "SIP debit failed and my bank charged Rs 500. For a Rs {amt} SIP that is ridiculous. Stopped it.",
    ],
    "Same-Category Overlap": [
        "I have five funds and realised three are {cat} with the same top holdings. Overlap is huge. Cancelled two SIPs after {m} months, now confused about what to keep.",
        "Started SIPs in a {cat} fund and later another {cat} fund thinking I was diversifying. Same stocks. Stopped both, too confusing.",
        "My portfolio is messy, 7 mutual funds, many overlapping. Did not know how to consolidate so I just stopped most SIPs at month {m}.",
        "Read on Reddit that my funds overlap 70%. I felt stupid. Cancelled the Rs {amt} SIP and haven't decided what to do since.",
        "Too many funds, similar holdings. Wanted to merge them but there is no easy way in the app, so I cancelled the extra SIPs after {m} months.",
    ],
    "Directionless Onboarding": [
        "Started a SIP because I saw a YouTube video, no real goal. After {m} months I did not know what I was saving for, so I stopped it.",
        "I never had a plan, just picked a {cat} fund and Rs {amt}. Around month {m} I lost interest and let the mandate lapse.",
        "Honestly I don't know why I started. Everyone said start SIP early. {m} months later I just cancelled, no goal in mind.",
        "Blindly started SIP as my first investment. Did not link it to anything. When a phone upgrade came up I stopped it after {m} months.",
        "What was I even investing for? No target, no timeline. Paused the SIP at {m} months and never resumed.",
    ],
    "Liquidity Shortage": [
        "Lost my job in month {m} of my SIP. Had to stop the Rs {amt} SIP and redeemed it to pay rent.",
        "Wedding expenses this year so money is tight. Stopped my SIP after {m} months, wish there was a way to skip a few months instead.",
        "Salary delayed twice, could not afford the Rs {amt} SIP. Cancelled it after {m} months, will restart when things settle.",
        "Medical emergency at home, needed cash. Withdrew everything from my {cat} SIP after {m} months.",
        "Rent went up and EMI started, so I am short on money. Reduced and then stopped my SIP at month {m}.",
        "Freelance income is irregular. Rs {amt} SIP was fine some months, impossible in others. Stopped after {m} months.",
    ],
    "Market Correction Panic": [
        "Market crashed and my {cat} SIP is down 9%. I got scared and stopped the SIP at {m} months, will wait for it to recover.",
        "Nifty falling every day, news says it will go lower. Panicked and paused my Rs {amt} SIP after {m} months.",
        "Saw the correction and my portfolio went red. First time investor, I was afraid of losing more so I redeemed everything at month {m}.",
        "With all the tariff and FII selling news, I am worried. Stopped SIPs for now after {m} months.",
        "The dip scared me. {m} months of SIP and now in loss. Paused until market is stable.",
    ],
    "Social Influence": [
        "My friends are making money in F&O and direct stocks. Mutual funds feel too slow. Stopped my Rs {amt} SIP after {m} months.",
        "Colleague said SIP is for old people and moved to crypto. I followed him and cancelled my SIP at month {m}.",
        "A finfluencer on Telegram said index funds are boring. Redeemed my SIP after {m} months to try stocks.",
        "Cousin told me my {cat} fund is bad. Stopped the SIP after {m} months, now not sure who to believe.",
    ],
    "Platform Bugs": [
        "App showed my SIP as cancelled when I never cancelled it. Support gave no response. After {m} months I moved to another platform.",
        "Units not updated for weeks and the statement showed the wrong amount. I don't trust it anymore, stopped my Rs {amt} SIP.",
        "SIP status showing pending for a month, money debited from bank. Customer care was useless. Stopped after {m} months.",
        "App crashes when I open the SIP page. Wrong NAV shown too. Switched platform after {m} months.",
    ],
}

_CONTEXT_SENTENCES = [
    "", "", "",
    " This was my first ever investment.",
    " I am new to mutual funds.",
    " I am a fresher in my first job.",
    " Not looking for advice on funds, just sharing my experience.",
    " Has anyone else gone through this?",
    " I do freelance work so income varies.",
    " Posting here in case it helps someone.",
]

_THEME_COUNTS = {  # sums to 220
    "Disappointment Valley": 48,
    "Bank & Auto-Debit Friction": 33,
    "Liquidity Shortage": 29,
    "Same-Category Overlap": 26,
    "Directionless Onboarding": 26,
    "Market Correction Panic": 26,
    "Social Influence": 16,
    "Platform Bugs": 16,
}

_AMOUNTS = [500, 500, 1000, 1000, 1000, 1500, 2000, 2000, 2500, 3000, 5000, 5000, 10000]


def _allocate(mix: dict[str, float], n: int) -> list[str]:
    """Deterministic quota allocation (largest remainder)."""
    raw = {k: v * n for k, v in mix.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n - sum(counts.values())]:
        counts[k] += 1
    return [k for k, c in counts.items() for _ in range(c)]


def _tenure_draw(rng: random.Random, center: int) -> int:
    """~60% of churn lands in the M10–14 cliff; tails spread across 1–36."""
    roll = rng.random()
    if roll < 0.60:
        return max(10, min(14, int(round(rng.gauss(12, 1.3)))))
    if roll < 0.82:
        return max(1, min(9, int(round(rng.gauss(min(center, 9) - 2, 2.2)))))
    return rng.randint(15, 30)


def build_benchmark_corpus(seed: int = 2026) -> pd.DataFrame:
    rng = random.Random(seed)
    sources = [s[:3] for s in _SOURCES]
    weights = [s[3] for s in _SOURCES]
    rows: list[dict] = []
    seen_text: set[str] = set()

    for theme, n in _THEME_COUNTS.items():
        kb = THEME_KB[theme]
        prevent = _allocate(kb["prevent_mix"], n)
        rng.shuffle(prevent)
        templates = _TEMPLATES[theme]
        for i in range(n):
            product, platform, url = rng.choices(sources, weights=weights)[0]
            tenure = _tenure_draw(rng, kb["tenure_center"])
            amount = rng.choice(_AMOUNTS[:9] if theme == "Liquidity Shortage" else _AMOUNTS)
            text = templates[i % len(templates)].format(m=tenure, amt=f"{amount:,}", cat=rng.choice(_CATEGORIES))
            text += rng.choice(_CONTEXT_SENTENCES)
            if platform == "Google Play Store" and product == "Groww":
                text = text.replace("the app", "the Groww app")
            prev = prevent[i]
            if prev == PREVENT_LOW and theme == "Liquidity Shortage" and not LIFE_EVENT_PATTERN.search(text):
                text = templates[0].format(m=tenure, amt=f"{amount:,}", cat=rng.choice(_CATEGORIES))
            while text in seen_text:  # keep every benchmark comment unique
                text += rng.choice([c for c in _CONTEXT_SENTENCES if c] or [" ."])
            seen_text.add(text)
            rows.append({
                "Product": product,
                "Source Platform": platform,
                "Source URL": url,
                "Original Comment": text,
                "User Type": detect_user_type(text, platform) if rng.random() < 0.5 else rng.choice(
                    ["First-time MF investor", "First-time MF investor", "Salaried first-jobber (22–27)",
                     "Tier-2/3 first-time investor", "Gig / variable-income earner"]),
                "SIP Tenure (Months)": tenure,
                "Monthly SIP Amount (INR)": amount,
                "SIP Stage": stage_for_tenure(tenure),
                "Problem Theme": theme,
                "Problem Behaviour": detect_behaviour(text, theme),
                "Problem Root Cause": kb["root_causes"][i % len(kb["root_causes"])],
                "Investor Emotional Trigger": detect_emotion(text, theme) if rng.random() < 0.6 else kb["emotions"][i % len(kb["emotions"])],
                "Churn Preventability": prev,
                "Recommended Groww Intervention": kb["intervention"],
                "Target North-Star Metric Impact": kb["nsm"],
            })

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return _coerce_types(df)


def load_demo_corpus() -> pd.DataFrame:
    """Synthetic fallback. Prefer load_real_benchmark()."""
    return build_benchmark_corpus()


# ---------------------------------------------------------------------------
# 6. SUMMARY METRICS (shared by dashboard and workbook)
# ---------------------------------------------------------------------------

def tenure_band(months) -> str:
    if months is None or pd.isna(months):
        return "Not stated"
    for label, lo, hi in TENURE_BANDS:
        if lo <= int(months) <= hi:
            return label
    return "Not stated"


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0}
    ten = df["SIP Tenure (Months)"]
    cliff = int(((ten >= 10) & (ten <= 14)).fillna(False).sum())
    addressable = int(df["Churn Preventability"].isin(ADDRESSABLE_LEVELS).sum())
    themes = []
    for theme in THEMES:
        sub = df[df["Problem Theme"] == theme]
        if sub.empty:
            continue
        emo = Counter(sub["Investor Emotional Trigger"]).most_common(1)[0][0]
        avg = sub["Monthly SIP Amount (INR)"].dropna()
        themes.append({
            "Problem Theme": theme,
            "Records": len(sub),
            "Share": len(sub) / n,
            "Avg Monthly SIP (INR)": float(avg.mean()) if len(avg) else None,
            "Median Tenure (Months)": float(sub["SIP Tenure (Months)"].dropna().median()) if sub["SIP Tenure (Months)"].notna().any() else None,
            "Dominant Emotion": emo,
            "Addressable Share": float(sub["Churn Preventability"].isin(ADDRESSABLE_LEVELS).mean()),
        })
    theme_df = pd.DataFrame(themes).sort_values("Records", ascending=False)
    known_tenure = int(ten.notna().sum())
    return {
        "n": n,
        "cliff": cliff,
        "cliff_share": cliff / known_tenure if known_tenure else 0.0,
        "known_tenure": known_tenure,
        "addressable": addressable,
        "addressable_share": addressable / n,
        "avg_sip": float(df["Monthly SIP Amount (INR)"].dropna().mean()) if df["Monthly SIP Amount (INR)"].notna().any() else None,
        "themes": theme_df,
        "top_theme": theme_df.iloc[0]["Problem Theme"] if not theme_df.empty else "—",
    }


# ---------------------------------------------------------------------------
# 7. PM GROWTH ROADMAP (shared by app + workbook)
# ---------------------------------------------------------------------------

ROADMAP = [
    {
        "Priority": "P0",
        "Feature": "SIP Sabbatical",
        "Themes": ["Liquidity Shortage", "Market Correction Panic"],
        "User Problem": "When cash is tight or markets fall, the only visible choices are ‘continue’ or ‘cancel’, so temporary stress becomes permanent churn.",
        "Solution Spec": "Replace the default ‘Cancel SIP’ path with a choice screen: Pause 1/2/3 instalments, Step down amount, or Cancel. Paused SIPs auto-resume; a single neutral reminder 5 days before resumption. Mandate stays intact.",
        "Guardrails": "Cancel remains one tap away (no dark patterns); no fear/urgency copy; no fund recommendations; resumption reminder is informational only.",
        "Success Metric": "Pause-instead-of-cancel ratio; % of paused SIPs resumed; M12 SIP retention",
        "Phase": "Q1 · MVP behind experiment flag (50/50 A/B on cancel flow)",
    },
    {
        "Priority": "P0",
        "Feature": "Pre-Debit Balance Predictor",
        "Themes": ["Bank & Auto-Debit Friction"],
        "User Problem": "Instalments bounce when the SIP date precedes salary credit; bank penalties make the SIP feel costly and investors cancel.",
        "Solution Spec": "T-3 day heads-up showing the upcoming debit amount and date, with one-tap ‘move my SIP date’ to after typical salary credit (user-entered). Mandate Health Check flags expired/broken mandates and guides re-registration.",
        "Guardrails": "Uses only data the user provides or explicitly consents to (e.g. AA-based balance check is opt-in); no account numbers displayed in full; no promotional content in the alert.",
        "Success Metric": "SIP instalment success rate; involuntary failure rate; M12 SIP retention",
        "Phase": "Q1 · Date-shift + mandate health · Q2 · opt-in balance signal",
    },
    {
        "Priority": "P1",
        "Feature": "Same-Category Overlap Consolidator",
        "Themes": ["Same-Category Overlap"],
        "User Problem": "Investors discover several SIPs in the same category with duplicated holdings and cancel many at once instead of simplifying.",
        "Solution Spec": "Portfolio view that groups the investor’s own SIPs by SEBI category and shows holding overlap %. A guided flow lets the investor choose which of their existing SIPs to keep and redirect the combined amount into it.",
        "Guardrails": "Shows only the investor’s own holdings; Groww never suggests which scheme to keep; factual overlap data with methodology disclosure.",
        "Success Metric": "Consolidated (not cancelled) SIP amount; SIP-book value retained at M12",
        "Phase": "Q2 · Overlap insight · Q3 · Guided consolidation flow",
    },
    {
        "Priority": "P1",
        "Feature": "Goal-Anchored SIP Setup",
        "Themes": ["Directionless Onboarding", "Disappointment Valley"],
        "User Problem": "SIPs started without a goal have no anchor when early returns look flat.",
        "Solution Spec": "Optional goal, horizon and amount step during SIP creation; neutral milestone check-ins at M3/M6/M9/M12 showing progress against the user’s own goal and units accumulated.",
        "Guardrails": "Goal step is skippable; projections shown as historical ranges with clear disclaimers; no product suggestions.",
        "Success Metric": "% of new SIPs with a linked goal; M12 retention of goal-linked vs. non-goal SIPs",
        "Phase": "Q2",
    },
    {
        "Priority": "P2",
        "Feature": "‘Your SIP in Context’ Card",
        "Themes": ["Disappointment Valley", "Market Correction Panic", "Social Influence"],
        "User Problem": "First-year SIP outcomes are read against FD returns or peers’ trading gains, without context.",
        "Solution Spec": "A factual card on the SIP detail page showing historical rolling-return ranges for the fund’s category by holding period, plus units accumulated to date.",
        "Guardrails": "Past-performance disclaimer; no forward-looking promises; no comparison to trading/F&O; no urgency language.",
        "Success Metric": "Cancellations in M10–14; card interaction rate",
        "Phase": "Q3",
    },
]


# ---------------------------------------------------------------------------
# 8. EXCEL EXPORT ENGINE (in-memory openpyxl)
# ---------------------------------------------------------------------------

NAVY = "1B365D"
EMERALD = "00D09C"
LIGHT = "F4F6F8"


def build_discovery_workbook(df: pd.DataFrame, provenance_note: str | None = None) -> bytes:
    """Build the 3-sheet PM Discovery Workbook and return it as bytes.
    Summary metrics are live Excel formulas over the dataset sheet, so
    filtering or editing rows in Excel recalculates the deck numbers."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    n = len(df)
    last = max(n + 1, 2)
    ds_name = f"Discovery Dataset ({n} Rows)"
    ds = f"'{ds_name}'"

    F = "Arial"
    hdr_font = Font(name=F, bold=True, color="FFFFFF", size=11)
    hdr_fill = PatternFill("solid", fgColor=NAVY)
    accent_fill = PatternFill("solid", fgColor=EMERALD)
    light_fill = PatternFill("solid", fgColor=LIGHT)
    body = Font(name=F, size=10)
    bold = Font(name=F, size=10, bold=True)
    title_font = Font(name=F, size=16, bold=True, color=NAVY)
    sub_font = Font(name=F, size=10, italic=True, color="5A6B82")
    link_font = Font(name=F, size=10, color="0563C1", underline="single")
    thin = Side(style="thin", color="D5DBE3")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap_top = Alignment(wrap_text=True, vertical="top")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    wb = Workbook()

    # ---------------- Sheet 1: dataset ----------------
    ws = wb.active
    ws.title = ds_name
    widths = [18, 26, 30, 50, 24, 12, 14, 22, 24, 22, 45, 20, 24, 50, 38]
    for c, (col, w) in enumerate(zip(COLUMNS, widths), start=1):
        cell = ws.cell(row=1, column=c, value=col)
        cell.font, cell.fill, cell.alignment, cell.border = hdr_font, hdr_fill, center, box
        ws.column_dimensions[get_column_letter(c)].width = w
    ws.row_dimensions[1].height = 34

    for r, rec in enumerate(df[COLUMNS].itertuples(index=False), start=2):
        for c, val in enumerate(rec, start=1):
            if val is None or (not isinstance(val, str) and pd.isna(val)):
                val = None
            elif hasattr(val, "item"):
                val = val.item()
            cell = ws.cell(row=r, column=c, value=val)
            cell.font, cell.alignment, cell.border = body, wrap_top, box
            if c == 3 and val:
                cell.hyperlink = val
                cell.font = link_font
            if c == 7 and val is not None:
                cell.number_format = "₹#,##0"
        if r % 2 == 0:
            for c in range(1, len(COLUMNS) + 1):
                ws.cell(row=r, column=c).fill = light_fill
    ws.freeze_panes = "E2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{last}"

    rng = lambda col: f"{ds}!${col}$2:${col}${last}"  # noqa: E731
    THEME, TEN, AMT, STAGE, PREV, COMMENT = (rng("I"), rng("F"), rng("G"), rng("H"), rng("M"), rng("D"))

    # ---------------- Sheet 2: summary metrics ----------------
    s2 = wb.create_sheet("Slide Deck Summary Metrics")
    s2.sheet_view.showGridLines = False
    for col, w in zip("ABCDEFG", [34, 14, 14, 18, 20, 22, 60]):
        s2.column_dimensions[col].width = w
    s2["A1"] = "SIP Retention AI Discovery Engine — Executive Summary"
    s2["A1"].font = title_font
    s2["A2"] = provenance_note or "Source: public web conversations, sanitised (zero PII). All figures below are live formulas over the dataset sheet."
    s2["A2"].font = sub_font
    s2.merge_cells("A2:G2")
    s2["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    s2.row_dimensions[2].height = 42

    def header_row(row: int, labels: list[str]):
        for i, lab in enumerate(labels):
            cell = s2.cell(row=row, column=1 + i, value=lab)
            cell.font, cell.fill, cell.alignment, cell.border = hdr_font, hdr_fill, center, box
        s2.row_dimensions[row].height = 30

    def section(row: int, text: str):
        cell = s2.cell(row=row, column=1, value=text)
        cell.font = Font(name=F, size=12, bold=True, color=NAVY)
        s2.cell(row=row, column=1).border = Border(bottom=Side(style="medium", color=EMERALD))

    # KPI block
    section(4, "Headline KPIs")
    kpis = [
        ("Total analysed records", f"=COUNTA({COMMENT})", "0"),
        ("Records with stated tenure", f"=COUNT({TEN})", "0"),
        ("Churn in Month 10–14 cliff", f'=COUNTIFS({TEN},">=10",{TEN},"<=14")', "0"),
        ("Cliff share of tenure-stated churn", "=IFERROR(B7/B6,0)", "0.0%"),
        ("Addressable by Product/UX & FinTech nudges", f'=COUNTIF({PREV},"High*")', "0"),
        ("Addressable share", "=IFERROR(B9/B5,0)", "0.0%"),
        ("Avg monthly SIP at churn (INR)", f"=IFERROR(AVERAGE({AMT}),0)", "₹#,##0"),
    ]
    for i, (label, formula, fmt) in enumerate(kpis):
        r = 5 + i
        a = s2.cell(row=r, column=1, value=label)
        b = s2.cell(row=r, column=2, value=formula)
        a.font, b.font = body, Font(name=F, size=11, bold=True, color=NAVY)
        b.number_format = fmt
        a.border = b.border = box
        if "share" in label.lower():
            b.fill = PatternFill("solid", fgColor="E6FAF5")

    # Driver breakdown
    row = 14
    section(row - 1, "Chart 2 · Root-Cause Driver Breakdown")
    header_row(row, ["Problem Theme", "Records", "% Share", "Avg Monthly SIP (INR)",
                     "Dominant Emotion*", "Addressable Records", "Recommended Groww Intervention"])
    summ = summarize(df)
    emo_map = {}
    if summ.get("n"):
        emo_map = dict(zip(summ["themes"]["Problem Theme"], summ["themes"]["Dominant Emotion"]))
    first = row + 1
    for i, theme in enumerate(THEMES):
        r = first + i
        vals = [
            theme,
            f'=COUNTIF({THEME},A{r})',
            f"=IFERROR(B{r}/$B$5,0)",
            f'=IFERROR(AVERAGEIFS({AMT},{THEME},A{r}),0)',
            emo_map.get(theme, "—"),
            f'=COUNTIFS({THEME},A{r},{PREV},"High*")',
            THEME_KB[theme]["intervention"],
        ]
        fmts = [None, "0", "0.0%", "₹#,##0", None, "0", None]
        for c, (v, fmt) in enumerate(zip(vals, fmts), start=1):
            cell = s2.cell(row=r, column=c, value=v)
            cell.font, cell.border, cell.alignment = body, box, wrap_top
            if fmt:
                cell.number_format = fmt
        s2.row_dimensions[r].height = 42
    tot = first + len(THEMES)
    for c, v in enumerate(["Total", f"=SUM(B{first}:B{tot-1})", f"=SUM(C{first}:C{tot-1})",
                           None, None, f"=SUM(F{first}:F{tot-1})", None], start=1):
        cell = s2.cell(row=tot, column=c, value=v)
        cell.font, cell.border = bold, box
        cell.number_format = "0.0%" if c == 3 else "0"
    s2.cell(row=tot + 1, column=1,
            value="*Dominant Emotion = most frequent Investor Emotional Trigger per theme at export time (static text).").font = sub_font

    # Tenure cliff
    row = tot + 4
    section(row - 1, "Chart 1 · The Month 10–14 Drop-off Cliff")
    header_row(row, ["Tenure Band", "From (M)", "To (M)", "Churn Records", "% of Tenure-Stated"])
    for i, (label, lo, hi) in enumerate(TENURE_BANDS):
        r = row + 1 + i
        vals = [label, lo, hi, f'=COUNTIFS({TEN},">="&B{r},{TEN},"<="&C{r})', "=IFERROR(D{0}/$B$6,0)".format(r)]
        for c, v in enumerate(vals, start=1):
            cell = s2.cell(row=r, column=c, value=v)
            cell.font, cell.border = body, box
            if c == 5:
                cell.number_format = "0.0%"
        if "Cliff" in label:
            for c in range(1, 6):
                s2.cell(row=r, column=c).fill = accent_fill
                s2.cell(row=r, column=c).font = Font(name=F, size=10, bold=True, color=NAVY)

    # Preventability
    row = row + len(TENURE_BANDS) + 3
    section(row - 1, "Chart 3 · Churn Preventability")
    header_row(row, ["Preventability", "Records", "% Share", "Addressable?"])
    for i, lvl in enumerate(PREVENTABILITY_LEVELS):
        r = row + 1 + i
        vals = [lvl, f'=COUNTIF({PREV},A{r})', f"=IFERROR(B{r}/$B$5,0)",
                "Yes" if lvl in ADDRESSABLE_LEVELS else "No"]
        for c, v in enumerate(vals, start=1):
            cell = s2.cell(row=r, column=c, value=v)
            cell.font, cell.border = body, box
            if c == 3:
                cell.number_format = "0.0%"

    # Stage
    row = row + len(PREVENTABILITY_LEVELS) + 3
    section(row - 1, "SIP Lifecycle Stage at Churn")
    header_row(row, ["SIP Stage", "Records", "% Share"])
    for i, st in enumerate(STAGES):
        r = row + 1 + i
        for c, v in enumerate([st, f'=COUNTIF({STAGE},A{r})', f"=IFERROR(B{r}/$B$5,0)"], start=1):
            cell = s2.cell(row=r, column=c, value=v)
            cell.font, cell.border = body, box
            if c == 3:
                cell.number_format = "0.0%"

    # ---------------- Sheet 3: roadmap ----------------
    s3 = wb.create_sheet("PM Growth Roadmap")
    s3.sheet_view.showGridLines = False
    s3["A1"] = "PM Growth Roadmap — Retention Features (compliance-first)"
    s3["A1"].font = title_font
    s3["A2"] = ("Design principles: no fund/stock recommendations · no FOMO, fear or urgency nudges · "
                "cancel always available in one tap · factual, disclaimed historical context only.")
    s3["A2"].font = sub_font
    s3.merge_cells("A2:I2")
    cols = ["Priority", "Feature", "Problem Themes Addressed", "Evidence: Records in Dataset",
            "User Problem", "Solution Spec", "Compliance Guardrails", "Success Metric (North-Star)", "Phase / Timeline"]
    for c, (lab, w) in enumerate(zip(cols, [9, 26, 28, 14, 45, 60, 45, 36, 30]), start=1):
        cell = s3.cell(row=4, column=c, value=lab)
        cell.font, cell.fill, cell.alignment, cell.border = hdr_font, hdr_fill, center, box
        s3.column_dimensions[get_column_letter(c)].width = w
    s3.row_dimensions[4].height = 32
    for i, item in enumerate(ROADMAP):
        r = 5 + i
        evidence = "=" + "+".join(f'COUNTIF({THEME},"{t}")' for t in item["Themes"])
        vals = [item["Priority"], item["Feature"], ", ".join(item["Themes"]), evidence,
                item["User Problem"], item["Solution Spec"], item["Guardrails"],
                item["Success Metric"], item["Phase"]]
        for c, v in enumerate(vals, start=1):
            cell = s3.cell(row=r, column=c, value=v)
            cell.font, cell.border, cell.alignment = body, box, wrap_top
        s3.cell(row=r, column=2).font = bold
        pcell = s3.cell(row=r, column=1)
        pcell.alignment = center
        pcell.font = Font(name=F, size=10, bold=True, color="FFFFFF" if item["Priority"] == "P0" else NAVY)
        pcell.fill = PatternFill("solid", fgColor=NAVY if item["Priority"] == "P0" else (EMERALD if item["Priority"] == "P1" else LIGHT))
        s3.row_dimensions[r].height = 120
    s3.freeze_panes = "C5"

    wb.active = 1  # open on the summary sheet
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


if __name__ == "__main__":  # quick CLI smoke test
    corpus, _meta = load_real_benchmark()
    note = (f"Real benchmark: {len(corpus)} public records") if corpus is not None else DEMO_PROVENANCE
    if corpus is None:
        print(_meta.get("error")); corpus = load_demo_corpus()
    s = summarize(corpus)
    print(f"Records: {s['n']} | Cliff share: {s['cliff_share']:.1%} | Addressable: {s['addressable_share']:.1%}")
    print(s["themes"][["Problem Theme", "Records", "Dominant Emotion"]].to_string(index=False))
    with open("PM_Discovery_Workbook.xlsx", "wb") as fh:
        fh.write(build_discovery_workbook(corpus, note))
    print("Wrote PM_Discovery_Workbook.xlsx")
