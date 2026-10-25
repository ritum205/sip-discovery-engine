"""
engine_core.py — SIP Retention AI Discovery Engine
====================================================
Ingestion, sanitisation, enrichment, multi-platform loading, and export module.
Supports Groww feedback across Google Play Store, Apple App Store, Reddit, and YouTube.
"""

from __future__ import annotations

import io
import json
import os
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

TENURE_BANDS = [
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
REAL_BENCHMARK_FILE = "real_benchmark_220.json"
REAL_EXCEL_FILE = "wb.xlsx"

USER_AGENT = "SIPRetentionDiscoveryEngine/1.0 (public research; zero PII)"

# ---------------------------------------------------------------------------
# 1. THEME KNOWLEDGE BASE
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
        "behaviours": ["Cancelled SIP", "Redeemed full corpus", "Paused SIP"],
        "root_causes": [
            "Return expectations anchored to FD/RD; no context on how equity returns vary over short holding periods",
            "First 9–12 months of a SIP show near-flat XIRR, read as 'the product does not work'",
            "Portfolio screen headlines absolute P&L, making a small drawdown feel like failure",
        ],
        "emotions": ["Disappointment", "Regret", "Impatience"],
        "prevent_mix": {PREVENT_NUDGE: 0.60, PREVENT_UX: 0.20, PREVENT_EDU: 0.20},
        "intervention": "‘Your SIP in Context’ card: historical rolling-return ranges by holding period + units accumulated",
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
        "behaviours": ["Let mandate lapse", "Cancelled SIP", "Switched platform"],
        "root_causes": [
            "Debit date collides with salary credit timing; bounce fee creates negative association with SIP",
            "Mandate expired or broke after a bank-account change with no proactive repair flow",
            "Repeated silent instalment failures — investor discovers gap months later",
        ],
        "emotions": ["Frustration", "Embarrassment", "Distrust"],
        "prevent_mix": {PREVENT_UX: 0.70, PREVENT_NUDGE: 0.20, PREVENT_EDU: 0.05, PREVENT_LOW: 0.05},
        "intervention": "Pre-Debit Balance Predictor (T-3 day alert + 1-tap date shift) and Mandate Health Check",
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
        "behaviours": ["Cancelled SIP", "Reduced number of SIPs", "Paused SIP"],
        "root_causes": [
            "Multiple SIPs started in same category at different times; investor discovers duplicates and cancels",
            "No portfolio view of category overlap, so cleanup defaults to cancelling rather than consolidating",
        ],
        "emotions": ["Confusion", "Overwhelm", "Self-doubt"],
        "prevent_mix": {PREVENT_UX: 0.70, PREVENT_EDU: 0.25, PREVENT_NUDGE: 0.05},
        "intervention": "Same-Category Overlap Consolidator: category-level overlap insight + guided merge flow",
        "nsm": "↑ Consolidated (not cancelled) SIP amount; ↑ M12 SIP retention",
        "tenure_center": 13,
    },
    "Directionless Onboarding": {
        "lexicon": [
            (r"\b(no|without a|didn'?t have a|never had a) (goal|plan|reason)\b", 4),
            (r"\bjust started\b|\bstarted because\b|\bsaw (a|on) (youtube|instagram|reel)\b", 2),
            (r"\bdon'?t (know|understand) (why|what)\b", 2),
            (r"\bwhat (am i|was i) (investing|saving) for\b", 4),
            (r"\bgoal\b", 1),
        ],
        "behaviours": ["Cancelled SIP", "Let mandate lapse", "Paused SIP"],
        "root_causes": [
            "SIP started without a goal or horizon; motivation fades during initial months",
            "Onboarding optimised for 2-minute conversion without connecting SIP to life goals",
        ],
        "emotions": ["Aimlessness", "Indifference", "Self-doubt"],
        "prevent_mix": {PREVENT_UX: 0.60, PREVENT_NUDGE: 0.20, PREVENT_EDU: 0.20},
        "intervention": "Goal-anchored SIP setup with neutral milestone progress check-ins at M3, M6, M9, M12",
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
        "behaviours": ["Cancelled SIP", "Redeemed full corpus", "Paused SIP"],
        "root_causes": [
            "Temporary cash-flow shock; the only visible options are continue or cancel",
            "SIP amount sized on peak income without emergency liquidity buffer",
        ],
        "emotions": ["Anxiety", "Stress", "Guilt"],
        "prevent_mix": {PREVENT_UX: 0.50, PREVENT_NUDGE: 0.25, PREVENT_LOW: 0.25},
        "intervention": "SIP Sabbatical: pause 1–3 instalments or step down amount instead of cancelling; auto-resume reminder",
        "nsm": "↑ Pause-instead-of-cancel ratio; ↑ SIP reactivation rate",
        "tenure_center": 10,
    },
    "Market Correction Panic": {
        "lexicon": [
            (r"\b(crash(ed|ing)?|correction|fall(ing)?|bloodbath|dip|down \d+%|sensex|nifty)\b", 3),
            (r"\b(scared|panic(ked)?|fear|afraid|worried)\b", 3),
            (r"\b(war|election|recession|tariff|fii selling)\b", 2),
        ],
        "behaviours": ["Paused SIP", "Cancelled SIP", "Redeemed full corpus"],
        "root_causes": [
            "First market drawdown experienced as a new investor; paper loss feels permanent",
            "Market volatility amplification without historical context in the app",
        ],
        "emotions": ["Fear", "Loss aversion", "Anxiety"],
        "prevent_mix": {PREVENT_NUDGE: 0.55, PREVENT_EDU: 0.25, PREVENT_UX: 0.15, PREVENT_LOW: 0.05},
        "intervention": "Calm-Market Context: historical drawdown-and-recovery ranges and unit accumulation context",
        "nsm": "↓ SIP stops during >5% drawdown weeks; ↑ M12 retention",
        "tenure_center": 11,
    },
    "Social Influence": {
        "lexicon": [
            (r"\b(friend|friends|colleague|cousin|brother|uncle|everyone)\b.{0,40}\b(said|told|making|earning|switched|moved)\b", 3),
            (r"\b(f&o|futures|options trading|crypto|intraday|direct stocks?)\b", 3),
            (r"\b(finfluencer|youtuber|telegram|tip)\b", 2),
        ],
        "behaviours": ["Cancelled SIP", "Redeemed full corpus", "Switched to direct stocks / F&O"],
        "root_causes": [
            "Peer narratives of fast trading returns make steady SIP feel too slow",
            "Unverified social media tips override the investor's original discipline",
        ],
        "emotions": ["Peer comparison", "Impatience", "Self-doubt"],
        "prevent_mix": {PREVENT_EDU: 0.50, PREVENT_NUDGE: 0.30, PREVENT_UX: 0.10, PREVENT_LOW: 0.10},
        "intervention": "Education module on holding period outcomes and the hidden cost of frequent strategy switching",
        "nsm": "↓ SIP-to-trading cannibalisation; ↑ M12 retention",
        "tenure_center": 12,
    },
    "Platform Bugs": {
        "lexicon": [
            (r"\b(bug|glitch|crash(es|ed)? (the )?app|app (is )?(not working|crashing|hangs))\b", 4),
            (r"\b(wrong|incorrect|not updated|mismatch)\b.{0,30}\b(nav|units|amount|value|statement|status)\b", 3),
            (r"\b(showing|shows|showed) (cancelled|failed|pending)\b", 3),
            (r"\b(support|customer care|ticket)\b.{0,30}\b(no response|useless|not respond)\b", 2),
        ],
        "behaviours": ["Switched platform", "Cancelled SIP", "Let mandate lapse"],
        "