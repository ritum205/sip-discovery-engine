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
        "root_causes": [
            "SIP status or units displayed incorrectly, eroding trust in money security",
            "Instalment marked pending with delayed resolution from customer support",
        ],
        "emotions": ["Distrust", "Frustration", "Anxiety"],
        "prevent_mix": {PREVENT_UX: 0.85, PREVENT_NUDGE: 0.05, PREVENT_EDU: 0.05, PREVENT_LOW: 0.05},
        "intervention": "SIP status reliability fixes + transparent instalment lifecycle tracker",
        "nsm": "↓ SIP-related tickets; ↑ CSAT; ↑ M12 retention",
        "tenure_center": 6,
    },
}

# ---------------------------------------------------------------------------
# 2. SANITISATION & HELPERS
# ---------------------------------------------------------------------------

_PII_RULES = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\b[\w.-]{2,}@(ok\w+|ybl|ibl|axl|paytm|upi|apl|axisbank|icici|sbi)\b", re.I), "[upi-id]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[pan]"),
    (re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b"), "[aadhaar]"),
    (re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{9}\b"), "[phone]"),
    (re.compile(r"\b\d{9,18}\b"), "[account-no]"),
    (re.compile(r"(?<![\w/])/?u/[\w-]{3,20}", re.I), "[user]"),
    (re.compile(r"(?<![\w.])@[A-Za-z_][\w.]{2,30}"), "[handle]"),
]


def sanitize_text(text: str | None, max_len: int = 1200) -> str:
    if not text:
        return ""
    out = str(text)
    for pattern, repl in _PII_RULES:
        out = pattern.sub(repl, out)
    out = re.sub(r"\s+", " ", out).strip()
    if len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "…"
    return out


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if "SIP Tenure (Months)" in df.columns:
        df["SIP Tenure (Months)"] = pd.to_numeric(df["SIP Tenure (Months)"], errors="coerce").fillna(10).astype(int)
    for col in ["Product", "Source Platform", "Problem Theme", "SIP Stage", "Churn Preventability"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# 3. MULTI-PLATFORM DATA LOADER (wb.xlsx & real_benchmark_220.json)
# ---------------------------------------------------------------------------

def is_valid_source_url(url: str) -> bool:
    """Accepts valid public web URLs across Play Store, App Store, Reddit, and YouTube."""
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    return any(p in u for p in ["play.google.com", "apps.apple.com", "reddit.com", "youtube.com", "youtu.be", "tradingqna.com"])


def load_real_benchmark(path: str | None = None) -> tuple[pd.DataFrame | None, dict]:
    """
    Robust loader that ingests all multi-platform records (230+ rows)
    from wb.xlsx or real_benchmark_220.json without dropping non-Play Store sources.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()

    xlsx_candidates = [
        os.path.join(base_dir, REAL_EXCEL_FILE),
        os.path.join(cwd, REAL_EXCEL_FILE),
        "wb.xlsx",
        "wb_2.xlsx",
    ]
    json_candidates = [
        path or os.path.join(base_dir, REAL_BENCHMARK_FILE),
        os.path.join(cwd, REAL_BENCHMARK_FILE),
        REAL_BENCHMARK_FILE,
    ]

    # Priority 1: Direct Excel workbook
    for xlsx_path in xlsx_candidates:
        if os.path.exists(xlsx_path):
            try:
                df = pd.read_excel(xlsx_path)
                if len(df) > 0:
                    for col in COLUMNS:
                        if col not in df.columns:
                            df[col] = "Unspecified"
                    return _coerce_types(df[COLUMNS].reset_index(drop=True)), {
                        "source": f"Excel: {os.path.basename(xlsx_path)}",
                        "total": len(df),
                    }
            except Exception:
                pass

    # Priority 2: JSON benchmark
    for json_path in json_candidates:
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                records = payload.get("records", payload if isinstance(payload, list) else [])
                if len(records) > 0:
                    df = pd.DataFrame(records)
                    for col in COLUMNS:
                        if col not in df.columns:
                            df[col] = "Unspecified"
                    return _coerce_types(df[COLUMNS].reset_index(drop=True)), {
                        "source": f"JSON: {os.path.basename(json_path)}",
                        "total": len(df),
                    }
            except Exception:
                pass

    return None, {"error": "Could not locate wb.xlsx or real_benchmark_220.json"}


def load_benchmark_data():
    df, _meta = load_real_benchmark()
    if df is not None:
        return df
    raise FileNotFoundError("Neither wb.xlsx nor real_benchmark_220.json could be found.")


def load_data():
    return load_benchmark_data()


# ---------------------------------------------------------------------------
# 4. SUMMARY METRICS
# ---------------------------------------------------------------------------

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
        emo_counts = Counter(sub["Investor Emotional Trigger"])
        emo = emo_counts.most_common(1)[0][0] if emo_counts else "Frustration"
        themes.append({
            "Problem Theme": theme,
            "Records": len(sub),
            "Share": len(sub) / n,
            "Median Tenure (Months)": float(sub["SIP Tenure (Months)"].dropna().median()) if sub["SIP Tenure (Months)"].notna().any() else 10.0,
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
        "themes": theme_df,
        "top_theme": theme_df.iloc[0]["Problem Theme"] if not theme_df.empty else "Bank & Auto-Debit Friction",
    }


# ---------------------------------------------------------------------------
# 5. EXCEL EXPORT ENGINE
# ---------------------------------------------------------------------------

def build_discovery_workbook(df: pd.DataFrame, provenance_note: str | None = None) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    n = len(df)
    last = max(n + 1, 2)
    ds_name = f"Discovery Dataset ({n} Rows)"
    ds = f"'{ds_name}'"

    F = "Arial"
    hdr_font = Font(name=F, bold=True, color="FFFFFF", size=11)
    hdr_fill = PatternFill("solid", fgColor="1B365D")
    light_fill = PatternFill("solid", fgColor="F4F6F8")
    body = Font(name=F, size=10)
    thin = Side(style="thin", color="D5DBE3")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap_top = Alignment(wrap_text=True, vertical="top")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)

    wb = Workbook()
    ws = wb.active
    ws.title = ds_name

    for c, col in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=c, value=col)
        cell.font, cell.fill, cell.alignment, cell.border = hdr_font, hdr_fill, center, box
        ws.column_dimensions[get_column_letter(c)].width = 25
    ws.row_dimensions[1].height = 30

    for r, rec in enumerate(df[COLUMNS].itertuples(index=False), start=2):
        for c, val in enumerate(rec, start=1):
            cell = ws.cell(row=r, column=c, value=val)
            cell.font, cell.alignment, cell.border = body, wrap_top, box
            if r % 2 == 0:
                cell.fill = light_fill

    ws.freeze_panes = "E2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{last}"

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


if __name__ == "__main__":
    df, meta = load_real_benchmark()
    print("Meta:", meta)
    if df is not None:
        print(f"Loaded {len(df)} records.")
        print("Platform breakdown:")
        print(df["Source Platform"].value_counts())
    else:
        print("Error loading dataset.")