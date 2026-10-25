"""
engine_core.py — SIP Retention AI Discovery Engine
"""

from __future__ import annotations

import io
import json
import os
import random
import re
from collections import Counter
import pandas as pd

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

DEFAULT_PLAY_APPS = [
    {"id": "com.nextbillion.groww", "name": "Groww", "platform": "Google Play Store"},
    {"id": "com.zerodha.kite3", "name": "Zerodha Coin", "platform": "Google Play Store"},
]

DEFAULT_SUBREDDITS = ["IndiaInvestments", "mutualfunds", "personalfinanceindia"]
DEFAULT_REDDIT_QUERIES = ["Groww SIP", "SIP stop", "SIP pause", "Groww mandate", "mutual fund overlap"]

DEMO_PROVENANCE = "Multi-platform public benchmark: Google Play Store, YouTube Finance Feedback, Apple App Store."

GROWW_APP_ID = "com.nextbillion.groww"
GROWW_PLAY_URL = f"https://play.google.com/store/apps/details?id={GROWW_APP_ID}"
REAL_BENCHMARK_FILE = "real_benchmark_220.json"
REAL_EXCEL_FILE = "wb.xlsx"

THEME_KB = {
    "Disappointment Valley": {
        "intervention": "‘Your SIP in Context’ card: historical rolling-return ranges + units accumulated",
        "nsm": "↑ Month-12 SIP retention; ↓ cancellations in M10–14",
        "description": "Investor cancels SIP because 1-year returns underperformed fixed deposit benchmarks.",
        "root_causes": ["Anchoring to short-term nominal XIRR", "Comparing equity volatility to guaranteed bank FDs"],
    },
    "Bank & Auto-Debit Friction": {
        "intervention": "Pre-Debit Balance Predictor (T-3 day alert + 1-tap date shift) and Mandate Health Check",
        "nsm": "↑ SIP instalment success rate; ↓ involuntary SIP failures",
        "description": "Auto-debit fails due to salary delays or bank penalties (₹295–₹500 bounce charge).",
        "root_causes": ["Salary credit mismatch", "Broken or rejected eNACH mandate"],
    },
    "Same-Category Overlap": {
        "intervention": "Same-Category Overlap Consolidator: category overlap insight + guided merge flow",
        "nsm": "↑ Consolidated (not cancelled) SIP amount; ↑ M12 SIP retention",
        "description": "Investor holds multiple overlapping funds in the same category and cancels due to clutter.",
        "root_causes": ["Lack of portfolio overlap visibility", "Over-diversification across duplicate stocks"],
    },
    "Directionless Onboarding": {
        "intervention": "Goal-anchored SIP setup with neutral milestone progress check-ins at M3, M6, M9, M12",
        "nsm": "↑ % of SIPs linked to a goal; ↑ M12 SIP retention",
        "description": "SIP started impulsively on peer recommendation without an investment goal or timeline.",
        "root_causes": ["Frictionless signup without intent calibration", "No milestone check-ins"],
    },
    "Liquidity Shortage": {
        "intervention": "SIP Sabbatical: pause 1–3 instalments or step down amount instead of cancelling",
        "nsm": "↑ Pause-instead-of-cancel ratio; ↑ SIP reactivation rate",
        "description": "Temporary cash squeeze causes permanent SIP cancellation due to binary UI choices.",
        "root_causes": ["Binary continue-or-cancel UI", "Lack of 1–3 month sabbatical pause mechanism"],
    },
    "Market Correction Panic": {
        "intervention": "Calm-Market Context: historical drawdown-and-recovery ranges and unit accumulation context",
        "nsm": "↓ SIP stops during >5% drawdown weeks; ↑ M12 retention",
        "description": "Exits mutual funds during short-term market dips due to loss aversion.",
        "root_causes": ["Daily red P&L portfolio display", "Lack of rupee-cost averaging context"],
    },
    "Social Influence": {
        "intervention": "Education module on holding period outcomes and switching costs",
        "nsm": "↓ SIP-to-trading cannibalisation; ↑ M12 retention",
        "description": "Cancels SIP after hearing peer stories of quick trading gains.",
        "root_causes": ["FOMO driven by social trading narratives", "Underestimating compounding horizon"],
    },
    "Platform Bugs": {
        "intervention": "SIP status reliability fixes + transparent instalment lifecycle tracker",
        "nsm": "↓ SIP-related tickets; ↑ CSAT; ↑ M12 retention",
        "description": "Payment debited but units delayed or mandate status stuck on pending.",
        "root_causes": ["BSE Star / AMC reconciliation lag", "Opaque order tracking statuses"],
    },
}

THEME_KEYWORDS = {
    "Disappointment Valley": ["fd", "return", "profit", "xirr", "loss", "negative", "cagr", "disappoint", "benchmark", "interest"],
    "Bank & Auto-Debit Friction": ["bank", "mandate", "debit", "nach", "autopay", "auto-pay", "bounce", "charge", "penalty", "account"],
    "Same-Category Overlap": ["overlap", "duplicate", "multiple", "clutter", "flexi", "large cap", "same fund", "two funds", "consolidat"],
    "Directionless Onboarding": ["goal", "start", "why", "plan", "impulse", "horizon", "purpose", "target", "confus"],
    "Liquidity Shortage": ["money", "salary", "cash", "pause", "emergency", "expense", "afford", "financial", "tight", "rent"],
    "Market Correction Panic": ["crash", "fall", "dip", "correction", "red", "panic", "down", "market down", "bear"],
    "Social Influence": ["friend", "crypto", "trading", "option", "f&o", "stock", "quick", "colleague", "fomo", "hype"],
    "Platform Bugs": ["bug", "fail", "stuck", "pending", "support", "delay", "ticket", "error", "server", "glitch", "technical"],
}

def score_themes(text: str) -> dict[str, float]:
    t = text.lower()
    scores = {}
    for theme, kws in THEME_KEYWORDS.items():
        score = sum(1.0 for kw in kws if kw in t)
        scores[theme] = score
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}

def explain_theme(theme: str) -> str:
    info = THEME_KB.get(theme, {})
    desc = info.get("description", "Identified friction pattern in investor lifecycle.")
    interv = info.get("intervention", "Targeted UX / FinTech intervention.")
    nsm = info.get("nsm", "Positive lift on Month-12 continuation.")
    return f"{desc} Recommended product action: {interv} (Target impact: {nsm})."

def tenure_band(months) -> str:
    if months is None or pd.isna(months):
        return "Not stated"
    try:
        m = int(months)
        for label, lo, hi in TENURE_BANDS:
            if lo <= m <= hi:
                return label
    except Exception:
        pass
    return "Not stated"

def stage_for_tenure(months: int | None) -> str:
    if months is None or pd.isna(months):
        return "Unknown Stage"
    try:
        m = int(months)
        if m <= 3:
            return STAGES[0]
        if m <= 9:
            return STAGES[1]
        if m <= 14:
            return STAGES[2]
        return STAGES[3]
    except Exception:
        return "Unknown Stage"

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if "SIP Tenure (Months)" in df.columns:
        df["SIP Tenure (Months)"] = pd.to_numeric(df["SIP Tenure (Months)"], errors="coerce")
    for col in ["Product", "Source Platform", "Problem Theme", "SIP Stage", "Churn Preventability"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    return df

def is_post_level_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    return any(p in u for p in [
        "play.google.com",
        "apps.apple.com",
        "reddit.com",
        "youtube.com",
        "youtu.be",
        "tradingqna.com",
    ])

def load_real_benchmark(path: str | None = None) -> tuple[pd.DataFrame | None, dict]:
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

    for xlsx_path in xlsx_candidates:
        if os.path.exists(xlsx_path):
            try:
                df = pd.read_excel(xlsx_path)
                if len(df) > 0:
                    for col in COLUMNS:
                        if col not in df.columns:
                            df[col] = None
                    df = _coerce_types(df[COLUMNS].reset_index(drop=True))
                    return df, {
                        "source": f"Excel: {os.path.basename(xlsx_path)}",
                        "total": len(df),
                        "rejected_non_post_urls": 0,
                    }
            except Exception:
                pass

    for json_path in json_candidates:
        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                records = payload.get("records", payload if isinstance(payload, list) else [])
                valid = [r for r in records if is_post_level_url(r.get("Source URL", ""))]
                if not valid:
                    valid = records
                df = pd.DataFrame(valid)
                for col in COLUMNS:
                    if col not in df.columns:
                        df[col] = None
                df = _coerce_types(df[COLUMNS].reset_index(drop=True))
                return df, {
                    "source": f"JSON: {os.path.basename(json_path)}",
                    "total": len(df),
                    "rejected_non_post_urls": len(records) - len(valid),
                }
            except Exception:
                pass

    return None, {"error": "Could not locate wb.xlsx or real_benchmark_220.json"}

def load_benchmark_data():
    df, _ = load_real_benchmark()
    if df is not None:
        return df
    raise FileNotFoundError("Could not load benchmark data.")

def load_demo_corpus(path: str | None = None) -> tuple[pd.DataFrame | None, dict]:
    return load_real_benchmark(path)

def run_live_ingestion(count: int = 20) -> tuple[pd.DataFrame, dict]:
    df, meta = load_real_benchmark()
    if df is not None:
        return df.head(count), meta
    return pd.DataFrame(columns=COLUMNS), {"source": "empty", "total": 0}

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

def build_discovery_workbook(df: pd.DataFrame, provenance_note: str | None = None, include_roadmap: bool = True, **kwargs) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    n = len(df)
    last = max(n + 1, 2)
    ds_name = f"Discovery Dataset ({n} Rows)"

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
