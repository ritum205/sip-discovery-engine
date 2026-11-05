"""
SIP Guard — AI-native MVP inside a Groww-style Mutual Funds dashboard.
NextLeap PM Capstone · Part 5: Build and Deploy an AI-Native MVP

Run locally:
    pip install -r requirements.txt
    streamlit run sip_guard_app.py

What is real and what is simulated
- The risk model is a real scikit-learn GradientBoostingClassifier, trained at start-up on SYNTHETIC debit logs
  (18,000 monthly debits, 1,500 simulated investors, 12 months). No real Groww or bank data is used.
- Every eval gate is computed live: precision and fairness from a time-split backtest (train on months 1–9,
  test on months 10–12), compliance and tone gates by scanning the approved copy library with regex.
- The investor (Ritu), the portfolio and the bank accounts are fictional. Fund names are used only as labels;
  no NAVs, returns or ratings of real funds are shown.
- Evidence figures (680 records, 83-case golden set, 35% / 62% / 11%, ₹295–₹885 bounce charges) come from the
  Groww SIP Discovery Engine and public bank fee schedules.

How SIP Guard knows a debit may fail (the "data tiers")
- Tier 1 · Mandate history (first-party, no extra consent): NACH / UPI AutoPay return codes on this mandate, e.g.
  "insufficient funds". Groww already receives these for every SIP.
- Tier 2 · Account Aggregator (RBI AA framework, explicit purpose-bound consent): balance and spending around the
  debit date. Only with this tier may the notice say anything about the investor's balance.
- Evaluators can switch AA consent off: the app retrains on mandate history only, shows the honest backtest, and
  SIP Guard falls back to an evidence-based reminder instead of guessing.

Concept prototype for a NextLeap capstone. Not affiliated with or endorsed by Groww.
"""
from __future__ import annotations

import html
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score

st.set_page_config(page_title="Mutual Funds · SIP Guard", page_icon="🛡️", layout="wide")

# =================================================================================================================
# MODULE 1 · DESIGN TOKENS, GLOBAL STYLING, NAVIGATION, PORTFOLIO SHELF
# =================================================================================================================
MINT, MINT_DK, MINT_TXT = "#00D09C", "#00B386", "#00805F"
NAVY, BODY, META = "#0F172A", "#44475B", "#7C7E8C"
CANVAS, BORDER, WHITE, RED = "#F8FAFC", "#ECEFF2", "#FFFFFF", "#EF4444"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded) {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }}
.stApp {{ background: {CANVAS}; }}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stHeader"], [data-testid="stDecoration"], [data-testid="stToolbar"] {{ display: none; }}
.block-container {{ padding-top: 0.8rem; padding-bottom: 2rem; max-width: 1260px; }}

/* top nav */
.nav {{ display: flex; align-items: center; gap: 18px; background: {WHITE}; border: 1px solid {BORDER};
  border-radius: 14px; padding: 10px 18px; }}
.logo {{ display: flex; align-items: center; gap: 8px; font-weight: 700; color: {NAVY}; font-size: 17px; white-space: nowrap; }}
.logo i {{ width: 28px; height: 28px; border-radius: 50%; display: inline-block;
  background: conic-gradient({MINT} 0 50%, #5367FF 50% 100%); }}
.search {{ flex: 1 1 auto; display: flex; align-items: center; gap: 8px; background: {CANVAS}; border: 1px solid {BORDER};
  border-radius: 10px; padding: 8px 12px; color: {META}; font-size: 14px; min-width: 0; max-width: 520px; }}
.nav .r {{ margin-left: auto; display: flex; align-items: center; gap: 16px; }}
.ico {{ position: relative; display: inline-flex; color: {NAVY}; }}
.dot {{ position: absolute; top: -1px; right: -1px; width: 8px; height: 8px; background: {RED}; border-radius: 50%;
  border: 2px solid {WHITE}; }}
.avatar {{ display: inline-flex; align-items: center; gap: 8px; font-size: 14px; font-weight: 600; color: {NAVY}; }}
.avatar b {{ width: 32px; height: 32px; border-radius: 50%; background: #E6FAF5; color: {MINT_TXT}; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center; font-size: 13px; }}
.subnav {{ display: flex; gap: 26px; padding: 12px 6px 0 6px; font-size: 15px; font-weight: 500; color: {META};
  border-bottom: 1px solid {BORDER}; }}
.subnav span {{ padding-bottom: 10px; }}
.subnav .on {{ color: {NAVY}; font-weight: 600; border-bottom: 3px solid {MINT}; }}
@media (max-width: 760px) {{ .search {{ display: none; }} .avatar span {{ display: none; }} }}

/* greeting + portfolio shelf */
.hi h2 {{ margin: 18px 0 2px 0; color: {NAVY}; font-size: 24px; font-weight: 700; letter-spacing: -0.01em; }}
.hi p {{ margin: 0; color: {BODY}; font-size: 14px; }}
.sum {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); background: {WHITE}; border: 1px solid {BORDER};
  border-radius: 16px; margin: 14px 0 18px 0; box-shadow: 0 1px 3px rgba(0,0,0,.04); }}
.sum > div {{ padding: 14px 18px; border-right: 1px solid {BORDER}; min-width: 0; }}
.sum > div:last-child {{ border-right: 0; }}
.sum .k {{ font-size: 12.5px; color: {META}; }}
.sum .v {{ font-size: 20px; font-weight: 700; color: {NAVY}; margin-top: 2px; letter-spacing: -0.01em; }}
.up {{ color: #00A57D !important; }}
@media (max-width: 760px) {{ .sum {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .sum > div:nth-child(2) {{ border-right: 0; }} .sum > div:nth-child(-n+2) {{ border-bottom: 1px solid {BORDER}; }} }}

/* cards, pills, text */
.card {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 16px; padding: 16px 18px; margin-bottom: 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,.04); }}
.card h3, .h3 {{ margin: 0 0 2px 0; font-size: 16px; color: {NAVY}; font-weight: 700; }}
.sub {{ font-size: 12.5px; color: {META}; line-height: 1.5; }}
.pill {{ display: inline-flex; align-items: center; gap: 4px; border-radius: 9999px; padding: 3px 10px; font-size: 11.5px;
  font-weight: 600; white-space: nowrap; }}
.p-mint {{ color: {MINT_TXT}; background: #E6FAF5; }} .p-red {{ color: #B91C1C; background: #FEE2E2; }}
.p-amber {{ color: #B45309; background: #FEF3C7; }} .p-grey {{ color: {BODY}; background: #F1F5F9; }}

/* SIP card (keyed containers so real buttons sit inside) */
.st-key-sip_ppfas, .st-key-sip_nippon, .st-key-evaluator {{ background: {WHITE}; border: 1px solid {BORDER};
  border-radius: 16px; padding: 16px 18px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.04); }}
.st-key-evaluator {{ border: 1px dashed #94A3B8; }}
.siphead {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }}
.fundico {{ width: 40px; height: 40px; border-radius: 10px; background: #EEF2FF; color: #4338CA; font-weight: 700;
  display: inline-flex; align-items: center; justify-content: center; font-size: 13px; flex: 0 0 auto; }}
.fn {{ font-size: 15px; font-weight: 600; color: {NAVY}; }}
.meta3 {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 14px 0 4px 0; }}
.meta3 .k {{ font-size: 11.5px; color: {META}; }} .meta3 .v {{ font-size: 14px; font-weight: 600; color: {NAVY}; }}

/* SIP Guard notice (calm by design: amber, no red, no countdown) */
.guard {{ border: 1px solid #FDE68A; background: #FFFBEB; border-radius: 14px; padding: 14px 16px; margin-top: 12px; }}
.guard .t {{ font-size: 15px; font-weight: 700; color: {NAVY}; }}
.guard .b {{ font-size: 13.5px; color: {NAVY}; margin-top: 6px; line-height: 1.55; }}
.guard .fee {{ font-size: 12.5px; color: {BODY}; margin-top: 8px; }}
.ok {{ border: 1px solid #A7F3D0; background: #ECFDF5; border-radius: 14px; padding: 12px 16px; font-size: 13.5px;
  color: #065F46; line-height: 1.55; margin-top: 12px; }}
.info {{ border: 1px solid {BORDER}; background: {CANVAS}; border-radius: 14px; padding: 12px 16px; font-size: 13.5px;
  color: {NAVY}; line-height: 1.55; margin-top: 12px; }}
.fail {{ border: 1px solid #FECACA; background: #FEE2E2; border-radius: 14px; padding: 12px 16px; font-size: 13.5px;
  color: #7F1D1D; line-height: 1.55; margin-top: 12px; }}

/* debit-history strip + "your SIP so far" (evidence the investor can verify) */
.hist {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 6px; margin: 12px 0 4px 0; }}
.hist div {{ text-align: center; font-size: 11.5px; color: {META}; }}
.hist i {{ display: flex; align-items: center; justify-content: center; height: 26px; border-radius: 8px;
  font-style: normal; font-weight: 700; font-size: 13px; margin-bottom: 3px; }}
.hist .y i {{ background: #E6FAF5; color: {MINT_TXT}; }} .hist .n i {{ background: #FDE68A; color: #92400E; }}
.hist .n {{ color: #92400E; font-weight: 600; }}
.stake {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; background: {WHITE};
  border: 1px solid #FDE68A; border-radius: 10px; padding: 8px 12px; margin-top: 10px; }}
.stake .k {{ font-size: 11.5px; color: {META}; }} .stake .v {{ font-size: 14px; font-weight: 700; color: {NAVY}; }}
.guard .src {{ font-size: 11.5px; color: {META}; margin-top: 8px; border-top: 1px dashed #FDE68A; padding-top: 8px; }}

/* status timeline */
.track {{ display: flex; align-items: flex-start; margin: 14px 0 6px 0; }}
.track .s {{ display: flex; flex-direction: column; align-items: center; gap: 4px; flex: 0 0 auto; font-size: 11.5px;
  font-weight: 600; color: {META}; text-align: center; max-width: 92px; }}
.track .c {{ width: 22px; height: 22px; border-radius: 50%; border: 2px solid #CBD5E1; background: {WHITE};
  display: inline-flex; align-items: center; justify-content: center; font-size: 12px; color: {WHITE}; }}
.track .s.done .c {{ background: {MINT}; border-color: {MINT}; }} .track .s.done {{ color: {NAVY}; }}
.track .s.bad .c {{ background: {RED}; border-color: {RED}; }} .track .s.bad {{ color: #B91C1C; }}
.track .ln {{ flex: 1 1 auto; height: 2px; background: #E2E8F0; margin: 10px 4px 0 4px; }}
.track .ln.done {{ background: {MINT}; }}

/* buttons */
.stButton button {{ border-radius: 8px; font-weight: 600; min-height: 44px; transition: all .2s cubic-bezier(.4,0,.2,1); }}
.stButton button[kind="primary"] {{ background: {MINT}; border: 0; color: {WHITE}; }}
.stButton button[kind="primary"]:hover {{ background: {MINT_DK}; color: {WHITE}; }}
.stButton button[kind="secondary"] {{ background: {WHITE}; border: 1px solid #E2E8F0; color: {BODY}; }}
.stButton button[kind="secondary"]:hover {{ border-color: {MINT}; color: {MINT_TXT}; background: #F6FFFC; }}

/* evaluator telemetry */
.tele {{ display: grid; grid-template-columns: minmax(0, 1fr) 70px; gap: 6px; padding: 8px 0;
  border-bottom: 1px solid #F1F5F9; font-size: 13px; color: {NAVY}; align-items: center; }}
.tele .bar {{ grid-column: 1 / -1; height: 6px; background: #F1F5F9; border-radius: 9999px; position: relative; }}
.tele .bar span {{ position: absolute; left: 0; top: 0; bottom: 0; border-radius: 9999px; background: {NAVY}; }}
.tele .v {{ text-align: right; font-weight: 600; }}
.prob {{ display: flex; align-items: baseline; gap: 10px; margin: 6px 0 4px 0; }}
.prob b {{ font-size: 34px; color: {NAVY}; letter-spacing: -0.02em; }}
.gate {{ display: flex; justify-content: space-between; gap: 10px; padding: 10px 0; border-bottom: 1px solid #F1F5F9; }}
.gate .t {{ font-size: 13px; font-weight: 600; color: {NAVY}; }} .gate .d {{ font-size: 12px; color: {META}; margin-top: 2px; }}
.note {{ font-size: 11.5px; color: {META}; border-top: 1px solid {BORDER}; padding-top: 10px; margin-top: 22px; }}
</style>
""", unsafe_allow_html=True)

esc = lambda x: html.escape(str(x), quote=True)  # noqa: E731


def keyed(key: str):
    """Container with a CSS key (Streamlit ≥ 1.39); plain container on older versions."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


def inr(x: float) -> str:
    """Indian digit grouping: 124380 -> ₹1,24,380."""
    s = f"{abs(int(round(x))):d}"
    head, tail = s[:-3], s[-3:]
    while len(head) > 2:
        tail = head[-2:] + "," + tail
        head = head[:-2]
    return ("-" if x < 0 else "") + "₹" + ((head + "," + tail) if head else tail)


def dmy(d: date) -> str:
    return f"{d.day} {d.strftime('%b')}"


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ---- Fictional account ------------------------------------------------------------------------------------------
USER, USER_FULL = "Ritu", "Ritu M"
TODAY = date(2026, 11, 2)            # demo clock: 3 days before the 5 Nov debit (T-3), so every option is still free
SKIP_CUTOFF_DAYS = 2                 # Groww: a skip within 2 days of the instalment applies only from the next one
SIPS = [
    {"id": "ppfas", "short": "PP", "name": "Parag Parikh Flexi Cap Fund", "plan": "Direct Growth",
     "cat": "Equity · Flexi Cap", "amt": 5000, "day": 5, "bank": "HDFC Bank ••4821", "invested": 60000,
     "current": 68360, "paid": 12, "since": "Sep 2025"},
    {"id": "nippon", "short": "NI", "name": "Nippon India Small Cap Fund", "plan": "Direct Growth",
     "cat": "Equity · Small Cap", "amt": 5000, "day": 12, "bank": "ICICI Bank ••7730", "invested": 60000,
     "current": 66833, "paid": 12, "since": "Nov 2025"},
]
AA_LINK = {"bank": "HDFC Bank ••4821", "via": "Account Aggregator", "purpose": "Warn you before SIP debits that "
           "may fail", "till": "31 Mar 2027"}  # fictional consent artefact shown to the investor
DAY_GAIN = 312
for s_ in SIPS:
    s_["next"] = date(2026, 11, s_["day"])
    s_["after"] = date(2026, 12, s_["day"])
GUARDED = SIPS[0]  # the debit SIP Guard is protecting in this demo

# =================================================================================================================
# MODULE 4a · RISK MODEL (trained on synthetic debit logs) + BACKTEST
# =================================================================================================================
FEATURES = ["bal_cover", "spend_vs_usual", "fails_6m", "lowbal_months_6m", "debit_day", "debits_same_day"]
MANDATE_FEATURES = ["fails_6m", "debit_day", "debits_same_day"]   # Tier 1: Groww already has these, no AA needed
FEATURE_LABEL = {
    "bal_cover": "Projected balance on debit day ÷ SIP amount",
    "spend_vs_usual": "Spending this month vs usual",
    "fails_6m": "Failed debits on this mandate (6 months)",
    "lowbal_months_6m": "Months with a low balance near this date (of 6)",
    "debit_day": "Debit day of month",
    "debits_same_day": "Debits on the same day",
}
ALERT_AT, FALLBACK_AT = 0.60, 0.30   # ≥60%: T-3 SIP Guard notice · 30–59%: standard T-1 reminder · <30%: silent
PRECISION_FLOOR = 0.60
COVERAGE_FLOOR = 0.20                # the notice must reach ≥20% of real failures, or it isn't worth shipping
FAIRNESS_MAX_GAP = 0.10              # max precision gap between PSU and private banks


def synthetic_debit_logs(seed: int = 42, n_users: int = 1500, months: int = 12) -> pd.DataFrame:
    """Simulated monthly SIP debits. Failure depends mostly on cash cover on the day and spending vs usual."""
    rng = np.random.default_rng(seed)
    n = n_users * months
    u = np.repeat(np.arange(n_users), months)
    month = np.tile(np.arange(1, months + 1), n_users)
    bank_type = np.where(rng.random(n_users) < 0.45, "PSU", "Private")[u]
    debit_day = rng.choice([1, 3, 5, 7, 10, 12, 15, 20, 25, 28], n_users)[u]
    tight = rng.beta(2, 4, n_users)[u]                                    # how stretched this investor usually is
    spend = np.clip(rng.lognormal(0, 0.2, n), 0.6, 2.2)
    cover = np.clip(np.exp(rng.normal(np.log(3.0) - 2.4 * tight, 0.6)) / spend, 0.05, 12)
    lowbal = np.clip(np.round(tight * 7 + rng.normal(0, 1.2, n)), 0, 6)
    fails = np.clip(rng.poisson(tight * 1.5, n), 0, 3)
    same_day = rng.choice([1, 2, 3], n_users, p=[0.6, 0.3, 0.1])[u]
    late = np.isin(debit_day, [25, 28]).astype(float)
    logit = (-3.1 - 2.4 * np.log(cover) + 1.4 * (spend - 1) + 0.5 * fails + 0.2 * lowbal
             + 0.25 * (same_day - 1) + 0.3 * late + rng.normal(0, 0.5, n))
    failed = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"month": month, "bank_type": bank_type, "bal_cover": cover, "spend_vs_usual": spend,
                         "fails_6m": fails, "lowbal_months_6m": lowbal, "debit_day": debit_day,
                         "debits_same_day": same_day, "failed": failed})


@st.cache_resource(show_spinner="Training the SIP Guard risk model on synthetic debit logs…")
def train_and_backtest(features: tuple[str, ...] = tuple(FEATURES)) -> dict:
    features = list(features)
    df = synthetic_debit_logs()
    train, test = df[df.month <= 9], df[df.month >= 10]          # time split: no peeking at the future
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=7)
    model.fit(train[features], train.failed)                     # bank type is NOT a feature (used only to audit)
    p = model.predict_proba(test[features])[:, 1]
    flag = p >= ALERT_AT
    by_bank = {}
    for b in ("PSU", "Private"):
        k = (test.bank_type == b).to_numpy()
        by_bank[b] = float(precision_score(test.failed[k], flag[k], zero_division=0))
    return {
        "model": model, "features": features,
        "rows": len(df), "train_rows": len(train), "test_rows": len(test),
        "base_rate": float(df.failed.mean()),
        "precision": float(precision_score(test.failed, flag, zero_division=0)),
        "recall": float(recall_score(test.failed, flag, zero_division=0)),
        "alerts": int(flag.sum()),
        "by_bank": by_bank,
        "medians": train[features].median().to_dict(),
    }


MODELS = {"aa": train_and_backtest(tuple(FEATURES)), "mandate": train_and_backtest(tuple(MANDATE_FEATURES))}

# Ritu's live feature vectors for the 5 Nov debit (evaluators can switch scenario to test the gates).
# "hist" = the last 6 debits on this mandate (first-party NACH / AutoPay return data); it matches fails_6m.
SCENARIOS = {
    "high": {"label": "Cash crunch this month",
             "x": {"bal_cover": 0.40, "spend_vs_usual": 1.60, "fails_6m": 2, "lowbal_months_6m": 5, "debit_day": 5,
                   "debits_same_day": 1},
             "hist": [("May", 1), ("Jun", 0), ("Jul", 1), ("Aug", 1), ("Sep", 0), ("Oct", 1)]},
    "unsure": {"label": "Borderline",
               "x": {"bal_cover": 0.70, "spend_vs_usual": 1.35, "fails_6m": 1, "lowbal_months_6m": 3, "debit_day": 5,
                     "debits_same_day": 1},
               "hist": [("May", 1), ("Jun", 1), ("Jul", 1), ("Aug", 0), ("Sep", 1), ("Oct", 1)]},
    "low": {"label": "Healthy balance",
            "x": {"bal_cover": 3.00, "spend_vs_usual": 0.90, "fails_6m": 0, "lowbal_months_6m": 0, "debit_day": 5,
                  "debits_same_day": 1},
            "hist": [("May", 1), ("Jun", 1), ("Jul", 1), ("Aug", 1), ("Sep", 1), ("Oct", 1)]},
}
NIPPON_X = {"bal_cover": 3.2, "spend_vs_usual": 1.0, "fails_6m": 0, "lowbal_months_6m": 0, "debit_day": 12,
            "debits_same_day": 1}


def active_model() -> dict:
    """Tier 2 model when the investor has given AA consent, otherwise the mandate-history-only model."""
    return MODELS["aa" if st.session_state.get("aa_consent", True) else "mandate"]


def score(x: dict, m: dict | None = None) -> float:
    m = m or active_model()
    return float(m["model"].predict_proba(pd.DataFrame([x])[m["features"]])[0, 1])


def drivers(x: dict) -> list[tuple[str, str, float]]:
    """How much each feature raises the score vs a typical (median) investor: p(x) − p(x with feature at median)."""
    m = active_model()
    base = score(x, m)
    out = []
    for f in m["features"]:
        x2 = dict(x)
        x2[f] = m["medians"][f]
        out.append((FEATURE_LABEL[f], f"{x[f]:g}", base - score(x2, m)))
    return sorted(out, key=lambda r: -r[2])


# =================================================================================================================
# APPROVED COPY LIBRARY (the app only fills slots; nothing is free-generated) + LINT RULES
# =================================================================================================================
TEMPLATES = {
    # Every factual claim in the notice maps to a data source the investor can see on the card:
    #   "{fails} of your last 6 debits"  -> mandate return history (Tier 1, shown as the 6-month strip)
    #   "balance was below {amt} ..."    -> Account Aggregator data (Tier 2, consent named in the source line)
    "notice_t": {"ok": True, "text": "⚡ SIP Guard: your {date} auto-debit may bounce"},
    "notice_b": {"ok": True, "text": "Your upcoming {amt} debit on {date} ({bank}) may not go through: {fails} of your "
                                     "last 6 debits on this mandate didn't go through, and in {lowbal} of the last 6 "
                                     "months your balance ran below {amt} around the {day_of}. Skip a ₹295–₹885 bank "
                                     "bounce charge and keep your SIP compounding: pick an option below while all of "
                                     "them are free. Nothing changes unless you choose."},
    "notice_b_mandate": {"ok": True, "text": "Your upcoming {amt} debit on {date} ({bank}) may not go through. "
                                             "{fails} of your last 6 debits on this mandate didn't go through due "
                                             "to low balance. Pick an option below while all of them are free. "
                                             "Nothing changes unless you choose."},
    "notice_hist": {"ok": True, "text": "Your last 6 debits for this SIP"},
    "notice_stake": {"ok": True, "text": "Skipping one month keeps your SIP running. If 3 instalments in a row are "
                                         "missed or skipped, the fund house stops the SIP and you'd need a new "
                                         "mandate to restart."},
    "notice_fee": {"ok": True, "text": "For reference: if a debit fails, banks usually charge ₹295–₹885 incl. GST."},
    "notice_src": {"ok": True, "text": "Why you're seeing this: your debit history on this mandate, plus {bank} "
                                       "balance data you shared via Account Aggregator (consent till {till}). We "
                                       "never read SMS."},
    "notice_src_mandate": {"ok": True, "text": "Why you're seeing this: the debit history on this mandate, as "
                                               "reported by {bank}. We never read SMS."},
    "reminder": {"ok": True, "text": "Reminder: {amt} will be debited from {bank} on {date}. Please keep enough "
                                     "balance."},
    "reminder_hist": {"ok": True, "text": "{fails} of your last 6 debits for this SIP didn't go through. Reminder: "
                                          "{amt} will be debited from {bank} on {date}."},
    "o_upi": {"ok": True, "text": "⚡ Skip this month's AutoPay & pay {amt} via UPI"},
    "o_upi_d": {"ok": True, "text": "Pay {amt} from any UPI app or another bank. The {date} AutoPay debit is not sent "
                                    "to {bank}, so it can't bounce."},
    "o_skip": {"ok": True, "text": "⏸️ Skip this month"},
    "o_skip_d": {"ok": True, "text": "This instalment isn't sent to your bank. Your SIP stays active and resumes on "
                                     "{after}."},
    "o_shift": {"ok": True, "text": "📅 Move future SIP dates"},
    "o_shift_d": {"ok": True, "text": "Pick a day that suits your cash flow. Applies from December."},
    "o_keep": {"ok": True, "text": "✅ Keep the debit as it is"},
    "o_keep_d": {"ok": True, "text": "Nothing changes. We'll send one reminder the day before."},
    "skip_rule": {"ok": True, "text": "Note: skips count toward the AMCs' rule that 3 missed or skipped instalments "
                                      "in a row cancel a SIP ({used} of 3 used after this skip)."},
    "recovery": {"ok": True, "text": "Your {date} debit didn't go through. You can still invest this month by paying "
                                     "via UPI."},
    "rogue": {"ok": False, "text": "Markets are rallying! Don't miss out: top up your best fund now for guaranteed "
                                   "returns."},
}
CONFIRM = {
    "upi": "<b>Paid {amt} via UPI.</b> AutoPay for {date} is paused for this instalment, so there's no duplicate "
           "debit and no bank charge. Units at the NAV of the payment date.",
    "skip": "<b>Skipped for November.</b> SIP remains active. Not sent to your bank, so no bank charge. "
            "Resumes automatically on {after}.",
    "shift": "<b>Date moved.</b> From December this SIP debits on the {day}. The {date} debit is still scheduled; "
             "we'll send one reminder on {t1}.",
    "keep": "<b>Debit kept.</b> We'll send one reminder on {t1} to keep {amt} in {bank}.",
}
STATUS = {
    "upi": "🟢 Paid via UPI · AutoPay paused (₹0 bank charge)",
    "skip": "🟢 Skipped · SIP active (₹0 bank charge)",
    "shift": "🟢 From Dec: debits on the {day}",
    "keep": "🔔 Kept · reminder on {t1}",
}
TOASTS = {
    "upi": "{amt} paid via UPI · AutoPay paused for {date}",
    "skip": "November instalment skipped · SIP stays active",
    "shift": "From December, this SIP debits on the {day}",
    "keep": "Debit kept · reminder on {t1}",
    "paid_after": "{amt} paid via UPI · units at the NAV of the payment date",
}
# Non-advisory guardrail (SEBI): no fund picks, amounts or market claims.
ADVICE = re.compile(r"\b(guarantee\w*|best fund|recommend\w*|buy|sell|rally\w*|top up your|switch to|"
                    r"outperform\w*|returns will|should invest)\b", re.I)
# Tone guardrail: no fear, FOMO or urgency in anything the investor sees.
PRESSURE = re.compile(r"\b(now|hurry|urgent\w*|immediately|avoid|last chance|don'?t miss|miss out|act fast|"
                      r"too late|lose|losing|only today|high risk|alert|warning|penalt\w*)\b", re.I)


def t(key: str, **kw) -> str:
    return TEMPLATES[key]["text"].format(**kw)


def strip_tags(x: str) -> str:
    return re.sub(r"<[^>]+>", "", x)


def library_texts() -> list[str]:
    return ([v["text"] for v in TEMPLATES.values() if v["ok"]] + list(CONFIRM.values()) + list(STATUS.values())
            + list(TOASTS.values()))


# =================================================================================================================
# MODULE 3 · STATE MACHINE (st.session_state)
# =================================================================================================================
ss = st.session_state
ss.setdefault("choice", None)          # None | upi | skip | shift | keep
ss.setdefault("shift_day", 8)
ss.setdefault("t0", None)              # None | not_presented | success | failed
ss.setdefault("paid_after", False)
ss.setdefault("balance_added", False)
ss.setdefault("flag_on", True)         # kill switch
ss.setdefault("eval_mode", False)
ss.setdefault("scenario", "high")
ss.setdefault("rogue", False)
ss.setdefault("prior_skips", 0)        # consecutive skips/misses before this instalment
ss.setdefault("aa_consent", True)      # Tier 2 data: Account Aggregator consent for HDFC ••4821
ss.setdefault("log", [])
ss.setdefault("pending_toast", None)   # toasts raised inside the dialog survive its st.rerun()
if ss.pending_toast:
    st.toast(ss.pending_toast)
    ss.pending_toast = None


def fmt(x: str) -> str:
    xs = SCENARIOS[ss.scenario]["x"]
    return x.format(amt=inr(GUARDED["amt"]), date=dmy(GUARDED["next"]), after=dmy(GUARDED["after"]),
                    bank=GUARDED["bank"], fund=GUARDED["name"], day=ordinal(ss.shift_day),
                    t1=dmy(GUARDED["next"] - timedelta(days=1)), fails=int(xs["fails_6m"]),
                    lowbal=int(xs["lowbal_months_6m"]), day_of=ordinal(GUARDED["day"]), till=AA_LINK["till"])


def tf(key: str) -> str:
    """Approved template with every slot filled from the live account (no free text)."""
    return fmt(TEMPLATES[key]["text"])


def log(event: str) -> None:
    ss.log = ss.log + [event]


def reset() -> None:
    ss.choice, ss.t0, ss.paid_after, ss.balance_added = None, None, False, False
    ss.log = []


def choose(c: str) -> None:
    ss.choice = c
    log({"upi": "UPI payment recorded · AutoPay presentation paused", "skip": "Skip request sent before cut-off",
         "shift": f"Future SIP date → {ordinal(ss.shift_day)}", "keep": "Kept · T-1 reminder queued"}[c])
    ss.pending_toast = fmt(TOASTS[c])


def simulate_debit_day() -> None:
    risky = ss.scenario == "high"
    if ss.choice in ("upi", "skip"):
        ss.t0 = "not_presented"
    elif ss.balance_added or not risky:
        ss.t0 = "success"
    else:
        ss.t0 = "failed"
    log(f"Debit day ({dmy(GUARDED['next'])}): " + {"not_presented": "not presented to bank", "success": "debited",
                                                  "failed": "bounced (insufficient balance)"}[ss.t0])


def pay_after_failure() -> None:
    ss.paid_after = True
    log("Paid via UPI after the failed debit")
    st.toast(fmt(TOASTS["paid_after"]))


# =================================================================================================================
# MODULE 4b · EVAL GATES (computed before rendering: they decide what the investor is allowed to see)
# =================================================================================================================
MODEL = active_model()
x_live = SCENARIOS[ss.scenario]["x"]
hist_live = SCENARIOS[ss.scenario]["hist"]
p_live = score(x_live)
band = "high" if p_live >= ALERT_AT else ("unsure" if p_live >= FALLBACK_AT else "low")
p_nippon = score(NIPPON_X)
model_ok = MODEL["precision"] >= PRECISION_FLOOR and MODEL["recall"] >= COVERAGE_FLOOR

shown = []
if ss.flag_on and band == "high" and ss.t0 is None and not ss.choice:
    # The balance claim is only allowed when AA data backs it; otherwise the mandate-only wording is used.
    shown += (["notice_t", "notice_b", "notice_hist", "notice_stake", "notice_fee", "notice_src"] if ss.aa_consent
              else ["notice_t", "notice_b_mandate", "notice_hist", "notice_stake", "notice_fee", "notice_src_mandate"])
    if ss.rogue:
        shown.append("rogue")
on_screen = " ".join(TEMPLATES[k]["text"] for k in shown)
lib_and_screen = " ".join(strip_tags(x) for x in library_texts()) + " " + on_screen
advice_hits = sorted({m.group(0) for m in ADVICE.finditer(lib_and_screen)})
tone_hits = sorted({m.group(0).lower() for m in PRESSURE.finditer(lib_and_screen)})
unapproved = [k for k in shown if not TEMPLATES[k]["ok"]]
gap = abs(MODEL["by_bank"]["PSU"] - MODEL["by_bank"]["Private"])

GATES = [
    ("Gate 1 · Precision ≥ 60% and coverage ≥ 20%", model_ok,
     f"{MODEL['precision']:.1%} of T-3 notices were real failures and they reached {MODEL['recall']:.1%} of all "
     f"failures, on a time-split backtest (train months 1–9, test months 10–12; {MODEL['test_rows']:,} synthetic "
     f"debits). Data: {'mandate history + Account Aggregator' if ss.aa_consent else 'mandate history only'}."),
    ("Gate 2 · Fairness across banks (gap ≤ 10 pts)", gap <= FAIRNESS_MAX_GAP,
     f"Precision PSU {MODEL['by_bank']['PSU']:.1%} vs private {MODEL['by_bank']['Private']:.1%} "
     f"(gap {gap * 100:.1f} pts). Bank type is not a model input; it is only used for this audit."),
    ("Gate 3 · Non-advisory guardrail (SEBI)", not advice_hits,
     "0 fund picks, amounts or market claims in the copy library or on screen." if not advice_hits
     else "Blocked: " + ", ".join(f"“{h}”" for h in advice_hits)),
    ("Gate 4 · Safe fallback (< 60% confidence)", True,
     "Inconclusive score → showing the standard T-1 reminder instead of a notice." if band == "unsure"
     else "Scores of 30–59% fall back to the standard T-1 reminder; below 30% nothing is shown."),
    ("Gate 5 · No fear, FOMO or urgency", not tone_hits,
     "0 pressure words in the approved copy library or on screen." if not tone_hits
     else "Blocked: " + ", ".join(f"“{h}”" for h in tone_hits)),
    ("Gate 6 · Template compliance", not unapproved,
     "100% pre-approved copy; slots only, no free text." if not unapproved
     else f"Blocked: {len(unapproved)} unapproved message(s) tried to render."),
]
gates_ok = all(ok for _, ok, _ in GATES)
guard_on = ss.flag_on and band == "high" and gates_ok

# =================================================================================================================
# RENDER · MODULE 1: NAV + GREETING + PORTFOLIO SHELF
# =================================================================================================================
SEARCH = ('<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="6.5" fill="none" '
          'stroke="currentColor" stroke-width="2"/><path d="M16 16l4.5 4.5" stroke="currentColor" stroke-width="2" '
          'stroke-linecap="round"/></svg>')
BELL = ('<svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 16V11a6 6 0 1 1 12 0v5l1.5 2h-15z'
        'M10 20a2 2 0 0 0 4 0" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>')
st.markdown(
    f'<div class="nav"><span class="logo"><i></i>Groww</span>'
    f'<span class="search">{SEARCH}Search mutual funds, SIPs…</span>'
    f'<div class="r"><span class="ico">{BELL}<span class="dot"></span></span>'
    f'<span class="avatar"><b>RM</b><span>{USER_FULL}</span></span></div></div>'
    f'<div class="subnav"><span class="on">Mutual Funds</span><span>SIP Dashboard</span><span>Orders</span></div>',
    unsafe_allow_html=True)

g_l, g_r = st.columns([3, 1.4], vertical_alignment="bottom")
with g_l:
    st.markdown(f'<div class="hi"><h2>Welcome back, {USER} 👋</h2><p>Your mutual fund portfolio and upcoming SIP '
                f'debits · today {dmy(TODAY)} 2026 (demo clock)</p></div>', unsafe_allow_html=True)
with g_r:
    st.toggle("🔍 Mentor / Evaluator Mode", key="eval_mode",
              help="Inspect the AI eval-gate architecture, model telemetry and the kill switch.")

INVESTED = sum(s["invested"] for s in SIPS)
CURRENT = sum(s["current"] for s in SIPS)
gain = CURRENT - INVESTED
st.markdown(
    f'<div class="sum"><div><div class="k">Total invested</div><div class="v">{inr(INVESTED)}</div></div>'
    f'<div><div class="k">Current value</div><div class="v">{inr(CURRENT)}</div></div>'
    f'<div><div class="k">1-day returns</div><div class="v up">+{inr(DAY_GAIN)} <span style="font-size:14px">'
    f'(+{DAY_GAIN / (CURRENT - DAY_GAIN) * 100:.2f}%)</span></div></div>'
    f'<div><div class="k">Total returns</div><div class="v up">+{inr(gain)} <span style="font-size:14px">'
    f'(+{gain / INVESTED * 100:.1f}%)</span></div></div></div>', unsafe_allow_html=True)

main, side = st.columns([1.65, 1], gap="large")


# =================================================================================================================
# RENDER · MODULE 2 + 3: RUNNING SIPS, SIP GUARD NOTICE, 1-TAP ACTION SHEET, TIMELINE
# =================================================================================================================
def tracker(stages: list[tuple[str, str]]) -> str:
    out = []
    for i, (label, s) in enumerate(stages):
        if i:
            out.append(f'<span class="ln {"done" if s in ("done", "bad") else ""}"></span>')
        mark = "✓" if s == "done" else ("!" if s == "bad" else "")
        out.append(f'<span class="s {s}"><span class="c">{mark}</span>{esc(label)}</span>')
    return '<div class="track">' + "".join(out) + "</div>"


@st.dialog("Handle your 5 Nov debit")
def action_sheet() -> None:
    g = GUARDED
    st.markdown(f'<div class="sub">{esc(g["name"])} · {inr(g["amt"])} · {dmy(g["next"])} · {esc(g["bank"])}. '
                f'All four options are free because it is more than {SKIP_CUTOFF_DAYS} days before the debit.</div>',
                unsafe_allow_html=True)
    st.write("")
    if st.button(tf("o_upi"), type="primary", use_container_width=True, key="d_upi"):
        choose("upi")
        st.rerun()
    st.caption(fmt(t("o_upi_d", amt="{amt}", date="{date}", bank="{bank}")))
    if st.button(t("o_skip"), use_container_width=True, key="d_skip"):
        choose("skip")
        st.rerun()
    st.caption(fmt(t("o_skip_d", after="{after}")) + " " + t("skip_rule", used=ss.prior_skips + 1))
    st.markdown(f'<div style="font-weight:600;font-size:14px;margin-top:6px">{esc(t("o_shift"))}</div>',
                unsafe_allow_html=True)
    ss.shift_day = st.select_slider("New SIP day", options=list(range(6, 29)), value=ss.shift_day,
                                    format_func=ordinal, label_visibility="collapsed")
    if st.button(f"Move to the {ordinal(ss.shift_day)} from December", use_container_width=True, key="d_shift"):
        choose("shift")
        st.rerun()
    st.caption(t("o_shift_d"))
    if st.button(t("o_keep"), use_container_width=True, key="d_keep"):
        choose("keep")
        st.rerun()
    st.caption(t("o_keep_d"))


with main:
    st.markdown(f'<div class="h3" style="margin:2px 0 10px 2px">Running SIPs <span class="sub">· {len(SIPS)} active · '
                f'{inr(sum(s["amt"] for s in SIPS))}/month</span></div>', unsafe_allow_html=True)

    # ---- SIP 1: the guarded debit ----
    g = GUARDED
    with keyed(f"sip_{g['id']}"):
        if ss.t0 is None:
            if guard_on and not ss.choice:
                pill = '<span class="pill p-amber">Needs a look</span>'
            elif ss.choice and ss.flag_on:
                pill = f'<span class="pill p-mint">{esc(fmt(STATUS[ss.choice]))}</span>'
            else:
                pill = '<span class="pill p-grey">Scheduled</span>'
        elif ss.t0 == "failed" and not ss.paid_after:
            pill = '<span class="pill p-red">Debit failed</span>'
        else:
            pill = '<span class="pill p-mint">Invested</span>'
        next_label = dmy(g["next"]) + (f" · then {ordinal(ss.shift_day)}" if ss.choice == "shift" else "")
        st.markdown(
            f'<div class="siphead"><div style="display:flex;gap:12px;align-items:center"><span class="fundico">'
            f'{g["short"]}</span><div><div class="fn">{esc(g["name"])}</div><div class="sub">{esc(g["plan"])} · '
            f'{esc(g["cat"])}</div></div></div>{pill}</div>'
            f'<div class="meta3"><div><div class="k">SIP amount</div><div class="v">{inr(g["amt"])}/mo</div></div>'
            f'<div><div class="k">Next debit</div><div class="v">{next_label}</div></div>'
            f'<div><div class="k">AutoPay</div><div class="v">{esc(g["bank"])}</div></div></div>',
            unsafe_allow_html=True)

        if ss.t0 is None:                                           # ---- before debit day ----
            if guard_on and not ss.choice:
                body_key, src_key = (("notice_b", "notice_src") if ss.aa_consent
                                     else ("notice_b_mandate", "notice_src_mandate"))
                hist = "".join(f'<div class="{"y" if ok else "n"}"><i>{"✓" if ok else "✕"}</i>{m}</div>'
                               for m, ok in hist_live)
                st.markdown(
                    f'<div class="guard"><div class="t">{esc(tf("notice_t"))}</div>'
                    f'<div class="b">{esc(tf(body_key))}</div>'
                    f'<div class="sub" style="margin-top:10px;font-weight:600;color:{NAVY}">{esc(tf("notice_hist"))}'
                    f'</div><div class="hist">{hist}</div>'
                    f'<div class="stake"><div><div class="k">Instalments paid</div><div class="v">{g["paid"]} since '
                    f'{esc(g["since"])}</div></div><div><div class="k">Invested</div><div class="v">'
                    f'{inr(g["invested"])}</div></div><div><div class="k">Current value</div><div class="v up">'
                    f'{inr(g["current"])}</div></div></div>'
                    f'<div class="fee">{esc(tf("notice_stake"))}</div>'
                    f'<div class="fee">{esc(tf("notice_fee"))}</div>'
                    + (f'<div class="b" style="color:#B91C1C">{esc(t("rogue"))}</div>' if "rogue" in shown else "")
                    + f'<div class="src">🔒 {esc(tf(src_key))}</div></div>', unsafe_allow_html=True)
                if st.button("See my options · 1 tap", type="primary", use_container_width=True, key="resolve"):
                    action_sheet()
                st.markdown(tracker([(f"Notified {dmy(TODAY)}", "done"), ("Your choice", ""),
                                     (f"Debit day {dmy(g['next'])}", ""), ("Units allotted", "")]),
                            unsafe_allow_html=True)
            elif ss.choice and ss.flag_on:
                st.markdown(f'<div class="ok">{fmt(CONFIRM[ss.choice])}</div>', unsafe_allow_html=True)
                if ss.choice == "skip":
                    st.caption(t("skip_rule", used=ss.prior_skips + 1))
                if ss.choice in ("shift", "keep"):
                    ss.balance_added = st.checkbox(f"I've added {inr(g['amt'])} to {g['bank']}",
                                                   value=ss.balance_added, key="bal_chk")
                st.markdown(tracker([(f"Notified {dmy(TODAY)}", "done"), ("Your choice", "done"),
                                     (f"Debit day {dmy(g['next'])}", ""), ("Units allotted", "")]),
                            unsafe_allow_html=True)
            elif ss.flag_on and band == "unsure":
                rem = tf("reminder_hist" if x_live["fails_6m"] > 0 else "reminder")
                st.markdown(f'<div class="info">🔔 {esc(rem)}'
                            '<div class="sub" style="margin-top:6px">Safe fallback: the model was not confident '
                            'enough for a SIP Guard notice.</div></div>', unsafe_allow_html=True)
            elif ss.flag_on and band == "high" and not gates_ok:
                rem = tf("reminder_hist" if x_live["fails_6m"] > 0 else "reminder")
                st.markdown(f'<div class="info">🔔 {esc(rem)}'
                            '<div class="sub" style="margin-top:6px">An eval gate failed, so SIP Guard fell back to '
                            'the standard reminder.</div></div>', unsafe_allow_html=True)
            elif ss.flag_on and x_live["fails_6m"] > 0:              # Tier 1 rule: past bounces always earn a T-1 nudge
                st.markdown(f'<div class="info">🔔 {esc(tf("reminder_hist"))}<div class="sub" style="margin-top:6px">'
                            'Low model score, but this mandate has bounced before, so a T-1 reminder is queued.'
                            '</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info">Auto-debit scheduled for {dmy(g["next"])}. No action needed.</div>',
                            unsafe_allow_html=True)
            st.button(f"⏩ Simulate debit day ({dmy(g['next'])})", use_container_width=True,
                      on_click=simulate_debit_day, key="sim_t0")
        else:                                                       # ---- debit day ----
            if ss.t0 == "not_presented":
                st.markdown(f'<div class="ok">✓ Nothing was presented to {esc(g["bank"])} on {dmy(g["next"])}, so '
                            'there was no bounce and no bank charge. '
                            + (f'{inr(g["amt"])} was invested via UPI.' if ss.choice == "upi"
                               else f'Instalment skipped; next debit {dmy(g["after"])}.') + '</div>',
                            unsafe_allow_html=True)
                st.markdown(tracker([("Notified", "done"), ("Your choice", "done"), ("Not presented", "done"),
                                     ("₹0 charge", "done")]), unsafe_allow_html=True)
            elif ss.t0 == "success" or ss.paid_after:
                st.markdown(f'<div class="ok">✓ {inr(g["amt"])} '
                            + ("paid via UPI after the failed debit." if ss.paid_after
                               else f'debited from {esc(g["bank"])}.') + ' Units are being allotted.</div>',
                            unsafe_allow_html=True)
                st.markdown(tracker([("Scheduled", "done"), ("Paid", "done"), ("Units allotted", "done")]),
                            unsafe_allow_html=True)
            elif not ss.flag_on or not gates_ok:
                st.markdown('<div class="fail"><b>SIP instalment failed.</b></div>'
                            '<div class="sub" style="margin-top:6px">Standard behaviour without SIP Guard: no warning '
                            'beforehand. The bank may charge ₹295–₹885 incl. GST, and 3 failed instalments in a row '
                            'cancel the SIP.</div>', unsafe_allow_html=True)
                st.markdown(tracker([("Scheduled", "done"), ("Debit bounced", "bad"), ("Miss 1 of 3", "bad")]),
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="fail">{esc(fmt(t("recovery", date="{date}")))}</div>', unsafe_allow_html=True)
                st.button(f"Pay {inr(g['amt'])} via UPI to keep this month's instalment", type="primary",
                          use_container_width=True, on_click=pay_after_failure, key="pay_after")
                st.markdown(tracker([("Scheduled", "done"), ("Debit bounced", "bad"), ("Recover via UPI", "")]),
                            unsafe_allow_html=True)
            st.button("↺ Replay from T-3", use_container_width=True, on_click=reset, key="replay")

    # ---- SIP 2: low risk, no intervention ----
    n = SIPS[1]
    with keyed(f"sip_{n['id']}"):
        st.markdown(
            f'<div class="siphead"><div style="display:flex;gap:12px;align-items:center"><span class="fundico">'
            f'{n["short"]}</span><div><div class="fn">{esc(n["name"])}</div><div class="sub">{esc(n["plan"])} · '
            f'{esc(n["cat"])}</div></div></div><span class="pill p-grey">Scheduled</span></div>'
            f'<div class="meta3"><div><div class="k">SIP amount</div><div class="v">{inr(n["amt"])}/mo</div></div>'
            f'<div><div class="k">Next debit</div><div class="v">{dmy(n["next"])}</div></div>'
            f'<div><div class="k">AutoPay</div><div class="v">{esc(n["bank"])}</div></div></div>',
            unsafe_allow_html=True)

    # ---- Holdings ----
    rows = "".join(
        f'<div style="display:flex;justify-content:space-between;gap:12px;padding:10px 0;border-top:1px solid {BORDER}">'
        f'<div><div class="fn" style="font-size:14px">{esc(s["name"])}</div><div class="sub">{s["paid"]} instalments '
        f'paid</div></div><div style="text-align:right"><div class="fn" style="font-size:14px">{inr(s["current"])}</div>'
        f'<div class="sub">Invested {inr(s["invested"])}</div></div></div>' for s in SIPS)
    st.markdown(f'<div class="card"><h3>Your holdings</h3>{rows}</div>', unsafe_allow_html=True)

    if ss.log:
        with st.expander("Activity log (this session)"):
            for e in ss.log:
                st.write("• " + e)

# =================================================================================================================
# RENDER · SIDE: SIP GUARD INFO, or MODULE 4: EVALUATOR TELEMETRY DRAWER
# =================================================================================================================
with side:
    if not ss.eval_mode:
        st.markdown(
            f'<div class="card"><h3>🛡️ SIP Guard</h3><div class="sub">Checks each upcoming AutoPay debit 3 days early. '
            f'If one looks likely to fail, you get one notice with 1-tap options while they are still free. It only '
            f'changes how this month\'s money moves, never which fund or how much.</div><div style="margin-top:10px">'
            f'<span class="pill {"p-mint" if ss.flag_on else "p-grey"}">{"On" if ss.flag_on else "Off"}</span></div>'
            f'</div>'
            f'<div class="card"><h3>AutoPay mandates</h3><div class="sub" style="font-size:13px;line-height:1.8">'
            + "".join(f'{esc(s["bank"])} · {esc(s["name"].split(" Fund")[0])} · {ordinal(s["day"])}<br>' for s in SIPS)
            + '</div></div>'
            + f'<div class="card"><h3>🔒 Data SIP Guard uses</h3><div class="sub" style="font-size:13px;line-height:1.7">'
              f'<b style="color:{NAVY}">Debit history</b> · success / failure of each AutoPay debit, as your bank '
              f'reports it to Groww.<br><b style="color:{NAVY}">Account Aggregator</b> · '
            + (f'{esc(AA_LINK["bank"])} linked for “{esc(AA_LINK["purpose"])}”, consent till {AA_LINK["till"]}. '
               f'Revoke any time.' if ss.aa_consent else 'Not linked. SIP Guard uses debit history only.')
            + '<br>Never used: SMS, contacts, or data for loans or ads.</div></div>', unsafe_allow_html=True)
    else:
        with keyed("evaluator"):
            st.markdown('<div class="h3">🔍 Evaluator telemetry</div><div class="sub">Inspect the AI eval-gate '
                        'architecture. Model figures are computed live on synthetic data.</div>',
                        unsafe_allow_html=True)
            st.toggle("Feature flag: Enable SIP Guard system", key="flag_on",
                      help="Kill switch. Off = standard Groww behaviour: no predictions, no notices.")
            tab_m, tab_t, tab_g, tab_p, tab_d, tab_c = st.tabs(
                ["Model", "Data tiers", "Eval gates", "Pilot plan", "Golden Set", "Controls"])

            with tab_m:
                verdict = {"high": ("p-red", "≥ 60% · T-3 SIP Guard notice"),
                           "unsure": ("p-amber", "30–59% · safe fallback: T-1 reminder"),
                           "low": ("p-mint", "< 30% · no message")}[band]
                st.markdown(f'<div class="prob"><b>{p_live:.0%}</b><span class="sub">failure probability · '
                            f'{esc(GUARDED["name"])}, {dmy(GUARDED["next"])}</span></div>'
                            f'<span class="pill {verdict[0]}">{verdict[1]}</span>'
                            f'<div class="sub" style="margin-top:6px">{esc(SIPS[1]["name"])}, '
                            f'{dmy(SIPS[1]["next"])}: {p_nippon:.0%} → no message.</div>', unsafe_allow_html=True)
                top = max(d for _, _, d in drivers(x_live)) or 1
                tele = "".join(
                    f'<div class="tele"><span>{esc(nm)} <span class="sub">({esc(v)})</span></span>'
                    f'<span class="v">{d * 100:+.0f} pts</span><div class="bar"><span style="width:'
                    f'{max(d, 0) / top * 100:.0f}%"></span></div></div>' for nm, v, d in drivers(x_live))
                st.markdown(f'<div style="margin-top:12px"><div class="sub" style="font-weight:600;color:{NAVY}">'
                            f'What drives this score</div>{tele}<div class="sub" style="margin-top:6px">Points = score '
                            f'drop if that input were typical (median).</div></div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="sub" style="margin-top:12px"><b style="color:{NAVY}">Model card.</b> '
                    f'Gradient-boosted trees (scikit-learn, 150 trees, depth 3) trained on {MODEL["train_rows"]:,} '
                    f'synthetic debits; base failure rate {MODEL["base_rate"]:.0%}. Backtest at the 60% threshold: '
                    f'precision {MODEL["precision"]:.1%}, recall {MODEL["recall"]:.1%}, {MODEL["alerts"]:,} notices. '
                    f'Inputs now: {"mandate debit history + Account Aggregator balance and spending" if ss.aa_consent else "mandate debit history only (AA consent off)"}. '
                    f'Never SMS.</div>', unsafe_allow_html=True)

            with tab_t:
                st.markdown(
                    "**How does Groww know the balance was low?** Only from data it is entitled to use:\n\n"
                    "| Tier | Source | Consent | What the notice may say |\n| --- | --- | --- | --- |\n"
                    "| 1 · Mandate history | NACH / UPI AutoPay return codes (e.g. *insufficient funds*) "
                    "that Groww already receives | None extra: part of running the SIP | "
                    "“2 of your last 6 debits didn't go through” |\n"
                    "| 2 · Account Aggregator | Bank statement via an RBI-licensed AA | Explicit, purpose-bound "
                    "(“warn you before SIP debits”), revocable | “…and your balance was below ₹5,000 around the 5th” |\n"
                    "| Never | SMS, contacts, data collected for loans | — | — |")
                a, m = MODELS["aa"], MODELS["mandate"]
                st.markdown(
                    "**Why Tier 2 is worth asking for** (same synthetic data, same 60% threshold):\n\n"
                    "| Inputs | Notices | Precision | Failures caught |\n| --- | --- | --- | --- |\n"
                    f"| Mandate history only | {m['alerts']:,} | {m['precision']:.0%} | {m['recall']:.1%} |\n"
                    f"| + Account Aggregator | {a['alerts']:,} | {a['precision']:.0%} | {a['recall']:.1%} |")
                st.caption("Bounce history alone barely clears the threshold for anyone, so without AA consent SIP "
                           "Guard does not guess: investors with past bounces get an evidence-based T-1 reminder "
                           "instead. Turn consent off under Controls to see it.")

            with tab_p:
                st.markdown(
                    "**Who:** active SIPs ≥ 6 months old with at least one insufficient-funds bounce in the last 6 "
                    "debits.  \n**When:** one notice at T-3 (push + this card); T-1 reminder as fallback.  \n"
                    "**Design:** 50 / 50 holdout at investor level for 3 debit cycles.\n\n"
                    "| Metric | Type | Success looks like |\n| --- | --- | --- |\n"
                    "| Flagged SIPs still active after 3 months | Primary | +5 pts vs holdout |\n"
                    "| Bounce rate on flagged debits | Secondary | −30% vs holdout |\n"
                    "| Skip or UPI chosen instead of a bounce | Secondary | ≥ 40% of notices acted on |\n"
                    "| SIP cancellations started from the notice | Guardrail | Not above holdout |\n"
                    "| Notices where the debit would have succeeded | Guardrail | ≤ 40% (precision ≥ 60%) |\n"
                    "| AA consent opt-in from the consent screen | Learning | Measure; decides Tier 2 rollout |")
                st.caption("Targets are proposed hypotheses for the pilot, not results.")

            with tab_g:
                gate_html = "".join(
                    f'<div class="gate"><div><div class="t">{esc(nm)}</div><div class="d">{esc(d)}</div></div>'
                    f'<span class="pill {"p-mint" if ok else "p-red"}">{"PASS" if ok else "FAIL"}</span></div>'
                    for nm, ok, d in GATES)
                st.markdown(gate_html, unsafe_allow_html=True)
                st.markdown(f'<div class="sub" style="margin-top:8px">{"All gates pass: SIP Guard may speak." if gates_ok else "A gate failed: SIP Guard is silenced and the app falls back to the standard reminder."}'
                            '</div>', unsafe_allow_html=True)

            with tab_d:
                st.markdown(
                    "| Discovery Engine finding | Evidence | How SIP Guard uses it |\n"
                    "| --- | --- | --- |\n"
                    "| Bank & auto-debit friction is #1 | 155 of 445 themed reviews (35%) | The event predicted: a bounced AutoPay debit |\n"
                    "| Operational friction dominates | 276 of 445 (62%) | Scope is operational fixes only |\n"
                    "| Market return fear | 48 of 445 (11%) | Out of scope; market copy blocked by Gate 3 |\n"
                    "| Bounce charges | ₹295–₹885 incl. GST | Shown once, as information |\n"
                    "| 3 failed instalments in a row cancel a SIP | AMC uniform rule | Skip counter shown (x of 3) |\n"
                    "| Survey: 7 of 17 ran short of cash | Money shortage, overspending, expenses | Balance-cover and spending inputs |\n"
                    "| Survey: only 3 of 17 check debit status | Failures are invisible | Notice arrives at T-3, unprompted |")
                st.caption("Grounded in the Discovery Engine's 680 records and 83-case golden set. The golden-set "
                           "file itself is not bundled in this app.")

            with tab_c:
                st.radio("Ritu's situation this month (changes the model inputs)", list(SCENARIOS), key="scenario",
                         format_func=lambda k: SCENARIOS[k]["label"], on_change=reset)
                st.toggle("Account Aggregator consent (Tier 2 balance data)", key="aa_consent", on_change=reset,
                          help="Off = SIP Guard uses mandate debit history only and may not mention balance.")
                st.toggle("Inject an unapproved marketing message (tests Gates 3, 5 and 6)", key="rogue")
                st.number_input("Consecutive skips/misses before this month", 0, 2, key="prior_skips")
                if st.button("↺ Reset demo", use_container_width=True):
                    reset()
                    st.rerun()

st.markdown('<div class="note">Concept prototype for a NextLeap product-management capstone. Not affiliated with or '
            'endorsed by Groww. Investor, portfolio and bank accounts are fictional; fund names are labels only. The '
            'risk model is trained on synthetic data. Bank bounce charges: ₹250–₹750 + 18% GST per failed debit, per '
            'public bank fee schedules.</div>', unsafe_allow_html=True)
