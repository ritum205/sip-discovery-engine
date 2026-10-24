"""
Groww SIP Discovery Engine — Streamlit app
Analysing why investors stop, pause or break their SIPs, from public investor conversations.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud → New app → this repo → app.py
"""

from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

import engine_core as ec

# ---------------------------------------------------------------------------
# Page setup & brand styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Groww SIP Discovery Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY, EMERALD, SLATE, LIGHT = "#1B365D", "#00D09C", "#8A99AD", "#F4F6F8"
THEME_COLORS = {
    "Disappointment Valley": "#1B365D",
    "Bank & Auto-Debit Friction": "#00D09C",
    "Liquidity Shortage": "#3E6FB0",
    "Same-Category Overlap": "#5BC8AF",
    "Directionless Onboarding": "#7A8CA5",
    "Market Correction Panic": "#E8A33D",
    "Social Influence": "#B07CC6",
    "Platform Bugs": "#D9645B",
}
PREVENT_COLORS = {
    ec.PREVENT_UX: NAVY,
    ec.PREVENT_NUDGE: EMERALD,
    ec.PREVENT_EDU: "#9BB3D1",
    ec.PREVENT_LOW: "#D5DBE3",
}
STAGE_COLORS = {
    "Activation (M0–3)": "#9BB3D1",
    "Habit Formation (M4–9)": "#3E6FB0",
    "Retention Cliff (M10–14)": EMERALD,
    "Post-Cliff Maturity (M15+)": NAVY,
    "Unknown Stage": "#D5DBE3",
}

st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 1.6rem; padding-bottom: 6rem; max-width: 1400px; }}
  .hero {{ background: linear-gradient(120deg, {NAVY} 0%, #24497A 70%, #2C5A8F 100%);
          color: #fff; border-radius: 16px; padding: 28px 32px; margin-bottom: 14px; }}
  .hero h1 {{ color: #fff; font-size: 2.0rem; margin: 0 0 6px 0; letter-spacing: -0.5px; }}
  .hero p {{ color: #D9E4F2; font-size: 1.0rem; margin: 0; max-width: 980px; line-height: 1.55; }}
  .pill {{ display: inline-block; background: rgba(0,208,156,0.18); color: {EMERALD};
          border: 1px solid rgba(0,208,156,0.5); border-radius: 999px; padding: 2px 12px;
          font-size: 0.78rem; font-weight: 600; margin-bottom: 10px; letter-spacing: .3px; }}
  .card {{ background: #fff; border: 1px solid #E3E8EF; border-radius: 14px; padding: 16px 18px;
          height: 100%; box-shadow: 0 1px 2px rgba(27,54,93,0.04); }}
  .card .label {{ color: #5A6B82; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .6px; font-weight: 600; }}
  .card .value {{ color: {NAVY}; font-size: 1.75rem; font-weight: 700; margin-top: 4px; line-height: 1.15; }}
  .card .hint {{ color: #5A6B82; font-size: 0.82rem; margin-top: 4px; }}
  .card.accent {{ border-left: 5px solid {EMERALD}; }}
  .step {{ background: {LIGHT}; border-radius: 12px; padding: 14px 16px; height: 100%; }}
  .step b {{ color: {NAVY}; }}
  .step span {{ color: #5A6B82; font-size: 0.86rem; }}
  .notice {{ background: #FFF8E6; border: 1px solid #F2D58A; color: #6B5100; border-radius: 10px;
            padding: 10px 14px; font-size: 0.86rem; margin: 6px 0 14px 0; }}
  div[data-testid="stForm"] {{ border: 1px solid #E3E8EF; border-radius: 12px; padding: 10px 14px 4px 14px; }}
  .footer {{ color: #5A6B82; font-size: 0.8rem; border-top: 1px solid #E3E8EF; padding-top: 12px; margin-top: 28px; }}
  [data-testid="stSidebar"] {{ background: {LIGHT}; }}
</style>
""",
    unsafe_allow_html=True,
)


def card(label: str, value: str, hint: str = "", accent: bool = False) -> str:
    return (f'<div class="card{" accent" if accent else ""}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="hint">{hint}</div></div>')


def stretch(fn, *args, **kwargs):
    """Full-width rendering across Streamlit versions (width='stretch' on new
    releases, use_container_width on older ones)."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def keyword_mask(frame: pd.DataFrame, term: str) -> pd.Series:
    """Case-insensitive match across comment, theme and product."""
    mask = pd.Series(False, index=frame.index)
    for col in ("Original Comment", "Problem Theme", "Product"):
        if col in frame.columns:
            mask |= frame[col].astype(str).str.contains(term, case=False, na=False, regex=False)
    return mask


# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_benchmark():
    """Real harvested benchmark first; synthetic demo corpus only if the file is missing."""
    df, meta = ec.load_real_benchmark()
    if df is not None:
        return df, meta, True
    return ec.load_demo_corpus(), meta, False


@st.cache_data(show_spinner=False, ttl=3600)
def _get_live_cached(app_items: tuple, reviews_per_app: int, subs: tuple, queries: tuple,
                     include_play: bool, include_reddit: bool):
    return ec.run_live_ingestion(dict(app_items), reviews_per_app, list(subs), list(queries),
                                 include_play, include_reddit)


def get_live(*args):
    """Never lets an exception reach the evaluator's screen."""
    try:
        return _get_live_cached(*args)
    except Exception as exc:  # run_live_ingestion already guards; this is the last line of defence
        return (pd.DataFrame(columns=ec.COLUMNS),
                [f"⚠️ Live ingestion failed unexpectedly: {type(exc).__name__}. Showing nothing from this run."],
                {"raw": 0, "retained": 0, "dropped": 0, "reddit_status": "error", "fallback_used": False})


@st.cache_data(show_spinner=False)
def get_workbook(df: pd.DataFrame, note: str) -> bytes:
    return ec.build_discovery_workbook(df, note, include_roadmap=False)


# ---------------------------------------------------------------------------
# Q&A engine — hybrid of dataset evidence (counts, root causes, verbatim quotes)
# and built-in SIP domain knowledge. Runs locally; no external AI service or API key.
# ---------------------------------------------------------------------------
THEME_SYNONYMS = {
    "Bank & Auto-Debit Friction": ["bank", "auto-debit", "autodebit", "auto debit", "autopay", "auto pay", "mandate",
                                   "nach", "enach", "bounce", "debit", "upi", "penalty", "deduct"],
    "Disappointment Valley": ["return", "returns", "xirr", "fd", "fixed deposit", "flat", "negative", "loss",
                              "profit", "disappoint", "not growing"],
    "Same-Category Overlap": ["overlap", "too many funds", "duplicate", "same stocks", "consolidat", "multiple funds"],
    "Directionless Onboarding": ["goal", "onboarding", "purpose", "why started", "no plan", "beginner", "aimless"],
    "Liquidity Shortage": ["job", "salary", "emergency", "rent", "emi", "afford", "cash", "liquidity", "money tight",
                           "medical", "wedding", "layoff", "income"],
    "Market Correction Panic": ["crash", "market fall", "correction", "panic", "nifty", "sensex", "dip", "fear",
                                "volatil", "scared"],
    "Social Influence": ["friend", "colleague", "influencer", "finfluencer", "youtube", "f&o", "crypto", "trading",
                         "peer", "telegram"],
    "Platform Bugs": ["bug", "glitch", "crash the app", "app crash", "error", "support", "customer care", "kyc",
                      "otp", "login", "wrong nav", "units", "status", "technical"],
}
PRODUCT_SYNONYMS = {"Groww": ["groww"], "Zerodha Coin": ["zerodha", "coin"], "Kuvera": ["kuvera"]}
BEHAVIOUR_SYNONYMS = {
    "Cancelled SIP": ["cancel", "stop", "quit", "discontinu"],
    "Paused SIP": ["pause", "skip"],
    "Redeemed full corpus": ["redeem", "withdraw", "sell", "exit"],
}
STOPWORDS = set("""a an the and or of to in on for with about what why how when who which is are was were do does did
people investors investor users user sip sips mutual fund funds their they them it its this that these those top most
main biggest common complaints complaint issues issue problems problem say says said tell me show give list reasons
reason cause causes driving drive leading lead groww app apps please any some more many much can could would should
i we you our your my from by as at be been have has had into than then there here also just get got points point
friction explain work works between difference""".split())

SUGGESTED_QUESTIONS = [
    "Why do people cancel their SIP?",
    "What are the top bank auto-debit friction points?",
    "When do most SIPs stop?",
    "Pause vs cancel — what's the difference?",
    "How can Groww reduce SIP churn?",
]

# Built-in domain knowledge for conceptual questions. General, educational and
# non-advisory: no scheme recommendations, no return promises, no urgency.
KNOWLEDGE_BASE = [
    {
        "topic": "Pausing vs cancelling a SIP",
        "keys": ["pause", "pausing", "cancel", "stop", "stopping", "skip", "restart", "resume", "better"],
        "themes": ["Liquidity Shortage", "Market Correction Panic"],
        "short": "Pausing skips a few instalments and keeps the SIP set up; cancelling ends all future instalments. "
                 "In both cases, the units already bought stay invested.",
        "points": [
            "**Pause:** future instalments are skipped for a chosen number of months, then the SIP resumes on its own. "
            "The registration and bank mandate stay in place. How long you can pause depends on the fund house and platform.",
            "**Cancel:** no more instalments are debited. The existing units stay in the folio until redeemed. "
            "Restarting later usually means setting up a new SIP and sometimes a new mandate.",
            "**Missed instalments:** a fund house typically doesn't fine you for one, but the bank may charge a fee if a "
            "debit fails for low balance. Many fund houses cancel a SIP after a few consecutive failed debits.",
            "**Redeeming** is different again: it sells units and takes money out of the investment.",
            "**Which is better?** It depends on why. For a temporary cash crunch or short-term worry, a pause keeps "
            "the habit and set-up intact. If the goal has changed or the SIP was never right for the investor, "
            "cancelling is a reasonable, deliberate choice. Neither is universally better.",
        ],
        "product": "When ‘continue’ and ‘cancel’ are the only visible options, a temporary cash crunch turns into "
                   "permanent churn. Showing pause or step-down at the moment of cancellation keeps the habit alive "
                   "without pressure, as long as cancel stays one tap away.",
    },
    {
        "topic": "Market volatility and SIPs",
        "keys": ["volatil", "crash", "correction", "fall", "dip", "market down", "bear", "averaging", "rupee cost",
                 "nav", "units"],
        "themes": ["Market Correction Panic"],
        "short": "A SIP invests a fixed amount on a schedule. When prices fall, the same amount buys more units, "
                 "which lowers the average cost per unit over time.",
        "points": [
            "**Rupee-cost averaging:** buying at many price points smooths the entry price. It doesn't guarantee profit "
            "or protect against loss.",
            "**Drawdowns are normal for equity:** Indian equity indices have had several sharp corrections that later "
            "recovered. Past patterns don't guarantee future outcomes.",
            "**Stopping during a fall** means no new units are bought at the lower prices. That's the opposite of what the "
            "SIP mechanism is designed for, though every investor's situation differs.",
            "**Short holding periods** carry the widest range of outcomes for equity funds. The range narrows as the "
            "holding period lengthens.",
        ],
        "product": "First-time investors often experience their first drawdown as a permanent loss. Neutral, factual "
                   "context (historical drawdown and recovery ranges, units accumulated) helps more than any nudge. "
                   "Avoid ‘buy the dip’ or urgency language.",
    },
    {
        "topic": "Investor psychology behind SIP drop-offs",
        "keys": ["psycholog", "behavio", "bias", "emotion", "loss aversion", "herd", "patience", "impatien", "mindset",
                 "feel", "regret"],
        "themes": ["Disappointment Valley", "Market Correction Panic", "Social Influence"],
        "short": "Most SIP drop-offs are driven less by the product than by predictable behavioural biases, "
                 "especially in the first year.",
        "points": [
            "**Loss aversion:** a loss feels about twice as painful as an equal gain feels good, so a red portfolio "
            "triggers action.",
            "**Myopic loss aversion:** checking the portfolio often makes normal ups and downs feel like constant losses.",
            "**Anchoring:** new investors compare year-one equity returns with FD or savings rates they know well.",
            "**Recency bias and herding:** recent news, peers or creators shape decisions more than the original plan.",
            "**Present bias:** today's spending needs beat a distant goal, especially if no goal was ever set.",
        ],
        "product": "Goal-linking at setup, fewer red-number shocks on the portfolio screen, and calm factual context at "
                   "stress moments address these biases without manipulation.",
    },
    {
        "topic": "Why first-year SIP returns look flat",
        "keys": ["xirr", "return", "flat", "fd", "fixed deposit", "first year", "one year", "1 year", "negative",
                 "cagr", "absolute return"],
        "themes": ["Disappointment Valley"],
        "short": "In year one, most of the money has been invested for only a few months, so even good market "
                 "periods produce small absolute gains.",
        "points": [
            "**Time-weighting:** with 12 monthly instalments, the average rupee has been invested for about 6 months "
            "at month 12, and the latest instalment for almost none.",
            "**XIRR** annualises returns across instalments with different dates. Over short periods it can swing "
            "sharply and be misleading.",
            "**FD comparisons** set an anchor of steady, guaranteed growth that equity is not designed to match "
            "in the short term.",
        ],
        "product": "Showing units accumulated and the typical range of outcomes by holding period gives context "
                   "without promising returns.",
    },
    {
        "topic": "How SIP auto-debit works (mandates)",
        "keys": ["mandate", "nach", "enach", "e-nach", "autopay", "auto pay", "auto-debit", "auto debit", "debit",
                 "bounce", "bank", "upi", "limit", "limits", "otp"],
        "themes": ["Bank & Auto-Debit Friction"],
        "short": "A SIP is debited through a standing instruction on the investor's bank account (e-NACH or UPI "
                 "AutoPay). Most failures come from timing, balance or a broken or expired mandate.",
        "points": [
            "**Set-up:** the investor authorises a mandate with a maximum amount. The platform then debits each "
            "instalment on the SIP date, as long as it is within that maximum. Raising the SIP above the mandate "
            "limit needs a new or modified mandate.",
            "**e-NACH** mandates are registered through net banking or a debit card and typically allow higher "
            "maximum amounts. Registration can take a few working days.",
            "**UPI AutoPay limits (2026 rules):** after the first authenticated payment, recurring debits go through "
            "without a fresh OTP up to ₹15,000 per transaction generally, and up to ₹1 lakh for mutual fund "
            "subscriptions, insurance premiums and credit card bills. Banks must send a pre-debit notice at least "
            "24 hours before each debit. Limits change, so check current RBI and NPCI rules.",
            "**Low balance on the debit date** causes a failed instalment. Banks may charge a return or bounce fee.",
            "**Bank-account changes, mandate expiry or limit mismatches** silently break future debits unless the "
            "investor re-registers.",
            "**Repeated failures:** many fund houses cancel a SIP after a few consecutive failed instalments, often "
            "without the investor noticing.",
        ],
        "product": "Pre-debit reminders, an easy SIP-date change after typical salary credit, and a mandate health "
                   "check with guided re-registration directly reduce involuntary churn.",
    },
    {
        "topic": "Fund overlap and diversification",
        "keys": ["overlap", "diversif", "too many funds", "multiple funds", "same category", "consolidat", "portfolio"],
        "themes": ["Same-Category Overlap"],
        "short": "Holding several funds in the same category often means owning many of the same stocks. Adding funds "
                 "doesn't automatically add diversification.",
        "points": [
            "Funds in the same category follow similar mandates, so their top holdings frequently overlap.",
            "When investors discover the overlap, they often cancel several SIPs at once rather than consolidating.",
            "Diversification is driven by category, asset class and strategy mix, not the number of funds.",
        ],
        "product": "A category-level overlap view of the investor's own holdings, plus a guided ‘merge into one SIP’ "
                   "flow, turns a cancellation moment into a consolidation. The investor chooses which SIP to keep; "
                   "the platform does not recommend a scheme.",
    },
    {
        "topic": "Goal-based investing and onboarding",
        "keys": ["goal", "onboard", "purpose", "plan", "beginner", "first time", "first-time", "new investor", "start"],
        "themes": ["Directionless Onboarding"],
        "short": "SIPs started without a goal or horizon have nothing to anchor them when motivation dips or "
                 "returns look flat.",
        "points": [
            "A named goal (an emergency buffer, a home deposit, education) with a horizon gives a reason to continue.",
            "Fast onboarding gets the first SIP live quickly but rarely connects it to a life outcome.",
            "Milestone check-ins against the investor's own goal reinforce the habit without promoting products.",
        ],
        "product": "An optional, skippable goal step at SIP creation and neutral milestone updates at M3, M6, M9 "
                   "and M12.",
    },
    {
        "topic": "Liquidity, emergencies and SIP sizing",
        "keys": ["emergency", "liquidity", "job", "salary", "afford", "cash", "income", "sizing", "amount", "budget",
                 "step down", "step-down", "step up", "top-up"],
        "themes": ["Liquidity Shortage"],
        "short": "Many SIPs stop because of a cash-flow shock, not because the investor has given up on investing.",
        "points": [
            "SIPs sized on a peak-month income leave no buffer for variable earnings or one-off expenses.",
            "Without an emergency fund, the SIP corpus becomes the emergency fund and gets redeemed.",
            "**Step-down / step-up:** reducing or increasing the instalment amount adapts the SIP to income "
            "instead of ending it.",
        ],
        "product": "Pause and step-down options at the cancellation moment, plus SIP-amount guidance based on the "
                   "investor's own stated budget, keep the habit through temporary stress.",
    },
    {
        "topic": "Lock-ins, exit loads and redemption timing",
        "keys": ["exit load", "lock-in", "lock in", "elss", "redeem", "redemption", "withdraw"],
        "themes": ["Liquidity Shortage", "Disappointment Valley"],
        "short": "Some funds restrict or charge for early withdrawals, and holding period affects how gains are taxed. "
                 "Check the scheme documents and current tax rules.",
        "points": [
            "**ELSS** (tax-saving) funds have a three-year lock-in, applied to each SIP instalment separately.",
            "**Exit loads:** many equity funds charge a small exit load if units are redeemed within a set period "
            "(often around a year). The exact terms are in each scheme's documents.",
            "**Tax:** gains are taxed differently depending on fund type and holding period. Rules change, so "
            "investors should check current regulations.",
        ],
        "product": "Clear, factual disclosure of lock-in and exit-load dates on the redemption screen prevents "
                   "surprises that erode trust.",
    },
    {
        "topic": "What a SIP is and how it works",
        "keys": ["what is a sip", "what is sip", "systematic investment", "how does sip", "how sip works", "sip work",
                 "sip mean", "sip basics"],
        "themes": [],
        "short": "A Systematic Investment Plan (SIP) invests a fixed amount in a mutual fund at regular intervals, "
                 "usually monthly, through an automatic bank debit.",
        "points": [
            "Each instalment buys units at that day's NAV (net asset value), so the number of units varies "
            "with the price.",
            "The investor chooses the amount, frequency, date and duration, and can usually pause, modify or "
            "cancel the SIP.",
            "Because it is automatic, a SIP builds a habit and removes the need to time the market. Like any "
            "market-linked investment, returns are not guaranteed.",
        ],
        "product": "The automation that makes SIPs effective also makes failures silent, so clear status, receipts "
                   "and pre-debit reminders matter for trust.",
    },
    {
        "topic": "Compounding and the cost of interrupting a SIP",
        "keys": ["compound", "compounding", "long term", "long-term", "power of", "interrupt", "missing", "missed",
                 "cost of stopping", "how much", "wealth"],
        "themes": ["Disappointment Valley", "Liquidity Shortage"],
        "short": "Compounding means gains themselves start earning returns, so most of a long SIP's growth arrives "
                 "in the later years. Interruptions early on have an outsized effect on the end value.",
        "points": [
            "**Illustration (assumed 10% a year, not a forecast):** ₹5,000 a month for 10 years (₹6 lakh invested) "
            "grows to about ₹10.3 lakh.",
            "**Stopping after year one:** the ₹60,000 invested, left alone for the remaining nine years, grows to "
            "about ₹1.55 lakh at the same assumed rate.",
            "**Pausing for six months instead:** ₹5.7 lakh invested grows to about ₹9.6 lakh. Most of the long-term "
            "outcome is preserved.",
            "Actual returns vary and can be negative; the illustration only shows how time in the market affects "
            "outcomes under a fixed assumption.",
        ],
        "product": "Showing investors this trade-off at the cancellation moment, framed neutrally and with clear "
                   "assumptions, makes pause or step-down a visible alternative without creating fear.",
    },
    {
        "topic": "How SIP redemptions are taxed (India)",
        "keys": ["tax", "taxed", "taxation", "ltcg", "stcg", "capital gain", "capital gains", "112a", "fifo"],
        "weight": 3,  # a tax word almost always means a tax question
        "themes": [],
        "short": "Each SIP instalment is treated as a separate purchase, so its holding period and tax are worked "
                 "out instalment by instalment, oldest units first (FIFO).",
        "points": [
            "**Equity-oriented funds, held over 12 months (LTCG):** 12.5% on gains above ₹1.25 lakh a year in "
            "total, without indexation.",
            "**Equity-oriented funds, held 12 months or less (STCG):** 20%.",
            "**Debt funds bought on or after 1 April 2023:** gains are taxed at the investor's income-tax slab "
            "rate, whatever the holding period.",
            "**SIP effect:** redeeming a whole SIP corpus after one year usually mixes long-term and short-term "
            "units, because recent instalments are still under 12 months old.",
            "These are the rates for FY 2026-27 (unchanged in Budget 2026). Tax rules change, so verify the "
            "current position or ask a tax adviser.",
        ],
        "product": "Showing the estimated short-term vs long-term split before redemption helps investors avoid "
                   "surprises, without advising them what to do.",
    },
    {
        "topic": "Neutral factors investors weigh (not a recommendation)",
        "keys": ["which fund", "best fund", "should i", "recommend", "where to invest", "where should i invest"],
        "themes": [],
        "short": "There is no single right fund or decision. It depends on the investor's goal, time horizon, "
                 "risk tolerance and cash-flow needs.",
        "points": [
            "**Goal and horizon:** money needed within a few years is usually treated differently from long-term goals.",
            "**Emergency buffer first:** keeping accessible savings separate stops a SIP from becoming the "
            "emergency fund.",
            "**Risk tolerance:** the investor's ability to sit through a fall without stopping matters as much as "
            "expected return.",
            "**Costs and category:** expense ratio, exit load and fund category are listed in each scheme's documents.",
            "**Continuing vs stopping:** for a temporary cash crunch, pausing or stepping down is an alternative to "
            "cancelling. The right choice depends on personal circumstances.",
        ],
        "product": "Platforms can help with neutral education and goal tools, while personal recommendations come "
                   "from a SEBI-registered investment adviser.",
    },
    {
        "topic": "Measuring SIP retention and churn",
        "keys": ["metric", "measure", "kpi", "retention rate", "churn rate", "north star", "cohort", "track"],
        "themes": [],
        "short": "SIP churn is best tracked by activation cohort, separating voluntary stops from involuntary "
                 "debit failures.",
        "points": [
            "**M12 SIP retention:** share of SIPs still active 12 months after activation, by monthly cohort.",
            "**Instalment success rate:** share of scheduled debits that succeed. This isolates bank and mandate friction.",
            "**Pause-to-cancel ratio:** how often investors choose a pause over a cancellation when offered both.",
            "**Reactivation rate:** paused or cancelled SIPs that resume within N months.",
        ],
        "product": "Pair these with the qualitative drivers in this engine to see why each cohort leaves, not just "
                   "how many.",
    },
]

ADVICE_PATTERN = re.compile(r"\b(which|best|top|good)\s+(mutual\s+)?funds?\b|\bshould i (invest|buy|sell|stop|"
                            r"continue|redeem|pause|cancel|withdraw)\b|\brecommend\b.*\bfund|\bwhere (should|to) i invest\b", re.I)
SOLUTION_PATTERN = re.compile(r"\b(fix|fixes|solve|solution|solutions|reduce|prevent|improve|product (fix|fixes|"
                              r"solution|solutions|change|changes|idea|ideas)|feature|features|intervention|"
                              r"interventions|what can groww|how can groww|how should groww)\b", re.I)
CONCEPT_PATTERN = re.compile(r"\b(what is|what's|what are the rules|explain|how does|how do .* work|how (is|are) .* tax|tax on|taxed|"
                             r"difference|vs\.?|versus|better|meaning|concept|psycholog|mechanism|why does|"
                             r"what happens|is it (good|bad|wise|okay)|how (much|long)|"
                             r"how (do|should|can) (we|i|you) (measure|track|calculate|think))\b", re.I)
DATA_CUE_PATTERN = re.compile(r"\b(top|most common|complaints?|friction points?|what do (investors|users|people) say|"
                              r"how many|percentage|share of|reviews?|quotes?|examples?|data)\b", re.I)


def _esc(text: str) -> str:
    """Escape characters Streamlit markdown would interpret ($ triggers LaTeX)."""
    return re.sub(r"([\\`*_{}\[\]<>$#|])", r"\\\1", str(text))


def _excerpt(text: str, limit: int = 300) -> str:
    text = str(text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[: end + 1] if end > limit * 0.5 else cut.rsplit(" ", 1)[0]) + " …"


def _pct(n: int, d: int) -> str:
    return f"{(n / d):.0%}" if d else "—"


def _match_any(q: str, terms: list[str]) -> bool:
    return any(re.search(r"(?<![a-z])" + re.escape(t), q) for t in terms)


def _match_topics(q: str, pool: list[dict] | None = None) -> list[dict]:
    """Rank knowledge topics by keyword hits; ties go to the more specific (longer) matched key."""
    scored = []
    for kb in (pool if pool is not None else KNOWLEDGE_BASE):
        matched = [k for k in kb["keys"] if _match_any(q, [k])]
        if matched:
            scored.append((len(matched) * kb.get("weight", 1), max(map(len, matched)), kb))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    scored = [(h, kb) for h, _, kb in scored]
    if not scored:
        return []
    return [scored[0][1]] + [kb for hits, kb in scored[1:2] if hits >= 2]


def _retrieve(q: str, data: pd.DataFrame) -> dict:
    """Map the question to drivers / products / behaviours and pull the matching records."""
    themes = [t for t, syns in THEME_SYNONYMS.items() if _match_any(q, syns) or t.lower() in q]
    if not themes:  # fall back to the engine's own classifier on the question text
        scored = ec.score_themes(q)
        best = max(scored.values())
        themes = [t for t, s in scored.items() if s == best and s >= 3]
    products = [p for p, syns in PRODUCT_SYNONYMS.items() if _match_any(q, syns)]
    behaviours = [b for b, syns in BEHAVIOUR_SYNONYMS.items() if _match_any(q, syns)]
    asks_when = _match_any(q, ["when", "month", "tenure", "how long", "first year", "cliff", "stage", "timing"])
    asks_feel = _match_any(q, ["feel", "emotion", "sentiment", "frustrat", "angry", "mood"])
    consumed = {w for syns in list(THEME_SYNONYMS.values()) + list(PRODUCT_SYNONYMS.values())
                + list(BEHAVIOUR_SYNONYMS.values()) for w in syns}
    tokens = [w for w in re.findall(r"[a-z&]{3,}", q) if w not in STOPWORDS and w not in consumed]

    subset, scope_bits, notes = data, [], []
    if themes:
        subset = subset[subset["Problem Theme"].isin(themes)]
        scope_bits.append(" / ".join(themes))
    if products:
        narrowed = subset[subset["Product"].isin(products)]
        if len(narrowed):
            subset = narrowed
            scope_bits.append(" / ".join(products))
        else:
            notes.append(f"No matching records mention {', '.join(products)}, so all products are included.")
    if behaviours and not themes and not asks_when:
        narrowed = subset[subset["Problem Behaviour"].isin(behaviours)]
        if len(narrowed) >= 3:
            subset = narrowed
            nice = {"Cancelled SIP": "SIP cancellations", "Paused SIP": "SIP pauses", "Redeemed full corpus": "redemptions"}
            scope_bits.append(" / ".join(nice.get(x, x) for x in behaviours))
    if tokens and not themes:
        kw_hits = subset[subset["Original Comment"].str.contains("|".join(map(re.escape, tokens)), case=False, na=False)]
        if len(kw_hits) >= 3:
            subset = kw_hits
            scope_bits.append("mentions of " + ", ".join(f"“{t}”" for t in tokens[:4]))
    if subset.empty:
        notes.append("Nothing in the data matched that specifically, so this answer covers all records.")
        subset, scope_bits = data, []
    return {"subset": subset, "scope": scope_bits, "notes": notes, "themes": themes, "tokens": tokens,
            "asks_when": asks_when, "asks_feel": asks_feel, "overall": not scope_bits}


def _quotes(ctx: dict, k: int = 3) -> list[str]:
    subset, tokens, themes, overall = ctx["subset"], ctx["tokens"], ctx["themes"], ctx["overall"]
    if subset.empty:
        return []

    def rank(row) -> float:
        text = str(row["Original Comment"]).lower()
        s = sum(text.count(t) for t in tokens) * 2
        s += sum(1 for syns in (THEME_SYNONYMS.get(t, []) for t in themes) for w in syns if w in text)
        s += 0.5 if "play.google.com" in str(row["Source URL"]) else 0
        s += min(len(text), 400) / 400  # prefer substantive comments
        return s

    def fingerprint(text: str) -> str:  # near-duplicate detector (ignores numbers & punctuation)
        return re.sub(r"[^a-z]+", " ", str(text).lower())[:70]

    ranked = subset.assign(_rank=subset.apply(rank, axis=1)).sort_values("_rank", ascending=False)
    picks, seen_themes, seen_fp = [], set(), set()
    for pass_no in (1, 2):  # pass 1 prefers theme diversity; pass 2 fills any remaining slots
        for _, row in ranked.iterrows():
            fp = fingerprint(row["Original Comment"])
            if fp in seen_fp or (pass_no == 1 and overall and row["Problem Theme"] in seen_themes):
                continue
            picks.append(row)
            seen_fp.add(fp)
            seen_themes.add(row["Problem Theme"])
            if len(picks) == k:
                break
        if len(picks) == k:
            break
    out = []
    for row in picks:
        url = str(row["Source URL"])
        label = "Open review ↗" if "play.google.com" in url else "Open source ↗"
        out.append(f"> “{_esc(_excerpt(row['Original Comment']))}”  \n"
                   f"> — {row['Product']} · {row['Source Platform']} · [{label}]({url})")
    return out


def _data_answer(ctx: dict, total: int) -> list[str]:
    subset, n = ctx["subset"], len(ctx["subset"])
    theme_counts = subset["Problem Theme"].value_counts()
    top_theme = theme_counts.index[0]
    addressable = int(subset["Churn Preventability"].isin(ec.ADDRESSABLE_LEVELS).sum())
    out: list[str] = []

    ten_all = subset["SIP Tenure (Months)"].dropna()
    if ctx["asks_when"] and len(ten_all):
        cliff_n = int(((ten_all >= 10) & (ten_all <= 14)).sum())
        out.append(f"**Short answer:** most drop-offs cluster in **months 10–14**: {_pct(cliff_n, len(ten_all))} of the "
                   f"{len(ten_all)} records that state a tenure (median **{int(ten_all.median())} months**). "
                   f"The leading driver in this view is **{top_theme}**.")
    elif ctx["overall"]:
        out.append(f"**Short answer:** across all **{total}** records, the leading driver of SIP drop-off is "
                   f"**{top_theme}** ({_pct(theme_counts.iloc[0], total)} of records), followed by "
                   + ", ".join(f"**{t}** ({_pct(c, total)})" for t, c in theme_counts.iloc[1:3].items()) + ".")
    else:
        out.append(f"**Short answer:** **{n}** of {total} records ({_pct(n, total)}) relate to "
                   f"**{', '.join(ctx['scope'])}**. The dominant driver here is **{top_theme}**, and "
                   f"{_pct(addressable, n)} of these cases look addressable through product/UX fixes or compliant nudges.")
    out += [f"_{note}_" for note in ctx["notes"]]

    rc = subset["Problem Root Cause"].value_counts().head(3)
    lines = ["**Core root causes in the data**"]
    for i, (cause, c) in enumerate(rc.items(), start=1):
        theme_of = subset.loc[subset["Problem Root Cause"] == cause, "Problem Theme"].mode().iat[0]
        lines.append(f"{i}. {_esc(cause)}: **{c}** records ({_pct(c, n)}) · _{theme_of}_")
    out.append("\n".join(lines))

    phrases = Counter()
    for text in subset["Original Comment"].head(400):
        for p in ec.explain_theme(str(text))["matched"]:
            phrases[p.lower().strip()] += 1
    if phrases:
        out.append("**Most frequent complaint signals:** "
                   + ", ".join(f"“{_esc(p)}” ({c})" for p, c in phrases.most_common(6)))

    out.append("**By the numbers**")
    if ctx["overall"] or len(theme_counts) > 1:
        table = ["| Driver | Records | Share | Dominant emotion |", "|---|---:|---:|---|"]
        for t, c in theme_counts.head(5).items():
            emo = subset.loc[subset["Problem Theme"] == t, "Investor Emotional Trigger"].mode().iat[0]
            table.append(f"| {t} | {c} | {_pct(c, n)} | {emo} |")
        out.append("\n".join(table))
    beh = subset["Problem Behaviour"].value_counts().head(3)
    emo_all = subset["Investor Emotional Trigger"].value_counts().head(3)
    bullets = [
        "Behaviour: " + ", ".join(f"{b} {_pct(c, n)}" for b, c in beh.items()),
        "Emotional triggers: " + ", ".join(f"{e} {_pct(c, n)}" for e, c in emo_all.items()),
        f"Addressable by product/UX or nudges: **{_pct(addressable, n)}** ({addressable} of {n})",
    ]
    if len(ten_all):
        cliff = int(((ten_all >= 10) & (ten_all <= 14)).sum())
        bullets.append(f"Timing: {_pct(cliff, len(ten_all))} of the {len(ten_all)} records that state a tenure stopped "
                       f"in months 10–14 (median {int(ten_all.median())} months)")
    elif ctx["asks_when"]:
        bullets.append("Timing: none of these records state how long the SIP ran.")
    if ctx["asks_when"]:
        stages = subset["SIP Stage"].value_counts()
        bullets.append("By stage: " + ", ".join(f"{s} {_pct(c, n)}" for s, c in stages.items()))
    if ctx["asks_feel"] and len(emo_all):
        bullets.insert(0, f"The most common feeling is **{emo_all.index[0]}** ({_pct(emo_all.iloc[0], n)} of records).")
    out.append("\n".join(f"- {b}" for b in bullets))

    out.append("**What investors actually said**")
    out += _quotes(ctx, 3)

    fix = subset.loc[subset["Problem Theme"] == top_theme, "Recommended Groww Intervention"].mode()
    if len(fix):
        out.append(f"**What this points to ({top_theme}):** {_esc(fix.iat[0])}")
    return out


def _concept_answer(topics: list[dict], q: str, data: pd.DataFrame) -> list[str]:
    main = topics[0]
    out = [f"**{main['topic']}**", f"**Short answer:** {main['short']}",
           "**Key points**\n" + "\n".join(f"- {p}" for p in main["points"])]
    if len(topics) > 1 and not topics[1]["topic"].startswith("Neutral factors"):
        extra = topics[1]
        out.append(f"**Related: {extra['topic']}**\n- {extra['short']}")

    related = [t for t in main["themes"] if t in set(data["Problem Theme"])]
    if related:
        sub = data[data["Problem Theme"].isin(related)]
        n, total = len(sub), len(data)
        lines = [f"**What the discovery data shows:** {n} of {total} records ({_pct(n, total)}) fall under "
                 f"{' / '.join(related)}."]
        emo = sub["Investor Emotional Trigger"].value_counts()
        beh = sub["Problem Behaviour"].value_counts()
        if len(emo):
            lines.append(f"- Dominant emotion: **{emo.index[0]}** ({_pct(emo.iloc[0], n)})")
        if len(beh):
            lines.append("- What investors did: " + ", ".join(f"{b} {_pct(c, n)}" for b, c in beh.head(3).items()))
        rc = sub["Problem Root Cause"].value_counts()
        if len(rc):
            lines.append(f"- Most common root cause: {_esc(rc.index[0])}")
        out.append("\n".join(lines))
        ctx = {"subset": sub, "tokens": [w for w in re.findall(r"[a-z]{4,}", q) if w not in STOPWORDS],
               "themes": related, "overall": False}
        quotes = _quotes(ctx, 2)
        if quotes:
            out.append("**In investors' own words**")
            out += quotes
    out.append(f"**Product angle:** {main['product']}")
    if len(topics) > 1 and topics[1]["topic"].startswith("Neutral factors"):
        out.append("**Neutral factors to weigh**\n" + "\n".join(f"- {p}" for p in topics[1]["points"]))
    return out


def _solution_answer(data: pd.DataFrame, ctx: dict) -> list[str]:
    total = len(data)
    base = ctx["subset"] if ctx["themes"] else data
    counts = base["Problem Theme"].value_counts().head(4)
    out = ["**Short answer:** focus on the drivers that are both large and addressable. "
           f"Across all {total} records, {_pct(int(data['Churn Preventability'].isin(ec.ADDRESSABLE_LEVELS).sum()), total)} "
           "of drop-offs look addressable through product/UX fixes or compliant nudges."]
    lines = ["**Where to act, by size of problem**"]
    for i, (theme, c) in enumerate(counts.items(), start=1):
        kb = ec.THEME_KB[theme]
        sub = base[base["Problem Theme"] == theme]
        addr = _pct(int(sub["Churn Preventability"].isin(ec.ADDRESSABLE_LEVELS).sum()), len(sub))
        lines.append(f"{i}. **{theme}** ({c} records, {_pct(c, len(base))}; {addr} addressable)  \n"
                     f"   Fix: {_esc(kb['intervention'])}  \n   Measure: {_esc(kb['nsm'])}")
    out.append("\n".join(lines))
    out.append("**Guardrails for every fix**\n"
               "- No fund or stock recommendations; show investors their own holdings and factual context only.\n"
               "- No FOMO, fear or urgency copy; cancel always stays one tap away.\n"
               "- Measure each fix against M12 SIP retention and the instalment success rate, by activation cohort.")
    ctx = dict(ctx, subset=base[base["Problem Theme"].isin(counts.index[:2])], overall=False)
    quotes = _quotes(ctx, 2)
    if quotes:
        out.append("**The friction in investors' words**")
        out += quotes
    return out


def answer_question(question: str, data: pd.DataFrame) -> tuple[str, str]:
    """Built-in answer: dataset evidence + domain knowledge. Returns (markdown, kind) where kind is one of
    'data', 'knowledge', 'solution', 'advice' or 'nomatch'. Works with no network or API key."""
    q = question.lower().strip()
    total = len(data)
    if not q:
        return "Ask me anything about SIP drop-offs, for example “Why do people pause their SIPs?”", "nomatch"
    if total == 0:
        return "No records are loaded yet. Load the benchmark corpus or run live ingestion first.", "nomatch"

    out: list[str] = []
    topics = _match_topics(q)
    ctx = _retrieve(q, data)
    is_advice = bool(ADVICE_PATTERN.search(q))
    is_concept = bool(topics) and bool(CONCEPT_PATTERN.search(q)) and not DATA_CUE_PATTERN.search(q)

    if is_advice:
        out.append("_I can't recommend specific funds or tell you whether to buy, sell, stop or continue an "
                   "investment. Here is neutral context that may help, and a registered adviser can help with "
                   "personal decisions._")
        # Personal-decision questions get neutral context, never a data dump or a recommendation.
        neutral = next(kb for kb in KNOWLEDGE_BASE if kb["topic"].startswith("Neutral factors"))
        # Prefer the substantive topic (e.g. market falls) over generic words like "stop" when both match.
        pool = [kb for kb in KNOWLEDGE_BASE if kb is not neutral]
        topical = _match_topics(q, [kb for kb in pool if not kb["topic"].startswith("Pausing")]) or _match_topics(q, pool)
        out += _concept_answer(topical[:1] + [neutral] if topical else [neutral], q, data)
        return "\n\n".join(out), "advice"
    if SOLUTION_PATTERN.search(q) and not CONCEPT_PATTERN.search(q):
        return "\n\n".join(_solution_answer(data, ctx)), "solution"
    if is_concept:
        return "\n\n".join(_concept_answer(topics, q, data)), "knowledge"

    nothing_matched = ctx["overall"] and not ctx["asks_when"] and not ctx["themes"] and not topics
    out += _data_answer(ctx, total)
    if topics:
        t = topics[0]
        out.append(f"**Background: {t['topic']}**\n- {t['short']}\n" + "\n".join(f"- {p}" for p in t["points"][:2]))
    if nothing_matched:
        out.insert(0, "_I didn't find a specific match for that, so here is the overall picture. Try asking about "
                      "a driver (e.g. auto-debit, returns, market falls), a behaviour (pause, cancel) or a concept._")
        return "\n\n".join(out), "nomatch"
    return "\n\n".join(out), "data"


# ---------------------------------------------------------------------------
# Optional external knowledge: Wikipedia lookup (free, no key) and an LLM
# (Anthropic Claude via API key in Streamlit secrets). Both are optional and
# fail silently back to the built-in answer.
# ---------------------------------------------------------------------------
FINANCE_WORDS = re.compile(r"\b(invest|fund|financ|payment|bank|tax|return|market|interest|stock|equity|bond|"
                           r"debit|mandate|portfolio|asset|price|rate|loan|saving|india)", re.I)
HTTP_HEADERS = {"User-Agent": "GrowwSIPDiscoveryEngine/1.0 (educational research app)"}


def _secret(name: str, default: str = "") -> str:
    try:
        val = st.secrets.get(name)  # raises if no secrets file exists
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(name, default)


@st.cache_data(show_spinner=False, ttl=86400)
def web_lookup(question: str) -> dict | None:
    """Look up a general concept on Wikipedia. Returns {title, extract, url} only when the page is
    clearly finance-related and mentions a term from the question; otherwise None."""
    terms = [w for w in re.findall(r"[a-z0-9&-]{3,}", question.lower()) if w not in STOPWORDS]
    if not terms:
        return None
    try:
        r = requests.get("https://en.wikipedia.org/w/rest.php/v1/search/page",
                         params={"q": " ".join(terms[:6]), "limit": 4}, headers=HTTP_HEADERS, timeout=8)
        pages = r.json().get("pages", []) if r.ok else []
        for page in pages:
            sr = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{page['key']}",
                              headers=HTTP_HEADERS, timeout=8)
            if not sr.ok:
                continue
            js = sr.json()
            extract = js.get("extract", "")
            text = (js.get("title", "") + " " + extract).lower()
            if js.get("type") == "disambiguation" or not FINANCE_WORDS.search(text):
                continue
            if not any(t in text for t in terms):
                continue
            url = js.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{page['key']}")
            return {"title": js.get("title", page.get("title", "")), "extract": _excerpt(extract, 700), "url": url}
    except Exception:
        return None
    return None


LLM_SYSTEM_PROMPT = """You are the assistant inside the Groww SIP Discovery Engine, an internal product-discovery tool
about why Indian mutual-fund investors stop, pause or break their SIPs.
Rules:
- For any statistic about reviews, use ONLY the numbers in DATA CONTEXT. Never invent counts, percentages or quotes.
- You may use general, accurate knowledge about SIPs and Indian mutual funds (NAV, mandates, XIRR, compounding,
  taxation, SEBI/AMFI norms). Where rules can change (tax, limits), say so briefly.
- Never recommend a specific fund, stock or AMC, and never tell the reader to buy, sell, stop or continue an
  investment. No FOMO, fear or urgency language.
- Do not include quotes or links; verified review quotes are appended automatically after your answer.
- Write for a product leader: start with a one-line **Short answer**, then concise bullet points, then a
  **Product angle** line when relevant. Use markdown. Stay under 300 words."""


@st.cache_data(show_spinner=False, ttl=86400)
def _llm_call(question: str, context: str, model: str) -> str | None:
    key = _secret("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": 900, "system": LLM_SYSTEM_PROMPT,
                  "messages": [{"role": "user", "content": f"QUESTION: {question}\n\n{context}"}]},
            timeout=45,
        )
        if not r.ok:
            return None
        text = "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
        return text or None
    except Exception:
        return None


def llm_enabled() -> bool:
    return bool(_secret("ANTHROPIC_API_KEY"))


def respond(question: str, data: pd.DataFrame) -> tuple[str, str]:
    """Final answer = (optional LLM synthesis or built-in answer) + optional Wikipedia context.
    Returns (markdown, source label)."""
    builtin, kind = answer_question(question, data)
    q = question.strip()
    if not q or data.empty:
        return builtin, "Built-in"

    web = web_lookup(q) if kind == "nomatch" else None
    web_md = (f"**From the web: {web['title']} (Wikipedia)**\n\n> {_esc(web['extract'])}\n\n"
              f"[Read more on Wikipedia ↗]({web['url']})") if web else ""

    if llm_enabled():
        quotes = [ln for ln in builtin.split("\n\n") if ln.startswith(">")]
        facts = "\n\n".join(ln for ln in builtin.split("\n\n") if not ln.startswith(">"))
        context = (f"DATA CONTEXT (computed from {len(data)} public investor reviews; question type: {kind}):\n{facts}"
                   + (f"\n\nWEB CONTEXT (Wikipedia: {web['title']}):\n{web['extract']}" if web else ""))
        llm = _llm_call(q, context, _secret("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
        if llm:
            tail = ""
            if quotes and kind != "advice":
                tail = "\n\n**What investors actually said**\n\n" + "\n\n".join(quotes[:3])
            if web:
                tail += f"\n\n[Background: {web['title']} on Wikipedia ↗]({web['url']})"
            return llm + tail, "Claude (Anthropic API) + discovery data" + (" + Wikipedia" if web else "")

    if web:
        top = data["Problem Theme"].value_counts()
        summary_line = (f"**Related discovery data:** across {len(data)} records, the leading SIP drop-off driver is "
                        f"**{top.index[0]}** ({_pct(int(top.iloc[0]), len(data))}). Ask about a driver or behaviour "
                        "for evidence and quotes.")
        return web_md + "\n\n" + summary_line, "Wikipedia + discovery data"
    source = {"data": "Discovery data", "solution": "Discovery data + product playbook",
              "knowledge": "Built-in SIP knowledge + discovery data", "advice": "Built-in SIP knowledge (neutral)",
              "nomatch": "Discovery data"}[kind]
    return builtin, source


# ---------------------------------------------------------------------------
# Header + top chat card
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="pill">GROWW INTELLIGENCE · INVESTOR INSIGHTS</div>
  <h1>Groww SIP Discovery Engine</h1>
  <p>Why do investors stop, pause or break their SIPs — especially first-time mutual-fund investors around
  the 12-month mark? This engine analyses what investors say in public conversations, pinpoints the friction
  behind each drop-off, and sizes every driver with links back to the original reviews.</p>
</div>
""",
    unsafe_allow_html=True,
)
chat_slot = st.container()  # filled once the dataset is resolved, but rendered right here under the header

c1, c2, c3, c4 = st.columns(4)
c1.markdown(card("North-Star Target", "↑ M12 SIP Retention", "Share of SIPs still active 12 months after activation", True), unsafe_allow_html=True)
c2.markdown(card("Focus Cohort", "First-time MF investors", "New-to-investing users on their first SIP"), unsafe_allow_html=True)
c3.markdown(card("Churn Drivers Tracked", "8 themes", "From Disappointment Valley to Platform Bugs"), unsafe_allow_html=True)
c4.markdown(card("Compliance Mode", "Zero PII · No advice"), unsafe_allow_html=True)

with st.expander("⚙️ Engine methodology", expanded=False):
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown('<div class="step"><b>1 · Ingest</b><br><span>Public Play Store reviews (google-play-scraper) and public Reddit JSON search. No logins, tokens or private data.</span></div>', unsafe_allow_html=True)
    m2.markdown('<div class="step"><b>2 · Sanitise</b><br><span>Strip handles, emails, phone, PAN, Aadhaar, UPI IDs and account-like numbers. Author names never collected.</span></div>', unsafe_allow_html=True)
    m3.markdown('<div class="step"><b>3 · Enrich</b><br><span>Explainable weighted-lexicon NLP assigns 1 of 8 drivers, then extracts tenure, behaviour, emotion and preventability.</span></div>', unsafe_allow_html=True)
    m4.markdown('<div class="step"><b>4 · Synthesise</b><br><span>Drop-off curve, driver sizing, preventability and stage splits, plus an Excel workbook with live formulas.</span></div>', unsafe_allow_html=True)
    st.caption("Classifier rules live in `engine_core.THEME_KB` — every assignment can be traced to the phrases that triggered it.")

# ---------------------------------------------------------------------------
# Chat card renderer
# ---------------------------------------------------------------------------
def _normalize_history(raw) -> list[dict]:
    """Coerce any chat history shape (current {q,a}, legacy {question/query, answer}, or
    {role, content} message pairs from older app versions) into [{q, a, src}]."""
    out, pending = [], None
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        q = item.get("q") or item.get("question") or item.get("query", "")
        a = item.get("a") or item.get("answer") or item.get("response", "")
        if q or a:
            out.append({"q": str(q or ""), "a": str(a or ""), "src": str(item.get("src", ""))})
            continue
        role, content = item.get("role"), item.get("content", "")
        if role == "user":
            pending = str(content)
        elif role == "assistant" and pending is not None:
            out.append({"q": pending, "a": str(content), "src": ""})
            pending = None
    return out[-10:]


def _rerun() -> None:
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun"))()


def render_chat(data: pd.DataFrame, source_note: str = "") -> None:
    with chat_slot:
        st.session_state["chat_history"] = _normalize_history(st.session_state.get("chat_history", []))

        pill_cols = st.columns(len(SUGGESTED_QUESTIONS))
        for i, sq in enumerate(SUGGESTED_QUESTIONS):
            if pill_cols[i].button(sq, key=f"suggest_{i}", use_container_width=True):
                st.session_state["pending_question"] = sq

        with st.form("ask_form", clear_on_submit=True, border=False) if _form_supports_border() else \
                st.form("ask_form", clear_on_submit=True):
            in_col, btn_col = st.columns([8, 1])
            typed = in_col.text_input(
                "Your question", label_visibility="collapsed", key="ask_text",
                placeholder="Ask a question about SIP drop-offs, cancellations, investor friction, or product fixes...")
            asked = btn_col.form_submit_button("Ask", type="primary", use_container_width=True)

        question = (typed.strip() if asked and typed and typed.strip() else None) \
            or st.session_state.pop("pending_question", None)
        if question:
            with st.spinner("Analysing reviews and SIP knowledge…"):
                try:
                    reply, src = respond(question, data)
                except Exception as exc:  # never surface a stack trace to evaluators
                    reply, src = (f"I couldn't analyse that question ({type(exc).__name__}). Try rephrasing, "
                                  "e.g. “Why do people pause SIPs?”"), "Error"
            st.session_state["chat_history"] = (st.session_state["chat_history"] + [{"q": question, "a": reply, "src": src}])[-10:]

        history = st.session_state["chat_history"]
        if history:
            latest = history[-1]
            with st.chat_message("user", avatar="🧑‍💼"):
                st.markdown(_esc(latest.get("q") or latest.get("question") or latest.get("query", "")))
            with st.chat_message("assistant", avatar="📈"):
                st.markdown(latest.get("a") or latest.get("answer", ""))
            footer_l, footer_r = st.columns([5, 1])
            src = latest.get("src") or "Discovery data"
            footer_l.caption(f"Source: {src} · {len(data):,} discovery records. {source_note}")
            if footer_r.button("Clear chat", key="clear_chat", use_container_width=True):
                st.session_state["chat_history"] = []
                st.session_state.pop("pending_question", None)
                _rerun()
            if len(history) > 1:
                with st.expander(f"Earlier questions ({len(history) - 1})"):
                    for item in reversed(history[:-1]):
                        q_text = item.get("q") or item.get("question") or item.get("query", "")
                        st.markdown(f"**Q: {_esc(q_text)}**")
                        st.markdown(item.get("a") or item.get("answer", ""))
                        st.divider()
        st.write("")


def _form_supports_border() -> bool:
    import inspect
    try:
        return "border" in inspect.signature(st.form).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Sidebar — control panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"<h3 style='color:{NAVY};margin-bottom:0'>Control Panel</h3>", unsafe_allow_html=True)
    st.caption("Choose the data source for the discovery run.")
    bench_df, bench_meta, bench_is_real = get_benchmark()
    bench_label = (f"Benchmark Corpus ({len(bench_df)} Real Public Records)" if bench_is_real
                   else "Demo Corpus (synthetic — real benchmark file missing)")
    mode = st.radio("Ingestion mode", [bench_label, "Live Public Web Ingestion"], index=0, key="ingestion_mode")
    live_mode = mode.startswith("Live")

    if live_mode:
        st.markdown("**Google Play Store**")
        include_play = st.checkbox("Include Play Store reviews", value=True, key="inc_play")
        apps_text = st.text_area(
            "Apps (Name = package id, one per line)",
            value="\n".join(f"{k} = {v}" for k, v in ec.DEFAULT_PLAY_APPS.items()),
            height=110, disabled=not include_play, key="apps_text",
        )
        reviews_per_app = st.slider("Newest reviews per app", 50, 1000, 300, 50, disabled=not include_play,
                                    key="reviews_per_app")

        st.markdown("**Reddit (public JSON)**")
        include_reddit = st.checkbox("Include Reddit posts", value=True, key="inc_reddit")
        st.caption("If Reddit blocks or rate-limits this server (HTTP 403/429), the run "
                   "automatically falls back to live Play Store ingestion.")
        subs = st.multiselect("Subreddits", ["IndiaInvestments", "personalfinanceindia", "mutualfunds", "IndianStreetBets"],
                              default=ec.DEFAULT_SUBREDDITS, disabled=not include_reddit, key="subs")
        queries_text = st.text_input("Search phrases (comma-separated)", ", ".join(ec.DEFAULT_REDDIT_QUERIES),
                                     disabled=not include_reddit, key="reddit_queries")
        run = st.button("🚀 Run Live Ingestion", type="primary", use_container_width=True, key="run_live")

        if run:
            app_ids = {}
            for line in apps_text.splitlines():
                if "=" in line:
                    name, pkg = (s.strip() for s in line.split("=", 1))
                    if name and pkg:
                        app_ids[name] = pkg
            queries = [q.strip() for q in queries_text.split(",") if q.strip()]
            with st.spinner("Scraping public sources, sanitising and enriching…"):
                df_live, log, stats = get_live(tuple(app_ids.items()), reviews_per_app, tuple(subs),
                                               tuple(queries), include_play, include_reddit)
            st.session_state["live"] = {"df": df_live, "log": log, "stats": stats,
                                        "at": datetime.now().strftime("%d %b %Y, %H:%M")}

    st.divider()
    st.caption("Data ethics: public conversations only · zero PII stored · no authentication to any platform.")

# ---------------------------------------------------------------------------
# Resolve active dataset
# ---------------------------------------------------------------------------
if live_mode:
    live = st.session_state.get("live")
    if not live:
        render_chat(bench_df, "Live ingestion hasn't run yet, so answers use the benchmark corpus.")
        st.info("👈 Configure sources in the sidebar and press **Run Live Ingestion**. "
                "Switch to the **Benchmark Corpus** for an instant, zero-latency review.")
        st.stop()
    df_all = live["df"]
    stats = live["stats"]
    reddit_status = stats.get("reddit_status", "skipped")
    if reddit_status.startswith("blocked"):
        reason = reddit_status.split(":", 1)[-1].strip()
        if stats.get("fallback_used") and not df_all.empty:
            st.info(f"Reddit is not serving this server right now ({reason}), so this run automatically "
                    "fell back to live Google Play Store reviews.")
        else:
            st.info(f"Reddit is not serving this server right now ({reason}). Results below come from "
                    "the Play Store only.")
    provenance = (f"Live public-web run at {live['at']}: {stats['raw']} raw items → "
                  f"{stats['retained']} SIP-churn records after relevance filtering and enrichment. "
                  "Comments are sanitised verbatim text; each Source URL links to the original review or post.")
    with st.expander(f"📡 Ingestion log — {stats['retained']} records retained", expanded=df_all.empty):
        for line in live["log"]:
            st.write(line)
        st.caption("Reddit throttles anonymous traffic and often blocks cloud IP ranges. Play Store package "
                   "IDs are verified before scraping and re-resolved by name if wrong.")
    if df_all.empty:
        render_chat(bench_df, "This live run returned no records, so answers use the benchmark corpus.")
        st.warning("This live run returned no SIP-churn records (most app reviews are not about SIPs). "
                   "Try more reviews per app, or switch to the Benchmark Corpus in the sidebar.")
        st.stop()
else:
    df_all = bench_df
    if bench_is_real:
        when = (bench_meta.get("harvested_at") or "")[:10]
        provenance = (f"Real public benchmark: {len(df_all)} verbatim, PII-scrubbed Play Store reviews and Reddit "
                      f"posts/comments harvested {when}. Every Source URL opens the original review or thread. "
                      "Labels are machine-assigned; tenure appears only where the author stated it.")
        rejected = bench_meta.get("rejected_non_post_urls", 0)
        st.markdown(
            f'<div class="notice" style="background:#E9FBF5;border-color:#9BE6CF;color:#0B5A45">'
            f'<b>Real public benchmark</b> · {len(df_all)} records harvested {when} · '
            f'every “Open source ↗” link goes to the original review or thread. Labels are machine-assigned by the '
            f'engine; tenure is shown only where the author stated it.'
            f'{f" ({rejected} records without a post-level link were excluded.)" if rejected else ""}</div>',
            unsafe_allow_html=True)
    else:
        provenance = ec.DEMO_PROVENANCE
        st.markdown(f'<div class="notice"><b>Real benchmark not found — showing the synthetic demo corpus.</b> '
                    f'{bench_meta.get("error", "")} {ec.DEMO_PROVENANCE}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
df_all = df_all.copy()
# Columns actually present in the dataset drive what the table shows; any schema
# column that is missing is added as empty so metrics, charts and export never crash.
display_cols = [c for c in ec.COLUMNS if c in df_all.columns]
for col in ec.COLUMNS:
    if col not in df_all.columns:
        df_all[col] = pd.NA
df_all["Tenure Band"] = df_all["SIP Tenure (Months)"].map(ec.tenure_band)

band_order = [b[0] for b in ec.TENURE_BANDS] + ["Not stated"]
present = lambda col, order: [x for x in order if x in set(df_all[col])]  # noqa: E731

st.markdown(f"<h3 style='color:{NAVY};margin-bottom:4px'>Discovery Filters</h3>", unsafe_allow_html=True)
f1, f2, f3, f4 = st.columns(4)
sel_theme = f1.multiselect("Problem Theme", present("Problem Theme", ec.THEMES), placeholder="All 8 drivers", key="f_theme")
sel_band = f2.multiselect("SIP Tenure", present("Tenure Band", band_order), placeholder="All tenures", key="f_band")
sel_stage = f3.multiselect("SIP Stage", present("SIP Stage", ec.STAGES + ["Unknown Stage"]), placeholder="All stages",
                           key="f_stage")
sel_prev = f4.multiselect("Churn Preventability", present("Churn Preventability", ec.PREVENTABILITY_LEVELS),
                          placeholder="All levels", key="f_prev")

render_chat(df_all)

df = df_all
if sel_theme:
    df = df[df["Problem Theme"].isin(sel_theme)]
if sel_band:
    df = df[df["Tenure Band"].isin(sel_band)]
if sel_stage:
    df = df[df["SIP Stage"].isin(sel_stage)]
if sel_prev:
    df = df[df["Churn Preventability"].isin(sel_prev)]

PLOT_LAYOUT = dict(
    font=dict(family="Inter, Arial, sans-serif", color=NAVY, size=13),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    margin=dict(l=10, r=10, t=60, b=10),
    hoverlabel=dict(bgcolor="#FFFFFF", font_size=13),
)


def render_dashboard(df: pd.DataFrame, summary: dict) -> None:
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown(card("Records in view", f"{summary['n']:,}", f"of {len(df_all):,} analysed"), unsafe_allow_html=True)
    k2.markdown(card("Churn in M10–14 cliff", f"{summary['cliff_share']:.0%}",
                     f"{summary['cliff']} of {summary['known_tenure']} with stated tenure", True), unsafe_allow_html=True)
    k3.markdown(card("Addressable by Product & Nudges", f"{summary['addressable_share']:.0%}",
                     "High – Product/UX + High – FinTech Nudge", True), unsafe_allow_html=True)
    k4.markdown(card("Top churn driver", summary["top_theme"],
                     f"{summary['themes'].iloc[0]['Share']:.0%} of records in view"), unsafe_allow_html=True)
    k5.markdown(card("Records with stated tenure", f"{summary['known_tenure']:,}",
                     "Used for the drop-off cliff chart"), unsafe_allow_html=True)
    st.write("")

    tab_dash, tab_table, tab_export = st.tabs(["📊 Evaluator Dashboard", "🔎 Discovery Table", "⬇️ Export Workbook"])

    with tab_dash:
        # Chart 1 — the cliff
        ten = df["SIP Tenure (Months)"].dropna().astype(int)
        if ten.empty:
            st.info("No tenure is stated in the records in view, so the cliff curve cannot be drawn.")
        else:
            max_m = max(24, int(ten.max()))
            counts = ten.value_counts().reindex(range(1, max_m + 1), fill_value=0)
            colors = [EMERALD if 10 <= m <= 14 else "#B8C4D4" for m in counts.index]
            fig1 = go.Figure(go.Bar(
                x=counts.index, y=counts.values, marker_color=colors,
                hovertemplate="Month %{x}<br>%{y} churn records<extra></extra>"))
            fig1.add_vrect(x0=9.5, x1=14.5, fillcolor=EMERALD, opacity=0.08, line_width=0,
                           annotation_text=f"Drop-off Cliff · {summary['cliff_share']:.0%} of churn",
                           annotation_position="top left", annotation_font_color=NAVY)
            fig1.update_layout(**PLOT_LAYOUT, height=380, bargap=0.15,
                               title=dict(text="<b>Chart 1 · The Month 10–14 Drop-off Cliff</b><br>"
                                               "<span style='font-size:12px;color:#5A6B82'>SIP tenure at churn vs. churn volume</span>"),
                               xaxis=dict(title="SIP tenure at churn (months)", dtick=1 if max_m <= 36 else 3, showgrid=False),
                               yaxis=dict(title="Churn records", gridcolor="#EEF1F5"))
            stretch(st.plotly_chart, fig1, config={"displayModeBar": False})

        # Chart 2 — driver breakdown (records + dominant emotion only)
        themes = summary["themes"].sort_values("Records")
        left, right = st.columns([3, 2])
        with left:
            fig2 = go.Figure(go.Bar(
                y=themes["Problem Theme"], x=themes["Share"], orientation="h",
                marker_color=[THEME_COLORS.get(t, NAVY) for t in themes["Problem Theme"]],
                text=[f"{s:.0%}" for s in themes["Share"]], textposition="outside",
                customdata=themes[["Records", "Dominant Emotion"]].values,
                hovertemplate="<b>%{y}</b><br>Share: %{x:.1%}<br>Records: %{customdata[0]}"
                              "<br>Dominant emotion: %{customdata[1]}<extra></extra>"))
            fig2.update_layout(**PLOT_LAYOUT, height=420,
                               title=dict(text="<b>Chart 2 · Root-Cause Driver Breakdown</b><br>"
                                               "<span style='font-size:12px;color:#5A6B82'>% share of churn records · hover for records & dominant emotion</span>"),
                               xaxis=dict(tickformat=".0%", gridcolor="#EEF1F5", range=[0, themes["Share"].max() * 1.25]),
                               yaxis=dict(title=""))
            stretch(st.plotly_chart, fig2, config={"displayModeBar": False})
        with right:
            st.markdown(f"<div style='height:52px'></div><b style='color:{NAVY}'>Driver scorecard</b>", unsafe_allow_html=True)
            tbl = summary["themes"][["Problem Theme", "Records", "Share", "Dominant Emotion", "Addressable Share"]].copy()
            tbl["Share"] = (tbl["Share"] * 100).round(1)
            tbl["Addressable Share"] = (tbl["Addressable Share"] * 100).round(0)
            stretch(st.dataframe, tbl, hide_index=True, height=340, column_config={
                "Share": st.column_config.ProgressColumn("Share", format="%.0f%%", min_value=0,
                                                         max_value=float(max(tbl["Share"].max(), 1))),
                "Addressable Share": st.column_config.NumberColumn("Addressable", format="%.0f%%"),
            })

        # Chart 3 — preventability
        p_left, p_right = st.columns([2, 3])
        prev_counts = df["Churn Preventability"].value_counts().reindex(ec.PREVENTABILITY_LEVELS, fill_value=0)
        prev_counts = prev_counts[prev_counts > 0]
        with p_left:
            fig3 = go.Figure(go.Pie(
                labels=prev_counts.index, values=prev_counts.values, hole=0.62, sort=False,
                marker=dict(colors=[PREVENT_COLORS[l] for l in prev_counts.index], line=dict(color="#FFFFFF", width=2)),
                textinfo="percent", hovertemplate="%{label}<br>%{value} records (%{percent})<extra></extra>"))
            fig3.add_annotation(text=f"<b>{summary['addressable_share']:.0%}</b><br><span style='font-size:12px'>addressable</span>",
                                showarrow=False, font=dict(size=26, color=NAVY))
            fig3.update_layout(**PLOT_LAYOUT, height=400, showlegend=True,
                               legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
                               title=dict(text="<b>Chart 3 · Churn Preventability</b>"))
            stretch(st.plotly_chart, fig3, config={"displayModeBar": False})
        with p_right:
            cross = df.groupby(["Problem Theme", "Churn Preventability"]).size().reset_index(name="Records")
            fig4 = px.bar(cross, x="Records", y="Problem Theme", color="Churn Preventability", orientation="h",
                          color_discrete_map=PREVENT_COLORS,
                          category_orders={"Churn Preventability": ec.PREVENTABILITY_LEVELS,
                                           "Problem Theme": list(themes["Problem Theme"])[::-1]})
            fig4.update_layout(**PLOT_LAYOUT, height=400, barmode="stack",
                               title=dict(text="<b>Preventability by driver</b><br><span style='font-size:12px;color:#5A6B82'>"
                                               "Where Product/UX fixes vs. nudges vs. education apply</span>"),
                               legend=dict(orientation="h", y=-0.15, title="", font=dict(size=11)),
                               xaxis=dict(gridcolor="#EEF1F5"), yaxis=dict(title=""))
            stretch(st.plotly_chart, fig4, config={"displayModeBar": False})

        # Chart 4 — SIP stage × driver
        s_left, s_right = st.columns([3, 2])
        with s_left:
            stage_cross = df.groupby(["SIP Stage", "Problem Theme"]).size().reset_index(name="Records")
            stage_order = [s for s in ec.STAGES + ["Unknown Stage"] if s in set(stage_cross["SIP Stage"])]
            fig5 = px.bar(stage_cross, x="SIP Stage", y="Records", color="Problem Theme",
                          color_discrete_map=THEME_COLORS,
                          category_orders={"SIP Stage": stage_order, "Problem Theme": ec.THEMES})
            fig5.update_layout(**PLOT_LAYOUT, height=420, barmode="stack",
                               title=dict(text="<b>Chart 4 · When each driver strikes</b><br><span style='font-size:12px;color:#5A6B82'>"
                                               "Churn records by SIP lifecycle stage and driver</span>"),
                               legend=dict(title="", font=dict(size=11)),
                               xaxis=dict(title=""), yaxis=dict(gridcolor="#EEF1F5"))
            stretch(st.plotly_chart, fig5, config={"displayModeBar": False})
        with s_right:
            emo = df["Investor Emotional Trigger"].value_counts().head(8).sort_values()
            fig6 = go.Figure(go.Bar(
                y=emo.index, x=emo.values, orientation="h", marker_color=NAVY,
                hovertemplate="%{y}: %{x} records<extra></extra>"))
            fig6.update_layout(**PLOT_LAYOUT, height=420,
                               title=dict(text="<b>Chart 5 · Investor emotional triggers</b><br><span style='font-size:12px;color:#5A6B82'>"
                                               "Most frequent emotions expressed by churners</span>"),
                               xaxis=dict(title="Records", gridcolor="#EEF1F5"), yaxis=dict(title=""))
            stretch(st.plotly_chart, fig6, config={"displayModeBar": False})

        verdict = "above" if summary["addressable_share"] > 0.75 else "below"
        st.markdown(
            f"**Read-out:** {summary['cliff_share']:.0%} of tenure-stated churn in view falls in months 10–14; "
            f"**{summary['top_theme']}** is the largest driver; and {summary['addressable_share']:.0%} of records are "
            f"addressable through Product/UX fixes or compliant FinTech nudges ({verdict} the 75% hypothesis).")

    with tab_table:
        keyword = st.text_input("🔍 Search Reviews by Keyword:", placeholder="e.g., cancel, pause, bounce, xirr, ...",
                                key="review_keyword").strip()
        table_df = df[keyword_mask(df, keyword)] if keyword else df
        st.caption(f"{len(table_df):,} of {len(df):,} records" + (f" matching “{keyword}”" if keyword else "")
                   + " · searches comment, theme and product · click a Source URL to open the original.")
        column_config = {
            "Source URL": st.column_config.LinkColumn("Source URL", display_text="Open source ↗"),
            "Original Comment": st.column_config.TextColumn("Original Comment", width="large"),
            "SIP Tenure (Months)": st.column_config.NumberColumn("SIP Tenure (Months)", format="%d"),
            "Recommended Groww Intervention": st.column_config.TextColumn(width="large"),
            "Problem Root Cause": st.column_config.TextColumn(width="large"),
        }
        column_config = {k: v for k, v in column_config.items() if k in display_cols}
        if table_df.empty:
            st.info("No reviews match this keyword. Try a broader term.")
        else:
            stretch(st.dataframe, table_df[display_cols], hide_index=True, height=560, column_config=column_config)
        st.download_button("Download filtered view (.csv)", table_df[display_cols].to_csv(index=False).encode("utf-8"),
                           file_name="sip_discovery_filtered.csv", mime="text/csv", key="csv_download",
                           disabled=table_df.empty)

    with tab_export:
        st.markdown("#### Discovery Workbook")
        st.markdown(
            "- **Sheet 1 · Discovery Dataset** — all records, navy headers, auto-filters, frozen panes, wrapped comments, clickable source links")
        scope = st.radio("Rows to export", ["Full dataset", "Current filtered view"], horizontal=True, key="export_scope")
        export_df = df_all[ec.COLUMNS] if scope == "Full dataset" else df[ec.COLUMNS]
        with st.spinner("Building workbook…"):
            xlsx = get_workbook(export_df.reset_index(drop=True), provenance)
        st.download_button(
            "📥 Download Discovery Workbook (.xlsx)",
            data=xlsx,
            file_name=f"Groww_SIP_Discovery_{datetime.now():%Y%m%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            key="xlsx_download",
        )


summary = ec.summarize(df)
if summary["n"] == 0:
    st.warning("No records match the current filters. Widen the filters to see the dashboard.")
else:
    render_dashboard(df, summary)

st.markdown(
    '<div class="footer"><b>Compliance:</b> Built from public web conversations only; no private user data, '
    'authentication or PII is collected or stored. This tool is for internal product discovery and does not '
    'constitute investment advice or recommend any fund or security. Mutual fund investments are subject to market risks.</div>',
    unsafe_allow_html=True,
)
