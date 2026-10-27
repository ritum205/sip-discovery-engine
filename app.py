"""
Groww SIP Discovery Engine — Streamlit app
Analysing why investors stop, pause or break their SIPs, from public investor conversations.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud → New app → this repo → app.py
"""

from __future__ import annotations

import html
import os
import re
import unicodedata
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
  /* Uniform KPI cards: identical fixed height; label row reserves two lines so every value sits on the same baseline */
  .card {{ height: 168px; min-height: 168px; display: flex; flex-direction: column; justify-content: flex-start;
           box-sizing: border-box; background: #fff; border: 1px solid #E3E8EF; border-radius: 12px; padding: 16px 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.04); overflow: hidden; margin-bottom: 8px; }}
  .card .label {{ color: #5A6B82; font-size: 0.74rem; text-transform: uppercase; letter-spacing: .6px; font-weight: 600;
                 line-height: 1.3; min-height: 2.6em; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
                 overflow: hidden; }}
  .card .value {{ color: {NAVY}; font-size: 1.7rem; font-weight: 700; line-height: 1.15; margin-top: 2px;
                 display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
  .card .value.sm {{ font-size: 1.15rem; line-height: 1.25; }}
  .card .hint {{ color: #5A6B82; font-size: 0.8rem; line-height: 1.35; margin-top: auto;
                display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }}
  div[data-testid="stMetric"], div[data-testid="metric-container"] {{
      background-color: #ffffff !important; border: 1px solid #e5e7eb !important; border-radius: 12px !important;
      padding: 16px 20px !important; min-height: 130px !important; height: 130px !important; display: flex !important;
      flex-direction: column !important; justify-content: space-between !important;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important; }}
  .card.accent {{ border-left: 5px solid {EMERALD}; }}
  .step {{ background: {LIGHT}; border-radius: 12px; padding: 14px 16px; height: 100%; }}
  .step b {{ color: {NAVY}; }}
  .step span {{ color: #5A6B82; font-size: 0.86rem; }}
  .notice {{ background: #FFF8E6; border: 1px solid #F2D58A; color: #6B5100; border-radius: 10px;
            padding: 10px 14px; font-size: 0.86rem; margin: 6px 0 14px 0; }}
  div[data-testid="stForm"] {{ border: 1px solid #E3E8EF; border-radius: 12px; padding: 10px 14px 4px 14px; }}
  .footer {{ color: #5A6B82; font-size: 0.8rem; border-top: 1px solid #E3E8EF; padding-top: 12px; margin-top: 28px; }}
  [data-testid="stSidebar"] {{ background: {LIGHT}; }}
  .scope-badge {{ display: inline-flex; align-items: center; gap: 8px; border: 1px solid rgba(0,208,156,.45);
                 background: rgba(0,208,156,.08); color: {NAVY}; border-radius: 999px; padding: 5px 14px;
                 font-size: 0.84rem; margin: 2px 0 14px 0; }}
  /* Pulsing "Ask" button. Several selectors because Streamlit's test-ids differ between versions;
     .st-key-ask_btn is the keyed container wrapping the button (Streamlit >= 1.39). */
  div[data-testid="stFormSubmitButton"] button,
  button[data-testid="stBaseButton-primaryFormSubmit"],
  button[data-testid="baseButton-primaryFormSubmit"],
  .st-key-ask_btn button {{
      background: linear-gradient(135deg, #00D09C 0%, #00B386 100%) !important;
      color: #fff !important; border: none !important; font-weight: 700 !important;
      animation: pulse-glow 2s infinite ease-in-out !important; }}
  @keyframes pulse-glow {{
      0%   {{ box-shadow: 0 0 0 0 rgba(0,208,156,0.7);  transform: scale(1); }}
      50%  {{ box-shadow: 0 0 0 10px rgba(0,208,156,0); transform: scale(1.02); }}
      100% {{ box-shadow: 0 0 0 0 rgba(0,208,156,0);    transform: scale(1); }} }}
  @media (prefers-reduced-motion: reduce) {{
      div[data-testid="stFormSubmitButton"] button, .st-key-ask_btn button {{ animation: none !important; }} }}
  /* Suggested-question chips */
  .st-key-suggest_chips button {{ border-radius: 999px !important; min-height: 54px; white-space: normal !important;
      line-height: 1.25 !important; font-size: 0.86rem !important; border: 1px solid rgba(0,208,156,.45) !important; }}
  .st-key-suggest_chips button:hover {{ background: rgba(0,208,156,.08) !important; }}
  /* Evidence cards */
  .ev-card {{ border: 1px solid #E3E8EF; border-radius: 12px; padding: 14px 16px; background: #fff; height: 100%;
             box-shadow: 0 1px 2px rgba(27,54,93,0.04); margin-bottom: 12px; }}
  .ev-meta {{ font-size: 0.82rem; color: #5A6B82; margin-bottom: 8px; }}
  .ev-meta b {{ color: {NAVY}; }}
  .ev-meta a {{ color: #2563EB; }}
  .ev-quote {{ border-left: 3px solid #9BB3D1; padding: 2px 0 2px 12px; color: #1F2937; font-size: 0.93rem; line-height: 1.5; }}
  .ev-quote {{ overflow-wrap: anywhere; word-break: break-word; }}
  .ev-trans {{ background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 14px; margin-top: 12px;
              box-sizing: border-box; max-width: 100%; overflow: hidden; }}
  .ai-badge {{ display: inline-flex; align-items: center; gap: 4px; background-color: #D1FAE5; border-radius: 12px;
              padding: 2px 8px; margin-bottom: 8px; max-width: 100%; white-space: nowrap; }}
  .ai-badge-icon {{ font-size: 14px; line-height: 1; flex-shrink: 0; }}
  .ai-badge-text {{ font-size: 11px; font-weight: 700; color: #065F46; text-transform: uppercase; letter-spacing: 0.05em;
                   overflow: hidden; text-overflow: ellipsis; }}
  .ev-trans-text {{ font-size: 13px; color: #065F46; line-height: 1.45; margin: 0; font-style: normal;
                   overflow-wrap: anywhere; word-break: break-word; white-space: normal; }}
  .ev-tags {{ margin-top: 8px; font-size: 0.76rem; color: #5A6B82; }}
</style>
""",
    unsafe_allow_html=True,
)


def card(label: str, value: str, hint: str = "", accent: bool = False) -> str:
    small = " sm" if len(str(value)) > 14 else ""  # long text values (e.g. a theme name) use a smaller size
    return (f'<div class="card{" accent" if accent else ""}" title="{html.escape(str(hint), quote=True)}">'
            f'<div class="label">{label}</div><div class="value{small}">{value}</div><div class="hint">{hint}</div></div>')


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
MASTER_SCHEMA = ["Product", "Source Platform", "Source URL", "Original Comment", "User Type", "SIP Tenure (Months)",
                 "SIP Stage", "Problem Theme", "Problem Behaviour", "Problem Root Cause", "Investor Emotional Trigger",
                 "Churn Preventability", "Year"]
SOURCE_FILES = ["wb.xlsx", "reddit_sip_reviews.csv", "youtube_sip_comments.csv"]
PLATFORMS = ["Play Store", "Reddit", "YouTube"]
VIEW_FULL, VIEW_DEEP = "Full Discovery Dataset (All Sources)", "Validated Insights Deep-Dive"
# wb.xlsx repeats each of its 32 comments ~7 times. Counts follow the raw extract by default;
# set True to drop exact duplicate rows (same comment + URL) from the combined dataset.
DROP_DUPLICATE_ROWS = False
# wb.xlsx labels were assigned in rotation and don't match the comments, so the Full view uses validated
# labels where a row is in the validated set and keyword auto-tags elsewhere. Set True to keep wb labels.
KEEP_WB_LABELS = False

AUTO_TAG_RULES = {'Disappointment Valley': [('\\b(no|zero|negative|flat|poor|low|bad)\\s+returns?\\b', 3), ('\\breturns?\\b.{0,30}\\b(fd|fixed deposit|savings account|rd)\\b', 3), ('\\b(fd|fixed deposit)\\b.{0,40}\\bbetter\\b', 3), ('\\b(red|in loss|losing money|lost money|xirr)\\b', 2), ("\\bnot worth\\b|\\bwaste of (time|money)\\b|\\bwhat'?s the point\\b", 2), ('\\b(a|one|1)\\s*(year|yr)\\b.{0,40}\\b(nothing|same|flat|loss)\\b', 3), ('\\bdisappoint', 2)], 'Bank & Auto-Debit Friction': [('\\bmandate\\b', 3), ('\\b(auto[\\s-]?pay|auto[\\s-]?debit|e-?nach|nach)\\b', 3), ('\\bbounce(d)?\\b|\\bdebit failed\\b|\\bpayment failed\\b|\\binsufficient (funds|balance)\\b', 3), ('\\b(penalty|bounce charge|charges? by (my )?bank)\\b', 2), ('\\bchang(e|ed|ing) (my )?bank\\b|\\bnew bank account\\b', 2), ('\\bupi\\b.{0,20}\\b(fail|limit|expired?)\\b', 2)], 'Same-Category Overlap': [('\\boverlap(ping)?\\b', 4), ('\\b(too many|multiple|\\d+|five|four|six|seven) (mutual )?funds\\b', 2), ('\\b(same|similar) (stocks?|holdings?|portfolio)\\b', 3), ('\\b(flexi ?cap|large ?cap|mid ?cap|small ?cap|index fund).{0,40}\\b(another|also|again)\\b', 2), ('\\bconsolidat', 3), ('\\b(confus|clutter|messy portfolio)', 2)], 'Directionless Onboarding': [("\\b(no|without a|didn'?t have a|never had a) (goal|plan|reason)\\b", 4), ('\\bjust started\\b|\\bstarted because\\b|\\bsaw (a|on) (youtube|instagram|reel)\\b', 2), ("\\bdon'?t (know|understand) (why|what)\\b", 2), ('\\b(random(ly)?|blindly|impulse)\\b', 2), ('\\bwhat (am i|was i) (investing|saving) for\\b', 4), ('\\bgoal\\b', 1), ('\\b(paisa|paise|pesa|money|amount|refund|invested amount)\\b.{0,50}\\b(wapas|vapas|back|credit\\w*|account|withdraw\\w*|milega|aayega|aaye)\\b', 3), ('cancel\\w*.{0,60}\\b(redeem\\w*|withdraw\\w*|nikal\\w*)\\b|\\b(redeem\\w*|withdraw\\w*)\\b.{0,60}cancel', 3), ('\\b(lock[- ]?in|lock period|3 yrs|3 years|elss)\\b', 2), ('\\b(regular plan|direct plan|by mistake|galti se|mistakenly|accidentally)\\b', 3)], 'Liquidity Shortage': [('\\b(lost|losing) (my )?job\\b|\\blaid off\\b|\\blayoff\\b|\\bjob change\\b|\\bnotice period\\b', 4), ('\\b(rent|emi|loan|wedding|medical|hospital|emergency|fees)\\b', 2), ("\\b(tight|short|no) (on )?(money|cash|funds)\\b|\\bcan'?t afford\\b|\\bcash ?crunch\\b", 4), ('\\bneeded (the )?money\\b|\\bneed(ed)? cash\\b', 3), ('\\bsalary (delayed|cut|late)\\b', 3), ('\\b(paise na ho|paisa nahi|no money|cash short|short of cash|salary)\\b', 3)], 'Market Correction Panic': [('\\b(crash(ed|ing)?|correction|fall(ing)?|bloodbath|dip|down \\d+%|sensex|nifty)\\b', 3), ('\\b(scared|panic(ked)?|fear|afraid|worried)\\b', 3), ('\\b(war|election|recession|tariff|fii selling)\\b', 2), ('\\bwill go (lower|down)\\b|\\bmarket (is|was) (down|falling)\\b', 3)], 'Social Influence': [('\\b(friend|friends|colleague|cousin|brother|uncle|everyone)\\b.{0,40}\\b(said|told|making|earning|switched|moved)\\b', 3), ('\\b(f&o|futures|options trading|crypto|intraday|direct stocks?)\\b', 3), ('\\b(finfluencer|youtuber|telegram|tip)\\b', 2), ('\\bmutual funds? (are|is) (for|too) (slow|boring|old)\\b', 3)], 'Platform Bugs': [('\\b(bug|glitch|crash(es|ed)? (the )?app|app (is )?(not working|crashing|hangs))\\b', 4), ('\\b(wrong|incorrect|not updated|mismatch)\\b.{0,30}\\b(nav|units|amount|value|statement|status)\\b', 3), ('\\b(showing|shows|showed) (cancelled|failed|pending)\\b', 3), ('\\b(support|customer care|ticket)\\b.{0,30}\\b(no response|useless|not respond)\\b', 2), ('\\b(otp|login|kyc)\\b.{0,20}\\b(issue|problem|fail)\\b', 2), ('cancel\\w*.{0,60}\\b(still|again|phir|fir|bhi|after)\\b.{0,40}\\b(deduct\\w*|debit\\w*|cut|kat\\w*|charge\\w*)', 4), ('\\b(deduct\\w*|debit\\w*|cut|kat\\w*)\\b.{0,40}\\b(after|even after|ke baad|ke bad)\\b.{0,20}cancel', 4), ('(cancel|pause|resume|restart)\\w*.{0,40}\\b(nahi|nhi|not|nai)\\b.{0,25}\\b(ho|hua|work\\w*|show\\w*|aa|dikh\\w*|allow\\w*)', 3), ('\\b(in progress|processing|process me|pending|progress me)\\b', 2), ('\\boption\\b.{0,25}\\b(nahi|nhi|not|nai|missing)\\b|\\b(no|not)\\b.{0,10}\\boption\\b', 3), ('\\b(registration not valid|cannot be cancelled|not allowed)\\b', 4)]}
_AUTO_TAG = {t: [(re.compile(p, re.I), w) for p, w in rules] for t, rules in AUTO_TAG_RULES.items()}
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
# Romanised Indic vocabulary for non-English detection (Hindi/Hinglish, Telugu, Tamil, Marathi).
# Strong words are rarely English; weak ones are short particles that only count alongside others.
_NON_ENGLISH_STRONG = set("""
hai hain nahi nhi kya kaise kese karna krna karu kru mera mere meri maine mene paisa paise raha rha rahi rhi gaya gya gyi
hoga hogi lekin toh wala wali jata jati kyu kyun abhi sakte skte chahiye pehle dubara hua hue kiya kiye diya liya mujhe
kitne kitna hoti hota hote tho padega bolo batana karni karte karke kare karen karne krne kre baad hamari humne hamara
kayse kaha agar agr jayega jayenge rahega apna apne aapka apka kuch koi bahut bohot jab tak wapas vapas milega aayega
kaat kat gaye
nenu nennu naaku naku meeru memu manam cheyali cheyyali chesanu chesa chesthunna chestunna chesthunnanu chestunnanu
chesina chesaru ayindi ayyindi ayyi avuthundi avutundi avuthmdhi avtundi undi undhi unnai unnayi vmnai ledu levu ledhu
raledu kaani kani malli vere enduku emaina enti eppudu ippudu ela kavali kaavali pettina pettanu kuda kooda anukondi
illa irukku iruku panna pannunga enna eppadi naan enakku venum vendam
aahe kasa kase mala tumhi
""".split())
_NON_ENGLISH_WEAK = set("kar kr ho se aur bhi bhai fir phir kab oka lo ka ki ke na k m h par ya".split())


def auto_tag_theme(text: str) -> str:
    """Keyword auto-tag for rows outside the validated set (about half agree with validated labels)."""
    scores = {t: sum(w for rx, w in rules if rx.search(str(text))) for t, rules in _AUTO_TAG.items()}
    theme, best = max(scores.items(), key=lambda kv: kv[1])
    return theme if best >= 2 else "Unclassified"


_EMOTION_RULES = [  # first match wins; applied only to auto-tagged (unvalidated) rows
    ("Fear of Penalty", r"\b(penalt\w*|bounce\w*|charges?|fine|lien|fee)\b"),
    ("Panic/Fear", r"\b(scared|panic\w*|fear|afraid|crash\w*|dar)\b"),
    ("Distrust", r"\b(scam\w*|fraud|cheat\w*|trust|hacked|without (my )?permission)\b"),
    ("Anxiety/Stress", r"\b(worried|tension|urgent|stress\w*|help|please|plz|pls|😢|😭)"),
    ("Confusion", r"\b(kaise|kese|how (do|to|can)|kya|confus\w*|understand|\?)"),
    ("Frustration", r"\b(again|still|not working|nahi ho|nhi ho|useless|worst|frustrat\w*|fed up)\b"),
    ("Disillusionment", r"\b(returns?|loss|negative|fd|not convincing)\b"),
]
_THEME_EMOTION = {"Bank & Auto-Debit Friction": "Frustration", "Platform Bugs": "Frustration",
                  "Directionless Onboarding": "Confusion", "Liquidity Shortage": "Anxiety/Stress",
                  "Market Correction Panic": "Panic/Fear", "Disappointment Valley": "Disillusionment",
                  "Social Influence": "FOMO", "Same-Category Overlap": "Confusion"}
_THEME_PREVENT = {"Bank & Auto-Debit Friction": ec.PREVENT_UX, "Platform Bugs": ec.PREVENT_UX,
                  "Directionless Onboarding": ec.PREVENT_EDU, "Liquidity Shortage": ec.PREVENT_UX,
                  "Market Correction Panic": ec.PREVENT_NUDGE, "Disappointment Valley": ec.PREVENT_NUDGE,
                  "Social Influence": ec.PREVENT_EDU, "Same-Category Overlap": ec.PREVENT_UX}


def infer_emotion(text: str, theme: str) -> str:
    """Keyword-inferred emotional trigger for auto-tagged rows (falls back to the theme's typical emotion)."""
    t = str(text).lower()
    for label, rx in _EMOTION_RULES:
        if re.search(rx, t):
            return label
    return _THEME_EMOTION.get(theme, "Anxiety / Frustration")


def infer_preventability(text: str, theme: str) -> str:
    """Keyword-inferred preventability for auto-tagged rows."""
    t = str(text).lower()
    if re.search(r"\b(lost (my )?job|laid off|medical|hospital|wedding|nri|abroad|out of india)\b", t):
        return ec.PREVENT_LOW
    if theme == "Bank & Auto-Debit Friction" and re.search(r"\b(balance|salary|penalt\w*|bounce\w*|reminder)\b", t):
        return ec.PREVENT_NUDGE
    return _THEME_PREVENT.get(theme, ec.NOT_ASSESSED)


# SIP lifecycle stage inferred from what the comment is about (every row gets one, unlike stated tenure)
LIFECYCLE_STAGES = ["AutoPay Setup & Verification", "Active SIP Execution", "Cancellation / Pause Request",
                    "Redemption / Withdrawal"]
LIFECYCLE_OTHER = "General / Other Feedback"
_LIFECYCLE_RULES = {
    "Redemption / Withdrawal": [
        (r"\b(redeem\w*|redemption|withdr[ae]w\w*|exit load|sell(ing)?|sold|nikal\w*|wapas|vapas|bapos|refund\w*)\b", 3),
        (r"\b(money|amount|paisa|paise|pasaa|paysa|ammount)\b.{0,40}\b(back|credit\w*|aata|aayega|milega|wapas)\b", 2),
        (r"\b(don'?t|do not|not|nahi|nhi)\b.{0,20}\b(withdraw\w*|redeem\w*)", -3),
        (r"\b(lumpsum|lump sum|lock[- ]?in|elss)\b", 1)],
    "Cancellation / Pause Request": [
        (r"\b(cancel\w*|cancell?ation|pause\w*|stop\w*|discontinu\w*|band kar\w*|close\w*)\b", 3),
        (r"\b(skip\w*|resume\w*|restart\w*|reactivat\w*|modify|step[- ]?down)\b", 2)],
    "AutoPay Setup & Verification": [
        (r"\b(set ?up|setup|register\w*|registration|approv\w*|verif\w*|activat\w*|link\w*|add(ed)? (a |new )?(bank|mandate))\b", 2),
        (r"\b(mandate|e-?nach|nach|auto[\s-]?pay|auto[\s-]?debit|upi limit|kyc)\b", 2),
        (r"\b(new sip|start(ed|ing)? (a |my )?sip|first sip|pehli sip)\b", 2)],
    "Active SIP Execution": [
        (r"\b(insufficient|failed|rejected|declined)\b", 3),
        (r"\b(deduct\w*|debit\w*|instal+ment|payment (failed|fail|pending)|bounce\w*|penalt\w*|charges?|kat\w*|cut)\b", 3),
        (r"\b(nav|units?|returns?|xirr|loss|profit|portfolio|growth|performance|negative|red)\b", 2),
        (r"\b(missed|pending|processing|in progress|order|allot\w*|not (reflect|showing|updated))\b", 2),
        (r"\b(market|crash\w*|correction|fall\w*|nifty|sensex)\b", 1)],
}
_LIFECYCLE_RX = {s: [(re.compile(p, re.I), w) for p, w in rules] for s, rules in _LIFECYCLE_RULES.items()}
_TIE_ORDER = ["Redemption / Withdrawal", "Cancellation / Pause Request", "AutoPay Setup & Verification", "Active SIP Execution"]

def infer_lifecycle_stage(text) -> str:
    """Which step of the SIP journey the friction happened in, from keywords in the comment."""
    t = str(text or "")
    scores = {s: sum(w for rx, w in rules if rx.search(t)) for s, rules in _LIFECYCLE_RX.items()}
    best = max(scores.values())
    if best < 2:
        return LIFECYCLE_OTHER
    return next(s for s in _TIE_ORDER if scores[s] == best)


def is_non_english(text) -> bool:
    """True for comments that need an English translation: any non-Latin script (Devanagari, Telugu, Tamil, etc.)
    or romanised Indic text such as Hinglish or Telugu typed in English letters. Emojis, the rupee sign and
    curly quotes are not letters, so they never trigger it; one stray Hindi word in an English comment doesn't either."""
    if text is None:
        return False
    try:
        if pd.isna(text):
            return False
    except (TypeError, ValueError):
        pass
    t = str(text)
    if not t.strip():
        return False
    if any(ch.isalpha() and not unicodedata.name(ch, "LATIN").startswith("LATIN") for ch in t):
        return True
    words = set(re.findall(r"[a-z]+", t.lower()))
    return 2 * len(words & _NON_ENGLISH_STRONG) + len(words & _NON_ENGLISH_WEAK) >= 3


needs_translation = is_non_english  # older name, kept for compatibility


def normalize_platform(value) -> str:
    v = str(value).lower()
    if "reddit" in v:
        return "Reddit"
    if "youtube" in v or "youtu.be" in v:
        return "YouTube"
    return "Play Store"  # Google Play Store and Apple App Store reviews


def _match_key(comment, url) -> tuple:
    """Encoding-proof row key: URL + the first 80 ASCII letters/digits of the comment. Survives files
    re-saved with garbled characters (₹, Hindi, dashes) because only ASCII alphanumerics are compared."""
    return (str(url).strip(), re.sub(r"[^a-z0-9]", "", str(comment).lower())[:80])


def _text_key(comment, url) -> tuple:
    return (" ".join(str(comment).split()).lower(), str(url).strip())



def _find_data_file(name: str):
    """Look next to app.py first, then in a data/ sub-folder and the working directory."""
    from pathlib import Path
    here = Path(__file__).resolve().parent
    for folder in (here, here / "data", Path.cwd(), Path.cwd() / "data"):
        if (folder / name).exists():
            return folder / name
    return None


def _clean_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise one source: drop Age, ensure the 13-column schema incl. Year, binary Product."""
    df = df.drop(columns=[c for c in df.columns if str(c).strip().lower() == "age"])
    for col in MASTER_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[MASTER_SCHEMA].copy()
    df["Original Comment"] = df["Original Comment"].map(lambda x: html.unescape(str(x)).strip() if pd.notna(x) else "")
    df["Product"] = ["Groww" if ("groww" in str(p).lower() or "groww" in str(c).lower()) else "Other"
                     for p, c in zip(df["Product"], df["Original Comment"])]
    df["Source Platform"] = df["Source Platform"].map(normalize_platform)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["SIP Tenure (Months)"] = pd.to_numeric(df["SIP Tenure (Months)"].map(ec.parse_tenure), errors="coerce").astype("Int64")
    return df


@st.cache_data(show_spinner=False)
def load_master_dataset():
    """wb.xlsx + reddit_sip_reviews.csv + youtube_sip_comments.csv -> df_master.
    Returns (df_master, info) where info has per-file row counts, missing files and duplicates removed."""
    frames, info = [], {"rows": {}, "missing": [], "errors": [], "extracted": 0, "duplicates": 0}
    for name in SOURCE_FILES:
        path = _find_data_file(name)
        if path is None:
            info["missing"].append(name)
            continue
        try:
            raw = pd.read_excel(path) if name.endswith(".xlsx") else pd.read_csv(path)
        except Exception as exc:
            info["errors"].append(f"{name}: {type(exc).__name__}")
            continue
        info["rows"][name] = len(raw)
        frames.append(_clean_schema(raw))
    if not frames:
        return pd.DataFrame(columns=MASTER_SCHEMA), info
    master = pd.concat(frames, ignore_index=True)
    info["extracted"] = len(master)
    if DROP_DUPLICATE_ROWS:
        master = master.drop_duplicates(subset=["Original Comment", "Source URL"]).reset_index(drop=True)
    info["duplicates"] = info["extracted"] - len(master)
    info["by_platform"] = master["Source Platform"].value_counts().reindex(PLATFORMS, fill_value=0).to_dict()
    return master, info


@st.cache_data(show_spinner=False)
def load_golden_set():
    """golden_set.xlsx (sheet 'Golden Set' or first sheet) or golden_set.csv -> (df_golden, file name, raw xlsx bytes)."""
    import io
    xlsx, csv = _find_data_file("golden_set.xlsx"), _find_data_file("golden_set.csv")
    try:
        if xlsx is not None:
            raw = xlsx.read_bytes()
            sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None)
            df = sheets.get("Golden Set", next(iter(sheets.values())))
            name = "golden_set.xlsx"
        elif csv is not None:
            df, raw, name = pd.read_csv(csv), None, "golden_set.csv"
        else:
            return None, None, None
    except Exception:
        return None, None, None
    df = df.drop(columns=[c for c in df.columns if str(c).strip().lower() == "age"])
    for col in MASTER_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[MASTER_SCHEMA].copy()
    df["Original Comment"] = df["Original Comment"].astype(str).map(html.unescape)
    df["Source Platform"] = df["Source Platform"].map(normalize_platform)
    df["Product"] = ["Groww" if ("groww" in str(p).lower() or "groww" in str(c).lower()) else "Other"
                     for p, c in zip(df["Product"], df["Original Comment"])]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["SIP Tenure (Months)"] = pd.to_numeric(df["SIP Tenure (Months)"].map(ec.parse_tenure), errors="coerce").astype("Int64")
    # Normalise label spellings (dash / encoding / case variants) so filters, charts and KPIs match
    df["Churn Preventability"] = df["Churn Preventability"].map(ec.normalize_preventability)
    df["Problem Theme"] = df["Problem Theme"].map(lambda v: ec.normalize_theme(v) if pd.notna(v) else v)
    df["Investor Emotional Trigger"] = df["Investor Emotional Trigger"].map(
        lambda v: str(v).strip() if pd.notna(v) and str(v).strip().lower() not in ec.MISSING_LABELS else pd.NA)
    return df, name, raw


@st.cache_data(show_spinner=False)
def load_golden_glosses() -> dict:
    """English glosses from the 'Reviewer Notes' sheet of golden_set.xlsx (if present), keyed by comment."""
    import io
    path = _find_data_file("golden_set.xlsx")
    if path is None:
        return {}
    try:
        sheets = pd.read_excel(io.BytesIO(path.read_bytes()), sheet_name=None)
        gs, notes = sheets.get("Golden Set"), sheets.get("Reviewer Notes")
        if gs is None or notes is None or len(gs) != len(notes) or notes.shape[1] < 3:
            return {}
        return {" ".join(html.unescape(str(c)).split()).lower(): str(g)
                for c, g in zip(gs["Original Comment"], notes.iloc[:, 2]) if not str(g).startswith("(English")}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def build_full_view(master: pd.DataFrame, golden: pd.DataFrame | None) -> pd.DataFrame:
    """Full Discovery dataset for analysis: validated labels where a row is in the validated set,
    keyword auto-tagged Problem Theme elsewhere (clearly marked in 'Label Source')."""
    df = master.copy()
    analytic = ["User Type", "SIP Tenure (Months)", "SIP Stage", "Problem Theme", "Problem Behaviour",
                "Problem Root Cause", "Investor Emotional Trigger", "Churn Preventability"]
    if not KEEP_WB_LABELS:
        df[analytic] = pd.NA
    df["Label Source"] = "Auto-tagged (keyword, unvalidated)"
    if golden is not None:
        lookup = {_match_key(c, u): i for i, (c, u) in enumerate(zip(golden["Original Comment"], golden["Source URL"]))}
        for idx, (c, u) in enumerate(zip(df["Original Comment"], df["Source URL"])):
            gi = lookup.get(_match_key(c, u))
            if gi is not None:
                for col in analytic:
                    df.at[idx, col] = golden.iloc[gi][col]
                df.at[idx, "Label Source"] = "Validated"
    unl = df["Label Source"] != "Validated"
    df.loc[unl, "Problem Theme"] = df.loc[unl, "Original Comment"].map(auto_tag_theme)
    df.loc[df["Problem Theme"] == "Unclassified", "Label Source"] = "Unclassified"
    auto = df["Label Source"] == "Auto-tagged (keyword, unvalidated)"
    df.loc[auto, "Investor Emotional Trigger"] = [infer_emotion(c, t) for c, t in
                                                  zip(df.loc[auto, "Original Comment"], df.loc[auto, "Problem Theme"])]
    df.loc[auto, "Churn Preventability"] = [infer_preventability(c, t) for c, t in
                                            zip(df.loc[auto, "Original Comment"], df.loc[auto, "Problem Theme"])]
    df["SIP Stage"] = df["SIP Stage"].fillna("Not stated")
    df["Churn Preventability"] = df["Churn Preventability"].fillna("Not assessed")
    df["SIP Tenure (Months)"] = pd.to_numeric(df["SIP Tenure (Months)"].map(ec.parse_tenure), errors="coerce").astype("Int64")
    df["Churn Preventability"] = df["Churn Preventability"].map(ec.normalize_preventability)
    kb = ec.THEME_KB
    df["Recommended Groww Intervention"] = df["Problem Theme"].map(lambda t: kb.get(t, {}).get("intervention", pd.NA))
    df["Target North-Star Metric Impact"] = df["Problem Theme"].map(lambda t: kb.get(t, {}).get("nsm", pd.NA))
    return df


@st.cache_data(show_spinner=False)
def dataset_to_xlsx(df: pd.DataFrame, sheet: str) -> bytes:
    """Plain formatted workbook: navy headers, wrapped text, clickable Source URL, filters."""
    import io
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active
    ws.title = sheet[:31]
    widths = {"Original Comment": 60, "Source URL": 40, "Problem Behaviour": 40, "Problem Root Cause": 40}
    for c, col in enumerate(df.columns, 1):
        cell = ws.cell(1, c, col)
        cell.font, cell.fill = Font(name="Arial", bold=True, color="FFFFFF"), PatternFill("solid", fgColor=NAVY[1:])
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[get_column_letter(c)].width = widths.get(col, 18)
    for r, row in enumerate(df.itertuples(index=False), 2):
        for c, v in enumerate(row, 1):
            v = None if (not isinstance(v, str) and pd.isna(v)) else (v.item() if hasattr(v, "item") else v)
            cell = ws.cell(r, c, v)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if df.columns[c - 1] == "Source URL" and v and str(v).startswith("http"):
                cell.hyperlink = str(v)
                cell.font = Font(name="Arial", size=10, color="0563C1", underline="single")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(df.columns))}{len(df) + 1}"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def get_benchmark():
    """Dashboard dataset: the hand-labelled golden set when available; otherwise the previous
    real benchmark (real_benchmark_220.json), then the synthetic demo corpus."""
    golden, name, _ = load_golden_set()
    if golden is not None and len(golden):
        df = golden.copy()
        kb = {t: v for t, v in ec.THEME_KB.items()}
        df["Recommended Groww Intervention"] = df["Problem Theme"].map(lambda t: kb.get(t, {}).get("intervention", pd.NA))
        df["Target North-Star Metric Impact"] = df["Problem Theme"].map(lambda t: kb.get(t, {}).get("nsm", pd.NA))
        return df, {"source": name, "harvested_at": "", "rejected_non_post_urls": 0}, True
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
                [f"⚠️ Live ingestion failed unexpectedly: {type(exc).__name__}: {exc}. Showing nothing from this run."],
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
friction explain work works between difference better best good bad worse worst than more less vs versus
compare comparison really actually way ways thing things know need want like""".split())

SUGGESTED_QUESTIONS = [
    "Why do investors pause or stop their SIPs?",
    "Why do AutoPay payments fail?",
    "What do investors say about returns in the first year?",
    "How do investors react to a market crash?",
    "What problems do investors face when starting a SIP?",
]

# ---------------------------------------------------------------------------
# SIP & wealth domain engine — built-in, zero-API-key knowledge.
# Each topic answers in three parts: explanation & trade-offs, investor
# behaviour insight, and Groww in-app best practice. Content is general and
# educational: no scheme recommendations, no return promises, no urgency.
# Figures that can change (tax, limits) carry their "as of" context.
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "topic": "SIP vs lumpsum",
        "keys": ["lumpsum", "lump sum", "lump-sum", "one time", "one-time", "onetime", "all at once", "stp",
                 "systematic transfer"],
        "weight": 3,
        "themes": ["Market Correction Panic", "Disappointment Valley"],
        "short": "A lumpsum puts the whole amount into the market at once; a SIP spreads it over regular "
                 "instalments. The trade-off is time in the market versus timing risk and regret.",
        "points": [
            "**Lumpsum:** all money is exposed from day one, so in rising markets it tends to end ahead. Studies of "
            "historical data, mostly from developed markets, find lumpsum beating staggered investing roughly "
            "two-thirds of the time. The cost is full exposure if the market falls soon after.",
            "**SIP:** suits regular income, needs no large sum upfront, and averages the purchase price across "
            "market levels. It lowers the risk of investing everything at a peak, but money waiting to be invested "
            "earns less in rising markets.",
            "**Middle path for a large sum:** park it in a low-volatility fund and move it gradually into equity "
            "through a Systematic Transfer Plan (STP).",
            "**The deciding factor is usually cash-flow, not maths:** most first-time investors invest from monthly "
            "income, so a SIP is the natural fit.",
        ],
        "behaviour": "Regret is asymmetric: investing a lumpsum just before a fall feels far worse than missing "
                     "gains by going slowly. That's why many people prefer SIPs even when lumpsum has the better "
                     "average outcome. SIP investors still churn when early returns look flat, because the "
                     "averaging benefit is invisible in year one.",
        "practice": [
            "Offer a neutral SIP / lumpsum / STP comparison based on the investor's own amount and horizon, with "
            "historical ranges rather than projected returns.",
            "Show ‘units bought at each price’ for SIPs so the averaging benefit becomes visible.",
            "Never frame either option with urgency (‘invest before the market rises’).",
        ],
    },
    {
        "topic": "Pausing vs cancelling a SIP",
        "keys": ["pause", "pausing", "cancel", "cancelling", "stop", "stopping", "skip", "restart", "resume"],
        "themes": ["Liquidity Shortage", "Market Correction Panic"],
        "short": "Pausing skips a few instalments and keeps the SIP set up; cancelling ends all future "
                 "instalments. In both cases, the units already bought stay invested.",
        "points": [
            "**Pause:** future instalments are skipped for a chosen number of months, then the SIP resumes on its "
            "own. The registration and bank mandate stay in place. Pause limits depend on the fund house and "
            "platform.",
            "**Cancel:** no more instalments are debited. Existing units stay in the folio until redeemed. "
            "Restarting means setting up a new SIP and sometimes a new mandate.",
            "**Redeem** is different again: it sells units and withdraws money.",
            "**Which is better?** For a temporary cash crunch or short-term worry, a pause keeps the habit and "
            "set-up intact. If the goal has changed or the SIP was never a fit, cancelling is a reasonable, "
            "deliberate choice. Neither is universally better.",
        ],
        "behaviour": "People rarely intend to quit investing for good. Most cancellations are reactions to a "
                     "short-term event: cash pressure, a scary week in the market, a failed debit. When the only "
                     "visible exit is ‘Cancel’, a temporary decision becomes permanent, and the effort of "
                     "restarting (status-quo bias) keeps it that way.",
        "practice": [
            "On the cancel screen, show Pause (1–3 months) and Step down amount alongside Cancel. Cancel stays "
            "one tap away.",
            "Auto-resume paused SIPs with a single neutral reminder a few days before.",
            "Track the pause-instead-of-cancel ratio and the reactivation rate.",
        ],
    },
    {
        "topic": "What a mutual fund is and how it works",
        "keys": ["what is mutual fund", "what is a mutual fund", "what are mutual funds", "mutual fund mean",
                 "how do mutual funds work", "how does a mutual fund work", "mutual fund basics", "what is an amc",
                 "what is nav"],
        "weight": 3,
        "themes": ["Directionless Onboarding", "Disappointment Valley"],
        "short": "A mutual fund pools money from many investors and invests it in a portfolio of stocks, bonds or "
                 "other assets, managed by a professional fund manager at an asset management company (AMC) "
                 "regulated by SEBI.",
        "points": [
            "**Units and NAV:** investors buy units at the day's net asset value (NAV), the per-unit value of the "
            "fund's holdings after expenses. The value of an investment rises and falls with the NAV.",
            "**Main types:** equity funds (shares, higher volatility), debt funds (bonds, money-market), hybrid "
            "funds (a mix) and index funds or ETFs (track a market index at low cost).",
            "**Costs:** an annual expense ratio is deducted inside the NAV; some funds charge an exit load on early "
            "redemption. Direct plans have lower expense ratios than regular plans.",
            "**Ways to invest:** a SIP (fixed amount at regular intervals) or a lumpsum (one-time amount).",
            "**Returns are market-linked and not guaranteed**, unlike a fixed deposit. Each scheme's documents set "
            "out its objective, risks and costs.",
        ],
        "behaviour": "Many first-time investors meet mutual funds through an app's 'start a SIP' button and treat "
                     "them like a recurring deposit. When the value fluctuates, the mismatch between that mental "
                     "model and market-linked reality is a common trigger for stopping.",
        "practice": [
            "Explain NAV, units and market-linked returns in one screen before the first SIP.",
            "Show units accumulated alongside value, so early fluctuations have context.",
        ],
    },
    {
        "topic": "What a SIP is and how it works",
        "keys": ["what is a sip", "what is sip", "systematic investment", "how does sip", "how sip works",
                 "sip work", "sip mean", "sip basics"],
        "weight": 2,
        "themes": [],
        "short": "A Systematic Investment Plan (SIP) invests a fixed amount in a mutual fund at regular intervals, "
                 "usually monthly, through an automatic bank debit.",
        "points": [
            "Each instalment buys units at that day's NAV (net asset value), so the number of units varies with "
            "the price.",
            "The investor chooses the amount, frequency, date and duration, and can usually pause, modify or "
            "cancel.",
            "Automation builds a habit and removes the need to time the market. Returns are market-linked and "
            "not guaranteed.",
        ],
        "behaviour": "Automation turns saving into a default, which is the most powerful behavioural lever there "
                     "is. The flip side is that people also stop paying attention, so failures and doubts surface "
                     "suddenly rather than gradually.",
        "practice": [
            "Pair automation with visibility: instalment receipts, units bought and a clear SIP status timeline.",
            "Send a pre-debit reminder so the debit never surprises the investor.",
        ],
    },
    {
        "topic": "Market crashes, volatility and SIPs",
        "keys": ["volatil", "crash", "correction", "fall", "falling", "dip", "market down", "bear", "averaging",
                 "rupee cost"],
        "themes": ["Market Correction Panic"],
        "short": "A SIP invests a fixed amount on a schedule. When prices fall, the same amount buys more units, "
                 "lowering the average cost per unit over time.",
        "points": [
            "**Rupee-cost averaging** smooths the entry price across market levels. It doesn't guarantee profit or "
            "protect against loss.",
            "**Drawdowns are normal for equity.** Indian indices have seen several sharp corrections that later "
            "recovered, but past patterns don't guarantee future outcomes.",
            "**Stopping during a fall** means no units are bought at the lower prices, which is the part the SIP "
            "mechanism relies on.",
            "**Short holding periods** carry the widest range of outcomes. The range narrows as the holding period "
            "lengthens.",
        ],
        "behaviour": "Loss aversion (losses hurt about twice as much as equal gains feel good) peaks during a "
                     "first-ever drawdown. News and social feeds amplify it, while the app shows a single red "
                     "number with no context.",
        "practice": [
            "During market-wide falls, show factual context: historical drawdown and recovery ranges and units "
            "accumulated. No ‘buy the dip’ prompts and no urgency.",
            "Lead the portfolio screen with invested amount and units, not just today's P&L.",
            "Measure SIP stops during weeks with a drawdown above 5%.",
        ],
    },
    {
        "topic": "What XIRR is and why first-year returns look flat",
        "keys": ["xirr", "cagr", "absolute return", "return", "returns", "flat", "fd", "fixed deposit",
                 "first year", "one year", "1 year", "negative"],
        "themes": ["Disappointment Valley"],
        "short": "XIRR is an annualised return that accounts for each instalment's date and amount. In year one, "
                 "most of the money has been invested for only a few months, so gains look small.",
        "points": [
            "**XIRR vs CAGR:** CAGR suits a single investment; XIRR handles many cash-flows on different dates, "
            "which is why it's used for SIPs.",
            "**Time-weighting:** at month 12, the average rupee in a monthly SIP has been invested for about six "
            "months, and the latest instalment for almost none.",
            "**Short periods mislead:** XIRR can swing sharply over a few months. Annualising a small gain or "
            "loss exaggerates it.",
            "**FD comparisons** set an anchor of steady, guaranteed growth that equity isn't designed to match "
            "in the short term.",
        ],
        "behaviour": "Anchoring to FD rates plus impatience creates a ‘disappointment valley’ around months 9–14. "
                     "The investor has paid in long enough to expect results but not long enough for compounding "
                     "to show.",
        "practice": [
            "Show units accumulated and historical outcome ranges by holding period next to returns.",
            "Explain XIRR in one line where it's displayed, and avoid annualising very short periods.",
        ],
    },
    {
        "topic": "Compounding and the cost of interrupting a SIP",
        "keys": ["compound", "compounding", "long term", "long-term", "power of", "interrupt", "cost of stopping",
                 "wealth", "how much will"],
        "themes": ["Disappointment Valley", "Liquidity Shortage"],
        "short": "Compounding means gains themselves earn returns, so most of a long SIP's growth arrives in the "
                 "later years. Early interruptions have an outsized effect on the end value.",
        "points": [
            "**Illustration (assumed 10% a year, not a forecast):** ₹5,000 a month for 10 years (₹6 lakh invested) "
            "grows to about ₹10.3 lakh.",
            "**Stopping after year one:** the ₹60,000 invested, left alone for nine more years, grows to about "
            "₹1.55 lakh at the same rate.",
            "**Pausing for six months instead:** ₹5.7 lakh invested grows to about ₹9.6 lakh, so most of the "
            "outcome is preserved.",
            "Actual returns vary and can be negative. The illustration only shows how time in the market matters.",
        ],
        "behaviour": "Humans think linearly about exponential growth, so stopping ‘just for now’ feels cheap. "
                     "Present bias makes today's cash need feel bigger than a distant, abstract gain.",
        "practice": [
            "At the cancel moment, show the investor's own pause vs cancel trade-off with stated assumptions, "
            "framed neutrally, without fear.",
            "Celebrate milestones (e.g. 12 instalments) based on consistency, not returns.",
        ],
    },
    {
        "topic": "How SIP auto-debit mandates work (e-NACH and UPI AutoPay)",
        "keys": ["mandate", "nach", "enach", "e-nach", "autopay", "auto pay", "auto-debit", "auto debit", "upi",
                 "limit", "limits", "otp", "registration"],
        "themes": ["Bank & Auto-Debit Friction"],
        "short": "A SIP is debited through a standing instruction on the investor's bank account: an e-NACH "
                 "mandate or UPI AutoPay. Most failures come from timing, balance, or a broken or expired mandate.",
        "points": [
            "**Mandate limit:** the investor authorises a maximum amount, and each instalment must be within it. "
            "Raising the SIP above the limit needs a new or modified mandate.",
            "**e-NACH** is registered through net banking or a debit card, typically allows higher limits, and "
            "can take a few working days to activate.",
            "**UPI AutoPay (2026 rules):** after the first authenticated payment, recurring debits need no fresh "
            "OTP up to ₹15,000 per transaction generally, and up to ₹1 lakh for mutual fund subscriptions, "
            "insurance premiums and credit card bills. Banks must send a pre-debit notice at least 24 hours "
            "before each debit. Check current RBI and NPCI rules, as limits change.",
            "**Silent breakage:** a bank-account change, mandate expiry or limit mismatch stops future debits "
            "unless the investor re-registers.",
        ],
        "behaviour": "Involuntary churn is the most preventable kind. The investor never decided to stop, but "
                     "a failed debit plus a bank charge creates a negative association, and fixing the mandate "
                     "feels like admin work.",
        "practice": [
            "Run a mandate health check (expiry, limit headroom, linked account) with one-tap guided "
            "re-registration.",
            "Offer an easy SIP-date shift to just after the investor's typical salary credit.",
            "Track instalment success rate separately from voluntary cancellations.",
        ],
    },
    {
        "topic": "Mandate bounce charges and failed instalments",
        "keys": ["charge", "charges", "bounce", "bounced", "penalty", "fees", "fine", "failed",
                 "insufficient", "missed", "miss", "not debited"],
        "weight": 2,
        "themes": ["Bank & Auto-Debit Friction"],
        "short": "A fund house typically doesn't charge for a missed SIP instalment, but the investor's bank "
                 "usually does when a mandate debit fails for low balance.",
        "points": [
            "**Bank charges:** banks levy a return or bounce fee on failed NACH or auto-debits, often a few hundred "
            "rupees plus GST. The amount varies by bank and is listed in its schedule of charges.",
            "**Fund house side:** a missed instalment is simply not invested. There is usually no penalty from the "
            "AMC, though the units for that month are not bought.",
            "**Repeated failures:** many fund houses cancel a SIP automatically after a few consecutive failed "
            "instalments, often three, and the investor may not notice.",
            "**Prevention:** keep a balance buffer before the SIP date, align the date with salary credit, or "
            "pause ahead of a known shortfall.",
        ],
        "behaviour": "A bounce fee on a ₹1,000 SIP feels like a punishment for investing. The investor blames "
                     "the SIP rather than the timing, and cancels to avoid feeling it again.",
        "practice": [
            "Send a pre-debit heads-up (amount and date) with a one-tap ‘skip this month’ option before the debit "
            "is attempted.",
            "After a failure, explain plainly what happened and offer date-change or pause.",
            "Alert the investor before an automatic cancellation after consecutive failures.",
        ],
    },
    {
        "topic": "How mutual fund and SIP redemptions are taxed (India)",
        "keys": ["tax", "taxed", "taxation", "ltcg", "stcg", "capital gain", "capital gains", "112a", "fifo"],
        "weight": 3,
        "themes": [],
        "short": "Each SIP instalment counts as a separate purchase, so holding period and tax are worked out "
                 "instalment by instalment, oldest units first (FIFO).",
        "points": [
            "**Equity-oriented funds held over 12 months (LTCG):** 12.5% on gains above ₹1.25 lakh a year in total, "
            "without indexation.",
            "**Equity-oriented funds held 12 months or less (STCG):** 20%.",
            "**Debt funds bought on or after 1 April 2023:** gains are taxed at the investor's income-tax slab "
            "rate, whatever the holding period.",
            "**SIP effect:** redeeming a whole SIP corpus after one year usually mixes long- and short-term units, "
            "because recent instalments are still under 12 months old.",
            "These are the FY 2026-27 rates (unchanged in Budget 2026). Tax rules change, so verify the current "
            "position or ask a tax adviser.",
        ],
        "behaviour": "Tax is rarely why people stop a SIP, but a surprise tax on an early redemption damages "
                     "trust, and investors often assume the whole corpus is ‘long-term’ after a year.",
        "practice": [
            "Before redemption, show an estimated short-term vs long-term split of the units being sold.",
            "Keep disclosures factual and avoid tax-saving urgency messaging.",
        ],
    },
    {
        "topic": "Lock-ins and exit loads",
        "keys": ["exit load", "lock-in", "lock in", "elss", "redeem", "redemption", "withdraw", "early exit"],
        "themes": ["Liquidity Shortage", "Disappointment Valley"],
        "short": "Some funds restrict or charge for early withdrawals. Check each scheme's documents for the "
                 "exact terms.",
        "points": [
            "**ELSS** (tax-saving) funds have a three-year lock-in, applied to each SIP instalment separately.",
            "**Exit loads:** many equity funds charge a small exit load if units are redeemed within a set period, "
            "often around a year. The exact terms are in the scheme documents.",
            "Under FIFO, the earliest instalments clear lock-in and exit-load periods first.",
        ],
        "behaviour": "Investors in a cash crunch discover lock-ins and exit loads only when they try to withdraw. "
                     "That turns a liquidity problem into a trust problem.",
        "practice": [
            "Show lock-in and exit-load dates per instalment on the holdings and redemption screens.",
            "Suggest a pause for temporary cash needs before the investor reaches the redeem flow.",
        ],
    },
    {
        "topic": "Step-up (top-up) and step-down SIPs",
        "keys": ["step up", "step-up", "top up", "top-up", "increase sip", "increase my sip", "step down",
                 "step-down", "reduce sip", "decrease", "lower amount"],
        "themes": ["Liquidity Shortage"],
        "short": "A step-up SIP raises the instalment by a fixed amount or percentage at set intervals. A step-down "
                 "reduces it. Both adapt the SIP to income instead of ending it.",
        "points": [
            "**Step-up:** aligns contributions with salary growth. Many fund houses and platforms support "
            "automatic annual top-ups.",
            "**Step-down:** reducing the amount during a tight period keeps the habit and mandate alive.",
            "**Mandate headroom:** a step-up may need a higher mandate limit, so set it with room to grow.",
        ],
        "behaviour": "An all-or-nothing frame (‘I can't afford ₹5,000, so I'll stop’) causes avoidable churn. "
                     "Giving people a smaller-amount option keeps them in the habit.",
        "practice": [
            "Offer step-down as a first-class option on the cancel screen.",
            "Prompt an optional step-up only at natural moments, such as the SIP anniversary, never with "
            "pressure.",
        ],
    },
    {
        "topic": "Direct vs regular mutual fund plans",
        "keys": ["direct plan", "regular plan", "direct vs regular", "regular vs direct", "expense ratio",
                 "commission"],
        "weight": 2,
        "themes": [],
        "short": "Direct and regular plans hold the same portfolio. A direct plan has no distributor commission, "
                 "so its expense ratio is lower; a regular plan includes the commission.",
        "points": [
            "The lower expense ratio of a direct plan compounds into a meaningful difference over long periods.",
            "A regular plan pays for distributor services, which some investors value.",
            "Expense ratios are published by each fund house and change over time.",
        ],
        "behaviour": "Small recurring costs are psychologically invisible (low salience). Investors focus on "
                     "headline returns, not the fees deducted inside NAV.",
        "practice": [
            "Display expense ratio clearly and consistently on fund and SIP pages.",
            "Keep the plan type visible in the investor's holdings.",
        ],
    },
    {
        "topic": "Choosing a SIP date",
        "keys": ["sip date", "which date", "best date", "debit date", "date of sip", "change date", "salary date"],
        "weight": 2,
        "themes": ["Bank & Auto-Debit Friction"],
        "short": "Over long periods the specific SIP date tends to matter far less than whether each debit "
                 "succeeds and the SIP keeps running.",
        "points": [
            "Choose a date a few days after salary credit, so the balance is there.",
            "Differences between dates usually even out over many years of instalments.",
            "Most platforms let you change the date, though it can take a cycle to apply.",
        ],
        "behaviour": "Investors over-think the ‘best date’ (an illusion of control) and under-think cash-flow "
                     "timing, which is what actually causes failed debits.",
        "practice": [
            "Default the SIP date suggestion to a few days after the investor's stated salary date.",
            "Make date changes self-serve and instant to understand.",
        ],
    },
    {
        "topic": "Fund overlap and diversification",
        "keys": ["overlap", "diversif", "too many funds", "multiple funds", "same category", "consolidat"],
        "themes": ["Same-Category Overlap"],
        "short": "Several funds in the same category often hold many of the same stocks. More funds doesn't "
                 "automatically mean more diversification.",
        "points": [
            "Funds in a category follow similar mandates, so their top holdings frequently overlap.",
            "Diversification comes from category, asset-class and strategy mix, not fund count.",
            "When overlap is discovered, investors often cancel several SIPs at once rather than consolidating.",
        ],
        "behaviour": "Choice overload and ‘diversification illusion’ lead to fund collecting. Discovering the "
                     "overlap then triggers self-doubt and a clean-up that looks like churn.",
        "practice": [
            "Show category-level overlap of the investor's own holdings.",
            "Offer a guided ‘merge into one SIP’ flow where the investor chooses what to keep. The platform does "
            "not recommend a scheme.",
        ],
    },
    {
        "topic": "Goal-based investing and onboarding",
        "keys": ["goal", "onboard", "purpose", "plan", "beginner", "first time", "first-time", "new investor",
                 "start investing"],
        "themes": ["Directionless Onboarding"],
        "short": "SIPs started without a goal or horizon have nothing to anchor them when motivation dips or "
                 "returns look flat.",
        "points": [
            "A named goal (an emergency buffer, a home deposit, education) with a horizon gives a reason to "
            "continue.",
            "Fast onboarding gets the first SIP live quickly but rarely connects it to a life outcome.",
            "Milestone check-ins against the investor's own goal reinforce the habit without promoting products.",
        ],
        "behaviour": "Without a goal, the SIP is just an expense line, so present bias wins whenever money is "
                     "tight. A named goal creates a mental account people are reluctant to raid.",
        "practice": [
            "Add an optional, skippable goal step at SIP creation.",
            "Send neutral milestone check-ins at M3, M6, M9 and M12 against the investor's own goal.",
        ],
    },
    {
        "topic": "Liquidity, emergencies and SIP sizing",
        "keys": ["emergency", "liquidity", "job", "salary", "afford", "cash", "income", "sizing", "budget"],
        "themes": ["Liquidity Shortage"],
        "short": "Many SIPs stop because of a cash-flow shock, not because the investor gave up on investing.",
        "points": [
            "SIPs sized on peak-month income leave no buffer for variable earnings or one-off expenses.",
            "Without an emergency fund, the SIP corpus becomes the emergency fund and gets redeemed.",
            "A commonly cited guideline is an emergency buffer of a few months' expenses held separately.",
        ],
        "behaviour": "Mental accounting: when there's no separate ‘emergency’ bucket, the investment becomes "
                     "the default source of cash, and redemption feels like a reasonable fix.",
        "practice": [
            "Offer pause and step-down before cancellation or redemption.",
            "Suggest a SIP amount based on the investor's own stated budget, with headroom.",
        ],
    },
    {
        "topic": "Investor psychology behind SIP drop-offs",
        "keys": ["psycholog", "behavio", "bias", "emotion", "loss aversion", "herd", "patience", "impatien",
                 "mindset", "regret", "fomo"],
        "themes": ["Disappointment Valley", "Market Correction Panic", "Social Influence"],
        "short": "Most SIP drop-offs are driven less by the product than by predictable behavioural biases, "
                 "especially in the first year.",
        "points": [
            "**Loss aversion:** a red portfolio triggers action out of proportion to its size.",
            "**Myopic loss aversion:** checking often makes normal ups and downs feel like constant losses.",
            "**Anchoring:** year-one equity returns are judged against familiar FD rates.",
            "**Herding and recency:** peers, creators and headlines override the original plan.",
            "**Present bias:** today's spending beats a distant goal, especially when no goal was set.",
        ],
        "behaviour": "These biases cluster around months 9–14: early enthusiasm has faded, compounding isn't "
                     "visible yet, and the investor has seen at least one drawdown.",
        "practice": [
            "Link goals at setup and reduce red-number shocks on the portfolio screen.",
            "Give calm, factual context at stress moments, with no manipulation or urgency.",
        ],
    },
    {
        "topic": "Measuring SIP retention and churn",
        "keys": ["metric", "measure", "kpi", "retention rate", "churn rate", "north star", "cohort", "track"],
        "themes": [],
        "short": "SIP churn is best tracked by activation cohort, separating voluntary stops from involuntary "
                 "debit failures.",
        "points": [
            "**M12 SIP retention:** share of SIPs still active 12 months after activation, by monthly cohort.",
            "**Instalment success rate:** share of scheduled debits that succeed, which isolates bank and "
            "mandate friction.",
            "**Pause-to-cancel ratio** and **reactivation rate** show whether softer exits are working.",
        ],
        "behaviour": "Aggregate churn hides the mix. Involuntary failures and voluntary exits need opposite "
                     "fixes, so blending them leads to the wrong interventions.",
        "practice": [
            "Report voluntary and involuntary churn separately, by cohort and tenure.",
            "Pair the numbers with the qualitative drivers in this engine to see why each cohort leaves.",
        ],
    },
    {
        "topic": "Neutral factors investors weigh (not a recommendation)",
        "keys": ["which fund", "best fund", "should i", "recommend", "where to invest", "where should i invest"],
        "themes": [],
        "short": "There is no single right fund or decision. It depends on the investor's goal, time horizon, "
                 "risk tolerance and cash-flow needs.",
        "points": [
            "**Goal and horizon:** money needed within a few years is usually treated differently from long-term "
            "goals.",
            "**Emergency buffer first:** separate savings stop a SIP becoming the emergency fund.",
            "**Risk tolerance:** being able to sit through a fall without stopping matters as much as expected "
            "return.",
            "**Costs and category:** expense ratio, exit load and category are in each scheme's documents.",
            "**Continuing vs stopping:** for a temporary crunch, pausing or stepping down is an alternative to "
            "cancelling. The right choice depends on personal circumstances.",
        ],
        "behaviour": "Under uncertainty people look for a single ‘right answer’ and defer to whoever sounds "
                     "confident, which is why unverified tips drive so many switches.",
        "practice": [
            "Provide neutral education and goal tools. Personal recommendations come from a SEBI-registered "
            "investment adviser.",
        ],
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
DATA_CUE_PATTERN = re.compile(r"\b(top|most common|complain\w*|friction|issues?|problems?|bugs?|glitch\w*|errors?|"
                              r"churn\w*|drop[- ]?offs?|why do (people|users|investors|customers)|reasons?|"
                              r"what do (investors|users|people|customers) say|users?|reviews?|quotes?|examples?|"
                              r"how many|percentage|share of|data|feedback|app)\b", re.I)


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
    asks_when = bool(re.search(
        r"\bwhen (do|does|did|are|is|will)\b.*\b(stop|cancel|pause|quit|drop|churn|break|leave)|\bwhich month\b|"
        r"\bwhat month\b|\bhow long\b|\btenure\b|\bcliff\b|\bat what (stage|point)\b|\btiming\b|"
        r"\bmonth \d+|\b\d+(st|nd|rd|th)? month", q))
    asks_start = bool(re.search(r"\b(start|starting|started|begin|beginning|new to|first sip|first[- ]time|"
                                r"onboard\w*|set ?up|setting up)\b", q))
    asks_feel = _match_any(q, ["feel", "emotion", "sentiment", "frustrat", "angry", "mood"])
    consumed = {w for syns in list(THEME_SYNONYMS.values()) + list(PRODUCT_SYNONYMS.values())
                + list(BEHAVIOUR_SYNONYMS.values()) for w in syns}
    tokens = [w for w in re.findall(r"[a-z&]{3,}", q) if w not in STOPWORDS and w not in consumed]

    subset, scope_bits, notes = data, [], []
    if asks_start and "User Type" in data.columns:
        early = data[(data["User Type"].astype(str) == "New") |
                     (pd.to_numeric(data["SIP Tenure (Months)"], errors="coerce") <= 1)]
        if len(early) >= 3:
            subset = early
            scope_bits.append("investors starting out (first SIP or first instalment)")
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
        bands = ten_all.map(ec.tenure_band).value_counts()
        out.append(f"**Short answer:** only **{len(ten_all)}** of these {n} records say how long the SIP ran. Among them "
                   f"the most common window is **{bands.index[0]}** ({bands.iloc[0]} records, median "
                   f"{int(ten_all.median())} months). The leading driver in this view is **{top_theme}**."
                   + (" Treat timing as indicative: the sample is small." if len(ten_all) < 20 else ""))
    elif ctx["overall"]:
        out.append(f"**Short answer:** across all **{total}** records, the leading driver of SIP drop-off is "
                   f"**{top_theme}** ({_pct(theme_counts.iloc[0], total)} of records), followed by "
                   + ", ".join(f"**{t}** ({_pct(c, total)})" for t, c in theme_counts.iloc[1:3].items()) + ".")
    else:
        out.append(f"**Short answer:** **{n}** of {total} records ({_pct(n, total)}) relate to "
                   f"**{', '.join(ctx['scope'])}**. The dominant driver here is **{top_theme}**, and "
                   f"{_pct(addressable, n)} of these cases look addressable through product fixes, nudges or education.")
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


def _data_tiein(themes: list[str], q: str, data: pd.DataFrame, n_quotes: int = 2) -> list[str]:
    """Short 'what the review data shows' block for a set of related drivers."""
    related = [t for t in themes if t in set(data["Problem Theme"])]
    if not related:
        return []
    sub = data[data["Problem Theme"].isin(related)]
    n, total = len(sub), len(data)
    lines = [f"**What the review data shows:** {n} of {total} reviews ({_pct(n, total)}) fall under "
             f"{' / '.join(related)}."]
    emo = sub["Investor Emotional Trigger"].value_counts()
    beh = sub["Problem Behaviour"].value_counts()
    if len(emo):
        lines.append(f"- Dominant emotion: **{emo.index[0]}** ({_pct(int(emo.iloc[0]), n)})")
    if len(beh):
        lines.append("- What investors did: " + ", ".join(f"{b} {_pct(int(c), n)}" for b, c in beh.head(3).items()))
    rc = sub["Problem Root Cause"].value_counts()
    if len(rc):
        lines.append(f"- Most common root cause: {_esc(rc.index[0])}")
    out = ["\n".join(lines)]
    ctx = {"subset": sub, "tokens": [w for w in re.findall(r"[a-z]{4,}", q) if w not in STOPWORDS],
           "themes": related, "overall": False}
    quotes = _quotes(ctx, n_quotes)
    if quotes:
        out += quotes
    return out


def _concept_answer(topics: list[dict], q: str, data: pd.DataFrame) -> list[str]:
    """Structured domain answer: explanation & trade-offs → behaviour insight → Groww best practice."""
    main = topics[0]
    neutral = topics[1] if len(topics) > 1 and topics[1]["topic"].startswith("Neutral factors") else None
    related = topics[1] if len(topics) > 1 and not neutral else None

    out = [f"### {main['topic']}",
           f"**Short answer:** {main['short']}",
           "**Direct explanation & trade-offs**\n" + "\n".join(f"- {p}" for p in main["points"])]
    if related:
        out.append(f"**Related: {related['topic']}**\n- {related['short']}")
    out.append(f"**Investor behaviour insight**\n\n{main['behaviour']}")
    out += _data_tiein(main["themes"], q, data)
    out.append("**Suggested Groww in-app best practice**\n" + "\n".join(f"- {p}" for p in main["practice"]))
    if neutral:
        out.append("**Neutral factors to weigh**\n" + "\n".join(f"- {p}" for p in neutral["points"]))
    return out


# Behavioural-economics lenses for questions outside the mapped topics
BEHAVIOUR_LENSES = [
    ("Loss aversion", ["loss", "lose", "losing", "down", "negative", "risk", "safe", "crash", "fall"],
     "losses weigh roughly twice as much as equal gains, so any visible drop pushes people to act."),
    ("Present bias", ["now", "today", "need", "money", "cash", "spend", "afford", "quick", "fast", "soon"],
     "near-term needs and rewards outweigh distant goals, so long-horizon investing loses out when money is tight."),
    ("Choice overload", ["which", "best", "better", "than", "options", "many", "compare", "choose", "select",
                         "vs", "or "],
     "too many similar options lead to delay, second-guessing or abandoning the decision altogether."),
    ("Anchoring", ["fd", "fixed", "interest", "percent", "%", "return", "expect", "rate"],
     "people judge outcomes against a familiar reference point (often FD rates), not against what the "
     "product is designed to do."),
    ("Herding & social proof", ["friend", "everyone", "people", "trend", "popular", "influencer", "youtube",
                                "reddit", "others"],
     "what peers and creators are doing overrides the investor's own plan, especially when uncertain."),
    ("Status-quo & default effect", ["auto", "default", "continue", "keep", "habit", "start", "stop", "change"],
     "whatever happens by default tends to persist. That is why automated SIPs stick, and why restarting "
     "after a stop is rare."),
    ("Mental accounting", ["emergency", "bonus", "salary", "budget", "bucket", "separate", "save", "saving"],
     "money is mentally labelled by source or purpose, which decides whether investments get raided."),
]


GENERAL_FINANCE_TERMS = set("""inflation gold silver stock stocks share shares equity debt bond bonds crypto bitcoin
insurance loan loans emi interest rate rates fd rd ppf epf nps elss etf etfs index nifty sensex portfolio wealth
money saving savings retirement pension budget expense income salary dividend nav amc sebi amfi kyc demat broker
broking trading invest investing investment investments investor fund funds mutual sip sips lumpsum returns risk
volatility recession economy rupee dollar real estate property house car goal goals plan planning tax""".split())


def _looks_meaningful(q: str, data: pd.DataFrame) -> bool:
    """Cheap guard against keyboard-mash input: at least one recognisable word must appear."""
    words = set(re.findall(r"[a-z]{3,}", q))
    if not words:
        return False
    if words & GENERAL_FINANCE_TERMS or words & STOPWORDS:
        return True
    kb_vocab = {w for kb in KNOWLEDGE_BASE for k in kb["keys"] for w in re.findall(r"[a-z]{3,}", k)}
    if words & kb_vocab:
        return True
    sample = " ".join(data["Original Comment"].astype(str).head(300)).lower()
    return any(w in sample for w in words if len(w) >= 4)


def _fallback_answer(question: str, data: pd.DataFrame) -> list[str]:
    """Analytical response for questions outside mapped topics: framed through investor retention and
    behavioural economics, with whatever the review data can contribute."""
    q = question.lower()
    subject = re.sub(r"\s+", " ", re.sub(r"[?!.]+$", "", question.strip()))[:90]
    lenses = [(name, why) for name, keys, why in BEHAVIOUR_LENSES if _match_any(q, keys)][:3]
    if len(lenses) < 2:
        defaults = [x for x in BEHAVIOUR_LENSES if x[0] in ("Present bias", "Loss aversion",
                                                            "Status-quo & default effect")]
        lenses += [(n, w) for n, _, w in defaults if n not in {l[0] for l in lenses}][: 3 - len(lenses)]

    out = [f"### Analytical view: “{_esc(subject)}”",
           "_This isn't one of the pre-mapped SIP topics, so here's how it plays out for SIP retention and "
           "investor behaviour, plus anything the review data adds._",
           "**Direct explanation & trade-offs**\n"
           "- Almost every SIP decision balances three tensions: **liquidity now vs growth later**, **certainty vs "
           "market-linked outcomes**, and **simplicity vs control**.\n"
           "- Choices that keep money invested consistently tend to preserve the long-term outcome. Choices that "
           "react to short-term events tend to interrupt compounding.\n"
           "- The right answer for any individual depends on their goal, horizon and cash-flow, so neutral "
           "information beats a single prescription."]
    out.append("**Investor behaviour insight**\n" + "\n".join(f"- **{n}:** {w[0].upper() + w[1:]}" for n, w in lenses))

    # What the reviews contribute: keyword overlap first, then nearest driver, then the overall mix
    tokens = [w for w in re.findall(r"[a-z]{4,}", q) if w not in STOPWORDS]
    hits = data[data["Original Comment"].str.contains("|".join(map(re.escape, tokens)), case=False, na=False)] \
        if tokens else data.iloc[0:0]
    if len(hits):
        top = hits["Problem Theme"].value_counts()
        out.append(f"**What the review data shows:** {len(hits)} of {len(data)} reviews ({_pct(len(hits), len(data))}) "
                   f"mention {', '.join(f'“{t}”' for t in tokens[:3])}. Most of them fall under "
                   f"**{top.index[0]}** ({_pct(int(top.iloc[0]), len(hits))}).")
        out += _quotes({"subset": hits, "tokens": tokens, "themes": [top.index[0]], "overall": False}, 2)
    else:
        top = data["Problem Theme"].value_counts().head(3)
        out.append("**What the review data shows:** no reviews mention this directly. The biggest drop-off "
                   "drivers overall are " + ", ".join(f"**{t}** ({_pct(int(c), len(data))})" for t, c in top.items())
                   + ". Ask about any of these for counts and quotes.")
    out.append("**Suggested Groww in-app best practice**\n"
               "- Put neutral, contextual education at the decision moment, not in a separate help page.\n"
               "- Make the habit-preserving option (pause, step-down, date change) as easy as the exit, with "
               "cancel still one tap away.\n"
               "- Measure the effect on M12 SIP retention by activation cohort, separating voluntary and "
               "involuntary churn.")
    return out


def _solution_answer(data: pd.DataFrame, ctx: dict) -> list[str]:
    total = len(data)
    base = ctx["subset"] if ctx["themes"] else data
    counts = base["Problem Theme"].value_counts().head(4)
    out = ["**Short answer:** focus on the drivers that are both large and addressable. "
           f"Across all {total} records, {_pct(int(data['Churn Preventability'].isin(ec.ADDRESSABLE_LEVELS).sum()), total)} "
           "of drop-offs look addressable through product fixes, nudges or education."]
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
    'data', 'knowledge', 'solution', 'advice' or 'analysis'. Works with no network or API key."""
    q = question.lower().strip()
    total = len(data)
    if not q:
        return "Ask me anything about SIP drop-offs, for example “Why do people pause their SIPs?”", "analysis"
    if total == 0:
        return "No records are loaded yet. Load the benchmark corpus or run live ingestion first.", "analysis"

    out: list[str] = []
    topics = _match_topics(q)
    ctx = _retrieve(q, data)
    is_advice = bool(ADVICE_PATTERN.search(q))
    data_question = bool(DATA_CUE_PATTERN.search(q)) or ctx["asks_when"]
    # Mapped investor topics answer from the domain engine unless the question is about reviews/complaints.
    is_concept = bool(topics) and not data_question

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

    has_review_signal = data_question or bool(ctx["themes"]) or not ctx["overall"]
    if not has_review_signal and not _looks_meaningful(q, data):
        return ("I'm not sure what you mean. Try one of the suggested questions above, or ask about SIP "
                "cancellations, auto-debit failures, returns, market falls, taxes or SIP vs lumpsum."), "analysis"
    if not has_review_signal:
        # Not a mapped topic and nothing review-specific: analytical behavioural-economics response.
        return "\n\n".join(_fallback_answer(question, data)), "analysis"
    out += _data_answer(ctx, total)
    if topics:
        t = topics[0]
        out.append(f"**Background: {t['topic']}**\n- {t['short']}\n" + "\n".join(f"- {p}" for p in t["points"][:2]))
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


LLM_SYSTEM_PROMPT = """You are the financial AI assistant for the Groww SIP Discovery Engine.
Your capabilities:
1. Conceptual, Educational & General Financial Guidance:
   If the user asks any question about mutual funds, SIPs, personal finance, investing, markets, or wealth-building—including but not limited to:
   - Mutual funds, index funds, active vs passive funds, direct vs regular plans, equity vs debt vs hybrid funds, ELSS, liquid funds, ETFs.
   - SIP (Systematic Investment Plan), lumpsum, STP (Systematic Transfer Plan), SWP (Systematic Withdrawal Plan), step-up SIP, top-up SIP, auto-pay, OTM, e-mandates.
   - Financial concepts: NAV, expense ratio, AUM, CAGR, XIRR, rolling returns, benchmark indices (Nifty, Sensex), tracking error, portfolio diversification, asset allocation, compound interest, power of compounding, time horizon, risk profile, inflation, rupee cost averaging.
   - Regulations & Taxation: SEBI, AMFI, AMC, registrar (CAMS/KFintech), LTCG, STCG, capital gains tax, exit load, lock-in period, KYC, nomination.
   - General investor behaviour: market volatility, bull/bear markets, corrections, holding period, fear & greed, emotional investing, financial planning, goal-based investing:
   -> Answer directly, clearly, objectively, and educationally like an expert financial educator.
   -> Explain the core concept, mechanics, benefits, and practical considerations.
   -> Do NOT force or restrict the answer into SIP churn, drop-off buckets, or review dataset numbers when not explicitly requested.

2. Retention, Churn & Discovery Engine Analytics:
   If the user specifically asks about SIP drop-offs, churn, cancellations, pauses, investor friction, bank auto-debit failures, app drop-off triggers, user reviews, or Groww product fixes:
   -> Ground your answer firmly in the provided DATA CONTEXT metrics, counts, and percentages.
   -> Provide actionable behavioural insights and product intervention recommendations.

Rules:
- Never recommend specific AMC schemes, fund names, or individual stocks.
- Never give direct investment advice (e.g., do not say "you must buy this").
- Keep answers structured with crisp markdown, clean sub-headings, and clear bullet points."""
def _llm_call(question: str, context: str, model: str = 'gemini-3.5-flash-lite') -> str | None:
    key = _secret('GEMINI_API_KEY') or _secret('ANTHROPIC_API_KEY')
    if not key:
        return None
    try:
        if _secret('GEMINI_API_KEY'):
            text, status = gemini_generate(LLM_SYSTEM_PROMPT, f"Question: {question}\n\nContext:\n{context[:4000]}")
            st.session_state["llm_status"] = status
            return text

        # Anthropic fallback
        r = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={'x-api-key': key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'},
            json={'model': model, 'max_tokens': 600, 'system': LLM_SYSTEM_PROMPT, 'messages': [{'role': 'user', 'content': f'QUESTION: {question}\n\n{context}'}]},
            timeout=30
        )
        if r.ok:
            return ''.join(b.get('text', '') for b in r.json().get('content', []) if b.get('type') == 'text').strip() or None
        return None
    except Exception as e:
        st.session_state["llm_status"] = {"ok": False, "model": None, "reason": f"{type(e).__name__}: {e}"[:160]}
        return None


GEMINI_DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]
GEMINI_LABELS = {"gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite", "gemini-3.8-flash": "Gemini 3.8 Flash"}


def _gemini_models() -> list[str]:
    custom = _secret("GEMINI_MODEL")
    return [m.strip() for m in custom.split(",") if m.strip()] if custom else GEMINI_DEFAULT_MODELS


def _gemini_reason(r) -> str:
    try:
        msg = r.json().get("error", {}).get("message", "")
    except Exception:
        msg = ""
    hint = {400: "bad request", 401: "API key rejected", 403: "API key not allowed for this project",
            404: "model not found", 429: "rate limit or quota reached", 500: "Gemini server error",
            503: "Gemini overloaded"}.get(r.status_code, "error")
    return f"HTTP {r.status_code} ({hint}){': ' + msg[:120] if msg else ''}"


def gemini_generate(system: str, user: str, json_mode: bool = False, deadline_s: float = 30.0,
                    max_tokens: int = 2048) -> tuple[str | None, dict]:
    """Calls Gemini with retries and an overall deadline. Never raises; returns (text or None, status)."""
    import time
    key = _secret("GEMINI_API_KEY")
    if not key:
        return None, {"ok": False, "model": None, "reason": "GEMINI_API_KEY not set"}
    start, reasons = time.monotonic(), []
    for model in _gemini_models():
        gen_cfg = {"maxOutputTokens": max_tokens}
        if "lite" not in model:
            gen_cfg["thinkingConfig"] = {"thinkingLevel": "low"}  # keeps latency down on Flash
        if json_mode:
            gen_cfg["responseMimeType"] = "application/json"
        payload = {"system_instruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user", "parts": [{"text": user}]}],
                   "generationConfig": gen_cfg}
        for attempt in range(2):
            remaining = deadline_s - (time.monotonic() - start)
            if remaining < 3:
                reasons.append("time budget used up")
                return None, {"ok": False, "model": None, "reason": "; ".join(reasons)}
            try:
                r = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    headers={"Content-Type": "application/json", "x-goog-api-key": key},
                    json=payload, timeout=(5, min(20.0, remaining)))
            except requests.exceptions.Timeout:
                reasons.append(f"{model}: timed out")
                break
            except requests.exceptions.RequestException as exc:
                reasons.append(f"{model}: {type(exc).__name__}")
                break
            if r.status_code == 200:
                try:
                    cand = (r.json().get("candidates") or [{}])[0]
                    parts = cand.get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                except Exception:
                    text, cand = "", {}
                if text:
                    return text, {"ok": True, "model": model, "reason": ""}
                reasons.append(f"{model}: empty reply ({cand.get('finishReason', 'no text')})")
                break
            if r.status_code == 400 and "thinkingConfig" in gen_cfg:
                gen_cfg.pop("thinkingConfig")  # older model versions reject thinking settings
                continue
            reasons.append(f"{model}: {_gemini_reason(r)}")
            if r.status_code in (429, 500, 503) and attempt == 0:
                time.sleep(1.5)
                continue
            break
    return None, {"ok": False, "model": None, "reason": "; ".join(reasons) or "no models configured"}


_TRANSLATE_SYSTEM = (
    "You translate Indian investor comments into natural, standard English. Inputs may be Hindi, Hinglish, Telugu, "
    "Tamil, Marathi or another Indian language, in native script or typed in English letters, often mixed with "
    "English. Fix obvious typos without changing what the user meant. Keep amounts, app, bank and fund names. "
    "Do not add commentary.")


class _TranslationUnavailable(Exception):
    """Raised inside cached translators so a failed call is never cached; callers treat it as 'no translation'."""


@st.cache_data(show_spinner=False, ttl=86400)
def _translate_batch_cached(texts: tuple) -> dict:
    import json
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    text, status = gemini_generate(
        _TRANSLATE_SYSTEM,
        f"Translate each numbered comment. Reply with a JSON array of {len(texts)} strings in the same order.\n\n{numbered}",
        json_mode=True, deadline_s=20.0, max_tokens=2048)
    if not text:
        raise _TranslationUnavailable(status.get("reason", ""))
    try:
        arr = json.loads(text)
    except Exception as exc:
        raise _TranslationUnavailable("unreadable JSON") from exc
    if isinstance(arr, dict):  # tolerate {"translations": [...]}
        arr = next((v for v in arr.values() if isinstance(v, list)), [])
    out = {t: _clean_translation(e) for t, e in zip(texts, arr if isinstance(arr, list) else [])}
    return {t: e for t, e in out.items() if e}


@st.cache_data(show_spinner=False, ttl=86400)
def _translate_one_cached(original_text: str) -> str:
    text, status = gemini_generate(
        _TRANSLATE_SYSTEM,
        f"Translate this comment. Reply with the English translation only.\n\nText: {original_text}",
        deadline_s=8.0, max_tokens=512)
    cleaned = _clean_translation(text)
    if not cleaned:
        raise _TranslationUnavailable(status.get("reason", ""))
    return cleaned


def _clean_translation(value) -> str:
    """Strips wrapping quotes, labels and whitespace from a model translation."""
    quotes = '"\u201c\u201d\''
    t = str(value or "").strip().strip(quotes).strip()
    t = re.sub(r"^(english( translation)?|translation)\s*:\s*", "", t, flags=re.I)
    return t.strip().strip(quotes).strip()


_TRANSLATE_COOLDOWN_S = 300  # after a failed call, skip translation for 5 minutes so reruns stay fast


def _translation_paused() -> bool:
    import time
    return time.time() < st.session_state.get("_translate_paused_until", 0)


def _pause_translation() -> None:
    import time
    st.session_state["_translate_paused_until"] = time.time() + _TRANSLATE_COOLDOWN_S


def translate_to_english(texts: tuple) -> dict:
    """Batch-translate non-English comments with Gemini. Returns {original: English}; {} if unavailable."""
    if not texts or not _secret("GEMINI_API_KEY") or _translation_paused():
        return {}
    try:
        return _translate_batch_cached(tuple(texts))
    except Exception:
        _pause_translation()
        return {}


def translate_comment_ai(original_text: str) -> str | None:
    """Single-comment translation, used when the batch call missed a comment. Cached; never raises."""
    if not original_text or not _secret("GEMINI_API_KEY") or _translation_paused():
        return None
    try:
        return _translate_one_cached(str(original_text))
    except Exception:
        _pause_translation()
        return None


def translate_comments(texts: tuple, max_single: int = 3) -> dict:
    """Batch first, then at most `max_single` one-by-one retries for anything the batch missed.
    Stops retrying after the first failure so a slow or down API can't stall the page."""
    out = translate_to_english(texts)
    for t in [t for t in texts if t not in out][:max_single]:
        one = translate_comment_ai(t)
        if not one:
            break
        out[t] = one
    return out


def llm_enabled() -> bool:
    return bool(_secret('GEMINI_API_KEY') or _secret('ANTHROPIC_API_KEY'))

def respond(question: str, data: pd.DataFrame) -> tuple[str, str]:
    """Final answer = (optional LLM synthesis or built-in answer) + optional Wikipedia context.
    Returns (markdown, source label)."""
    builtin, kind = answer_question(question, data)
    q = question.strip()
    if not q or data.empty:
        return builtin, "Built-in"

    web = web_lookup(q) if kind == "analysis" else None

    if llm_enabled():
        quotes = [ln for ln in builtin.split("\n\n") if ln.startswith(">")]
        facts = "\n\n".join(ln for ln in builtin.split("\n\n") if not ln.startswith(">"))
        context = (f"DATA CONTEXT (computed from {len(data)} public investor reviews; question type: {kind}):\n{facts}"
                   + (f"\n\nWEB CONTEXT (Wikipedia: {web['title']}):\n{web['extract']}" if web else ""))
        model_name = "gemini-3.5-flash-lite" if _secret('GEMINI_API_KEY') else _secret("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        llm = _llm_call(q, context, model_name)
        if llm:
            tail = ""
            if quotes and kind != "advice":
                tail = "\n\n**What investors actually said**\n\n" + "\n\n".join(quotes[:3])
            if web:
                tail += f"\n\n[Background: {web['title']} on Wikipedia ↗]({web['url']})"
            _st = st.session_state.get("llm_status", {})
            provider_label = (f"Google {GEMINI_LABELS.get(_st.get('model'), _st.get('model') or 'Gemini')}"
                              if _secret('GEMINI_API_KEY') else "Claude")
            return llm + tail, f"{provider_label} + discovery data" + (" + Wikipedia" if web else "")

    source = {"data": "Review data synthesis", "solution": "Review data + product playbook",
              "knowledge": "SIP domain engine + review data", "advice": "SIP domain engine (neutral)",
              "analysis": "Behavioural analysis + review data"}[kind]
    if web:
        builtin += f"\n\n**Reference:** [{web['title']} on Wikipedia ↗]({web['url']}): {_esc(_excerpt(web['extract'], 260))}"
        source += " + Wikipedia"
    if llm_enabled() and _secret('GEMINI_API_KEY'):
        _reason = st.session_state.get("llm_status", {}).get("reason", "")
        source += f" · built-in engine answered (Gemini unavailable: {_reason})" if _reason else ""
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

df_master, master_info = load_master_dataset()
df_golden, golden_file, golden_raw = load_golden_set()

c1, c2, c3, c4 = st.columns(4)
_groww_n = int((df_master["Product"] == "Groww").sum())
_bp = master_info.get("by_platform", {})
c1.markdown(card("Total Reviews Extracted", f"{master_info['extracted']:,}",
                 " · ".join(f"{int(_bp.get(p, 0)):,} {p}" for p in PLATFORMS), True), unsafe_allow_html=True)
c2.markdown(card("Product / Platform", f"{_groww_n:,} | {len(df_master) - _groww_n:,}", "Groww | Other"),
            unsafe_allow_html=True)
c3.markdown(card("Validated Insights Dataset", f"{0 if df_golden is None else len(df_golden):,}",
                 "Deep-dive validated cancellation cases"), unsafe_allow_html=True)
_years = df_master["Year"].dropna()
c4.markdown(card("Longitudinal Span", f"{int(_years.min())} – {int(_years.max())}" if len(_years) else "—",
                 f"Coverage across {len(_years):,} dated reviews" if len(_years) else "No Year values found"),
            unsafe_allow_html=True)
st.markdown('<div class="scope-badge">🌐 Ingested &amp; analyzed user sentiment across English, Hindi (हिन्दी), '
            'and Hinglish</div>', unsafe_allow_html=True)
if master_info["missing"] or master_info["errors"]:
    st.warning("Some source files could not be loaded: "
               + ", ".join(master_info["missing"] + master_info["errors"])
               + ". Place them next to app.py; KPIs and exports use the files that loaded.")

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
        if llm_enabled():
            _models = " → ".join(GEMINI_LABELS.get(m, m) for m in _gemini_models()) if _secret("GEMINI_API_KEY") else "Claude"
            st.caption(f"🟢 **AI Enabled:** Google {_models} (Live Synthesis)" if _secret("GEMINI_API_KEY")
                       else "🟢 **AI Enabled:** Claude (Live Synthesis)")
        st.session_state["chat_history"] = _normalize_history(st.session_state.get("chat_history", []))

        with _keyed_container("suggest_chips"):
            pill_cols = st.columns(len(SUGGESTED_QUESTIONS))
            for i, sq in enumerate(SUGGESTED_QUESTIONS):
                pill_cols[i].button(sq, key=f"suggest_{i}", use_container_width=True,
                                    on_click=_queue_question, args=(sq,))

        with st.form("ask_form", clear_on_submit=True, border=False) if _form_supports_border() else \
                st.form("ask_form", clear_on_submit=True):
            in_col, btn_col = st.columns([8, 1])
            typed = in_col.text_input(
                "Your question", label_visibility="collapsed", key="ask_text",
                placeholder="Ask a question about SIP drop-offs, cancellations, investor friction, or product fixes...")
            with btn_col:
                with _keyed_container("ask_btn"):
                    asked = st.form_submit_button("✨ Ask AI", type="primary", use_container_width=True)

        question = (typed.strip() if asked and typed and typed.strip() else None) \
            or st.session_state.pop("pending_question", None)
        if question:
            with st.spinner("Analysing reviews and SIP knowledge…"):
                try:
                    reply, src = respond(question, data)
                except Exception as exc:  # never surface a stack trace to evaluators
                    reply, src = (f"I couldn't analyse that question ({type(exc).__name__}: {exc}). Try rephrasing, "
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


def _queue_question(q: str) -> None:
    """Chip click: put the question in the input box and queue it for the engine on this run."""
    st.session_state["pending_question"] = q
    st.session_state["ask_text"] = q


def _keyed_container(key: str):
    """st.container(key=...) adds a .st-key-<key> CSS class (Streamlit >= 1.39); plain container otherwise."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


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
    if "Recommended Groww Intervention" not in bench_df.columns:
        bench_df["Recommended Groww Intervention"] = "Proactive communication and guidance"
    bench_label = (f"Benchmark Corpus ({len(bench_df)} Real Public Records)" if bench_is_real
                   else "Demo Corpus (synthetic — real benchmark file missing)")
    mode = "Benchmark Corpus"
    live_mode = False

    if live_mode:
        st.markdown("**Google Play Store**")
        include_play = st.checkbox("Include Play Store reviews", value=True, key="inc_play")
        apps_text = st.text_area(
            "Apps (Name = package id, one per line)",
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
    if bench_is_real and bench_meta.get("source", "").startswith("golden_set"):
        provenance = (f"AI golden benchmark ({bench_meta['source']}): {len(df_all)} hand-labelled public reviews and "
                      "comments. Every Source URL opens the original review, thread or video.")
        st.markdown(
            '<div class="notice" style="background:#E9FBF5;border-color:#9BE6CF;color:#0B5A45">'
            'Every “Open source ↗” link opens the original review, thread or video. Tenure is shown only where '
            'the investor stated it.</div>', unsafe_allow_html=True)
    elif bench_is_real:
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
chat_data = df_all  # Ask AI always answers from the validated set (it relies on its root-cause labels)
st.markdown(f"<h3 style='color:{NAVY};margin-bottom:4px'>Discovery Filters</h3>", unsafe_allow_html=True)
view_mode = st.radio("View Mode", [VIEW_FULL, VIEW_DEEP], index=0, horizontal=True, key="view_mode")
if view_mode == VIEW_FULL and len(df_master):
    df_all = build_full_view(df_master, df_golden)
    st.caption("Full view: validated rows keep their labels. Other rows are auto-tagged from keywords (theme, emotion and "
               "preventability are inferred, not validated); see the Label Source column. Tenure is only shown where stated.")
df_all = df_all.copy()
# Columns actually present in the dataset drive what the table shows; any schema
# column that is missing is added as empty so metrics, charts and export never crash.
display_cols = [c for c in ec.COLUMNS if c in df_all.columns] + [c for c in ("Year", "Label Source") if c in df_all.columns]
for col in ec.COLUMNS:
    if col not in df_all.columns:
        df_all[col] = pd.NA
df_all["Source Platform"] = df_all["Source Platform"].map(normalize_platform)
df_all["Churn Preventability"] = df_all["Churn Preventability"].map(ec.normalize_preventability)
df_all["Problem Theme"] = df_all["Problem Theme"].map(lambda v: ec.normalize_theme(v) if pd.notna(v) else v)
df_all["SIP Tenure (Months)"] = pd.to_numeric(df_all["SIP Tenure (Months)"].map(ec.parse_tenure), errors="coerce").astype("Int64")
df_all["Tenure Band"] = df_all["SIP Tenure (Months)"].map(ec.tenure_band)
df_all["Lifecycle Stage"] = df_all["Original Comment"].map(infer_lifecycle_stage)
if "Year" in df_all.columns:
    df_all["Year"] = pd.to_numeric(df_all["Year"], errors="coerce").astype("Int64")
display_cols.append("Lifecycle Stage")

band_order = [b[0] for b in ec.TENURE_BANDS] + ["Not stated"]
present = lambda col, order: [x for x in order if x in set(df_all[col].dropna())]  # noqa: E731

f0, f1, f2, f3, f4, f5 = st.columns(6)
sel_src = f0.multiselect("Source", present("Source Platform", PLATFORMS), placeholder="All sources", key="f_src")
sel_theme = f1.multiselect("Problem Theme", present("Problem Theme", ec.THEMES + ["Unclassified"]),
                           placeholder="All 8 drivers", key="f_theme")
sel_band = f2.multiselect("SIP Tenure", present("Tenure Band", band_order), placeholder="All tenures", key="f_band")
_stage_opts = present("SIP Stage", ec.STAGES + ["Unknown Stage"])
_stage_opts += sorted(str(x) for x in set(df_all["SIP Stage"].dropna()) if x not in _stage_opts)
sel_stage = f3.multiselect("SIP Stage", _stage_opts, placeholder="All stages", key="f_stage")
_prev_opts = present("Churn Preventability", ec.PREVENTABILITY_LEVELS)
_prev_opts += sorted(str(x) for x in set(df_all["Churn Preventability"].dropna()) if str(x) not in _prev_opts)
sel_prev = f4.multiselect("Churn Preventability", _prev_opts,
                          placeholder="All levels", key="f_prev")
_year_opts = (sorted({int(y) for y in df_all["Year"].dropna()}, reverse=True) if "Year" in df_all.columns else [])
sel_year = f5.multiselect("Year", _year_opts, placeholder="All years", key="f_year")

render_chat(chat_data)

df = df_all
if sel_src:
    df = df[df["Source Platform"].isin(sel_src)]
if sel_theme:
    df = df[df["Problem Theme"].isin(sel_theme)]
if sel_band:
    df = df[df["Tenure Band"].isin(sel_band)]
if sel_stage:
    df = df[df["SIP Stage"].isin(sel_stage)]
if sel_prev:
    df = df[df["Churn Preventability"].isin(sel_prev)]
if sel_year:
    df = df[df["Year"].isin(sel_year)]


def render_evidence(view: pd.DataFrame, k: int = 6) -> None:
    """Representative quote cards from the rows currently in view, with AI translations for non-English comments."""
    filtered = bool(sel_src or sel_theme or sel_band or sel_stage or sel_prev or sel_year)
    st.markdown(f"<h4 style='color:{NAVY};margin:18px 0 2px'>Evidence Highlights</h4>", unsafe_allow_html=True)
    st.caption(("Representative comments for the current filters" if filtered else
                "Representative comments from the active view") + f" · {len(view):,} rows in view")
    pool = view.copy()
    pool["_k"] = pool["Original Comment"].map(lambda t: " ".join(str(t).split()).lower())
    pool = pool[pool["_k"].str.len().between(40, 900)].drop_duplicates("_k")
    if pool.empty:
        st.info("No comments to highlight for this selection.")
        return
    val = (pool["Label Source"] == "Validated") if "Label Source" in pool.columns else pd.Series(True, index=pool.index)
    pool["_score"] = (val.astype(int) * 3 + pool["Problem Theme"].isin(ec.THEMES).astype(int)
                      + pool["Product"].eq("Groww").astype(int) * 0.5 + pool["_k"].str.len().clip(upper=400) / 400)
    ranked = pool.sort_values("_score", ascending=False)
    chosen, per_src, per_theme = [], Counter(), Counter()
    for cap in (2, k):  # first pass keeps sources and themes varied; second fills any gaps
        for idx, r in ranked.iterrows():
            if len(chosen) == k:
                break
            if idx in chosen or per_src[r["Source Platform"]] >= cap or per_theme[r["Problem Theme"]] >= cap:
                continue
            chosen.append(idx)
            per_src[r["Source Platform"]] += 1
            per_theme[r["Problem Theme"]] += 1
    picks = ranked.loc[chosen]
    glosses = load_golden_glosses()
    non_english = {c for c in picks["Original Comment"].astype(str) if is_non_english(c)}
    need = tuple(c for c in picks["Original Comment"].astype(str)
                 if c in non_english and _text_key(c, "")[0] not in glosses)
    ai = translate_comments(need) if need else {}
    cols = st.columns(3)
    for i, (_, r) in enumerate(picks.iterrows()):
        comment = str(r["Original Comment"])
        year = "" if pd.isna(r["Year"]) else f" · {int(r['Year'])}"
        url = html.escape(str(r["Source URL"]), quote=True)
        meta = (f"<b>R{i + 1}</b> · {html.escape(str(r['Source Platform']))} · {html.escape(str(r['Product']))}{year} · "
                f"<a href=\"{url}\" target=\"_blank\">Open source ↗</a>")
        trans = (glosses.get(_text_key(comment, "")[0]) or ai.get(comment)) if comment in non_english else None
        trans_html = (f'<div class="ev-trans"><div class="ai-badge"><span class="ai-badge-icon">🌐</span>'
                      f'<span class="ai-badge-text">Translated to English by AI</span></div>'
                      f'<p class="ev-trans-text">{html.escape(trans)}</p></div>') if trans else ""
        tags = " · ".join(html.escape(str(r[c])) for c in ("Problem Theme", "SIP Stage")
                          if pd.notna(r[c]) and str(r[c]) not in ("Not stated", "Unclassified"))
        cols[i % 3].markdown(
            f'<div class="ev-card"><div class="ev-meta">{meta}</div>'
            f'<div class="ev-quote">“{html.escape(_excerpt(comment, 420))}”</div>{trans_html}'
            f'{f"<div class=ev-tags>{tags}</div>" if tags else ""}</div>', unsafe_allow_html=True)
    missing = [c for c in non_english if not (glosses.get(_text_key(c, "")[0]) or ai.get(c))]
    if missing:
        st.caption("English translations for non-English comments appear when the Gemini API key is configured."
                   if not _secret("GEMINI_API_KEY") else
                   "Some non-English comments couldn't be translated right now; the app will try again in a few minutes.")


PLOT_LAYOUT = dict(
    font=dict(family="Inter, Arial, sans-serif", color=NAVY, size=13),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    margin=dict(l=10, r=10, t=60, b=10),
    hoverlabel=dict(bgcolor="#FFFFFF", font_size=13),
)


# =============================================================================
# KPI SHELF — 5 uniform SaaS cards in a responsive CSS grid (Groww emerald accents)
# Rendered at the top of render_dashboard(); replaces the old st.columns(5) metric cards.
# Requires at module level: `import html`, `import pandas as pd`, `import streamlit as st`, `import engine_core as ec`
# =============================================================================
KPI_SHELF_CSS = """
<style>
.kpi-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin: 6px 0 18px 0; }
.kpi-card { height: 140px; box-sizing: border-box; background: #FFFFFF; border: 1px solid #EAECF0; border-radius: 12px;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.05); padding: 14px 16px; display: flex; flex-direction: column;
            transition: all 0.2s ease; position: relative; overflow: hidden; container-type: inline-size; }
.kpi-card::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: #00D09C; opacity: 0; transition: opacity 0.2s ease; }
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(16, 24, 40, 0.08); border-color: #D0F5EA; }
.kpi-card:hover::before { opacity: 1; }
.kpi-head { height: 18px; display: flex; align-items: center; }
.kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #667085;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-metric { height: 42px; display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-top: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; line-height: 1.1; color: #101828; letter-spacing: -0.02em; min-width: 0; }
.kpi-value.text { font-size: 17px; line-height: 1.22; display: -webkit-box; -webkit-line-clamp: 2;
                  -webkit-box-orient: vertical; overflow: hidden; word-break: break-word; }
.kpi-pill { display: inline-block; margin-top: 2px; font-size: 11px; font-weight: 500; color: #475467; background: #F2F4F7;
            border-radius: 999px; padding: 2px 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 100%; }
.kpi-badge { flex-shrink: 0; font-size: 10.5px; font-weight: 600; border-radius: 999px; padding: 2px 8px; white-space: nowrap; }
.kpi-badge.green { color: #027A48; background: #ECFDF3; }
.kpi-badge.emerald { color: #00805F; background: rgba(0, 208, 156, 0.14); }
.kpi-badge.red { color: #B42318; background: #FEF3F2; }
.kpi-badge.amber { color: #B54708; background: #FFFAEB; }
.kpi-badge.grey { color: #475467; background: #F2F4F7; }
.kpi-foot { margin-top: auto; font-size: 12px; color: #667085; line-height: 1.3; display: -webkit-box; -webkit-line-clamp: 2;
            -webkit-box-orient: vertical; overflow: hidden; }
.kpi-pill .short { display: none; }
@container (max-width: 250px) { .kpi-pill .full { display: none; } .kpi-pill .short { display: inline; }
                                .kpi-value { font-size: 26px; } .kpi-badge { font-size: 10px; padding: 2px 6px; } }
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 700px)  { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 460px)  { .kpi-grid { grid-template-columns: minmax(0, 1fr); } }
</style>
"""


def _kpi_card(label: str, value: str, foot: str, badge: tuple[str, str] | None = None,
              pill: str | None = None, text_value: bool = False, pill_short: str | None = None,
              badge_below: bool = False) -> str:
    """One card as a single-line HTML string (no line breaks, so Streamlit's markdown never turns it into a code block)."""
    esc = lambda x: html.escape(str(x), quote=True)  # noqa: E731
    badge_html = f'<span class="kpi-badge {badge[1]}">{esc(badge[0])}</span>' if badge else ""
    below_html = ""
    if badge_html and badge_below:  # long text values need the full row width, so the badge drops below them
        below_html, badge_html = f'<div style="margin-top:4px">{badge_html}</div>', ""
    pill_html = (f'<div><span class="kpi-pill" title="{esc(pill)}"><span class="full">{esc(pill)}</span>'
                 f'<span class="short">{esc(pill_short or pill)}</span></span></div>') if pill else ""
    return (f'<div class="kpi-card">'
            f'<div class="kpi-head"><span class="kpi-label" title="{esc(label)}">{esc(label)}</span></div>'
            f'<div class="kpi-metric"><span class="kpi-value{" text" if text_value else ""}" title="{esc(value)}">{esc(value)}</span>{badge_html}</div>'
            f'{below_html}{pill_html}'
            f'<div class="kpi-foot" title="{esc(foot)}">{esc(foot)}</div>'
            f'</div>')


def render_kpi_shelf(view: pd.DataFrame, summary: dict) -> None:
    """Five uniform KPI cards for the rows currently in view (`view` = filtered df, `summary` = ec.summarize(view))."""
    n = int(summary.get("n", len(view)))

    # Card 1 · Records in view, with a per-source breakdown pill
    src = view["Source Platform"].astype(str).value_counts() if "Source Platform" in view else pd.Series(dtype=int)
    breakdown = " · ".join(f"{int(src.get(p, 0)):,} {p}" for p in ("Play Store", "Reddit", "YouTube"))
    breakdown_short = " · ".join(f"{int(src.get(p, 0)):,} {s}" for p, s in
                                 (("Play Store", "Play"), ("Reddit", "Reddit"), ("YouTube", "YT")))

    # Card 2 · Critical SIP stage: the lifecycle step with the most friction (inferred for every row)
    stage_counts = (view["Lifecycle Stage"].value_counts() if "Lifecycle Stage" in view else pd.Series(dtype=int))
    stage_counts = stage_counts[stage_counts.index != LIFECYCLE_OTHER]
    if stage_counts.sum():
        top_stage = str(stage_counts.index[0])
        stage_share = round(100 * stage_counts.iloc[0] / max(len(view), 1))
    else:
        top_stage, stage_share = "—", 0
    tenure_count = int(summary.get("known_tenure", 0))

    # Card 3 · Addressable friction: High or Medium preventability among assessed rows
    prev = view["Churn Preventability"].map(ec.normalize_preventability) if "Churn Preventability" in view else pd.Series(dtype=str)
    assessed = prev[prev.isin(ec.PREVENTABILITY_LEVELS)]
    addressable_pct = round(100 * assessed.map(ec.is_addressable).mean()) if len(assessed) else 0

    # Card 4 · Top churn driver, as a share of rows that carry a problem theme
    themed = view["Problem Theme"][view["Problem Theme"].isin(ec.THEMES)] if "Problem Theme" in view else pd.Series(dtype=str)
    if len(themed):
        top_counts = themed.value_counts()
        top_driver, top_share = str(top_counts.index[0]), round(100 * top_counts.iloc[0] / len(themed))
        top_foot = f"{top_share}% of total friction volume"
    else:
        top_driver, top_foot = "—", "No themed rows in view"

    cards = [
        _kpi_card("Records in view", f"{n:,}", "Total user sentiment extracted", pill=breakdown,
                  pill_short=breakdown_short),
        _kpi_card("Critical SIP Stage", top_stage, "Primary lifecycle drop-off point",
                  badge=(f"{stage_share}% of friction", "emerald") if stage_share else None,
                  text_value=True, badge_below=True),
        _kpi_card("Addressable Friction", f"{addressable_pct}%" if len(assessed) else "—",
                  "High & Medium preventability cases", badge=("High UX Fix", "emerald")),
        _kpi_card("Top Churn Driver", top_driver, top_foot, text_value=True),
        _kpi_card("Cohort Tenures", f"{tenure_count:,}", "Available for cohort cliff curve",
                  badge=("Stated Duration", "emerald")),
    ]
    st.markdown(KPI_SHELF_CSS + '<div class="kpi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def _life_readout(view: pd.DataFrame) -> str:
    """Opening clause of the dashboard read-out: the lifecycle stage with the most friction."""
    counts = view["Lifecycle Stage"].value_counts() if "Lifecycle Stage" in view else pd.Series(dtype=int)
    counts = counts[counts.index != LIFECYCLE_OTHER]
    if not counts.sum():
        return ""
    return (f"**{counts.index[0]}** is where most friction happens ({counts.iloc[0] / max(len(view), 1):.0%} of "
            f"records in view); ")


def render_dashboard(df: pd.DataFrame, summary: dict) -> None:
    render_kpi_shelf(df, summary)
    st.write("")

    render_evidence(df)
    tab_dash, tab_table, tab_export = st.tabs(["📊 Evaluator Dashboard", "🔎 Discovery Table", "⬇️ Export Workbook"])

    with tab_dash:
        # Chart 1 — friction by SIP lifecycle stage (inferred for every row, so it is never sparse)
        life_order = LIFECYCLE_STAGES + [LIFECYCLE_OTHER]
        life_counts = df["Lifecycle Stage"].value_counts().reindex(life_order, fill_value=0)
        if int(life_counts.sum()) == 0:
            st.info("No records in view, so the lifecycle chart cannot be drawn.")
        else:
            top_life = life_counts[LIFECYCLE_STAGES].idxmax()
            life_share = life_counts / max(int(life_counts.sum()), 1)
            life_driver = []
            for stg in life_order:
                sub = df.loc[(df["Lifecycle Stage"] == stg) & df["Problem Theme"].isin(ec.THEMES), "Problem Theme"]
                life_driver.append(sub.value_counts().index[0] if len(sub) else "—")
            plot_order = life_order[::-1]  # Plotly draws horizontal bars bottom-up; keep journey order top-down
            idx = [life_order.index(x) for x in plot_order]
            bar_colors = [EMERALD if stg == top_life else ("#CBD5E1" if stg == LIFECYCLE_OTHER else "#0F172A")
                          for stg in plot_order]
            fig1 = go.Figure(go.Bar(
                y=plot_order, x=[int(life_counts[x]) for x in plot_order], orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f"{int(life_counts[x]):,} · {life_share[x]:.0%}" for x in plot_order], textposition="outside",
                cliponaxis=False,
                customdata=[[f"{life_share[x]:.1%}", life_driver[k]] for x, k in zip(plot_order, idx)],
                hovertemplate="<b>%{y}</b><br>%{x:,} records (%{customdata[0]} of view)"
                              "<br>Top driver here: %{customdata[1]}<extra></extra>"))
            fig1.update_layout(**PLOT_LAYOUT, height=380, bargap=0.35,
                               title=dict(text="<b>Chart 1 · SIP Friction by Lifecycle Stage</b><br>"
                                               "<span style='font-size:12px;color:#5A6B82'>Where in the SIP journey investors "
                                               "hit friction · stage inferred from each comment · hover for the top driver</span>"),
                               xaxis=dict(title="Records", gridcolor="#EEF1F5", range=[0, life_counts.max() * 1.22]),
                               yaxis=dict(title="", tickfont=dict(size=13, color="#0F172A")))
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
        _assessed_share = (df["Churn Preventability"].map(ec.is_addressable).sum()
                           / max(int(prev_counts.sum()), 1))
        with p_left:
            if prev_counts.empty:
                st.info("No rows in view have a preventability assessment yet.")
            else:
                fig3 = go.Figure(go.Pie(
                    labels=prev_counts.index, values=prev_counts.values, hole=0.62, sort=False,
                    marker=dict(colors=[PREVENT_COLORS[l] for l in prev_counts.index], line=dict(color="#FFFFFF", width=2)),
                    textinfo="percent", hovertemplate="%{label}<br>%{value} records (%{percent})<extra></extra>"))
                fig3.add_annotation(text=f"<b>{_assessed_share:.0%}</b><br><span style='font-size:12px'>addressable</span>",
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
                y=emo.index, x=emo.values, orientation="h", marker_color=["#00D09C", "#1E293B", "#4A6FA5", "#E5A93C", "#D9534F", "#6C7A89", "#66C2A5", "#8A2BE2"],
                hovertemplate="%{y}: %{x} records<extra></extra>"))
            fig6.update_layout(**PLOT_LAYOUT, height=420,
                               title=dict(text="<b>Chart 5 · Investor emotional triggers</b><br><span style='font-size:12px;color:#5A6B82'>"
                                               "Most frequent emotions expressed by churners</span>"),
                               xaxis=dict(title="Records", gridcolor="#EEF1F5"), yaxis=dict(title=""))
            stretch(st.plotly_chart, fig6, config={"displayModeBar": False})

        verdict = "above" if _assessed_share > 0.75 else "below"  # same basis as the KPI card and Chart 3
        st.markdown(
            f"**Read-out:** {_life_readout(df)}**{summary['top_theme']}** is the largest driver; and "
            f"{_assessed_share:.0%} of assessed records are addressable through product fixes, nudges or education "
            f"({verdict} the 75% hypothesis).")

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
        st.markdown("#### Datasets")
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "📥 Download Combined Dataset (.xlsx)",
                data=dataset_to_xlsx(df_master, "Combined Discovery Dataset") if len(df_master) else b"",
                file_name=f"Groww_SIP_Combined_Dataset_{datetime.now():%Y%m%d}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=not len(df_master), key="dl_combined", use_container_width=True)
        with ex2:
            st.download_button(
                "🏅 Download Validated Insights Dataset (.xlsx)",
                data=golden_raw or (dataset_to_xlsx(df_golden, "Golden Set") if df_golden is not None else b""),
                file_name="validated_insights_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                disabled=df_golden is None, key="dl_golden", use_container_width=True)
        st.divider()
        st.markdown("#### Discovery Workbook")
        scope = st.radio("Rows to export", ["Full dataset", "Current filtered view"], horizontal=True, key="export_scope")
        _ecols = ec.COLUMNS + [c for c in ("Year", "Label Source") if c in df_all.columns]
        export_df = df_all[_ecols] if scope == "Full dataset" else df[_ecols]
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


summary = ec.summarize(df) if df["Problem Theme"].isin(ec.THEMES).any() else {"n": 0}
if summary["n"] == 0 and len(df):
    st.info("None of the rows in view have a Problem Theme yet, so the dashboard charts are hidden.")
    render_evidence(df)
elif summary["n"] == 0:
    st.warning("No records match the current filters. Widen the filters to see the dashboard.")
else:
    render_dashboard(df, summary)

st.markdown(
    '<div class="footer"><b>Compliance:</b> Built from public web conversations only; no private user data, '
    'authentication or PII is collected or stored. This tool is for internal product discovery and does not '
    'constitute investment advice or recommend any fund or security. Mutual fund investments are subject to market risks.</div>',
    unsafe_allow_html=True,
)
