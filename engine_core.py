"""
engine_core.py — SIP Retention AI Discovery Engine
"""
from __future__ import annotations
import io, json, os, random, re
from collections import Counter
import pandas as pd

COLUMNS = ['Product', 'Source Platform', 'Source URL', 'Original Comment', 'User Type', 'SIP Tenure (Months)', 'SIP Stage', 'Problem Theme', 'Problem Behaviour', 'Problem Root Cause', 'Investor Emotional Trigger', 'Churn Preventability', 'Recommended Groww Intervention', 'Target North-Star Metric Impact']
THEMES = ['Disappointment Valley', 'Bank & Auto-Debit Friction', 'Same-Category Overlap', 'Directionless Onboarding', 'Liquidity Shortage', 'Market Correction Panic', 'Social Influence', 'Platform Bugs']
PREVENT_UX = 'High – Product/UX Fix'
PREVENT_NUDGE = 'High – FinTech Nudge'
PREVENT_EDU = 'Medium – Investor Education'
PREVENT_LOW = 'Low – External / Life Event'
PREVENTABILITY_LEVELS = [PREVENT_UX, PREVENT_NUDGE, PREVENT_EDU, PREVENT_LOW]
ADDRESSABLE_LEVELS = {PREVENT_UX, PREVENT_NUDGE}
STAGES = ['Activation (M0–3)', 'Habit Formation (M4–9)', 'Retention Cliff (M10–14)', 'Post-Cliff Maturity (M15+)']
TENURE_BANDS = [('0–3 months', 0, 3), ('4–6 months', 4, 6), ('7–9 months', 7, 9), ('10–14 months (Cliff)', 10, 14), ('15–24 months', 15, 24), ('25–36 months', 25, 36), ('37+ months', 37, 600)]
GROWW_APP_ID = 'com.nextbillion.groww'
GROWW_PLAY_URL = f'https://play.google.com/store/apps/details?id={GROWW_APP_ID}'
REAL_BENCHMARK_FILE = 'real_benchmark_220.json'
REAL_EXCEL_FILE = 'wb.xlsx'

THEME_KB = {
    'Disappointment Valley': {'intervention': 'Your SIP in Context card: historical rolling-return ranges + units accumulated', 'nsm': 'Month-12 SIP retention'},
    'Bank & Auto-Debit Friction': {'intervention': 'Pre-Debit Balance Predictor and Mandate Health Check', 'nsm': 'SIP instalment success rate'},
    'Same-Category Overlap': {'intervention': 'Same-Category Overlap Consolidator', 'nsm': 'Consolidated SIP amount'},
    'Directionless Onboarding': {'intervention': 'Goal-anchored SIP setup', 'nsm': '% of SIPs linked to a goal'},
    'Liquidity Shortage': {'intervention': 'SIP Sabbatical: pause 1-3 instalments', 'nsm': 'Pause-instead-of-cancel ratio'},
    'Market Correction Panic': {'intervention': 'Calm-Market Context: drawdown ranges', 'nsm': 'Down-market retention'},
    'Social Influence': {'intervention': 'Education module on holding periods', 'nsm': 'Discipline retention'},
    'Platform Bugs': {'intervention': 'SIP status reliability fixes', 'nsm': 'Support tickets reduced'}
}

def tenure_band(months):
    if months is None or pd.isna(months):
        return 'Not stated'
    try:
        m = int(months)
        for label, lo, hi in TENURE_BANDS:
            if lo <= m <= hi:
                return label
    except Exception:
        pass
    return 'Not stated'

def stage_for_tenure(months):
    if months is None or pd.isna(months):
        return 'Unknown Stage'
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
        return 'Unknown Stage'

def _coerce_types(df):
    if 'SIP Tenure (Months)' in df.columns:
        df['SIP Tenure (Months)'] = pd.to_numeric(df['SIP Tenure (Months)'], errors='coerce')
    for col in ['Product', 'Source Platform', 'Problem Theme', 'SIP Stage', 'Churn Preventability']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str).str.strip()
    return df

def is_post_level_url(url):
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    return any(p in u for p in ['play.google.com', 'apps.apple.com', 'reddit.com', 'youtube.com', 'youtu.be', 'tradingqna.com'])

def load_real_benchmark(path=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    xlsx_candidates = [os.path.join(base_dir, REAL_EXCEL_FILE), os.path.join(cwd, REAL_EXCEL_FILE), 'wb.xlsx', 'wb_2.xlsx']
    json_candidates = [path or os.path.join(base_dir, REAL_BENCHMARK_FILE), os.path.join(cwd, REAL_BENCHMARK_FILE), REAL_BENCHMARK_FILE]
    for xp in xlsx_candidates:
        if os.path.exists(xp):
            try:
                df = pd.read_excel(xp)
                if len(df) > 0:
                    for col in COLUMNS:
                        if col not in df.columns:
                            df[col] = None
                    return _coerce_types(df[COLUMNS].reset_index(drop=True)), {'source': f'Excel: {os.path.basename(xp)}', 'total': len(df), 'rejected_non_post_urls': 0}
            except Exception:
                pass
    for jp in json_candidates:
        if jp and os.path.exists(jp):
            try:
                with open(jp, 'r', encoding='utf-8') as fh:
                    payload = json.load(fh)
                records = payload.get('records', payload if isinstance(payload, list) else [])
                valid = [r for r in records if is_post_level_url(r.get('Source URL', ''))] or records
                df = pd.DataFrame(valid)
                for col in COLUMNS:
                    if col not in df.columns:
                        df[col] = None
                return _coerce_types(df[COLUMNS].reset_index(drop=True)), {'source': f'JSON: {os.path.basename(jp)}', 'total': len(df), 'rejected_non_post_urls': len(records) - len(valid)}
            except Exception:
                pass
    return None, {'error': 'Could not locate wb.xlsx or real_benchmark_220.json'}

def load_benchmark_data():
    df, _ = load_real_benchmark()
    if df is not None:
        return df
    raise FileNotFoundError('Could not load benchmark data.')

def summarize(df):
    n = len(df)
    if n == 0:
        return {'n': 0}
    ten = df['SIP Tenure (Months)']
    cliff = int(((ten >= 10) & (ten <= 14)).fillna(False).sum())
    addressable = int(df['Churn Preventability'].isin(ADDRESSABLE_LEVELS).sum())
    themes = []
    for theme in THEMES:
        sub = df[df['Problem Theme'] == theme]
        if sub.empty:
            continue
        emo_counts = Counter(sub['Investor Emotional Trigger'])
        emo = emo_counts.most_common(1)[0][0] if emo_counts else 'Frustration'
        themes.append({
            'Problem Theme': theme,
            'Records': len(sub),
            'Share': len(sub) / n,
            'Median Tenure (Months)': float(sub['SIP Tenure (Months)'].dropna().median()) if sub['SIP Tenure (Months)'].notna().any() else 10.0,
            'Dominant Emotion': emo,
            'Addressable Share': float(sub['Churn Preventability'].isin(ADDRESSABLE_LEVELS).mean())
        })
    theme_df = pd.DataFrame(themes).sort_values('Records', ascending=False)
    known_tenure = int(ten.notna().sum())
    return {'n': n, 'cliff': cliff, 'cliff_share': cliff / known_tenure if known_tenure else 0.0, 'known_tenure': known_tenure, 'addressable': addressable, 'addressable_share': addressable / n, 'themes': theme_df, 'top_theme': theme_df.iloc[0]['Problem Theme'] if not theme_df.empty else 'Bank & Auto-Debit Friction'}

def build_discovery_workbook(df, provenance_note=None):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = f'Discovery Dataset ({len(df)} Rows)'
    for c, col in enumerate(COLUMNS, start=1):
        ws.cell(row=1, column=c, value=col)
    for r, rec in enumerate(df[COLUMNS].itertuples(index=False), start=2):
        for c, val in enumerate(rec, start=1):
            ws.cell(row=r, column=c, value=val)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
