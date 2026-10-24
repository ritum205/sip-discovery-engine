"""
Groww SIP Churn Discovery & Investor Friction Engine — Streamlit app
Groww Growth · Analysing SIP churners and the friction behind their drop-off.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud → New app → this repo → app.py
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engine_core as ec

# ---------------------------------------------------------------------------
# Page setup & brand styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Groww SIP Churn Discovery & Investor Friction Engine",
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

st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
  .hero {{ background: linear-gradient(120deg, {NAVY} 0%, #24497A 70%, #2C5A8F 100%);
          color: #fff; border-radius: 16px; padding: 28px 32px; margin-bottom: 18px; }}
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
  .roadmap {{ background: #fff; border: 1px solid #E3E8EF; border-radius: 14px; padding: 18px 20px; margin-bottom: 12px; }}
  .roadmap h4 {{ color: {NAVY}; margin: 0 0 6px 0; }}
  .prio {{ display: inline-block; border-radius: 6px; padding: 1px 8px; font-size: 0.75rem; font-weight: 700; margin-right: 8px; }}
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
    return ec.build_discovery_workbook(df, note)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <div class="pill">GROWW GROWTH · PM DISCOVERY</div>
  <h1>Groww SIP Churn Discovery &amp; Investor Friction Engine</h1>
  <p><b>Focus:</b> investors who stop, pause or break their SIPs — especially first-time mutual-fund investors
  dropping off around the 12-month mark. This engine analyses what churners say in public conversations,
  pinpoints the friction behind each drop-off, sizes every churn driver, and maps it to a compliant product fix.</p>
</div>
""",
    unsafe_allow_html=True,
)

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
    m4.markdown('<div class="step"><b>4 · Synthesise</b><br><span>Cliff curve, driver sizing, preventability split, and a 3-sheet PM workbook with live Excel formulas.</span></div>', unsafe_allow_html=True)
    st.caption("Classifier rules live in `engine_core.THEME_KB` — every assignment can be traced to the phrases that triggered it.")

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
        include_play = st.checkbox("Include Play Store reviews", value=True)
        apps_text = st.text_area(
            "Apps (Name = package id, one per line)",
            value="\n".join(f"{k} = {v}" for k, v in ec.DEFAULT_PLAY_APPS.items()),
            height=110,
            disabled=not include_play,
        )
        reviews_per_app = st.slider("Newest reviews per app", 50, 1000, 300, 50, disabled=not include_play)

        st.markdown("**Reddit (public JSON)**")
        include_reddit = st.checkbox("Include Reddit posts", value=True)
        st.caption("If Reddit blocks or rate-limits this server (HTTP 403/429), the run "
                   "automatically falls back to live Play Store ingestion.")
        subs = st.multiselect("Subreddits", ["IndiaInvestments", "personalfinanceindia", "mutualfunds", "IndianStreetBets"],
                              default=ec.DEFAULT_SUBREDDITS, disabled=not include_reddit)
        queries_text = st.text_input("Search phrases (comma-separated)", ", ".join(ec.DEFAULT_REDDIT_QUERIES),
                                     disabled=not include_reddit)
        run = st.button("🚀 Run Live Ingestion", type="primary", use_container_width=True)

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
sel_theme = f1.multiselect("Problem Theme", present("Problem Theme", ec.THEMES), placeholder="All 8 drivers")
sel_band = f2.multiselect("SIP Tenure", present("Tenure Band", band_order), placeholder="All tenures")
sel_stage = f3.multiselect("SIP Stage", present("SIP Stage", ec.STAGES + ["Unknown Stage"]), placeholder="All stages")
sel_prev = f4.multiselect("Churn Preventability", present("Churn Preventability", ec.PREVENTABILITY_LEVELS),
                          placeholder="All levels")

df = df_all
if sel_theme:
    df = df[df["Problem Theme"].isin(sel_theme)]
if sel_band:
    df = df[df["Tenure Band"].isin(sel_band)]
if sel_stage:
    df = df[df["SIP Stage"].isin(sel_stage)]
if sel_prev:
    df = df[df["Churn Preventability"].isin(sel_prev)]

summary = ec.summarize(df)
if summary["n"] == 0:
    st.warning("No records match the current filters.")
    st.stop()

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

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_dash, tab_table, tab_road, tab_export = st.tabs(
    ["📊 Evaluator Dashboard", "🔎 Discovery Table", "🗺️ PM Growth Roadmap", "⬇️ Export Workbook"])

PLOT_LAYOUT = dict(
    font=dict(family="Inter, Arial, sans-serif", color=NAVY, size=13),
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    margin=dict(l=10, r=10, t=60, b=10),
    hoverlabel=dict(bgcolor="#FFFFFF", font_size=13),
)

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

    # Chart 2 — driver breakdown
    themes = summary["themes"].sort_values("Records")
    left, right = st.columns([3, 2])
    with left:
        fig2 = go.Figure(go.Bar(
            y=themes["Problem Theme"], x=themes["Share"], orientation="h",
            marker_color=[THEME_COLORS.get(t, NAVY) for t in themes["Problem Theme"]],
            text=[f"{s:.0%}" for s in themes["Share"]], textposition="outside",
            customdata=themes[["Records", "Dominant Emotion"]].values,
            hovertemplate="<b>%{y}</b><br>Share: %{x:.1%} (%{customdata[0]} records)"
                          "<br>Dominant emotion: %{customdata[1]}<extra></extra>"))
        fig2.update_layout(**PLOT_LAYOUT, height=420,
                           title=dict(text="<b>Chart 2 · Root-Cause Driver Breakdown</b><br>"
                                           "<span style='font-size:12px;color:#5A6B82'>% share of churn records · hover for details</span>"),
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
        cross = (df.groupby(["Problem Theme", "Churn Preventability"]).size().reset_index(name="Records"))
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

    verdict = "above" if summary["addressable_share"] > 0.75 else "below"
    st.markdown(
        f"**Read-out:** {summary['cliff_share']:.0%} of tenure-stated churn in view falls in months 10–14; "
        f"**{summary['top_theme']}** is the largest driver; and {summary['addressable_share']:.0%} of records are "
        f"addressable through Product/UX fixes or compliant FinTech nudges ({verdict} the 75% hypothesis).")

with tab_table:
    keyword = st.text_input("🔍 Search Reviews by Keyword:", placeholder="e.g., cancel, pause, sip, ...",
                            key="review_keyword")
    table_df = df
    if keyword.strip():
        kw = keyword.strip()
        searchable = [c for c in ["Original Comment", "Problem Theme", "Product"] if c in table_df.columns]
        mask = pd.Series(False, index=table_df.index)
        for col in searchable:
            mask |= table_df[col].astype(str).str.contains(kw, case=False, na=False, regex=False)
        table_df = table_df[mask]

    st.caption(f"{len(table_df):,} of {len(df):,} records" + (f" matching “{keyword.strip()}”" if keyword.strip() else "")
               + " · click a Source URL to open the original review or thread.")
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
                       file_name="sip_churn_discovery_filtered.csv", mime="text/csv",
                       disabled=table_df.empty, key="csv_download")

with tab_road:
    st.markdown("Every feature is designed to **retain without pressure**: no fund or stock recommendations, "
                "no FOMO/fear/urgency copy, and cancellation always one tap away.")
    for item in ec.ROADMAP:
        evidence = int(df_all["Problem Theme"].isin(item["Themes"]).sum())
        bg = NAVY if item["Priority"] == "P0" else (EMERALD if item["Priority"] == "P1" else "#D5DBE3")
        fg = "#FFFFFF" if item["Priority"] == "P0" else NAVY
        st.markdown(
            f"""<div class="roadmap"><h4><span class="prio" style="background:{bg};color:{fg}">{item['Priority']}</span>{item['Feature']}</h4>
            <div style="color:#5A6B82;font-size:.85rem;margin-bottom:8px">Addresses: {', '.join(item['Themes'])} · <b>{evidence}</b> supporting records · {item['Phase']}</div>
            <p style="margin:4px 0"><b>Problem.</b> {item['User Problem']}</p>
            <p style="margin:4px 0"><b>Spec.</b> {item['Solution Spec']}</p>
            <p style="margin:4px 0"><b>Guardrails.</b> {item['Guardrails']}</p>
            <p style="margin:4px 0"><b>Success metric.</b> {item['Success Metric']}</p></div>""",
            unsafe_allow_html=True)

with tab_export:
    st.markdown("#### Official PM Discovery Workbook")
    st.markdown(
        "- **Sheet 1 · Discovery Dataset** — all records, navy headers, auto-filters, frozen panes, wrapped comments, clickable source links")
    scope = st.radio("Rows to export", ["Full dataset", "Current filtered view"], horizontal=True, key="export_scope")
    export_df = df_all[ec.COLUMNS] if scope == "Full dataset" else df[ec.COLUMNS]
    with st.spinner("Building workbook…"):
        xlsx = get_workbook(export_df.reset_index(drop=True), provenance)
    st.download_button(
        "📥 Download Official PM Discovery Workbook (.xlsx)",
        data=xlsx,
        file_name=f"Groww_SIP_Churn_Discovery_{datetime.now():%Y%m%d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        key="xlsx_download",
    )

st.markdown(
    '<div class="footer"><b>Compliance:</b> Built from public web conversations only; no private user data, '
    'authentication or PII is collected or stored. This tool is for internal product discovery and does not '
    'constitute investment advice or recommend any fund or security. Mutual fund investments are subject to market risks.</div>',
    unsafe_allow_html=True,
)
