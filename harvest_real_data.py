#!/usr/bin/env python3
"""
harvest_real_data.py — build the REAL benchmark for the SIP Retention AI Discovery Engine.

Run once on your own machine (home/office IPs are far less likely to be
rate-limited by Reddit than cloud servers), then commit the output file:

    pip install -r requirements.txt
    python harvest_real_data.py                  # ~10–20 min with polite rate limits
    git add real_benchmark_220.json && git commit -m "Real benchmark" && git push

What it does
------------
1. Pulls public Google Play reviews for Groww, Zerodha Coin and Kuvera
   (package IDs are verified against the Play Store; a wrong ID is re-resolved
   by name search and the resolution is printed).
2. Pulls public Reddit posts AND comments from r/IndiaInvestments,
   r/personalfinanceindia and r/mutualfunds that match SIP drop-off phrases.
3. Keeps only items that (a) talk about SIPs / mutual funds, (b) describe
   stopping, pausing, failing, redeeming or breaking, and (c) classify into one
   of the 8 churn drivers with enough confidence.
4. Strips PII (handles, emails, phones, PAN, Aadhaar, UPI IDs, account numbers).
   Author names are never collected.
5. Stores the exact post-/review-level URL for every record:
     Reddit  → https://www.reddit.com/r/<sub>/comments/<id>/<slug>/[<comment_id>/]
     Play    → https://play.google.com/store/apps/details?id=<pkg>&reviewId=<id>
   Reddit links are re-checked before saving and dropped if the content was
   removed or deleted.
6. Writes real_benchmark_220.json (meta + records in the 15-column schema, plus
   audit fields prefixed with "_") and real_benchmark_review.csv for a quick
   manual QA pass.

Optional manual curation
------------------------
Create label_overrides.csv with columns  source_id,action,value  where action is
`exclude` (value ignored) or `theme` (value = one of the 8 driver names), then
re-run with --reuse-raw to apply your corrections without re-scraping.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

import engine_core as ec

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CACHE = os.path.join(HERE, ".harvest_raw_cache.json")

REDDIT_SUBS = ["IndiaInvestments", "personalfinanceindia", "mutualfunds"]
REDDIT_QUERIES = [
    "stopped SIP", "stop my SIP", "cancel SIP", "cancelled SIP", "pause SIP",
    "paused SIP", "SIP mandate failed", "SIP autopay failed", "SIP bounce",
    "SIP returns after one year", "SIP negative returns", "redeem SIP",
    "too many mutual funds overlap", "SIP market crash", "lost job SIP",
]


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 1. Collect raw public items
# ---------------------------------------------------------------------------

def collect_raw(args) -> list[dict]:
    raw: list[dict] = []

    if not args.skip_play:
        apps = dict(ec.DEFAULT_PLAY_APPS)
        for override in args.app or []:  # --app "Kuvera=com.example.id"
            name, _, pkg = override.partition("=")
            apps[name.strip()] = pkg.strip()
        log(f"\n▶ Google Play Store — up to {args.play_reviews} reviews × {len(apps)} apps × 2 sort orders")
        items, lines = ec.fetch_play_store_reviews(apps, count_per_app=args.play_reviews,
                                                   sorts=("NEWEST", "MOST_RELEVANT"))
        for line in lines:
            log("  " + line)
        raw += items

    if not args.skip_reddit:
        log(f"\n▶ Reddit — {len(REDDIT_SUBS)} subreddits × {len(REDDIT_QUERIES)} queries (posts + matching comments)")
        churn_comment = lambda t: bool(ec.RELEVANCE_PATTERN.search(t) and ec.CHURN_SIGNAL_PATTERN.search(t))  # noqa: E731
        items, lines, status = ec.fetch_reddit_posts(
            REDDIT_SUBS, REDDIT_QUERIES, limit=50, timeframe="all", pause_s=args.reddit_pause,
            include_comments=True, max_wait_s=60, comment_filter=churn_comment)
        for line in lines:
            log("  " + line)
        if status.startswith("blocked"):
            log("  ↪ Continuing with what was collected. Re-run later (or from another network) to add more Reddit data.")
        raw += items

    return raw


# ---------------------------------------------------------------------------
# 2. Enrich, filter, de-duplicate
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()[:220]


def load_overrides(path: str) -> dict[str, tuple[str, str]]:
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["source_id"].strip(): (row["action"].strip().lower(), (row.get("value") or "").strip())
                for row in csv.DictReader(fh) if row.get("source_id")}


def enrich(raw: list[dict], min_score: int, overrides: dict) -> list[dict]:
    out, seen = [], set()
    for item in raw:
        sid = item.get("source_id", "")
        action, value = overrides.get(sid, ("", ""))
        if action == "exclude":
            continue
        rec = ec.enrich_record(item["product"], item["platform"], item["url"], item["text"],
                               require_churn_signal=True)
        if rec is None:
            continue
        audit = ec.explain_theme(rec["Original Comment"])
        if action == "theme" and value in ec.THEMES and value != rec["Problem Theme"]:
            # Re-derive theme-dependent fields for the corrected theme
            kb = ec.THEME_KB[value]
            text = rec["Original Comment"]
            rec.update({
                "Problem Theme": value,
                "Problem Behaviour": ec.detect_behaviour(text, value),
                "Problem Root Cause": ec.pick_root_cause(text, value),
                "Investor Emotional Trigger": ec.detect_emotion(text, value),
                "Churn Preventability": ec.detect_preventability(text, value),
                "Recommended Groww Intervention": kb["intervention"],
                "Target North-Star Metric Impact": kb["nsm"],
            })
            audit["manual_override"] = True
        elif audit["score"] < min_score:
            continue
        key = _norm(rec["Original Comment"])
        if key in seen:
            continue
        seen.add(key)
        rec.update({
            "_source_id": sid,
            "_posted_at": item.get("posted_at"),
            "_rating_or_score": item.get("rating"),
            "_theme_score": audit["score"],
            "_theme_margin": audit["margin"],
            "_matched_phrases": "; ".join(audit["matched"]),
            "_manual_override": bool(audit.get("manual_override")),
        })
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# 3. Verify Reddit links still resolve to live content
# ---------------------------------------------------------------------------

def verify_reddit_links(records: list[dict], pause_s: float) -> list[dict]:
    import time

    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": ec.USER_AGENT})
    kept = []
    reddit = [r for r in records if r["Source URL"].startswith("https://www.reddit.com/")]
    log(f"\n▶ Verifying {len(reddit)} Reddit permalinks (drops removed/deleted content)")
    for i, rec in enumerate(records):
        url = rec["Source URL"]
        if not url.startswith("https://www.reddit.com/"):
            kept.append(rec)
            continue
        try:
            data = ec._reddit_get(session, url.rstrip("/") + "/.json", {"raw_json": 1}, max_wait_s=60)
        except ec.RedditBlocked as exc:
            log(f"  ⚠ Verification stopped ({exc}); keeping remaining links unverified.")
            kept.extend(r for r in records[i:])
            return kept
        time.sleep(pause_s)
        ok = False
        if isinstance(data, list) and data:
            sid = rec.get("_source_id", "")
            if sid.startswith("t1_"):
                listing = data[1]["data"]["children"] if len(data) > 1 else []
                body = listing[0]["data"].get("body", "") if listing else ""
                ok = bool(listing) and not ec._removed(body)
            else:
                post = data[0]["data"]["children"][0]["data"]
                ok = not post.get("removed_by_category") and post.get("title")
        if ok:
            kept.append(rec)
        else:
            log(f"  ✗ dropped (no longer public): {url}")
    return kept


# ---------------------------------------------------------------------------
# 4. Select and write
# ---------------------------------------------------------------------------

def select(records: list[dict], target: int) -> list[dict]:
    """Keep the `target` most confidently classified records. Themes are NOT
    rebalanced — the real distribution is the finding."""
    records = sorted(records, key=lambda r: (r["_theme_score"], r["_theme_margin"]), reverse=True)
    return records[:target]


def write_outputs(records: list[dict], out_path: str, args, raw_count: int) -> None:
    records = sorted(records, key=lambda r: (ec.THEMES.index(r["Problem Theme"]), r["Source Platform"]))
    by_platform = Counter(r["Source Platform"].split(" ·")[0] for r in records)
    meta = {
        "dataset": "SIP Retention AI Discovery Engine — real public benchmark",
        "harvested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_count": len(records),
        "raw_items_scanned": raw_count,
        "sources": {
            "google_play_apps": ec.DEFAULT_PLAY_APPS if not args.skip_play else {},
            "subreddits": REDDIT_SUBS if not args.skip_reddit else [],
            "reddit_queries": REDDIT_QUERIES if not args.skip_reddit else [],
        },
        "records_by_platform": dict(by_platform),
        "records_by_theme": dict(Counter(r["Problem Theme"] for r in records)),
        "method": ("Verbatim public text (PII-scrubbed). Kept only items mentioning SIPs/MFs, describing a "
                   "stop/pause/failure/redemption, and scoring ≥ %d on the weighted churn-driver lexicon. "
                   "Labels are machine-assigned by engine_core.py; tenure and amount are populated only "
                   "when the author stated them." % args.min_score),
        "privacy": "No author names, avatars or account identifiers collected. Handles, emails, phones, "
                   "PAN, Aadhaar, UPI IDs and long digit strings replaced with placeholders.",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "records": records}, fh, ensure_ascii=False, indent=1)

    review_path = os.path.join(HERE, "real_benchmark_review.csv")
    cols = ec.COLUMNS + [k for k in records[0] if k.startswith("_")] if records else ec.COLUMNS
    with open(review_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)

    log(f"\n✅ Wrote {len(records)} records → {os.path.basename(out_path)}")
    log(f"   QA sheet → {os.path.basename(review_path)} (open in Excel; add fixes to label_overrides.csv)")
    log("   By platform: " + ", ".join(f"{k}: {v}" for k, v in by_platform.most_common()))
    log("   By theme:    " + ", ".join(f"{k}: {v}" for k, v in Counter(r['Problem Theme'] for r in records).most_common()))
    stated = sum(1 for r in records if r["SIP Tenure (Months)"] is not None)
    log(f"   Tenure stated in {stated}/{len(records)} records (the cliff chart uses these).")


def main() -> int:
    p = argparse.ArgumentParser(description="Harvest the real public benchmark for the SIP Retention engine.")
    p.add_argument("--target", type=int, default=220, help="max records to keep (default 220)")
    p.add_argument("--play-reviews", type=int, default=3000, help="reviews per app per sort order (default 3000)")
    p.add_argument("--min-score", type=int, default=3, help="minimum classifier score to keep (default 3)")
    p.add_argument("--reddit-pause", type=float, default=2.0, help="seconds between Reddit requests (default 2.0)")
    p.add_argument("--app", action="append", help='override/add a Play app: --app "Kuvera=com.pkg.id"')
    p.add_argument("--skip-play", action="store_true")
    p.add_argument("--skip-reddit", action="store_true")
    p.add_argument("--no-verify", action="store_true", help="skip re-checking Reddit permalinks")
    p.add_argument("--reuse-raw", action="store_true", help="re-label the cached raw pull without scraping again")
    p.add_argument("--out", default=os.path.join(HERE, ec.REAL_BENCHMARK_FILE))
    args = p.parse_args()

    if args.reuse_raw and os.path.exists(RAW_CACHE):
        with open(RAW_CACHE, encoding="utf-8") as fh:
            raw = json.load(fh)
        log(f"Re-using {len(raw)} cached raw items from {os.path.basename(RAW_CACHE)}")
    else:
        raw = collect_raw(args)
        with open(RAW_CACHE, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, ensure_ascii=False)
    if not raw:
        log("\n❌ Nothing was collected. Check your internet connection and that google-play-scraper is installed.")
        return 1

    overrides = load_overrides(os.path.join(HERE, "label_overrides.csv"))
    if overrides:
        log(f"Applying {len(overrides)} manual overrides from label_overrides.csv")
    records = enrich(raw, args.min_score, overrides)
    records = [r for r in records if ec.is_post_level_url(r["Source URL"])]
    log(f"\n▶ {len(raw)} raw items → {len(records)} confident SIP-churn records")

    if not args.no_verify and not args.skip_reddit:
        records = verify_reddit_links(select(records, int(args.target * 1.3)), args.reddit_pause)
    records = select(records, args.target)

    if len(records) < args.target:
        log(f"\n⚠ Only {len(records)} records met the bar (target {args.target}). Options: raise "
            "--play-reviews, lower --min-score to 2, or re-run Reddit later. The app works with any count.")
    if not records:
        return 1
    write_outputs(records, args.out, args, len(raw))
    return 0


if __name__ == "__main__":
    sys.exit(main())
