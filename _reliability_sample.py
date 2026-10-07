"""
Draw the reliability sample of the Stage 1 protocol (section 3.6) and freeze the list of development reports.

500 basic-school analyses of the non-held-out half, 125 per inspection era, a simple random sample within era:
the eligible analyses of an era are ordered by the SHA-256 of "<SEED>:<analysis_id>" and the first 125 are taken.
The draw uses identifiers and inspection dates only. Not eligible: reports that are anchor sources and reports
scored in a development run (every report in the scoring cache except those of the website's own rubric).

    python _reliability_sample.py            # freezes the development list from the cache, draws, writes both files
    python _reliability_sample.py --check    # redraws from the frozen list and compares with the sample file

Outputs (text files, so that their hashes do not depend on a library version):
data/development_report_ids.csv (report_id) and data/reliability_sample.csv (era, analysis_id, report_id, inspection_date).
"""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys

import pandas as pd

from paper1_analysis import DATA_DIR, anchor_report_ids, era, load_analyses
from scoring import CACHE_DB

SEED = "paper1-reliability-sample-v1"
PER_ERA = 125
ERAS = ["≤2010", "2011–15", "2016–19", "2020+"]
DEVELOPMENT, SAMPLE = DATA_DIR / "development_report_ids.csv", DATA_DIR / "reliability_sample.csv"


def development_reports_from_cache() -> list[str]:
    """Reports scored in development: all cache rows but the website rubric, whose rows are keyed by analysis."""
    with sqlite3.connect(CACHE_DB) as c:
        return sorted(r for r, in c.execute("SELECT DISTINCT report_id FROM calls WHERE run_id NOT LIKE 'webrubric%'"))


def draw(development: set[str]) -> pd.DataFrame:
    a = load_analyses()
    half_a = set(pd.read_parquet(DATA_DIR / "half_a_red_izos.parquet").red_izo)
    a = a[a.red_izo.isin(half_a) & a.insp_date.notna() & ~a.report_id.isin(development | anchor_report_ids())].copy()
    a["era"] = era(a.insp_date)
    a["order"] = [hashlib.sha256(f"{SEED}:{x}".encode()).hexdigest() for x in a.analysis_id]
    out = a.sort_values("order").groupby("era", sort=False).head(PER_ERA)
    out = out.assign(inspection_date=out.insp_date.dt.strftime("%Y-%m-%d"), era=pd.Categorical(out.era, ERAS))
    return out.sort_values(["era", "analysis_id"])[["era", "analysis_id", "report_id", "inspection_date"]].reset_index(drop=True)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    check = ap.parse_args().check
    if not check:
        pd.DataFrame({"report_id": development_reports_from_cache()}).to_csv(DEVELOPMENT, index=False)
    sample = draw(set(pd.read_csv(DEVELOPMENT, dtype=str).report_id))
    if sample.era.value_counts().min() < PER_ERA or len(sample) != PER_ERA * len(ERAS):
        sys.exit(f"an era has fewer than {PER_ERA} eligible analyses: {sample.era.value_counts().to_dict()}")
    if check:
        same = sample.astype(str).equals(pd.read_csv(SAMPLE, dtype=str))
        print("reliability sample reproduced" if same else "MISMATCH with data/reliability_sample.csv")
        sys.exit(0 if same else 1)
    sample.to_csv(SAMPLE, index=False)
    print(f"{len(sample)} analyses of {sample.report_id.nunique()} reports, {sample.era.value_counts().to_dict()}")
    print(f"seed {SEED}\n{DEVELOPMENT.name} sha256 {sha256(DEVELOPMENT)}\n{SAMPLE.name} sha256 {sha256(SAMPLE)}")
