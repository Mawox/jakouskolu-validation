"""
Paper 1 — registered primary analysis (protocol sections 3.2–3.5, 3.7 "H1", 3.8).

Reads local files only. Steps: linked sample from dates alone (3.4), criterion scales
(3.3, ``mapa_utils.scale_scores`` on the frozen item-score file), score of an analysis
from its three runs (3.2), score of a school (3.3), H1 with Holm adjustment and the H1
part of the validity gate (3.7, 3.8, 3.5). The gate has two outcomes: "pass" needs N ≥ 300,
a Holm-significant test (adjusted p ≤ 0.05) and r ≥ 0.15; anything else is "fail", and the
three conditions are reported beside it. The reliability gates of section 3.6 are evaluated
elsewhere; a dimension that passes here still needs them.

Descriptive output, without any test: correlations with the parent scales (N, r, 95 %
interval), and the block "pattern" with the correlation of every school score with the
primary criterion of every dimension and the correlations among the four school scores.

Scores file: parquet, long format, one row per analysis × run × dimension.
    analysis_id  str    ``analysis_id`` of the legacy-version row (report × school) in data/analyses.parquet
    report_id    str    must equal the report of that row
    run          int    1, 2 or 3
    dimension    str    teaching_quality | leadership_management | school_climate | student_support
    score        float  run score 1, 2, 3 or 4; null if the run returned null or has no valid response
    valid        bool   the run has a schema-valid response (the same for the four dimensions of a run)
Every linked analysis needs all twelve rows; a run without a valid response after five
requests is a row with ``valid = false``, not a missing row. Analyses outside the linked
sample are ignored.

Guard: the entry point reads a scores file only if its parquet schema metadata carry
``paper1_scores = synthetic`` or its SHA-256 equals the content of ``registered_scores.sha256``
beside this module. Synthetic files come from ``--synthetic`` and serve tests and dry runs.

Usage:
    paper1_analysis.py --counts                      linked sample and criterion counts, no scores
    paper1_analysis.py --synthetic OUT.parquet [--rho 0.2] [--seed 1]
    paper1_analysis.py SCORES.parquet [--out RESULT.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from _mapa_linkage_audit import LEGACY_VERSION
from mapa_utils import DATA_DIR, DIMS, MAPA_DIR, fieldwork_dates, load_mapa_long, scale_scores

HERE = Path(__file__).parent
REGISTERED_HASH_FILE = HERE / "registered_scores.sha256"
SYNTHETIC_KEY, SYNTHETIC_VALUE = b"paper1_scores", b"synthetic"
DIMENSIONS = list(DIMS.values())
WINDOW_DAYS = 730
FLOOR_R = 0.15
ALPHA = 0.05
MIN_N = 300
COLUMNS = ["analysis_id", "report_id", "run", "dimension", "score", "valid"]
_N01 = NormalDist()


class ScoresRefused(RuntimeError):
    pass


# --- linked sample (3.4) -------------------------------------------------------

def load_waves() -> pd.DataFrame:
    """School-waves of the current questionnaire with the wave date (median answer timestamp)."""
    w = load_mapa_long()[["red_izo", "school_year"]].drop_duplicates()
    w = w.merge(fieldwork_dates()[["red_izo", "school_year", "median"]], on=["red_izo", "school_year"], how="left")
    if w["median"].isna().any():
        raise ValueError(f"{int(w['median'].isna().sum())} school-waves have no wave date")
    return w.rename(columns={"median": "wave_date"}).reset_index(drop=True)


def load_analyses() -> pd.DataFrame:
    """Basic-school analyses of the legacy version index: identifiers and dates only, never a score column."""
    a = pd.read_parquet(DATA_DIR / "analyses.parquet",
                        columns=["analysis_id", "izo", "red_izo", "report_id", "target_school_type", "inspection_date", "version_id"])
    a = a[(a.version_id == LEGACY_VERSION) & (a.target_school_type == "základní")].copy()
    a["insp_date"] = pd.to_datetime(a.inspection_date, errors="coerce")
    return a[["analysis_id", "izo", "red_izo", "report_id", "insp_date"]].reset_index(drop=True)


def link(waves: pd.DataFrame, inspections: pd.DataFrame) -> pd.DataFrame:
    """Every wave × inspection pair of a school; gap_days = inspection date minus wave date, both as calendar dates."""
    insp = inspections[["red_izo", "report_id", "insp_date"]].dropna().drop_duplicates()
    m = waves.merge(insp, on="red_izo")
    m["gap_days"] = (m.insp_date.dt.normalize() - m.wave_date.dt.normalize()).dt.days.astype(float)
    return m


def select_pairs(pairs: pd.DataFrame, window: float = WINDOW_DAYS) -> pd.DataFrame:
    """One pair per school: smallest absolute gap, then earlier wave date, earlier inspection, lower report id as text."""
    p = pairs[pairs.gap_days.abs() <= window].assign(abs_gap=lambda x: x.gap_days.abs(), wave_day=lambda x: x.wave_date.dt.normalize(),
                                                     insp_day=lambda x: x.insp_date.dt.normalize())
    p = p.sort_values(["red_izo", "abs_gap", "wave_day", "insp_day", "report_id"], kind="mergesort")
    return p.groupby("red_izo", sort=False).head(1).drop(columns=["abs_gap", "wave_day", "insp_day"]).reset_index(drop=True)


def anchor_report_ids() -> set[str]:
    return {a["report_id"] for a in json.loads((DATA_DIR / "rubric_v2_anchors.json").read_text())}


def era(dates: pd.Series) -> pd.Series:
    y = dates.dt.year
    return pd.Series(np.select([y <= 2010, y <= 2015, y <= 2019], ["≤2010", "2011–15", "2016–19"], "2020+"), index=dates.index)


def linked_sample() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Primary sample (one row per school), its basic-school analyses, and the counts of section 3.4."""
    analyses = load_analyses()
    registry = set(pd.read_parquet(DATA_DIR / "schools.parquet", columns=["red_izo"]).red_izo.dropna())
    waves = load_waves()
    waves = waves[waves.red_izo.isin(registry)]
    pairs = link(waves, analyses)
    within = pairs[pairs.gap_days.abs() <= WINDOW_DAYS]
    selected = select_pairs(pairs)
    anchors = anchor_report_ids()
    sample = selected[~selected.report_id.isin(anchors)].reset_index(drop=True)
    sample["era"] = era(sample.insp_date)
    units = analyses.merge(sample[["red_izo", "report_id"]], on=["red_izo", "report_id"])
    gap_y = sample.gap_days.abs() / 365.25
    halves = {h: set(pd.read_parquet(DATA_DIR / f"half_{h}_red_izos.parquet").red_izo) for h in "ab"}
    counts = {
        "schools_under_rule": int(selected.red_izo.nunique()),
        "linked_school_waves": int(within.groupby(["red_izo", "school_year"]).ngroups),
        "distinct_inspections": int(within.report_id.nunique()),
        "schools_within_365_days": int(pairs[pairs.gap_days.abs() <= 365].red_izo.nunique()),
        "schools_within_1095_days": int(pairs[pairs.gap_days.abs() <= 1095].red_izo.nunique()),
        "anchors_in_prompt": len(anchors),
        "selected_reports_that_are_anchor_sources": sorted(selected.report_id[selected.report_id.isin(anchors)]),
        "primary_sample_schools": int(len(sample)),
        "first_selected_inspection": str(sample.insp_date.min().date()),
        "last_selected_inspection": str(sample.insp_date.max().date()),
        "first_linked_inspection": str(within.insp_date.min().date()),
        "selected_by_era": sample.era.value_counts().to_dict(),
        "gap_years_quartiles": [float(gap_y.quantile(q)) for q in (0.25, 0.5, 0.75)],
        "inspection_precedes_survey": int((sample.gap_days < 0).sum()),
        "half_a": int(sample.red_izo.isin(halves["a"]).sum()),
        "half_b": int(sample.red_izo.isin(halves["b"]).sum()),
        "analyses_of_selected_reports": int(len(units)),
        "schools_with_more_than_one_analysis": int((units.groupby("red_izo").size() > 1).sum()),
    }
    return sample, units, counts


# --- criterion scales (3.3) ----------------------------------------------------

def criteria(sample: pd.DataFrame, items: pd.DataFrame | None = None) -> pd.DataFrame:
    """School × dimension: primary criterion, parent scale and pupil-group pattern of the selected wave.

    Primary is the pupil composite (group "Z" of ``scale_scores``: mean of the standardised scales of
    pupils 1–5 and 6–9, or the one that exists) for teaching, climate and support, and the teacher
    scale for leadership. Scales are computed on all school-waves and subset afterwards.
    """
    if items is None:
        items = pd.read_parquet(MAPA_DIR / "mapa_item_scores.parquet")
    sc = scale_scores(items).merge(sample[["red_izo", "school_year"]], on=["red_izo", "school_year"])
    sc["dimension"] = sc.dim.map(DIMS)
    key = ["red_izo", "dimension"]
    primary = sc[(sc.role == "primary") & sc.group.isin(["Z", "U"])].set_index(key).scale_z.rename("primary")
    parents = sc[sc.group == "R"].set_index(key).scale_z.rename("parents")
    groups = (sc[sc.group.isin(["Z1", "Z2"])].sort_values("group").groupby(key).group.agg("+".join)
              .map({"Z1+Z2": "both", "Z1": "grades 1–5 only", "Z2": "grades 6–9 only"}).rename("pupil_groups"))
    grid = pd.MultiIndex.from_product([sample.red_izo, DIMENSIONS], names=key)
    return pd.concat([primary, parents, groups], axis=1).reindex(grid).reset_index()


def benchmark(crit: pd.DataFrame) -> dict:
    """Survey-only: primary criterion against the parent scale on the schools of the sample that have both."""
    out = {}
    for d, g in crit.groupby("dimension"):
        ok = g[g.primary.notna() & g.parents.notna()]
        out[d] = _described(ok.primary, ok.parents)
    return out


def criterion_counts(crit: pd.DataFrame) -> dict:
    out = {}
    for d, g in crit.groupby("dimension"):
        out[d] = {"primary": int(g.primary.notna().sum()), "parents": int(g.parents.notna().sum())}
        if d != DIMS["LM"]:
            out[d]["pupil_groups"] = g[g.primary.notna()].pupil_groups.value_counts().to_dict()
    return out


# --- scores (3.2, 3.3) ---------------------------------------------------------

def check_runs(runs: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Validates the long scores frame and returns the rows of the linked analyses."""
    missing = [c for c in COLUMNS if c not in runs.columns]
    if missing:
        raise ValueError(f"scores file lacks columns {missing}")
    r = runs[runs.analysis_id.isin(set(units.analysis_id))][COLUMNS].copy()
    if not r.dimension.isin(DIMENSIONS).all() or not r.run.isin([1, 2, 3]).all():
        raise ValueError("unknown dimension or run outside 1–3")
    if r.valid.isna().any() or r.valid.dtype != bool:
        raise ValueError("valid must be a boolean without nulls")
    if r.duplicated(["analysis_id", "run", "dimension"]).any():
        raise ValueError("duplicate analysis × run × dimension rows")
    if not r.score.dropna().isin([1, 2, 3, 4]).all():
        raise ValueError("run score outside 1, 2, 3, 4")
    if r.score[~r.valid].notna().any():
        raise ValueError("a run without a valid response carries a score")
    if (r.groupby(["analysis_id", "run"]).valid.nunique() > 1).any():
        raise ValueError("valid differs between the dimensions of one run")
    n = r.groupby("analysis_id").size().reindex(units.analysis_id, fill_value=0)
    if (n != 3 * len(DIMENSIONS)).any():
        raise ValueError(f"{int((n != 3 * len(DIMENSIONS)).sum())} linked analyses lack some of their twelve rows")
    if len(r.merge(units[["analysis_id", "report_id"]], on=["analysis_id", "report_id"])) != len(r):
        raise ValueError("report_id does not match the analyses index")
    r["score"] = r.score.astype(float)
    return r


def analysis_scores(runs: pd.DataFrame) -> pd.DataFrame:
    """Score of an analysis per dimension from its three runs (3.2).

    Mean of the run scores if at least two runs return one; null if at least two return null;
    otherwise a scoring failure. Returns analysis_id, dimension, status (score | null | failure),
    score, n_scored, n_null, n_invalid.
    """
    r = runs.assign(scored=runs.score.notna(), null=runs.valid & runs.score.isna(), invalid=~runs.valid)
    g = r.groupby(["analysis_id", "dimension"])
    out = g.agg(n_runs=("run", "nunique"), n_rows=("run", "size"), n_scored=("scored", "sum"), n_null=("null", "sum"),
                n_invalid=("invalid", "sum"), score=("score", "mean")).reset_index()
    if ((out.n_runs != 3) | (out.n_rows != 3)).any():
        raise ValueError("every analysis × dimension needs exactly runs 1–3")
    out["status"] = np.select([out.n_scored >= 2, out.n_null >= 2], ["score", "null"], "failure")
    out["score"] = out.score.where(out.status == "score")
    return out.drop(columns=["n_runs", "n_rows"])


def school_scores(scores: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Score of a school per dimension (3.3): the mean of the non-null scores of its analyses of the report.

    Without any score the school is a failure if an analysis failed, else null.
    """
    s = scores.merge(units[["analysis_id", "red_izo"]], on="analysis_id")
    g = s.assign(failed=s.status == "failure").groupby(["red_izo", "dimension"])
    out = g.agg(score=("score", "mean"), n_analyses=("analysis_id", "nunique"), n_scored=("score", "count"), n_failed=("failed", "sum")).reset_index()
    out["status"] = np.select([out.n_scored > 0, out.n_failed > 0], ["score", "failure"], "null")
    return out


# --- H1 and the gate (3.5, 3.7, 3.8) --------------------------------------------

def correlation(x, y) -> dict:
    """Pearson r, one-sided p (H0: ρ ≤ 0) and two-sided 95 % interval by the Fisher transformation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    out: dict = {"n": n, "r": None, "ci95": None, "p": None}
    if n < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:  # a constant score cannot be estimated
        return out
    r = float(np.corrcoef(x, y)[0, 1])
    with np.errstate(divide="ignore"):  # r = ±1 gives an infinite z
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    half = _N01.inv_cdf(0.975) * se
    return {"n": n, "r": r, "ci95": [float(np.tanh(z - half)), float(np.tanh(z + half))], "p": float(1 - _N01.cdf(z / se))}


def holm(ps: list[float | None]) -> list[float]:
    """Holm-adjusted p-values; a test that cannot be computed stays in the family with p = 1."""
    p = np.array([1.0 if v is None else v for v in ps])
    order = np.argsort(p, kind="mergesort")
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty(len(p))
    out[order] = adj
    return [float(v) for v in out]


def gate(r: float | None, n: int, p_holm: float) -> dict:
    """H1 part of the validity gate (pass | fail) with the three conditions a pass needs."""
    cond = {"n_at_least_300": n >= MIN_N, "significant": r is not None and p_holm <= ALPHA, "r_at_least_floor": r is not None and r >= FLOOR_R}
    return {"h1_gate": "pass" if all(cond.values()) else "fail", **cond}


def _described(x, y) -> dict:
    c = correlation(x, y)
    return {"n": c["n"], "r": c["r"], "ci95": c["ci95"]}


def pattern(table: pd.DataFrame) -> dict:
    """Descriptive: school score of d against the primary criterion of e, and the school scores among themselves."""
    score = table.pivot(index="red_izo", columns="dimension", values="score")
    crit = table.pivot(index="red_izo", columns="dimension", values="primary")
    out: dict = {"score_by_criterion": {}, "among_scores": {}}
    for i, d in enumerate(DIMENSIONS):
        out["score_by_criterion"][d] = {}
        for e in DIMENSIONS:
            ok = score[d].notna() & crit[e].notna()
            out["score_by_criterion"][d][e] = _described(score[d][ok], crit[e][ok])
        for e in DIMENSIONS[i + 1:]:
            ok = score[d].notna() & score[e].notna()
            out["among_scores"][f"{d} × {e}"] = _described(score[d][ok], score[e][ok])
    return out


def _distribution(s: pd.Series) -> dict:
    return {f"{k:.3f}": int(v) for k, v in s.value_counts().sort_index().items()}


def _status(g: pd.DataFrame) -> dict:
    return {"schools": int(len(g)), "scores": int((g.status == "score").sum()), "nulls": int((g.status == "null").sum()),
            "failures": int((g.status == "failure").sum()), "distribution": _distribution(g.score.dropna())}


def h1(table: pd.DataFrame, two_runs: pd.DataFrame) -> dict:
    """Primary results, the descriptive parent correlations and the pattern block from the school × dimension table.

    table: red_izo, dimension, score, status, primary, parents, pupil_groups, era.
    two_runs: analysis-level rows (red_izo, dimension, third) of scores resting on two runs.
    """
    primary: dict = {}
    parents: dict = {}
    for d in DIMENSIONS:
        t = table[table.dimension == d]
        ok = t[t.score.notna() & t.primary.notna()]
        tw = two_runs[two_runs.dimension == d]
        res = correlation(ok.score, ok.primary) | {"distribution_analysed": _distribution(ok.score)} | _status(t)
        res["schools_with_criterion"] = int(t.primary.notna().sum())
        res["two_run_scores"] = {k: {"linked_analyses": int((tw.third == k).sum()),
                                     "of_analysed_schools": int(((tw.third == k) & tw.red_izo.isin(ok.red_izo)).sum())}
                                 for k in ("third_run_null", "third_run_invalid")}
        res["by_era"] = {k: _status(g) for k, g in t.groupby("era")}
        if d != DIMS["LM"]:
            res["by_pupil_groups"] = {k: _status(t[t.pupil_groups == k]) | _described(g.score, g.primary) for k, g in ok.groupby("pupil_groups")}
        primary[d] = res
        okp = t[t.score.notna() & t.parents.notna()]
        parents[d] = _described(okp.score, okp.parents)
    for d, adj in zip(DIMENSIONS, holm([primary[d]["p"] for d in DIMENSIONS])):
        primary[d] |= {"p_holm": adj} | gate(primary[d]["r"], primary[d]["n"], adj)
    pat = pattern(table)
    for d in DIMENSIONS:  # the diagonal is the primary estimate itself, not a recomputation in another row order
        pat["score_by_criterion"][d][d] = {k: primary[d][k] for k in ("n", "r", "ci95")}
    return {"primary": primary, "secondary_parents": parents, "pattern": pat}


def analyse(runs: pd.DataFrame, sample: pd.DataFrame, units: pd.DataFrame, crit: pd.DataFrame) -> dict:
    a = analysis_scores(check_runs(runs, units))
    table = school_scores(a, units).merge(crit, on=["red_izo", "dimension"]).merge(sample[["red_izo", "era"]], on="red_izo")
    two = a[(a.status == "score") & (a.n_scored == 2)].merge(units[["analysis_id", "red_izo"]], on="analysis_id")
    return h1(table, two.assign(third=np.where(two.n_null == 1, "third_run_null", "third_run_invalid")))


# --- guard and entry point ------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_scores_file(path: Path) -> dict:
    """Refuses a scores file that is neither marked synthetic nor the registered one. Reads no score."""
    digest = sha256(path)
    if (pq.read_schema(path).metadata or {}).get(SYNTHETIC_KEY) == SYNTHETIC_VALUE:
        return {"kind": "synthetic", "sha256": digest}
    if REGISTERED_HASH_FILE.exists() and digest == REGISTERED_HASH_FILE.read_text().split()[0].lower():
        return {"kind": "registered", "sha256": digest}
    raise ScoresRefused(
        f"Refused: {path} is not marked synthetic and its SHA-256 ({digest}) "
        f"{'differs from' if REGISTERED_HASH_FILE.exists() else 'cannot be checked, because there is no'} {REGISTERED_HASH_FILE.name}. "
        "No association between LLM scores and Mapa školy is computed before the scores file of the registered run is recorded."
    )


def run(path: Path) -> dict:
    info = check_scores_file(path)
    sample, units, counts = linked_sample()
    crit = criteria(sample)
    return {"scores_file": {"path": str(path), **info}, "sample": counts, "criteria": criterion_counts(crit), "benchmark": benchmark(crit),
            **analyse(pd.read_parquet(path, columns=COLUMNS), sample, units, crit)}


# --- synthetic scores (tests and dry runs only) ----------------------------------

def synthetic_runs(units: pd.DataFrame, rho: float | dict[str, float], seed: int, run_sd: float = 0.5, p_null: float = 0.03,
                   p_run_null: float = 0.02, p_invalid: float = 0.01, cuts: tuple[float, float, float] = (-1.0, 0.1, 1.1)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic run scores for ``units`` (analysis_id, report_id and one criterion column per dimension).

    Per analysis and dimension a latent value correlates at ``rho`` with the standardised criterion
    (pure noise where the criterion is missing); each run adds noise and cuts the value into 1–4.
    An analysis is null in all runs with ``p_null``, a single run score with ``p_run_null``, and a run
    has no valid response with ``p_invalid``. Returns the long scores frame and the latent values.
    """
    rng = np.random.default_rng(seed)
    n = len(units)
    invalid = rng.random((n, 3)) < p_invalid
    rows, latent = [], {}
    for d in DIMENSIONS:
        rh = rho[d] if isinstance(rho, dict) else rho
        c = units[d].to_numpy(float)
        c = np.nan_to_num((c - np.nanmean(c)) / np.nanstd(c))
        mix = np.where(np.isnan(units[d].to_numpy(float)), 1.0, np.sqrt(1 - rh**2))
        latent[d] = rh * c + mix * rng.standard_normal(n)
        value = latent[d][:, None] + run_sd * rng.standard_normal((n, 3))
        score = 1.0 + (value[:, :, None] > np.array(cuts)).sum(axis=2)
        null = (rng.random(n) < p_null)[:, None] | (rng.random((n, 3)) < p_run_null)
        score[null | invalid] = np.nan
        for k in range(3):
            rows.append(pd.DataFrame({"analysis_id": units.analysis_id.to_numpy(), "report_id": units.report_id.to_numpy(), "run": k + 1,
                                      "dimension": d, "score": score[:, k], "valid": ~invalid[:, k]}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(latent).assign(analysis_id=units.analysis_id.to_numpy())


def write_scores(runs: pd.DataFrame, path: Path, synthetic: bool) -> None:
    t = pa.Table.from_pandas(runs[COLUMNS], preserve_index=False)
    if synthetic:
        t = t.replace_schema_metadata({**(t.schema.metadata or {}), SYNTHETIC_KEY: SYNTHETIC_VALUE})
    pq.write_table(t, path)


def write_synthetic(path: Path, rho: float, seed: int) -> None:
    sample, units, _ = linked_sample()
    wide = criteria(sample).pivot(index="red_izo", columns="dimension", values="primary").reset_index()
    runs, _ = synthetic_runs(units.merge(wide, on="red_izo"), rho, seed)
    write_scores(runs, path, synthetic=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scores", nargs="?", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--counts", action="store_true")
    ap.add_argument("--synthetic", type=Path, metavar="OUT")
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.synthetic:
        write_synthetic(a.synthetic, a.rho, a.seed)
        return
    if a.counts:
        sample, _, counts = linked_sample()
        crit = criteria(sample)
        res = {"sample": counts, "criteria": criterion_counts(crit), "benchmark": benchmark(crit)}
    elif a.scores:
        try:
            res = run(a.scores)
        except ScoresRefused as e:
            sys.exit(str(e))
    else:
        ap.error("give a scores file, --counts or --synthetic OUT")
    text = json.dumps(res, indent=2, ensure_ascii=False)
    a.out.write_text(text + "\n") if a.out else print(text)


if __name__ == "__main__":
    main()
