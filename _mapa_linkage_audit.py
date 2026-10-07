"""
Mapa školy ↔ ČŠI linkage audit (research plan phase 0.3).

Counts what the Mapa export can validate: schools and waves, registry match,
inspections within ±W years of a wave, the change-on-change panel, participation
selection versus all ZŠ, and the questionnaire break in 2016/17.
Writes data/mapa/mapa_linkage_audit_results.json. No AI-score × Mapa-score
correlation is computed here (locked until the rubric and mapping are registered).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mapa_utils import ITEMS, MAPA_DIR, MAPA_FIRST_YEAR, load_mapa_long, load_mapa_questions, score_items, wave_date

DATA_DIR = Path(__file__).parent / "data"
LEGACY_VERSION = "v_693327a1"
WINDOWS = (1, 2, 3)


def main() -> None:
    out: dict = {}
    raw = load_mapa_long(current_instrument_only=False)
    q = load_mapa_questions()
    d = raw[raw.school_year >= MAPA_FIRST_YEAR]

    # --- questionnaire continuity
    qy = raw.groupby("qid").agg(first=("school_year", "min"), last=("school_year", "max"), nyears=("school_year", "nunique"))
    old = qy[qy["last"] < MAPA_FIRST_YEAR]
    out["questionnaire"] = {
        "qids_total": int(len(qy)),
        "qids_only_2016_17": int(len(old)),
        "qids_current_instrument": int((qy["first"] >= MAPA_FIRST_YEAR).sum()),
        "qids_present_every_year_since_2017_18": int((qy[qy["first"] >= MAPA_FIRST_YEAR].nyears == d.school_year.nunique()).sum()),
        "scored_items_in_codebook": len(ITEMS),
        "scored_items_by_dim": pd.Series([v["dim"] for v in ITEMS.values()]).value_counts().to_dict(),
    }

    # --- waves
    waves = d.groupby(["red_izo", "school_year"]).agg(groups=("group", lambda s: "".join(sorted(set(s))))).reset_index()
    waves["wave_date"] = wave_date(waves.school_year)
    fw_path = MAPA_DIR / "mapa_fieldwork_dates.parquet"
    if fw_path.exists():
        fw = pd.read_parquet(fw_path)[["red_izo", "school_year", "median"]]
        waves = waves.merge(fw, on=["red_izo", "school_year"], how="left")
        waves["wave_date"] = waves["median"].fillna(waves.wave_date)
        waves = waves.drop(columns=["median"])
        out["wave_date_source"] = "observed median answer timestamp per school-wave; nominal 15 March fallback"
    resp = (d.groupby(["red_izo", "school_year", "group", "grade"], dropna=False).n_respondents.max()
            .groupby(level=["red_izo", "school_year", "group"]).sum().reset_index())  # respondents summed over grades
    out["waves"] = {
        "school_years": sorted(d.school_year.unique()),
        "schools": int(waves.red_izo.nunique()),
        "school_waves": int(len(waves)),
        "waves_per_year": waves.school_year.value_counts().sort_index().to_dict(),
        "waves_per_school": waves.groupby("red_izo").size().value_counts().sort_index().to_dict(),
        "school_waves_by_group": resp.group.value_counts().to_dict(),
        "median_respondents_by_group": resp.groupby("group").n_respondents.median().to_dict(),
        "share_school_wave_group_with_lt10_respondents": resp.groupby("group").n_respondents.apply(lambda s: float((s < 10).mean())).round(3).to_dict(),
        "waves_excluded_2016_17": int(raw[raw.school_year < MAPA_FIRST_YEAR].groupby(["red_izo", "school_year"]).ngroups),
    }

    # --- registry + analyses
    schools = pd.read_parquet(DATA_DIR / "schools.parquet")
    mapa_schools = set(waves.red_izo)
    reg = schools[schools.red_izo.isin(mapa_schools)]
    a = pd.read_parquet(DATA_DIR / "analyses.parquet", columns=["izo", "red_izo", "report_id", "target_school_type", "inspection_date", "version_id"])
    a = a[a.version_id == LEGACY_VERSION].copy()  # one row per report × school; linkage uses dates only
    a["insp_date"] = pd.to_datetime(a.inspection_date, errors="coerce")
    az = a[a.target_school_type == "základní"]
    out["registry"] = {
        "mapa_schools_in_registry": int(reg.red_izo.nunique()),
        "mapa_schools_missing_from_registry": sorted(mapa_schools - set(schools.red_izo.dropna())),
        "mapa_schools_with_zs_analysis": int(az[az.red_izo.isin(mapa_schools)].red_izo.nunique()),
        "entity_composition": reg.groupby("red_izo").school_type.apply(lambda x: "+".join(sorted(set(x)))).value_counts().to_dict(),
        "zs_inspections_per_mapa_school_median": float(az[az.red_izo.isin(mapa_schools)].groupby("red_izo").report_id.nunique().median()),
    }

    # --- cross-sectional linkage
    m = waves.merge(az[["red_izo", "izo", "report_id", "insp_date"]], on="red_izo", how="left")
    m["gap_d"] = (m.insp_date - m.wave_date.dt.normalize()).dt.days  # calendar days, as protocol 3.4 and paper1_analysis.link
    m["gap_y"] = m.gap_d / 365.25
    link = {}
    for w in WINDOWS:
        mm = m[m.gap_d.abs() <= 365 * w]
        per_group = {}
        for g in ["R", "U", "Z1", "Z2"]:
            wg = resp[resp.group == g][["red_izo", "school_year"]]
            per_group[g] = int(mm.merge(wg, on=["red_izo", "school_year"]).groupby(["red_izo", "school_year"]).ngroups)
        link[f"pm{w}y"] = {
            "school_waves_with_zs_inspection": int(mm.groupby(["red_izo", "school_year"]).ngroups),
            "schools": int(mm.red_izo.nunique()),
            "distinct_inspections": int(mm.report_id.nunique()),
            "school_waves_by_group": per_group,
        }
    near = m.dropna(subset=["gap_y"]).assign(ag=lambda x: x.gap_y.abs()).sort_values("ag").groupby(["red_izo", "school_year"]).head(1)
    link["nearest_inspection_gap_years"] = {k: round(float(v), 2) for k, v in near.ag.describe().to_dict().items()}
    out["linkage"] = link

    # --- change-on-change panel: two waves of the same school, each with a distinct inspection within ±W
    coc = {}
    for w in WINDOWS:
        mm = m[m.gap_d.abs() <= 365 * w].dropna(subset=["report_id"])
        pairs = 0
        schools_ok = set()
        for rz, g in mm.groupby("red_izo"):
            wv = g.groupby("school_year").report_id.apply(set)
            if len(wv) < 2:
                continue
            ys = list(wv.index)
            ok = False
            for i in range(len(ys)):
                for j in range(i + 1, len(ys)):
                    if wv.iloc[i] != wv.iloc[j] and (wv.iloc[i] - wv.iloc[j]) and (wv.iloc[j] - wv.iloc[i]):
                        ok = True
            if ok:
                schools_ok.add(rz)
                pairs += 1
        coc[f"pm{w}y"] = {"schools_with_two_waves_and_two_distinct_inspections": len(schools_ok)}
    coc["schools_with_ge2_waves"] = int((waves.groupby("red_izo").size() >= 2).sum())
    out["change_on_change"] = coc

    # --- selection profile vs all ZŠ
    zs = schools[schools.school_type == "základní"].copy()
    zs["mapa"] = zs.red_izo.isin(mapa_schools)
    prof = {"zs_in_registry": int(len(zs)), "zs_in_mapa": int(zs.mapa.sum())}
    for col in ["founder_type", "region"]:
        ct = pd.crosstab(zs[col], zs.mapa, normalize="columns").round(3)
        prof[col] = {k: {"all_zs": float(ct.loc[k, False]), "mapa": float(ct.loc[k, True])} for k in ct.index}
    prof["capacity_median"] = zs.groupby("mapa").capacity.median().to_dict()
    prof["capacity_median"] = {("mapa" if k else "all_zs"): float(v) for k, v in prof["capacity_median"].items()}
    out["selection"] = prof

    # --- halves (if frozen)
    hb = DATA_DIR / "half_b_red_izos.parquet"
    if hb.exists():
        b = set(pd.read_parquet(hb).red_izo)
        linked2 = set(m[m.gap_d.abs() <= 730].red_izo)
        out["halves"] = {
            "mapa_schools_half_A": int(len(mapa_schools - b)),
            "mapa_schools_half_B": int(len(mapa_schools & b)),
            "linked_pm2y_half_A": int(len(linked2 - b)),
            "linked_pm2y_half_B": int(len(linked2 & b)),
        }

    # --- item score matrix (Mapa only)
    scores = score_items(d)
    scores.to_parquet(MAPA_DIR / "mapa_item_scores.parquet", index=False)
    out["item_scores"] = {
        "rows": int(len(scores)),
        "school_waves_scored": int(scores.groupby(["red_izo", "school_year"]).ngroups),
        "items_scored": int(scores.qid.nunique()),
        "mean_score01_by_dim": scores.groupby("dim").score01.mean().round(3).to_dict(),
    }

    (MAPA_DIR / "mapa_linkage_audit_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
