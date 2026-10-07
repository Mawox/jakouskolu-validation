"""Tests of paper1_analysis.py. Every score here is hand-written or synthetic; no real LLM score is read."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import paper1_analysis as p1  # noqa: E402
from paper1_analysis import DIMENSIONS  # noqa: E402

D = DIMENSIONS[0]
HAS_DATA = (p1.DATA_DIR / "analyses.parquet").exists() and (p1.MAPA_DIR / "mapa_item_scores.parquet").exists()


def runs_of(cases: dict[str, list]) -> pd.DataFrame:
    """cases: analysis_id -> three run results; a number is a run score, None a null, "x" no valid response."""
    rows = [(a, f"r_{a}", k + 1, D, None if v in (None, "x") else float(v), v != "x") for a, vs in cases.items() for k, v in enumerate(vs)]
    return pd.DataFrame(rows, columns=p1.COLUMNS)


# --- score of an analysis --------------------------------------------------------

@pytest.mark.parametrize("vals, status, score", [
    ([3, 3, 4], "score", 10 / 3),
    ([1, 2, 4], "score", 7 / 3),
    ([2, 3, None], "score", 2.5),
    ([None, 4, 1], "score", 2.5),
    ([None, None, 3], "null", None),
    ([None, None, None], "null", None),
    (["x", 2, 3], "score", 2.5),
    ([2, "x", 3], "score", 2.5),
    ([2, 3, "x"], "score", 2.5),
    (["x", None, None], "null", None),
    ([None, "x", None], "null", None),
    ([None, None, "x"], "null", None),
    (["x", 3, None], "failure", None),
    ([3, "x", None], "failure", None),
    ([None, 3, "x"], "failure", None),
    (["x", "x", 3], "failure", None),
    (["x", "x", None], "failure", None),
    (["x", "x", "x"], "failure", None),
])
def test_analysis_score_rule(vals, status, score) -> None:
    s = p1.analysis_scores(runs_of({"a": vals})).iloc[0]
    assert s.status == status
    assert (pd.isna(s.score) if score is None else s.score == pytest.approx(score))
    assert s.n_scored + s.n_null + s.n_invalid == 3


def test_analysis_score_needs_three_runs() -> None:
    with pytest.raises(ValueError):
        p1.analysis_scores(runs_of({"a": [3, 3, 3]}).iloc[:2])


def test_school_score_rule() -> None:
    cases = {"a1": [3, 3, 3], "a2": [4, 4, 4],         # school S1: two scored analyses
             "b1": [2, 2, 2], "b2": [None, None, 4],   # S2: score and null
             "c1": [None, None, None], "c2": ["x", "x", 1],  # S3: null and failure
             "d1": [1, 2, "x"], "d2": ["x", 1, None],  # S4: score and failure
             "e1": [None, None, None]}                 # S5: null
    units = pd.DataFrame({"analysis_id": list(cases), "red_izo": ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4", "S5"]})
    s = p1.school_scores(p1.analysis_scores(runs_of(cases)), units).set_index("red_izo")
    assert s.status.to_dict() == {"S1": "score", "S2": "score", "S3": "failure", "S4": "score", "S5": "null"}
    assert s.score.dropna().to_dict() == {"S1": 3.5, "S2": 2.0, "S4": 1.5}


def full_runs(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame([(a, f"r_{a}", k, d, 2.0, True) for a in ids for k in (1, 2, 3) for d in DIMENSIONS], columns=p1.COLUMNS)


def test_check_runs() -> None:
    units = pd.DataFrame({"analysis_id": ["a", "b"], "report_id": ["r_a", "r_b"], "red_izo": ["S1", "S2"]})
    good = full_runs(["a", "b", "outside"])
    assert set(p1.check_runs(good, units).analysis_id) == {"a", "b"}
    bad = {
        "missing row": good.iloc[1:],
        "duplicate": pd.concat([good, good.iloc[:1]]),
        "score 5": good.assign(score=[5.0] + [2.0] * (len(good) - 1)),
        "score 2.5": good.assign(score=[2.5] + [2.0] * (len(good) - 1)),
        "invalid run with a score": good.assign(valid=[False] + [True] * (len(good) - 1)),
        "run 4": good.assign(run=good.run.replace(3, 4)),
        "wrong report": good.assign(report_id="r_a"),
        "valid differs within a run": good.assign(valid=[False] + [True] * (len(good) - 1), score=[None] + [2.0] * (len(good) - 1)),
    }
    for name, frame in bad.items():
        with pytest.raises(ValueError):
            p1.check_runs(frame, units)
            pytest.fail(name)


# --- linkage ---------------------------------------------------------------------

def test_linkage_tie_breaks() -> None:
    ts = pd.Timestamp
    waves = pd.DataFrame([
        ("A", "2019/20", ts("2020-03-01")), ("A", "2021/22", ts("2022-03-01 12:00")),
        ("B", "2019/20", ts("2020-01-01")), ("B", "2020/21", ts("2020-01-21")),
        ("C", "2019/20", ts("2020-01-01")),
        ("D", "2019/20", ts("2020-01-01")),
        ("E", "2019/20", ts("2020-01-01")),
        ("F", "2019/20", ts("2020-01-02")),
        ("G", "2019/20", ts("2020-01-02 10:00")),
    ], columns=["red_izo", "school_year", "wave_date"])
    insp = pd.DataFrame([
        ("A", "a_far", ts("2019-01-01")), ("A", "a_near", ts("2022-02-27")),   # smallest gap wins, whichever wave
        ("B", "b1", ts("2020-01-11")),                                         # 10 days from both waves: earlier wave
        ("C", "c_late", ts("2020-01-06")), ("C", "c_early", ts("2019-12-27")),  # same gap either side: earlier inspection
        ("D", "10_x", ts("2020-02-01")), ("D", "9_x", ts("2020-02-01")),       # same day: lower report id (as text)
        ("E", "e_out", ts("2022-01-01")),                                      # 731 days: not linked
        ("F", "f_in", ts("2018-01-02")),                                       # 730 days exactly: linked
        ("G", "g_in", ts("2018-01-02")),                                       # 730 calendar days, the time of day does not count: linked
    ], columns=["red_izo", "report_id", "insp_date"])
    pairs = p1.link(waves, insp)
    sel = p1.select_pairs(pairs).set_index("red_izo")
    assert sel.report_id.to_dict() == {"A": "a_near", "B": "b1", "C": "c_early", "D": "10_x", "F": "f_in", "G": "g_in"}
    assert sel.school_year.to_dict() == {"A": "2021/22", "B": "2019/20", "C": "2019/20", "D": "2019/20", "F": "2019/20", "G": "2019/20"}
    assert sel.loc["A", "gap_days"] == -2
    # the input order must not matter
    shuffled = p1.select_pairs(p1.link(waves.iloc[::-1], insp.sample(frac=1, random_state=3))).set_index("red_izo")
    assert shuffled.report_id.to_dict() == sel.report_id.to_dict() and shuffled.school_year.to_dict() == sel.school_year.to_dict()
    assert set(p1.select_pairs(pairs, window=5).red_izo) == {"A", "C"}


def test_era() -> None:
    d = pd.Series(pd.to_datetime(["2010-12-31", "2011-01-01", "2015-12-31", "2016-01-01", "2019-12-31", "2020-01-01"]))
    assert p1.era(d).tolist() == ["≤2010", "2011–15", "2011–15", "2016–19", "2016–19", "2020+"]


# --- H1, Holm and the gate ---------------------------------------------------------

def test_holm() -> None:
    assert p1.holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert p1.holm([0.012, 0.012, 0.012, 0.012]) == pytest.approx([0.048] * 4)
    assert p1.holm([0.0125, 0.5, None, 0.9]) == pytest.approx([0.05, 1.0, 1.0, 1.0])
    assert p1.holm([0.03, 0.03, 0.03, 0.001]) == pytest.approx([0.09, 0.09, 0.09, 0.004])


def test_correlation_matches_formula_and_protocol_table() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(541)
    y = 0.2 * x + rng.standard_normal(541)
    c = p1.correlation(x, y)
    r = np.corrcoef(x, y)[0, 1]
    z = np.arctanh(r) * np.sqrt(538)
    assert c["r"] == pytest.approx(r) and c["n"] == 541
    assert c["p"] == pytest.approx(1 - p1._N01.cdf(z))
    assert c["ci95"] == pytest.approx([np.tanh(np.arctanh(r) - 1.959964 / np.sqrt(538)), np.tanh(np.arctanh(r) + 1.959964 / np.sqrt(538))], abs=1e-6)
    assert c["ci95"][0] < r < c["ci95"][1]
    # critical r of protocol table 3 at α = 0.0125: 0.096 (N = 541) and 0.129 (N = 300)
    for n, crit in ((541, 0.096), (300, 0.129)):
        assert np.tanh(p1._N01.inv_cdf(1 - 0.0125) / np.sqrt(n - 3)) == pytest.approx(crit, abs=5e-4)
    assert p1.correlation(-x, y)["p"] > 0.99  # one-sided: a negative correlation is not significant
    assert p1.correlation(np.full(50, 3.0), y[:50]) == {"n": 50, "r": None, "ci95": None, "p": None}


def test_gate() -> None:
    def g(r, n, p):
        out = p1.gate(r, n, p)
        return out["h1_gate"], out["n_at_least_300"], out["significant"], out["r_at_least_floor"]
    assert g(0.1500001, 541, 0.001) == ("pass", True, True, True)
    assert g(0.15, 541, 0.05) == ("pass", True, True, True)
    assert g(0.1499999, 541, 0.001) == ("fail", True, True, False)
    assert g(0.30, 541, 0.0500001) == ("fail", True, False, True)
    assert g(0.30, 299, 0.0001) == ("fail", False, True, True)
    assert g(0.30, 300, 0.0001) == ("pass", True, True, True)
    assert g(None, 541, 1.0) == ("fail", True, False, False)
    assert g(-0.2, 541, 1.0) == ("fail", True, False, False)
    assert g(0.10, 200, 0.2) == ("fail", False, False, False)


def with_exact_r(score: np.ndarray, r: float, rng: np.random.Generator) -> np.ndarray:
    """A criterion whose sample correlation with ``score`` is exactly r."""
    s = (score - score.mean()) / score.std()
    e = rng.standard_normal(len(s))
    e -= e.mean() + (e @ s / len(s) - e.mean() * s.mean()) * s
    e /= e.std()
    return r * s + np.sqrt(1 - r**2) * e


def table_for(n: int, rs: dict[str, float], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in DIMENSIONS:
        score = rng.integers(3, 13, n) / 3
        crit = with_exact_r(score, rs[d], rng)
        rows.append(pd.DataFrame({"red_izo": [f"s{i}" for i in range(n)], "dimension": d, "score": score, "status": "score",
                                  "primary": crit, "parents": crit, "pupil_groups": "both", "era": "2020+"}))
    return pd.concat(rows, ignore_index=True)


NO_TWO_RUNS = pd.DataFrame(columns=["red_izo", "dimension", "third"])


def test_h1_gate_on_constructed_correlations() -> None:
    rs = dict(zip(DIMENSIONS, [0.151, 0.149, 0.30, 0.02]))
    res = p1.h1(table_for(500, rs), NO_TWO_RUNS)["primary"]
    for d in DIMENSIONS:
        assert res[d]["r"] == pytest.approx(rs[d], abs=1e-9) and res[d]["n"] == 500
    assert [res[d]["h1_gate"] for d in DIMENSIONS] == ["pass", "fail", "pass", "fail"]
    assert [(res[d]["n_at_least_300"], res[d]["significant"], res[d]["r_at_least_floor"]) for d in DIMENSIONS] == [
        (True, True, True), (True, True, False), (True, True, True), (True, False, False)]
    assert [res[d]["p_holm"] for d in DIMENSIONS] == pytest.approx(p1.holm([res[d]["p"] for d in DIMENSIONS]))
    small = p1.h1(table_for(299, rs), NO_TWO_RUNS)
    assert {(v["h1_gate"], v["n_at_least_300"]) for v in small["primary"].values()} == {("fail", False)}
    # parents: descriptive, the same correlation here because the table uses one criterion for both
    for d in DIMENSIONS:
        assert set(small["secondary_parents"][d]) == {"n", "r", "ci95"}
        assert small["secondary_parents"][d] == {k: small["primary"][d][k] for k in ("n", "r", "ci95")}


def test_pattern_block() -> None:
    a, b = DIMENSIONS[0], DIMENSIONS[1]
    score = {a: [1.0, 2.0, 3.0, 4.0, 2.5, np.nan], b: [4.0, 3.0, 2.0, 1.0, np.nan, 2.0]}
    crit = {a: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], b: [1.0, -1.0, 1.0, -1.0, 0.0, np.nan]}
    rows = [(f"s{i}", d, score.get(d, [3.0] * 6)[i], crit.get(d, [np.nan] * 6)[i]) for d in DIMENSIONS for i in range(6)]
    t = pd.DataFrame(rows, columns=["red_izo", "dimension", "score", "primary"])
    pat = p1.pattern(t)
    sc = pat["score_by_criterion"]
    assert set(sc) == set(DIMENSIONS) and all(set(v) == set(DIMENSIONS) for v in sc.values())
    assert all(set(c) == {"n", "r", "ci95"} for v in sc.values() for c in v.values())
    assert sc[a][a]["n"] == 5 and sc[a][a]["r"] == pytest.approx(np.corrcoef(score[a][:5], crit[a][:5])[0, 1])
    assert sc[a][b]["n"] == 5 and sc[a][b]["r"] == pytest.approx(np.corrcoef(score[a][:5], crit[b][:5])[0, 1])
    assert sc[b][a]["n"] == 5 and sc[b][a]["r"] == pytest.approx(np.corrcoef(np.delete(score[b], 4), np.delete(crit[a], 4))[0, 1])
    assert sc[b][b]["n"] == 4 and sc[b][b]["r"] == pytest.approx(np.corrcoef(score[b][:4], crit[b][:4])[0, 1])
    assert sc[a][DIMENSIONS[2]] == {"n": 0, "r": None, "ci95": None}
    assert sc[DIMENSIONS[2]][a] == {"n": 6, "r": None, "ci95": None}  # constant score
    among = pat["among_scores"]
    assert len(among) == 6 and all(set(v) == {"n", "r", "ci95"} for v in among.values())
    assert among[f"{a} × {b}"]["n"] == 4 and among[f"{a} × {b}"]["r"] == pytest.approx(-1.0)
    assert among[f"{a} × {DIMENSIONS[2]}"] == {"n": 5, "r": None, "ci95": None}
    # the diagonal equals the primary result
    full = p1.h1(table_for(400, dict(zip(DIMENSIONS, [0.3, 0.2, 0.1, 0.0]))), NO_TWO_RUNS)
    for d in DIMENSIONS:
        assert full["pattern"]["score_by_criterion"][d][d] == {k: full["primary"][d][k] for k in ("n", "r", "ci95")}


def test_h1_uses_schools_with_score_and_criterion_only() -> None:
    t = table_for(400, dict.fromkeys(DIMENSIONS, 0.3))
    first = t.dimension == D
    t.loc[first & (t.index % 400 < 30), ["score", "status"]] = [np.nan, "null"]
    t.loc[first & (t.index % 400).isin(range(30, 40)), ["score", "status"]] = [np.nan, "failure"]
    t.loc[first & (t.index % 400).isin(range(380, 400)), "primary"] = np.nan
    r = p1.h1(t, NO_TWO_RUNS)["primary"][D]
    assert (r["n"], r["nulls"], r["failures"], r["scores"], r["schools"], r["schools_with_criterion"]) == (340, 30, 10, 360, 400, 380)
    assert sum(r["distribution"].values()) == 360 and sum(r["distribution_analysed"].values()) == 340
    # a constant score cannot be estimated and does not pass; its test stays in the Holm family
    t.loc[first, "score"] = 3.0
    res = p1.h1(t, NO_TWO_RUNS)["primary"]
    assert res[D]["r"] is None and res[D]["p_holm"] == 1.0 and res[D]["h1_gate"] == "fail"
    assert [res[d]["p_holm"] for d in DIMENSIONS] == pytest.approx(p1.holm([None] + [res[d]["p"] for d in DIMENSIONS[1:]]))


# --- synthetic generator ------------------------------------------------------------

def fake_units(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    u = pd.DataFrame({"analysis_id": [f"a{i}" for i in range(n)], "report_id": [f"r{i}" for i in range(n)], "red_izo": [f"s{i}" for i in range(n)]})
    for d in DIMENSIONS:
        u[d] = rng.standard_normal(n)
    return u


def test_synthetic_generator_recovers_the_correlation() -> None:
    rho = dict(zip(DIMENSIONS, [0.0, 0.15, 0.3, 0.5]))
    u = fake_units(40_000, 1)
    runs, latent = p1.synthetic_runs(u, rho, seed=2)
    assert len(runs) == 40_000 * 12 and runs.score.dropna().isin([1, 2, 3, 4]).all()
    assert not runs.valid.all() and runs.score[~runs.valid].isna().all()
    s = p1.analysis_scores(p1.check_runs(runs, u))
    assert set(s.status) == {"score", "null", "failure"}
    sixths = set(np.round(s.score.dropna() * 6).astype(int))  # thirds and halves between 1 and 4
    assert sixths <= set(range(6, 25, 2)) | set(range(6, 25, 3)) and {8, 15} <= sixths
    wide = s.pivot(index="analysis_id", columns="dimension", values="score").reindex(u.analysis_id)
    for d in DIMENSIONS:
        lat = latent.set_index("analysis_id")[d].reindex(u.analysis_id).to_numpy()
        se = 1 / np.sqrt(len(u) - 3)
        assert abs(np.arctanh(np.corrcoef(lat, u[d])[0, 1]) - np.arctanh(rho[d])) < 4 * se
        ok = wide[d].notna().to_numpy()
        r_crit = p1.correlation(wide[d][ok], u[d][ok])["r"]
        r_lat = np.corrcoef(wide[d][ok], lat[ok])[0, 1]
        assert 0.8 < r_lat < 1
        # for a normal criterion, corr(score, criterion) = rho × corr(score, latent)
        assert abs(r_crit - rho[d] * r_lat) < 4 / np.sqrt(ok.sum())


def test_pipeline_on_synthetic_scores() -> None:
    u = fake_units(600, 5)
    runs, _ = p1.synthetic_runs(u, 0.4, seed=6)
    sample = pd.DataFrame({"red_izo": u.red_izo, "era": "2020+"})
    crit = u.melt(id_vars="red_izo", value_vars=DIMENSIONS, var_name="dimension", value_name="primary").assign(parents=np.nan, pupil_groups="both")
    res = p1.analyse(runs, sample, u, crit)
    a = p1.analysis_scores(runs)
    for d in DIMENSIONS:
        r, ad = res["primary"][d], a[a.dimension == d]
        assert r["h1_gate"] == "pass" and 0.2 < r["r"] < 0.5 and r["ci95"][0] < r["r"] < r["ci95"][1]
        assert (r["scores"], r["nulls"], r["failures"]) == tuple(int((ad.status == k).sum()) for k in ("score", "null", "failure"))
        assert r["n"] == r["scores"] and r["scores"] + r["nulls"] + r["failures"] == 600
        two = ad[(ad.status == "score") & (ad.n_scored == 2)]
        assert r["two_run_scores"]["third_run_null"]["linked_analyses"] == int((two.n_null == 1).sum()) > 0
        assert r["two_run_scores"]["third_run_invalid"]["linked_analyses"] == int((two.n_invalid == 1).sum()) > 0
        assert res["secondary_parents"][d] == {"n": 0, "r": None, "ci95": None}
        assert res["pattern"]["score_by_criterion"][d][d] == {k: r[k] for k in ("n", "r", "ci95")}
        assert all(res["pattern"]["score_by_criterion"][d][e]["n"] == r["n"] for e in DIMENSIONS)
    assert len(res["pattern"]["among_scores"]) == 6 and all(v["n"] > 500 and abs(v["r"]) < 0.2 for v in res["pattern"]["among_scores"].values())


# --- guard -----------------------------------------------------------------------

def test_guard(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p1, "REGISTERED_HASH_FILE", tmp_path / "registered_scores.sha256")
    runs, _ = p1.synthetic_runs(fake_units(20, 1), 0.2, seed=1)
    plain, marked = tmp_path / "plain.parquet", tmp_path / "marked.parquet"
    p1.write_scores(runs, plain, synthetic=False)
    p1.write_scores(runs, marked, synthetic=True)
    assert p1.check_scores_file(marked)["kind"] == "synthetic"
    with pytest.raises(p1.ScoresRefused, match="not marked synthetic"):
        p1.check_scores_file(plain)
    with pytest.raises(p1.ScoresRefused):
        p1.run(plain)  # refused before any data file is read
    p1.REGISTERED_HASH_FILE.write_text("0" * 64 + "\n")
    with pytest.raises(p1.ScoresRefused, match="differs from"):
        p1.check_scores_file(plain)
    p1.REGISTERED_HASH_FILE.write_text(p1.sha256(plain) + "  plain.parquet\n")
    assert p1.check_scores_file(plain)["kind"] == "registered"


def test_cli_refuses_unmarked_file(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(p1, "REGISTERED_HASH_FILE", tmp_path / "registered_scores.sha256")
    runs, _ = p1.synthetic_runs(fake_units(5, 1), 0.2, seed=1)
    p1.write_scores(runs, tmp_path / "plain.parquet", synthetic=False)
    monkeypatch.setattr(sys, "argv", ["paper1_analysis.py", str(tmp_path / "plain.parquet")])
    with pytest.raises(SystemExit) as e:
        p1.main()
    assert "Refused" in str(e.value.code) and capsys.readouterr().out == ""


# --- real linkage and survey data, synthetic scores -----------------------------------

@pytest.mark.skipif(not HAS_DATA, reason="data files not present")
def test_real_sample_with_synthetic_scores(tmp_path) -> None:
    sample, units, counts = p1.linked_sample()
    assert counts["schools_under_rule"] == 558 and counts["linked_school_waves"] == 714 and counts["distinct_inspections"] == 665
    assert sample.red_izo.is_unique and len(units) >= len(sample)
    crit = p1.criteria(sample)
    assert len(crit) == 4 * len(sample)
    z = crit[crit.dimension != p1.DIMS["LM"]]
    assert z[z.primary.notna()].pupil_groups.notna().all() and z[z.primary.isna()].pupil_groups.isna().all()
    # the pupil composite is the mean of the two pupil scales, or the one that exists
    sc = p1.scale_scores(pd.read_parquet(p1.MAPA_DIR / "mapa_item_scores.parquet"))
    wide = sc[sc.group.isin(["Z1", "Z2", "Z"])].pivot(index=["red_izo", "school_year", "dim"], columns="group", values="scale_z")
    assert np.allclose(wide["Z"], wide[["Z1", "Z2"]].mean(axis=1))
    path = tmp_path / "synthetic.parquet"
    p1.write_synthetic(path, rho=0.35, seed=11)
    res = p1.run(path)
    assert res["scores_file"]["kind"] == "synthetic"
    for d in DIMENSIONS:
        r = res["primary"][d]
        assert r["schools"] == len(sample) and r["n"] <= res["criteria"][d]["primary"]
        # the cut into four values and run noise attenuate 0.35 by a factor of about 0.9
        assert abs(np.arctanh(r["r"]) - np.arctanh(0.35 * 0.9)) < 4 / np.sqrt(r["n"] - 3)
        assert r["h1_gate"] == "pass" and r["n_at_least_300"] and r["significant"] and r["r_at_least_floor"]
        assert set(res["secondary_parents"][d]) == {"n", "r", "ci95"} and res["secondary_parents"][d]["n"] > 400
        assert res["pattern"]["score_by_criterion"][d][d] == {k: r[k] for k in ("n", "r", "ci95")}
    assert len(res["pattern"]["among_scores"]) == 6


def test_benchmark_is_survey_only() -> None:
    crit = pd.DataFrame({"red_izo": list("abcdef") * 2, "dimension": [DIMENSIONS[0]] * 6 + [DIMENSIONS[1]] * 6,
                         "primary": [1.0, 2.0, 3.0, 4.0, 5.0, None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                         "parents": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0, None, None, None, None, None, None]})
    b = p1.benchmark(crit)
    assert b[DIMENSIONS[0]]["n"] == 5 and b[DIMENSIONS[0]]["r"] == pytest.approx(np.corrcoef([1, 2, 3, 4, 5], [2, 1, 4, 3, 6])[0, 1])
    assert b[DIMENSIONS[1]] == {"n": 0, "r": None, "ci95": None}
