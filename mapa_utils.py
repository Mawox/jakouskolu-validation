"""
Mapa školy (Scio parent/pupil/teacher survey) — loader and item scoring.

Raw export (two header-less CSVs, ';'-separated, 'NULL' = missing) lives in
data/mapa/. Aggregated answer counts per school year × REDIZO × respondent
group × grade × question × option. Respondent groups: R = parents,
U = teachers, Z1 = pupils grades 1–5, Z2 = pupils grades 6–9.

Score direction: every scored item is oriented so that HIGHER = BETTER and
rescaled to 0–1 (``score01``). "Nedovedu posoudit"-type options are dropped
from the denominator. Multi-select items are scored as the share of
respondents choosing the reference option (``count / n_respondents``).

The 2016/17 wave used a different questionnaire (174 of 177 question ids are
unique to that year, only 3 texts recur later) and is excluded from
``MAPA_FIRST_YEAR`` onward analyses.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MAPA_DIR = DATA_DIR / "mapa"
MAPA_FIRST_YEAR = "2017/18"  # first wave of the current questionnaire
GROUP_LABELS = {"R": "rodiče", "U": "učitelé", "Z1": "žáci 1.–5. r.", "Z2": "žáci 6.–9. r."}


# --- raw CSV -> parquet -------------------------------------------------------

def build_mapa_parquets() -> None:
    q = pd.read_csv(MAPA_DIR / "mapa_skoly_otazky.csv", sep=";", header=None,
                    encoding="utf-8-sig", dtype=str, na_values=["NULL"], keep_default_na=False)
    q.columns = ["qid", "battery", "text"] + [chr(65 + i) for i in range(q.shape[1] - 3)]
    q.to_parquet(MAPA_DIR / "mapa_questions.parquet", index=False)

    d = pd.read_csv(MAPA_DIR / "mapa_skoly_data.csv", sep=";", header=None,
                    encoding="utf-8-sig", dtype=str, na_values=["NULL"], keep_default_na=False)
    d.columns = ["school_year", "red_izo", "group", "grade", "qid", "n_respondents", "answer", "count"]
    d["group"] = d["group"].str.strip()
    for c in ["grade", "n_respondents", "count"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d.to_parquet(MAPA_DIR / "mapa_long.parquet", index=False)


def load_mapa_long(current_instrument_only: bool = True) -> pd.DataFrame:
    d = pd.read_parquet(MAPA_DIR / "mapa_long.parquet")
    if current_instrument_only:
        d = d[d.school_year >= MAPA_FIRST_YEAR]
    return d


def load_mapa_questions() -> pd.DataFrame:
    return pd.read_parquet(MAPA_DIR / "mapa_questions.parquet")


def wave_date(school_year: pd.Series) -> pd.Series:
    """Nominal wave date: 15 March of the second calendar year of the school year.

    Median fieldwork date across school-waves falls in February–March; use
    ``fieldwork_dates()`` for the observed per-school-wave median where available.
    """
    return pd.to_datetime(school_year.str[:4].astype(int).add(1).astype(str) + "-03-15")


def fieldwork_dates() -> pd.DataFrame:
    """Observed fieldwork dates per school-wave (min/median/max of answer timestamps).

    Built by _mapa_reliability.py from the respondent-level export.
    """
    return pd.read_parquet(MAPA_DIR / "mapa_fieldwork_dates.parquet")


# --- item codebook -------------------------------------------------------------
# kind: "desc"  = option A is best, letters descend (A=k … last=1); excluded letters dropped
#       "asc"   = option A is worst, letters ascend (A=1 … last=k)
#       "share" = multi-select or binary; score = share of respondents choosing `ref`
#       "share_not" = 1 - share choosing `ref`
# dim: DRAFT a-priori mapping to the AI rubric dimension (TQ teaching_quality,
#      LM leadership_management, SC school_climate, SS student_support, GL global).
#      The mapping is locked in the preregistration (plan phase 1.2), not here.

ITEMS: dict[str, dict] = {
    # ---- parents (R) ----
    "166272": dict(group="R", kind="desc", k=4, exclude="E", dim="TQ", label="R: pedagogické kvality vyučujících"),
    "166273": dict(group="R", kind="desc", k=4, exclude="E", dim="TQ", label="R: celková úroveň výuky"),
    "185609": dict(group="R", kind="share", ref="A", dim="TQ", label="R: žádný předmět s výhradami"),
    "185610": dict(group="R", kind="share_not", ref="A", dim="TQ", label="R: má předmět s obzvlášť dobrou výukou"),
    "166271": dict(group="R", kind="desc", k=4, exclude="E", dim="LM", label="R: spokojenost s vedením školy"),
    "166267": dict(group="R", kind="desc", k=4, exclude="E", dim="LM", label="R: předávání informací rodičům"),
    "166268": dict(group="R", kind="desc", k=4, exclude="E", dim="SC", label="R: vztahy mezi žáky"),
    "166270": dict(group="R", kind="desc", k=4, exclude="E", dim="SS", label="R: komunikace s třídním"),
    "166280": dict(group="R", kind="desc", k=4, exclude="E", dim="GL", label="R: doporučil by školu"),
    "166281": dict(group="R", kind="desc", k=5, exclude="", dim="GL", label="R: známka škole (1–5)"),
    # ---- teachers (U) ----
    "166208": dict(group="U", kind="desc", k=4, exclude="", dim="LM", label="U: přístup vedení"),
    "166214": dict(group="U", kind="desc", k=4, exclude="EF", dim="LM", label="U: opora vedení – problémy se žáky"),
    "166215": dict(group="U", kind="desc", k=4, exclude="EF", dim="LM", label="U: opora vedení – psychická zátěž"),
    "166216": dict(group="U", kind="desc", k=4, exclude="EF", dim="LM", label="U: opora vedení – vztahy v pedag. sboru"),
    "166217": dict(group="U", kind="desc", k=4, exclude="EF", dim="LM", label="U: opora vedení – konflikty s rodiči"),
    "166218": dict(group="U", kind="desc", k=4, exclude="", dim="LM", label="U: vedení oceňuje práci"),
    "166207": dict(group="U", kind="desc", k=4, exclude="", dim="LM", label="U: systém administrativy"),
    "166202": dict(group="U", kind="desc", k=4, exclude="", dim="LM", label="U: příležitosti k profesnímu rozvoji"),
    "166220": dict(group="U", kind="desc", k=4, exclude="", dim="LM", label="U: spokojenost se zaměstnáním"),
    "166219": dict(group="U", kind="desc", k=4, exclude="", dim="SC", label="U: příjemná atmosféra"),
    "166209": dict(group="U", kind="desc", k=4, exclude="", dim="SC", label="U: vztahy v učitelském sboru"),
    "166593": dict(group="U", kind="desc", k=4, exclude="", dim="SC", label="U: chování žáků vůči učitelům"),
    "175640": dict(group="U", kind="share", ref="B", dim="SC", label="U: někdo umí řešit šikanu"),
    "166206": dict(group="U", kind="desc", k=4, exclude="", dim="TQ", label="U: připravenost žáků na výuku"),
    # ---- pupils grades 1–5 (Z1) ----
    "166071": dict(group="Z1", kind="share", ref="A", dim="TQ", label="Z1: baví mě, co se učíme"),
    "166525": dict(group="Z1", kind="desc", k=3, exclude="", dim="TQ", label="Z1: mám rád učitele"),
    "166078": dict(group="Z1", kind="desc", k=3, exclude="", dim="TQ", label="Z1: učitel nechá říct názor"),
    "166079": dict(group="Z1", kind="desc", k=3, exclude="", dim="SS", label="Z1: učitel pomůže"),
    "207930": dict(group="Z1", kind="share", ref="A", dim="SS", label="Z1: rád za třídního"),
    "166076": dict(group="Z1", kind="desc", k=3, exclude="", dim="SC", label="Z1: kamarádi ve třídě"),
    "166077": dict(group="Z1", kind="desc", k=3, exclude="", dim="SC", label="Z1: chodím rád do školy"),
    "166523": dict(group="Z1", kind="desc", k=3, exclude="", dim="SC", label="Z1: bez strachu z dětí"),
    "166073": dict(group="Z1", kind="desc", k=3, exclude="", dim="SC", label="Z1: bez strachu z dospělých"),
    "207928": dict(group="Z1", kind="desc", k=3, exclude="", dim="SC", label="Z1: bez posmívání/ubližování"),
    "166072": dict(group="Z1", kind="desc", k=5, exclude="", dim="GL", label="Z1: známka škole (1–5)"),
    # ---- pupils grades 6–9 (Z2) ----
    "166131": dict(group="Z2", kind="desc", k=4, exclude="E", dim="TQ", label="Z2: vyučující jsou dobří pedagogové"),
    "166138": dict(group="Z2", kind="desc", k=4, exclude="E", dim="TQ", label="Z2: výuka je kvalitní"),
    "166144": dict(group="Z2", kind="asc", k=4, exclude="E", dim="TQ", label="Z2: netřeba zlepšit pedag. schopnosti"),
    "166151": dict(group="Z2", kind="asc", k=4, exclude="E", dim="TQ", label="Z2: netřeba zlepšit kvalitu výuky"),
    "166156": dict(group="Z2", kind="share", ref="A", dim="TQ", label="Z2: žádný nudný předmět"),
    "185606": dict(group="Z2", kind="share_not", ref="A", dim="TQ", label="Z2: má záživný předmět"),
    "166124": dict(group="Z2", kind="desc", k=4, exclude="", dim="TQ", label="Z2: ve třídě se dobře učí (klid)"),
    "166130": dict(group="Z2", kind="desc", k=4, exclude="E", dim="SC", label="Z2: vztahy mezi žáky přátelské"),
    "166143": dict(group="Z2", kind="asc", k=4, exclude="E", dim="SC", label="Z2: netřeba zlepšit vztahy"),
    "166120": dict(group="Z2", kind="desc", k=4, exclude="", dim="SC", label="Z2: bez slovního ubližování"),
    "166121": dict(group="Z2", kind="desc", k=4, exclude="", dim="SC", label="Z2: bez fyzického ubližování"),
    "207995": dict(group="Z2", kind="desc", k=4, exclude="", dim="SC", label="Z2: bez kyberšikany"),
    "166119": dict(group="Z2", kind="share", ref="A", dim="SC", label="Z2: nesetkal se se šikanou"),
    "175079": dict(group="Z2", kind="desc", k=4, exclude="", dim="SC", label="Z2: cítím se ve třídě dobře"),
    "175078": dict(group="Z2", kind="desc", k=4, exclude="", dim="SC", label="Z2: chodím rád do školy"),
    "166118": dict(group="Z2", kind="desc", k=3, exclude="", dim="SC", label="Z2: někdo umí řešit šikanu"),
    "166123": dict(group="Z2", kind="desc", k=3, exclude="", dim="SS", label="Z2: důvěryhodný dospělý ve škole"),
    "185607": dict(group="Z2", kind="share_not", ref="A", dim="SS", label="Z2: má vstřícný předmět"),
    "166152": dict(group="Z2", kind="share", ref="A", dim="SS", label="Z2: žádný nepříjemný učitel"),
    "207992": dict(group="Z2", kind="desc", k=4, exclude="", dim="SS", label="Z2: rád za třídního"),
}


def score_items(d: pd.DataFrame | None = None) -> pd.DataFrame:
    """School-wave × item scores (0–1, higher = better) for every item in ITEMS.

    Returns columns: red_izo, school_year, group, qid, dim, label, n_valid, score01.
    Grades are pooled within the respondent group.
    """
    if d is None:
        d = load_mapa_long()
    d = d[d.qid.isin(ITEMS)].copy()
    rows = []
    for (rz, sy, g, qid), sub in d.groupby(["red_izo", "school_year", "group", "qid"]):
        spec = ITEMS[qid]
        if spec["kind"] in ("share", "share_not"):
            n = sub.groupby("grade", dropna=False).n_respondents.first().sum()
            hit = sub.loc[sub.answer == spec["ref"], "count"].sum()
            if n <= 0:
                continue
            s = hit / n
            if spec["kind"] == "share_not":
                s = 1 - s
            rows.append((rz, sy, g, qid, spec["dim"], spec["label"], int(n), float(s)))
            continue
        sub = sub[~sub.answer.isin(list(spec["exclude"]))]
        sub = sub[sub.answer.str.len() == 1]
        idx = sub.answer.map(lambda a: ord(a) - 65)
        k = spec["k"]
        val = (k - idx) if spec["kind"] == "desc" else (idx + 1)
        val = val.where((val >= 1) & (val <= k))
        w = sub["count"].where(val.notna())
        n = w.sum()
        if not n or n <= 0:
            continue
        mean = (val * w).sum() / n
        rows.append((rz, sy, g, qid, spec["dim"], spec["label"], int(n), float((mean - 1) / (k - 1))))
    return pd.DataFrame(rows, columns=["red_izo", "school_year", "group", "qid", "dim", "label", "n_valid", "score01"])


DROPPED_ITEMS = {"185606", "185607", "185610"}  # subject-list items, ICC(1) < 0.01
SINGLE_ITEMS = {"175640", "166593"}  # teachers on pupils (bullying handled, behaviour): reported singly; the teacher climate scale covers staff climate only
DIMS = {"TQ": "teaching_quality", "LM": "leadership_management", "SC": "school_climate", "SS": "student_support"}


def scale_role(group: str, dim: str) -> str:
    """A-priori role of a respondent group's scale for a rubric dimension (rubric-v2-mapa-mapping.md)."""
    if dim == "LM":
        return {"U": "primary", "R": "secondary"}.get(group, "none")
    return {"Z": "primary", "Z1": "primary", "Z2": "primary", "R": "secondary", "U": "discriminant"}[group]


def scale_scores(items: pd.DataFrame | None = None, min_n: int = 5) -> pd.DataFrame:
    """School-wave × respondent group × dimension scale scores.

    Items are z-standardised within item × school year across schools and then averaged,
    because the item set varies by year and items differ in base rate; scale scores are then
    re-standardised to unit variance per group × dimension. A scale score needs valid school
    means for at least half of the scale's items administered in that school year. Group "Z"
    is the pupil composite: mean of the Z1 and Z2 scale scores that exist for the school-wave.
    Returns columns: red_izo, school_year, group, dim, scale_z, n_items, role.
    """
    d = score_items() if items is None else items
    d = d[d.dim.isin(DIMS) & ~d.qid.astype(str).isin(DROPPED_ITEMS | SINGLE_ITEMS) & (d.n_valid >= min_n)].copy()
    d["z"] = d.groupby(["qid", "school_year"]).score01.transform(lambda x: (x - x.mean()) / x.std())
    key = ["red_izo", "school_year", "group", "dim"]
    out = d.groupby(key).z.agg(scale_z="mean", n_items="count").reset_index()
    asked = d.groupby(["school_year", "group", "dim"]).qid.nunique().rename("n_asked").reset_index()
    out = out.merge(asked, on=["school_year", "group", "dim"])
    out = out[2 * out.n_items >= out.n_asked].drop(columns="n_asked")
    out["scale_z"] = out.groupby(["group", "dim"]).scale_z.transform(lambda x: (x - x.mean()) / x.std())
    pupils = out[out.group.isin(["Z1", "Z2"])].groupby(["red_izo", "school_year", "dim"]).agg(scale_z=("scale_z", "mean"), n_items=("n_items", "sum")).reset_index()
    pupils["group"] = "Z"
    out = pd.concat([out, pupils[key + ["scale_z", "n_items"]]], ignore_index=True)
    out["role"] = [scale_role(g, dm) for g, dm in zip(out.group, out.dim)]
    return out


# --- respondent-level export (mapa_skoly_data_XML.csv) --------------------------
# One row per respondent: school_year; red_izo; group; grade; respondent_id; XML of
# <ItemAnswer id="CIS-00166250" responded="2018-01-30T15:23:53"><Choice key="A" /></ItemAnswer>.
# Free-text answers (<OpenItem answerValue="...">) are NOT kept — only a flag that a
# non-empty text was entered. Output: data/mapa/mapa_respondents.parquet (long).

_ITEM_RE = None


def build_respondent_parquet(chunk_rows: int = 20_000) -> None:
    import csv
    import re
    import sys

    import pyarrow as pa
    import pyarrow.parquet as pq

    csv.field_size_limit(sys.maxsize)
    item_re = re.compile(r'<ItemAnswer id="CIS-0*(\d+)" responded="([^"]*)">(.*?)</ItemAnswer>', re.S)
    choice_re = re.compile(r'<Choice key="([^"]*)"')
    open_re = re.compile(r'<OpenItem [^>]*answerValue="([^"]*)"')
    schema = pa.schema([
        ("school_year", pa.string()), ("red_izo", pa.string()), ("group", pa.string()),
        ("grade", pa.int16()), ("respondent_id", pa.string()), ("qid", pa.string()),
        ("choice", pa.string()), ("responded", pa.string()), ("has_text", pa.bool_()),
    ])
    writer = pq.ParquetWriter(MAPA_DIR / "mapa_respondents.parquet", schema, compression="zstd")
    cols = {name: [] for name in schema.names}
    n_resp = 0

    def flush():
        if cols["qid"]:
            writer.write_table(pa.table({k: pa.array(v, type=schema.field(k).type) for k, v in cols.items()}))
            for v in cols.values():
                v.clear()

    with open(MAPA_DIR / "mapa_skoly_data_XML.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh, delimiter=";"):
            if len(row) < 6:
                continue
            sy, rz, grp, grade, rid, xml = row[0], row[1], row[2].strip(), row[3], row[4], row[5]
            grade_i = int(grade) if grade not in ("", "NULL") else -1
            n_resp += 1
            for qid, ts, body in item_re.findall(xml):
                choices = choice_re.findall(body)
                texts = open_re.findall(body)
                has_text = any(t.strip() for t in texts)
                if not choices:
                    choices = [None]
                for ch in choices:
                    cols["school_year"].append(sy); cols["red_izo"].append(rz); cols["group"].append(grp)
                    cols["grade"].append(grade_i); cols["respondent_id"].append(rid); cols["qid"].append(qid)
                    cols["choice"].append(ch); cols["responded"].append(ts[:19] or None); cols["has_text"].append(has_text)
            if n_resp % chunk_rows == 0:
                flush()
                print(f"{n_resp} respondents", flush=True)
    flush()
    writer.close()
    print(f"done: {n_resp} respondents")


def load_mapa_respondents(current_instrument_only: bool = True):
    import pandas as pd
    d = pd.read_parquet(MAPA_DIR / "mapa_respondents.parquet")
    if current_instrument_only:
        d = d[d.school_year >= MAPA_FIRST_YEAR]
    return d
