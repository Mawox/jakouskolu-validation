"""
Rubric v2 (research plan phase 1.2): four dimensions of jakouskolu.cz, defined from
the children's perspective. Construct: "what the school is like for children, as
read from the ČŠI inspection report". Single source of truth for definitions,
scale anchors and the null rule; the prompt and the human-readable document are
generated from this module. Anchor excerpts come from data/rubric_v2_anchors.json.

    python rubric_v2.py        # prints PROMPT_HASH, rewrites rubric-v2-anchors.md
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ANCHORS_PATH = Path(__file__).parent / "data" / "rubric_v2_anchors.json"
ANCHORS_MD = Path(__file__).parent / "rubric-v2-anchors.md"

CONSTRUCT = (
    "Hodnotíme, jaká je škola pro děti, podle toho, co o ní napsala Česká školní inspekce. "
    "Nehodnotíme administrativu, soulad dokumentů, kvalifikaci ani výsledky testů, pokud zpráva "
    "nepopisuje jejich dopad na děti."
)

DIMENSIONS: dict[str, dict[str, str | list[str]]] = {
    "teaching_quality": {
        "label": "Jak se děti učí",
        "definition": (
            "Jak výuka vypadá z pohledu dítěte: jaké metody učitelé používají a zda sedí těmto dětem, "
            "nakolik jsou děti aktivní a zapojené, zda učitelé přizpůsobují výuku jednotlivcům, "
            "zda děti dostávají zpětnou vazbu, která jim pomáhá se učit, a zda se učí věci, které dávají smysl."
        ),
        "counts": [
            "metody a jejich vhodnost pro tyto děti", "aktivita a zapojení dětí v hodinách",
            "individualizace a diferenciace", "zpětná vazba a hodnocení jako podpora učení",
            "smysluplnost obsahu, propojení s životem",
        ],
        "excluded": ["soulad ŠVP s RVP", "kvalifikace učitelů", "výsledky testů a soutěží", "dokumentace výuky"],
    },
    "leadership_management": {
        "label": "Jak vedení ovlivňuje život dětí",
        "definition": (
            "Zda vedení školy ví, jak se ve třídách učí a jak se dětem daří, a zda s tím něco dělá: "
            "chodí do hodin a výsledky promítá do praxe, podporuje učitele v práci s dětmi, má představu, "
            "kam školu vede, a řeší problémy, které děti zažívají. "
            "Výjimka jen pro tuto oblast, která má přednost před škálou a pravidly pro důkazy: když vedení pravidelně "
            "hospituje a na zjištění navazuje zpětnou vazbou učitelům nebo opatřeními, stačí to na stupeň 3 i bez "
            "popsaného přínosu pro děti, pokud jsou výhrady v oblasti nejvýše dílčí. "
            "Stejnou přednost má pravidlo pro stupeň 4 v této oblasti: stačí jedno zjištění místo dvou, pokud vedení výuku nebo "
            "výsledky dětí sleduje soustavně, výhrady k práci vedení jsou nejvýše dílčí a zpráva aspoň u jednoho opatření vedení "
            "popisuje, co se díky němu u dětí nebo v hodinách změnilo (například klesl počet neprospívajících, snížila se absence, "
            "zlepšily se výsledky, změnila se výuka předmětu, učitelé začali v hodinách používat jiné metody). Takové zjištění "
            "hledej v celé zprávě, i v částech o výsledcích vzdělávání. Obecná věta, že se něco „pozitivně odráží v kvalitě "
            "vzdělávání“, na stupeň 4 nestačí."
        ),
        "counts": [
            "vedení zná výuku z první ruky a reaguje", "podpora a rozvoj učitelů s dopadem na děti",
            "koncepce školy viditelná v praxi", "řešení problémů a stabilita, kterou děti zažívají",
        ],
        "excluded": ["účetnictví a hospodaření", "formální dokumentace", "organizační struktura bez dopadu na děti"],
    },
    "school_climate": {
        "label": "Klima pro děti",
        "definition": (
            "Jak se děti ve škole cítí: zda jsou v bezpečí, jak se k sobě chovají děti navzájem, "
            "jak s nimi jednají dospělí, zda se děti mohou vyjádřit a zda škola předchází strachu, posmívání "
            "a šikaně a řeší je."
        ),
        "counts": [
            "vztahy mezi dětmi", "vztah dospělých k dětem, respekt", "hlas dětí, možnost vyjádřit názor",
            "prevence a řešení šikany a rizikového chování", "bezpečí dětí, pokud zpráva popisuje skutečný dopad (například opakovaná zranění)",
        ],
        "excluded": [
            "stav budovy, vybavení a technická bezpečnost (dohled, revize, BOZP), pokud zpráva popisuje jen riziko, a ne dopad na děti",
            "adaptace nových dětí a přechody (patří do podpory)", "atmosféra v učitelském sboru",
            "vztahy s rodiči a zřizovatelem bez popsaného dopadu na děti",
        ],
    },
    "student_support": {
        "label": "Komu a jak škola pomáhá",
        "definition": (
            "Zda škola rozpozná děti, které něco potřebují, a zda jim opravdu pomáhá: děti s obtížemi v učení, "
            "nadané, cizinci, děti v těžké situaci; jak fungují poradenství a adaptace při přechodech, "
            "a jak škola spolupracuje s rodiči kvůli konkrétnímu dítěti."
        ),
        "counts": [
            "rozpoznání potřeb a reálná pomoc", "práce s dětmi s obtížemi i nadanými",
            "poradenství pro děti a rodiče v praxi", "adaptace a přechody", "spolupráce s rodiči kvůli dítěti",
        ],
        "excluded": [
            "výčet dokumentů a pozic bez popsané praxe", "počty podpůrných opatření bez popsaného dopadu",
            "běžná diferenciace výuky pro celou třídu (patří do výuky)",
            "prevence a řešení šikany a rizikového chování (patří do klimatu)",
        ],
    },
}

SCALE: dict[int, str] = {
    1: "Převažují doložené nedostatky: konkrétní slabiny v oblasti převažují nad pozitivy rozsahem nebo závažností. Dopad na děti nemusí být popsán.",
    2: "Běžný standard: zpráva oblast hodnotí jen obecně bez doložených přínosů pro děti, nebo jsou doložená pozitiva a doložené nedostatky zhruba v rovnováze (pozitiva s podstatnými výhradami).",
    3: "Převažují doložená pozitiva: zpráva popisuje konkrétní praxi, která dětem prospívá, a výhrady jsou jen dílčí.",
    4: "Výrazně nadstandardní praxe: aspoň dvě věcně různá konkrétní zjištění s popsaným přínosem pro děti, uplatňovaná soustavně. Slova jako „vynikající“ nebo „příkladný“ sama nestačí.",
}

NULL_RULE = (
    "Pokud zpráva o oblasti mlčí nebo k ní uvádí jen výčet dokumentů, funkcí a programů, skóre je null. "
    "Null není známka; je to informace, že zpráva o oblasti mlčí. Neodhaduj z toho, co ve zprávě chybí: "
    "to, že zpráva nepopisuje nápravu nebo pomoc, neznamená, že neexistují."
)

PROCEDURE = [
    "Najdi v textu zprávy konkrétní pozorování k dané oblasti.",
    "Porovnej je s definicí oblasti a se stupni škály.",
    "Přiřaď skóre podle toho, co v pasážích o oblasti převažuje, nebo null, pokud zpráva o oblasti mlčí. Většina zpráv uvádí pozitiva i výhrady; to samo neznamená stupeň 2: rozhoduje, zda jsou výhrady dílčí (3), podstatné (2), nebo převažují (1).",
    "Uveď jednu nebo dvě doslovné citace ze zprávy (každá do 25 slov), které skóre dokládají; pokud žádnou nenajdeš, evidence je prázdná a skóre null.",
]

EVIDENCE_RULE = (
    "Důkaz je doslovná citace ze zprávy k dané oblasti. Pro stupně 1, 3 a 4 musí popisovat konkrétní pozorování nebo zjištění: "
    "co se ve škole děje a s kým, u stupňů 3 a 4 i co z toho děti mají. Zkušenost dítěte se počítá i bez měřeného výsledku, například využitá "
    "možnost říct názor nebo poskytnutá úprava výuky. Obecné hodnocení bez popsané praxe („klima je příznivé“, „podpora je zajištěna“) "
    "a popis procesů, u kterých zpráva neříká, co z nich děti nebo učitelé ve třídách mají (škola plánuje, zjišťuje, vyhodnocuje, schází se), "
    "stačí nejvýše na stupeň 2. Důkazem není výčet dokumentů, funkcí a programů ani nic z toho, co se v oblasti nepočítá. "
    "Skóre a důkazy spolu drží: skóre 1–4 má vždy jednu nebo dvě citace, null má vždy prázdný seznam citací."
)

QUOTE_RULE = (
    "Citace kopíruj ze zprávy přesně, znak po znaku, bez parafráze a bez uvozovek. Každá citace má nejvýše 25 slov; "
    "vynechané místo uvnitř citace označ „…“. Necituj příklady z této instrukce, jsou z jiných zpráv."
)

PARENT_RULES = [
    "3–5 vět pro rodiče, kteří školu vybírají: jaká je škola pro děti podle této inspekční zprávy.",
    "Piš prostě a konkrétně, v činném rodě, bez odborných pojmů inspekce (ŠVP, RVP, SVP, ŠPP, hospitace); když je potřebuješ, vysvětli je běžnými slovy.",
    "Opírej se jen o to, co zpráva popisuje; neuváděj skóre ani čísla oblastí. Pokud zpráva o dětech říká málo, napiš to.",
]


def load_anchors(path: Path = ANCHORS_PATH) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _dimension_block(key: str, anchors: list[dict]) -> str:
    d = DIMENSIONS[key]
    lines = [f"### {key}: {d['label']}", str(d["definition"]),
             f"Počítá se: {'; '.join(d['counts'])}.", f"Nepočítá se: {'; '.join(d['excluded'])}."]
    own = [a for a in anchors if a["dimension"] == key]
    if own:
        lines.append("Příklady z jiných zpráv (ilustrují stupně, nejsou důkazem pro hodnocenou zprávu a jsou delší než citace, které vracíš):")
        for score in [*SCALE, None]:
            for a in (a for a in own if a["score"] == score):
                lines.append(f"- {'null' if score is None else score}: „{a['text']}“ – {a['why']}")
    return "\n".join(lines)


def build_prompt(anchors: list[dict] | None = None) -> str:
    anchors = load_anchors() if anchors is None else anchors
    dims = "\n\n".join(_dimension_block(k, anchors) for k in DIMENSIONS)
    scale = "\n".join(f"- {k}: {v}" for k, v in SCALE.items())
    steps = "\n".join(f"{i}. {x}" for i, x in enumerate(PROCEDURE, 1))
    parent = "\n".join(f"  - {x}" for x in PARENT_RULES)
    return f"""Hodnotíš inspekční zprávu České školní inspekce pro výzkum. Text zprávy dostaneš ve zprávě uživatele.

## Co hodnotíme
{CONSTRUCT}

## Oblasti
Každou ze čtyř oblastí hodnotíš samostatně podle její definice.

{dims}

## Škála (stejná pro všechny oblasti)
{scale}

## Když zpráva oblast nepopisuje
{NULL_RULE}

## Důkazy
{EVIDENCE_RULE}
{QUOTE_RULE}

## Postup pro každou oblast
{steps}

## Výstup
Vrať JSON podle schématu.
- rubric_scores.<oblast>.evidence: 0–2 doslovné citace podle pravidel výše.
- rubric_scores.<oblast>.justification: 1–3 věcné věty pro výzkumníka – která zjištění rozhodla o skóre a proč ne sousední stupeň; u null jedna věta, co zpráva k oblasti obsahuje nebo že mlčí.
- rubric_scores.<oblast>.score: 1–4, nebo null podle pravidla výše.
- parent_text: samostatný text pro web jakouskolu.cz. Pravidla stylu níže platí jen pro toto pole, ne pro evidence a justification:
{parent}

## Kontrola před odesláním
- Každá oblast má skóre 1–4 s jednou až dvěma citacemi, nebo null s prázdnými citacemi.
- Každá citace je doslovně ve zprávě a má nejvýše 25 slov.
- Žádná citace ani skóre se neopírá o to, co se v oblasti nepočítá.
- Skóre vychází jen z textu zprávy, ne z odhadu toho, co ve zprávě chybí.
"""


def _dimension_schema(key: str) -> dict:
    return {
        "type": "object",
        "properties": {
            "evidence": {"type": "array", "maxItems": 2,
                         "items": {"type": "string", "description": "doslovná citace ze zprávy, max 25 slov"},
                         "description": "0–2 doslovné citace; prázdné právě tehdy, když score je null"},
            "justification": {"type": "string", "description": "1–3 věcné věty pro výzkumníka"},
            "score": {"anyOf": [{"type": "integer", "enum": [1, 2, 3, 4]}, {"type": "null"}],
                      "description": "1–4, nebo null, když zpráva oblast konkrétně nepopisuje"},
        },
        "required": ["evidence", "justification", "score"],
        "additionalProperties": False,
    }


SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rubric_scores": {
            "type": "object",
            "properties": {k: _dimension_schema(k) for k in DIMENSIONS},
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        "parent_text": {"type": "string", "description": "3–5 vět pro rodiče, prostým jazykem"},
    },
    "required": ["rubric_scores", "parent_text"],
    "additionalProperties": False,
}

def validate(parsed: dict) -> str | None:
    """Shape check of one response; None when valid. Normalises in place: Gemini returns
    integer enums as strings ("3", "null")."""
    rs = parsed.get("rubric_scores")
    if not isinstance(rs, dict) or not isinstance(parsed.get("parent_text"), str):
        return "missing rubric_scores or parent_text"
    for k in DIMENSIONS:
        e = rs.get(k)
        if not isinstance(e, dict) or not {"evidence", "justification", "score"} <= e.keys():
            return f"{k}: missing fields"
        if isinstance(e["score"], str) and e["score"].strip() in ("1", "2", "3", "4", "null"):
            e["score"] = None if e["score"].strip() == "null" else int(e["score"])
        if e["score"] not in (1, 2, 3, 4, None) or isinstance(e["score"], bool):
            return f"{k}: score {e['score']!r}"
        if not isinstance(e["evidence"], list) or len(e["evidence"]) > 2 or not all(isinstance(q, str) for q in e["evidence"]):
            return f"{k}: evidence"
        if (e["score"] is None) != (not [q for q in e["evidence"] if q.strip()]):
            return f"{k}: score {e['score']!r} with {len(e['evidence'])} quotes"
        if not isinstance(e["justification"], str) or not e["justification"].strip():
            return f"{k}: empty justification"
    if not parsed["parent_text"].strip():
        return "empty parent_text"
    return None


COMPONENT_FOCUS = (
    "Tato zpráva pokrývá více součástí školy. Hodnoť pouze součást {target} škola. "
    "Pasáže o ostatních součástech ignoruj a citace ber jen z pasáží o této součásti; "
    "pokud zpráva o této součásti v dané oblasti mlčí, skóre je null.\n\n"
)


def user_prompt(report_text: str, target_school_type: str | None = None, multi_component: bool = False) -> str:
    focus = COMPONENT_FOCUS.format(target=target_school_type) if multi_component and target_school_type else ""
    return focus + report_text


SYSTEM_PROMPT = build_prompt()
PROMPT_HASH = hashlib.sha256(
    (SYSTEM_PROMPT + COMPONENT_FOCUS + json.dumps(SCHEMA, sort_keys=True, ensure_ascii=False)).encode()
).hexdigest()

# Locked on 4 October 2026 (protocol section 3.2). A prompt with another hash is a new rubric version,
# not an edit of this one.
LOCKED_PROMPT_HASH = "d0a1dd8b9b2049e661cfddfc7881487eba1538ee653237077843c339d2f2db42"
if PROMPT_HASH != LOCKED_PROMPT_HASH:
    raise RuntimeError(f"prompt changed after the lock: {PROMPT_HASH} != {LOCKED_PROMPT_HASH}")


def anchors_markdown(anchors: list[dict] | None = None) -> str:
    anchors = load_anchors() if anchors is None else anchors
    lines = ["# Rubrika v2 – kotvící úryvky\n",
             "Jeden až čtyři úryvky ze skutečných inspekčních zpráv na každý stupeň škály a oblast a jeden příklad pro null. Úryvky jsou doslovné, „…“ označuje vynechaný text. "
             "Jsou součástí promptu; změna kteréhokoli z nich mění PROMPT_HASH.\n"]
    for key, d in DIMENSIONS.items():
        lines.append(f"\n## {d['label']} (`{key}`)\n")
        for score, text in [*SCALE.items(), (None, NULL_RULE)]:
            lines.append(f"\n### {'null' if score is None else score} – {text}\n")
            for a in (a for a in anchors if a["dimension"] == key and a["score"] == score):
                lines.append(f"- **{a['report_id']}** ({a['era']}, {a['school_type']}) – {a['why']}\n  > {a['text']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ANCHORS_MD.write_text(anchors_markdown())
    print(f"PROMPT_HASH {PROMPT_HASH}  prompt {len(SYSTEM_PROMPT)} chars, {len(load_anchors())} anchors -> {ANCHORS_MD.name}")
