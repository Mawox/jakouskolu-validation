"""
The scoring request in one place, for the research runs and for production (vendored with rubric_v2.py and scoring.py):
model, reasoning effort, schema name and number of attempts, the rule for the component instruction, and the rule that
decides whether an analysis may be scored before the hold on surveyed organisations is released.

    python scoring_request.py        # rewrites fixtures/: the request bodies of both calls for a made-up report
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path

import httpx

from scoring import ScoringSpec, _call_openrouter

SPEC = ScoringSpec("openai/gpt-6-luna", "high", label="luna")
SCHEMA_NAME = "rubric_v2"
ATTEMPTS = 5  # requests per call before the item counts as failed
FIXTURES = Path(__file__).parent / "fixtures"


def multi_component(school_types_of_report: Iterable[str | None]) -> bool:
    """True when the report is analysed for more than one school type. Count every component the report covers,
    before any filter (school type, hold, already analysed); a false value sends the whole report to the model."""
    return len({t for t in school_types_of_report if t}) > 1


def is_held(red_izo: str | None, held_red_izos: set[str]) -> bool:
    """Whether the hold forbids scoring this analysis. The hold covers every analysis of a surveyed organisation,
    whatever its school type: the kindergarten and the secondary school are analysed from the same reports as the
    basic school. Fails closed: an analysis without red_izo is held."""
    return not red_izo or str(red_izo) in held_red_izos


def request_body(spec: ScoringSpec, system_prompt: str, user_prompt: str, schema: dict, schema_name: str) -> dict:
    """The JSON body scoring._call_openrouter sends, captured without a network call."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(500, text="captured")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            try:
                await _call_openrouter(client, spec, system_prompt, user_prompt, schema, schema_name)
            except RuntimeError:
                pass

    asyncio.run(run())
    return seen[0]


if __name__ == "__main__":
    import rubric_v2
    import web_texts

    report = (FIXTURES / "report.txt").read_text()
    record = json.loads((FIXTURES / "record.json").read_text())
    bodies = {
        "scoring_request.json": request_body(SPEC, rubric_v2.SYSTEM_PROMPT, rubric_v2.user_prompt(report, record["target_school_type"], True),
                                             rubric_v2.SCHEMA, SCHEMA_NAME),
        "page_request.json": request_body(web_texts.SPEC, web_texts.SYSTEM_PROMPT, web_texts.user_message(record, report, True),
                                          web_texts.SCHEMA, web_texts.SCHEMA_NAME),
    }
    for name, body in bodies.items():
        (FIXTURES / name).write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n")
        print(name, len(json.dumps(body)), "bytes")

    # The web rubric (the website's own pipeline): prompt, request and one worked case with the expected results
    # of every step after the call, for the tests of the production port.
    import copy

    import web_rubric

    (FIXTURES / "web_rubric_prompt.txt").write_text(web_rubric.SYSTEM_PROMPT)
    message = web_rubric.user_message(report, record["target_school_type"], True)
    body = request_body(web_rubric.SPECS["luna"], web_rubric.SYSTEM_PROMPT, message, web_rubric.SCHEMA, web_rubric.SCHEMA_NAME)
    (FIXTURES / "web_rubric_request.json").write_text(json.dumps(body, ensure_ascii=False, indent=1) + "\n")
    response = json.loads((FIXTURES / "web_rubric_response.json").read_text())
    focus = "Analyzuj zprávu z pohledu druhého stupně. Ignoruj první stupeň."
    case = {"target_school_type": record["target_school_type"], "multi_component": True, "user_message": message,
            "profile_focus": focus, "user_message_with_profile_focus": web_rubric.user_message(report, record["target_school_type"], True, focus),
            "response": copy.deepcopy(response), "validate": web_rubric.validate(copy.deepcopy(response))}
    grounded = copy.deepcopy(response)
    case["quotes"], case["quotes_found"] = web_rubric.ground(grounded, report)
    case.update(grounded=grounded, consistency=web_rubric.consistency(grounded), wording=web_rubric.wording(grounded), stored=web_rubric.to_stored(grounded))
    (FIXTURES / "web_rubric_case.json").write_text(json.dumps(case, ensure_ascii=False, indent=1) + "\n")
    print("web_rubric_case.json:", case["quotes_found"], "of", case["quotes"], "quotes found;", "consistency", case["consistency"], "wording", case["wording"])
