"""
Provider-agnostic scoring layer (research plan phase 1.1).

One interface for scoring inspection reports with any chat model behind OpenRouter
(and, later, native batch adapters). Every call is written through a SQLite cache
keyed by (report_id, run_id, provider, model, prompt_hash, config_hash); the
API-returned model id, upstream provider, system fingerprint, token usage and cost
are stored per call so that version stability can be asserted across a corpus.

Secrets: OPENROUTER_API_KEY from the environment or projects/jakouskolu-research/.env.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx
import numpy as np

PROJECT_DIR = Path(__file__).parent
CACHE_DB = PROJECT_DIR / "cache" / "scoring_cache.sqlite"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
BATCHES_URL = "https://openrouter.ai/api/v1/batches"
RETRIES: dict[str, int] = {}  # model -> number of retried calls (truncated / malformed / transport)


def load_env() -> None:
    env = PROJECT_DIR / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@dataclass(frozen=True)
class ScoringSpec:
    model: str                      # OpenRouter slug, e.g. "z-ai/glm-5.3-flash"
    reasoning: str = "medium"       # off | minimal | low | medium | high | xhigh | max
    temperature: float | None = None  # None = provider default
    max_tokens: int = 32000
    provider: str = "openrouter"
    label: str = ""

    def config_hash(self) -> str:
        d = asdict(self); d.pop("label")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def prompt_hash(system_prompt: str, schema: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(system_prompt.encode()); h.update(b"\x1f"); h.update(json.dumps(schema, sort_keys=True).encode())
    return h.hexdigest()[:16]


# --- cache ------------------------------------------------------------------------

def _ensure_cache() -> None:
    CACHE_DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(CACHE_DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS calls (
            cache_key TEXT PRIMARY KEY, report_id TEXT NOT NULL, run_id TEXT NOT NULL,
            provider TEXT NOT NULL, model TEXT NOT NULL, prompt_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
            returned_model TEXT, upstream_provider TEXT, system_fingerprint TEXT,
            input_tokens INTEGER, output_tokens INTEGER, reasoning_tokens INTEGER, cost_usd REAL, latency_s REAL,
            response_json TEXT NOT NULL, created_at REAL NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY, model TEXT NOT NULL, prompt_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
            jobs_json TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL, finished_at REAL)""")


def _key(report_id: str, run_id: str, spec: ScoringSpec, phash: str) -> str:
    return hashlib.sha256(f"{report_id}|{run_id}|{spec.provider}|{spec.model}|{phash}|{spec.config_hash()}".encode()).hexdigest()


def _cache_get(key: str) -> dict[str, Any] | None:
    with sqlite3.connect(CACHE_DB) as c:
        row = c.execute("SELECT response_json FROM calls WHERE cache_key=?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def _cache_put(key: str, report_id: str, run_id: str, spec: ScoringSpec, phash: str, resp: dict[str, Any]) -> None:
    m = resp["meta"]
    with sqlite3.connect(CACHE_DB) as c:
        c.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (key, report_id, run_id, spec.provider, spec.model, phash, spec.config_hash(),
                   m.get("returned_model"), m.get("upstream_provider"), m.get("system_fingerprint"),
                   m.get("input_tokens"), m.get("output_tokens"), m.get("reasoning_tokens"), m.get("cost_usd"), m.get("latency_s"),
                   json.dumps(resp, ensure_ascii=False), time.time()))


def cache_calls(model: str | None = None) -> list[dict[str, Any]]:
    _ensure_cache()
    q = "SELECT report_id, run_id, model, returned_model, upstream_provider, system_fingerprint, input_tokens, output_tokens, reasoning_tokens, cost_usd, latency_s FROM calls"
    args: tuple = ()
    if model:
        q += " WHERE model=?"; args = (model,)
    with sqlite3.connect(CACHE_DB) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(q, args)]


# --- OpenRouter adapter ------------------------------------------------------------

def _reasoning_param(spec: ScoringSpec) -> dict[str, Any]:
    if spec.reasoning == "off":
        return {"enabled": False}
    return {"effort": spec.reasoning}


def strict_schema(schema: Any) -> Any:
    """OpenAI strict-mode variant: additionalProperties=false on objects, all properties
    required, unsupported keywords (minimum/maximum) folded into the description."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k in ("minimum", "maximum"):
                continue
            out[k] = strict_schema(v)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        if "minimum" in schema or "maximum" in schema:
            rng = f" Range {schema.get('minimum', '')}–{schema.get('maximum', '')}."
            out["description"] = (out.get("description", "") + rng).strip()
        return out
    if isinstance(schema, list):
        return [strict_schema(x) for x in schema]
    return schema


def request_body(spec: ScoringSpec, system_prompt: str, user_prompt: str, schema: dict[str, Any], schema_name: str) -> dict[str, Any]:
    """The chat-completions body of one call; the synchronous path and the batch path send the same one."""
    body: dict[str, Any] = {
        "model": spec.model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "response_format": {"type": "json_schema", "json_schema": {"name": schema_name, "strict": True,
                                                                  "schema": strict_schema(schema) if spec.model.startswith("openai/") else schema}},
        "reasoning": _reasoning_param(spec),
        "max_tokens": spec.max_tokens,
        "usage": {"include": True},
        "provider": {"require_parameters": True},
    }
    if spec.temperature is not None:
        body["temperature"] = spec.temperature
    return body


def read_completion(data: dict[str, Any], schema: dict[str, Any], latency: float | None,
                    validate: Callable[[dict[str, Any]], str | None] | None = None) -> dict[str, Any]:
    """Parsed and validated result of one chat completion; raises RuntimeError on an unusable response."""
    if "error" in data:
        raise RuntimeError(f"API error: {json.dumps(data['error'])[:300]}")
    choice = data["choices"][0]
    content = choice["message"].get("content")
    if not content:
        raise RuntimeError(f"empty content (finish_reason={choice.get('finish_reason')}, native={choice.get('native_finish_reason')})")
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or not all(k in parsed for k in schema.get("required", [])):
        raise RuntimeError(f"unexpected JSON shape: {str(parsed)[:120]}")
    if validate and (err := validate(parsed)):
        raise RuntimeError(f"invalid response ({err}): {content[:200]}")
    u = data.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    return {
        "rubric": parsed,
        "meta": {
            "returned_model": data.get("model"), "upstream_provider": data.get("provider"),
            "system_fingerprint": data.get("system_fingerprint"),
            "input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
            "reasoning_tokens": det.get("reasoning_tokens"), "cost_usd": u.get("cost"),
            "latency_s": None if latency is None else round(latency, 2), "finish_reason": choice.get("finish_reason"),
        },
    }


async def _call_openrouter(client: httpx.AsyncClient, spec: ScoringSpec, system_prompt: str, user_prompt: str,
                           schema: dict[str, Any], schema_name: str,
                           validate: Callable[[dict[str, Any]], str | None] | None = None) -> dict[str, Any]:
    t0 = time.time()
    r = await client.post(OPENROUTER_URL, json=request_body(spec, system_prompt, user_prompt, schema, schema_name), timeout=600)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return read_completion(r.json(), schema, time.time() - t0, validate)


async def score_reports(jobs: Iterable[tuple[str, str, str]], system_prompt: str, spec: ScoringSpec,
                        schema: dict[str, Any], schema_name: str = "rubric_scores", concurrency: int = 8,
                        max_retries: int = 5, budget_usd: float | None = None, progress_every: int = 25,
                        validate: Callable[[dict[str, Any]], str | None] | None = None) -> list[dict[str, Any]]:
    """Score (report_id, run_id, report_text) triples. Returns [{report_id, run_id, response}] in input order.

    Cached calls are free and do not count against budget_usd; new spend stops when the budget is exceeded.
    validate(parsed) returns an error string for responses to retry instead of caching.
    """
    load_env(); _ensure_cache()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY missing (set it in .env)")
    phash = prompt_hash(system_prompt, schema)
    sem = asyncio.Semaphore(concurrency)
    jobs = list(jobs)
    results: list[dict[str, Any]] = [{} for _ in jobs]
    spent = 0.0; done = 0
    headers = {"Authorization": f"Bearer {key}", "HTTP-Referer": "https://jakouskolu.cz", "X-Title": "jakouskolu research"}

    async with httpx.AsyncClient(headers=headers) as client:
        async def worker(i: int, report_id: str, run_id: str, text: str) -> None:
            nonlocal spent, done
            ck = _key(report_id, run_id, spec, phash)
            resp = _cache_get(ck)
            cached = resp is not None
            if resp is None:
                async with sem:
                    if budget_usd is not None and spent > budget_usd:
                        results[i] = {"report_id": report_id, "run_id": run_id, "response": None, "error": "budget"}
                        return
                    for attempt in range(max_retries):
                        try:
                            resp = await _call_openrouter(client, spec, system_prompt, text, schema, schema_name, validate)
                            break
                        except (RuntimeError, ValueError, TypeError, httpx.HTTPError, KeyError) as exc:
                            RETRIES[spec.model] = RETRIES.get(spec.model, 0) + 1
                            delay = min(60.0, 2 ** attempt + random.uniform(0, 1.5))
                            print(f"  [{spec.model} {report_id}/{run_id}] attempt {attempt + 1}: {repr(exc)[:600]}; retry in {delay:.0f}s", file=sys.stderr, flush=True)
                            await asyncio.sleep(delay)
                    if resp is None:
                        results[i] = {"report_id": report_id, "run_id": run_id, "response": None, "error": "failed"}
                        return
                    _cache_put(ck, report_id, run_id, spec, phash, resp)
                    spent += resp["meta"].get("cost_usd") or 0.0
            results[i] = {"report_id": report_id, "run_id": run_id, "response": resp, "cached": cached}
            done += 1
            if done % progress_every == 0 or done == len(jobs):
                print(f"  [{spec.label or spec.model}] {done}/{len(jobs)} scored, new spend ${spent:.2f}", flush=True)

        await asyncio.gather(*(worker(i, rid, run, txt) for i, (rid, run, txt) in enumerate(jobs)))
    return results


def _open_batches(spec: ScoringSpec, phash: str) -> list[tuple[str, dict[str, list[str]], float]]:
    """The open batches of a model and prompt: id, jobs by cache key, and the time of submission."""
    with sqlite3.connect(CACHE_DB) as c:
        rows = c.execute("SELECT batch_id, jobs_json, created_at FROM batches WHERE status='open' AND model=? AND prompt_hash=? AND config_hash=?",
                         (spec.model, phash, spec.config_hash())).fetchall()
    return [(b, json.loads(j), created) for b, j, created in rows]


def score_reports_batch(jobs: Iterable[tuple[str, str, str]], system_prompt: str, spec: ScoringSpec, schema: dict[str, Any],
                        schema_name: str = "rubric_scores", chunk: int = 200, poll_s: float = 30.0, max_new: int | None = None,
                        validate: Callable[[dict[str, Any]], str | None] | None = None,
                        transport: httpx.BaseTransport | None = None, offline_s: float = 6 * 3600.0,
                        settle_s: float = 90.0, wait: bool = True, credit_waits: int = 30,
                        credit_wait_s: float = 120.0, give_up_s: float | None = None) -> dict[str, int]:
    """Send the uncached jobs through the OpenRouter Batch API (half price, results within 24 hours) and cache the
    valid responses under the keys score_reports uses. A job whose response is missing or invalid is left uncached:
    call score_reports afterwards to request it synchronously. Batches are recorded in the cache database, so an
    interrupted run collects its open batches before it submits anything new. max_new caps the jobs submitted;
    transport replaces the network in tests.

    With wait=False the call only submits: jobs that are neither cached nor in an open batch are sent and nothing is
    collected, so a long list can be submitted part by part before the first batch is waited for. A later call with
    wait=True collects every open batch of this prompt and model, whichever call submitted it. jobs is read once and
    only one chunk of texts is held at a time.

    A batch has 24 hours at the provider and a few take most of them. With give_up_s a batch that is still running
    that long after its submission is not waited for: it stays open, a later call collects it if it completes, and
    its jobs stay uncached, so score_reports requests them synchronously.

    The network may be away while a batch runs: polling goes on for offline_s before it gives up. A submission whose
    answer is lost may have been accepted, and the API has no idempotency key, so the batch is looked up in the
    workspace's list after settle_s (a new batch is not listed at once) instead of being sent a second time."""
    load_env(); _ensure_cache()
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "HTTP-Referer": "https://jakouskolu.cz", "X-Title": "jakouskolu research"}
    phash = prompt_hash(system_prompt, schema)
    counts = {"cached": 0, "pending": 0, "submitted": 0, "valid": 0, "invalid": 0, "failed": 0, "missing": 0, "left_running": 0, "cost_usd": 0.0}

    def collect(client: httpx.Client, batch_id: str, by_key: dict[str, list[str]], submitted_at: float) -> None:
        misses, offline_since, shown = 0, None, None
        while True:
            try:
                r = client.get(f"{BATCHES_URL}/{batch_id}", timeout=120)
            except httpx.TransportError as exc:
                offline_since = offline_since or time.monotonic()
                if time.monotonic() - offline_since > offline_s:
                    raise RuntimeError(f"batch {batch_id}: no connection for {offline_s / 3600:.1f} h ({exc!r})") from exc
                time.sleep(poll_s)
                continue
            offline_since = None
            if r.status_code != 200:
                # a batch is not readable for a short while after it is accepted; server errors pass too
                misses += 1
                if misses > 40 or r.status_code not in (404, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"batch {batch_id}: HTTP {r.status_code}: {r.text[:300]}")
                time.sleep(poll_s)
                continue
            data = r.json()
            if data["status"] in ("completed", "failed", "expired", "cancelled"):
                break
            if (progress := (data["status"], (data.get("request_counts") or {}).get("completed"))) != shown:
                shown = progress
                print(f"  [batch {batch_id}] {data['status']}: {data.get('request_counts')}", flush=True)
            if give_up_s is not None and time.time() - submitted_at > give_up_s:
                counts["left_running"] += len(by_key)
                print(f"  [batch {batch_id}] still {data['status']} {(time.time() - submitted_at) / 3600:.1f} h after submission: not waited for", flush=True)
                return
            time.sleep(poll_s)
        if data["status"] != "completed":  # no results come with a batch that failed, expired or was cancelled
            print(f"  [batch {batch_id}] {data['status']}: {json.dumps(data.get('error'))[:500]}", file=sys.stderr, flush=True)
        results = data.get("results") or []
        # A batch reports its cost as one figure. Each request gets a share by its tokens, output weighted five
        # times input as in the price list, so that the shares of a batch add up to what was charged.
        def weight(res: dict[str, Any]) -> float:
            u = ((res.get("response") or {}).get("body") or {}).get("usage") or {}
            return (u.get("prompt_tokens") or 0) + 5 * (u.get("completion_tokens") or 0)
        batch_cost, total_weight = float((data.get("usage") or {}).get("cost") or 0), sum(weight(r) for r in results)
        seen: set[str] = set()
        for res in results:
            key = res.get("custom_id")
            if key not in by_key or key in seen:
                raise RuntimeError(f"batch {batch_id}: result for an unknown or repeated job {key!r}")
            seen.add(key)
            report_id, run_id = by_key[key]
            body = (res.get("response") or {}).get("body")
            if res.get("error") or not body or (res.get("response") or {}).get("status_code") != 200:
                counts["failed"] += 1
                continue
            try:
                resp = read_completion(body, schema, None, validate)
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                counts["invalid"] += 1
                print(f"  [batch {report_id}/{run_id}] {repr(exc)[:300]}", file=sys.stderr, flush=True)
                continue
            if resp["meta"]["cost_usd"] is None and total_weight:
                resp["meta"]["cost_usd"] = round(batch_cost * weight(res) / total_weight, 6)
            resp["meta"]["batch_id"] = batch_id
            _cache_put(key, report_id, run_id, spec, phash, resp)
            counts["valid"] += 1
        counts["missing"] += len(by_key) - len(seen)
        counts["cost_usd"] += float((data.get("usage") or {}).get("cost") or 0)
        with sqlite3.connect(CACHE_DB) as c:
            c.execute("UPDATE batches SET status=?, finished_at=? WHERE batch_id=?", (data["status"], time.time(), batch_id))
        print(f"  [batch {batch_id}] {data['status']}: {data.get('request_counts')}, cost {(data.get('usage') or {}).get('cost')}", flush=True)

    def unrecorded(client: httpx.Client, n: int, since: float) -> str | None:
        """The batch of n requests that the workspace has held since `since` and the cache database does not know."""
        with sqlite3.connect(CACHE_DB) as c:
            known = {b for (b,) in c.execute("SELECT batch_id FROM batches")}
        r = client.get(BATCHES_URL, params={"created_after": int(since) - 5, "limit": 100}, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"batch list: HTTP {r.status_code}: {r.text[:300]}")
        found = [b["id"] for b in r.json()["data"]
                 if b["id"] not in known and str(b.get("model") or "").startswith(spec.model)  # listed under the dated slug
                 and (b.get("request_counts") or {}).get("total") in (n, 0, None)]
        if len(found) > 1:
            raise RuntimeError(f"batch submit: {len(found)} unrecorded batches of {n} requests ({found}); which one is this run's has to be decided by hand")
        return found[0] if found else None

    def submit(client: httpx.Client, requests: list[dict[str, Any]]) -> str:
        """The id of the batch that holds `requests`: a new one, or the one an unanswered submission left behind.
        A submission reserves credit at the synchronous price until the batch is done; one that is refused for
        credit (HTTP 402) is sent again after a wait, since an automatic top-up or a finished batch frees credit."""
        unanswered = refused = 0
        while unanswered < 3:
            started = time.time()
            try:
                # endpoint and model have to precede requests in the body: the API reads it as a stream
                r = client.post(BATCHES_URL, json={"endpoint": "/v1/chat/completions", "model": spec.model, "requests": requests}, timeout=600)
            except httpx.TransportError as exc:
                print(f"  [batch submit] no answer ({exc!r}); looking for the batch before sending again", file=sys.stderr, flush=True)
                time.sleep(settle_s)
                waited = time.monotonic()
                while True:
                    try:
                        batch_id = unrecorded(client, len(requests), started)
                        break
                    except httpx.TransportError as again:
                        if time.monotonic() - waited > offline_s:
                            raise RuntimeError(f"batch submit: no connection for {offline_s / 3600:.1f} h ({again!r})") from again
                        time.sleep(poll_s)
                if batch_id:
                    print(f"  [batch {batch_id}] found: the submission had been accepted", file=sys.stderr, flush=True)
                    return batch_id
                unanswered += 1
                continue
            if r.status_code == 402 and refused < credit_waits:
                refused += 1
                print(f"  [batch submit] HTTP 402, no credit for the reservation; waiting {credit_wait_s:.0f} s ({refused}/{credit_waits})", file=sys.stderr, flush=True)
                time.sleep(credit_wait_s)
                continue
            if r.status_code not in (200, 202):
                raise RuntimeError(f"batch submit: HTTP {r.status_code}: {r.text[:500]}")
            return r.json()["id"]
        raise RuntimeError("batch submit: three submissions without an answer and no batch in the workspace's list")

    with httpx.Client(headers=headers, transport=transport) as client:
        pending: set[str] = set()
        for batch_id, by_key, submitted_at in _open_batches(spec, phash):
            pending |= set(by_key)
            if wait:
                collect(client, batch_id, by_key, submitted_at)
        submitted: list[tuple[str, dict[str, list[str]]]] = []
        part: list[tuple[str, str, str, str]] = []

        def flush() -> None:
            if not part:
                return
            requests = []
            for key, _, _, text in part:
                body = request_body(spec, system_prompt, text, schema, schema_name)
                body.pop("model")
                requests.append({"custom_id": key, "body": body})
            batch_id = submit(client, requests)
            by_key = {key: [report_id, run_id] for key, report_id, run_id, _ in part}
            with sqlite3.connect(CACHE_DB) as c:
                c.execute("INSERT INTO batches VALUES (?,?,?,?,?,?,?,?)", (batch_id, spec.model, phash, spec.config_hash(), json.dumps(by_key), "open", time.time(), None))
            submitted.append((batch_id, by_key))
            counts["submitted"] += len(part)
            print(f"  [batch {batch_id}] submitted {len(part)} requests ({counts['submitted']} in this call)", flush=True)
            part.clear()

        for report_id, run_id, text in jobs:
            key = _key(report_id, run_id, spec, phash)
            if _cache_get(key) is not None:
                counts["cached"] += 1
            elif key in pending:  # in an open batch (wait=False), or collected above without a usable response
                counts["pending"] += 1
            elif max_new is None or counts["submitted"] + len(part) < max_new:
                part.append((key, report_id, run_id, text))
                if len(part) == chunk:
                    flush()
        flush()
        if wait:
            for batch_id, by_key in submitted:
                collect(client, batch_id, by_key, time.time())
    return counts


def extract_scores(result: dict[str, Any], cols: tuple[str, ...]) -> dict[str, int | None]:
    rubric = ((result.get("response") or {}).get("rubric") or {}).get("rubric_scores", {})
    out: dict[str, int | None] = {}
    for c in cols:
        v = (rubric.get(c) or {}).get("score")
        out[c] = int(v) if isinstance(v, (int, float)) else None
    return out


def usage_summary(results: list[dict]) -> dict[str, Any]:
    metas = [r["response"]["meta"] for r in results if r.get("response")]
    if not metas:
        return {}
    def med(k):
        v = [m[k] for m in metas if m.get(k) is not None]; return float(np.median(v)) if v else None
    models = sorted({m.get("returned_model") for m in metas}); provs = sorted({m.get("upstream_provider") for m in metas})
    return {"n_calls": len(metas), "input_tokens_med": med("input_tokens"), "output_tokens_med": med("output_tokens"),
            "reasoning_tokens_med": med("reasoning_tokens"), "cost_usd_total": round(sum(m.get("cost_usd") or 0 for m in metas), 4),
            "cost_usd_per_call": round(sum(m.get("cost_usd") or 0 for m in metas) / len(metas), 5), "latency_s_med": med("latency_s"),
            "returned_models": models, "upstream_providers": provs,
            "truncated": sum(1 for m in metas if m.get("finish_reason") == "length"),
            "retries": None}


# --- evidence grounding --------------------------------------------------------------

_QUOTES = dict.fromkeys(map(ord, "„“”‚‘’\"'«»‹›`´"), None)
_DASHES = dict.fromkeys(map(ord, "–—−‒‐‑"), "-")
PAGE_BREAK_GAP = 160
_ELLIPSIS = re.compile(r"…|\.\.\.+|\[\s*(?:…|\.\.\.)\s*\]|\(\s*(?:…|\.\.\.)\s*\)")


def normalise_text(s: str) -> str:
    """Case-folded, quote-free, dash-unified, hyphenation-joined, whitespace-collapsed text."""
    s = unicodedata.normalize("NFKC", s).replace("\xad", "").translate(_QUOTES).translate(_DASHES)
    s = re.sub(r"[-•·_]+", " ", s)
    s = re.sub(r"-[ \t]*\n\s*", "-", s)
    return re.sub(r"\s+", " ", s).strip().casefold()


def _compact(s: str) -> str:
    return re.sub(r"[\s\-]+", "", s)


def _quote_parts(quote: str) -> list[str]:
    return [p for p in (normalise_text(x).strip(" .,;:") for x in _ELLIPSIS.split(quote)) if len(p) >= 3]


class GroundingIndex:
    """Report text prepared once for repeated quote lookups."""

    def __init__(self, text: str):
        self.norm = normalise_text(text)
        self.compact = _compact(self.norm)

    def _across_page_break(self, part: str) -> bool:
        """The part occurs with one page break (page number, running header) inside it."""
        words = part.split(" ")
        for i in range(2, len(words) - 1):
            head, tail = " ".join(words[:i]), " ".join(words[i:])
            if len(head) < 15 or len(tail) < 15:
                continue
            start = self.norm.find(head)
            while start != -1:
                if self.norm.find(tail, start + len(head), start + len(head) + PAGE_BREAK_GAP + len(tail)) != -1:
                    return True
                start = self.norm.find(head, start + 1)
        return False

    def contains(self, quote: str) -> bool:
        """Every segment of the quote occurs in the text, each after the one before it. A segment that is found
        only across a page break does not move the position the next segment is searched from."""
        parts = _quote_parts(quote)
        pos = 0
        for part in parts:
            compact = _compact(part)
            at = self.compact.find(compact, pos)
            if at != -1:
                pos = at + len(compact)
            elif not self._across_page_break(part):
                return False
        return bool(parts)


def is_grounded(quote: str, text: str) -> bool:
    return GroundingIndex(text).contains(quote)


def grounding_record(parsed: dict[str, Any] | None, text: str, dims: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Per dimension: score, number of quotes, number grounded in the report text."""
    idx = GroundingIndex(text)
    out: dict[str, dict[str, Any]] = {}
    for d in dims:
        entry = (parsed or {}).get(d) or {}
        quotes = [q for q in (entry.get("evidence") or []) if isinstance(q, str) and q.strip()]
        v = entry.get("score")
        out[d] = {"score": int(v) if isinstance(v, (int, float)) else None, "n_quotes": len(quotes),
                  "n_grounded": sum(idx.contains(q) for q in quotes)}
    return out


def grounding_metrics(records: list[dict[str, dict[str, Any]]], eras: list[str], dims: Iterable[str]) -> dict[str, Any]:
    """Quote-grounding rate, ungrounded-score share (non-null score with zero grounded quotes) and
    null rate per dimension, overall and by era. records[i] is grounding_record() for a report of eras[i];
    failed calls (None) are skipped."""
    def summarise(rs: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for d in dims:
            cells = [r[d] for r in rs]
            nq = sum(c["n_quotes"] for c in cells); ng = sum(c["n_grounded"] for c in cells)
            scored = [c for c in cells if c["score"] is not None]
            res[d] = {"n": len(cells), "quotes": nq,
                      "quote_grounding_rate": round(ng / nq, 3) if nq else None,
                      "ungrounded_score_share": round(sum(c["n_grounded"] == 0 for c in scored) / len(scored), 3) if scored else None,
                      "null_rate": round(1 - len(scored) / len(cells), 3) if cells else None,
                      "null_with_evidence": sum(c["score"] is None and c["n_quotes"] > 0 for c in cells)}
        return res
    dims = list(dims)
    pairs = [(r, e) for r, e in zip(records, eras) if r is not None]
    out = {"all": summarise([r for r, _ in pairs])}
    for e in sorted({e for _, e in pairs}):
        out[e] = summarise([r for r, x in pairs if x == e])
    return out
