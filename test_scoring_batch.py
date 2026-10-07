"""
Tests of the Batch API path of scoring.py against a fake OpenRouter (no network, no spend).

    uv run --with pytest --with httpx --with numpy pytest test_scoring_batch.py -q
"""

from __future__ import annotations

import json

import httpx
import pytest

import scoring
from scoring import ScoringSpec, _cache_get, _key, prompt_hash, score_reports_batch

SPEC = ScoringSpec("openai/gpt-6-luna", "high")
SCHEMA = {"type": "object", "properties": {"score": {"type": "integer"}}, "required": ["score"], "additionalProperties": False}
PROMPT = "system prompt"


def validate(parsed: dict) -> str | None:
    return None if parsed["score"] in (1, 2, 3, 4) else "score out of range"


class FakeOpenRouter:
    """Accepts batches, hides a new batch for `hidden` polls, runs it for `running` polls, then completes it.
    `answer(custom_id, body)` returns the result object of one request, or None to leave the request out."""

    def __init__(self, answer, hidden: int = 1, running: int = 1, final_status: str = "completed"):
        self.answer, self.hidden, self.running, self.final_status = answer, hidden, running, final_status
        self.batches: dict[str, dict] = {}
        self.polls: dict[str, int] = {}
        self.submissions: list[dict] = []
        self.lost_answers = 0   # submissions that are accepted while the connection drops before the answer
        self.lost_requests = 0  # submissions that never arrive
        self.offline_polls = 0  # polls during which the network is away
        self.no_credit = 0      # submissions refused with HTTP 402
        self.lists = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        if request.method == "POST" and request.url.path == "/api/v1/batches":
            if self.lost_requests:
                self.lost_requests -= 1
                raise httpx.ConnectError("Temporary failure in name resolution", request=request)
            if self.no_credit:
                self.no_credit -= 1
                return httpx.Response(402, json={"error": {"message": "Insufficient credits"}})
            body = json.loads(request.content)
            assert list(body)[:2] == ["endpoint", "model"], "endpoint and model precede requests"
            self.submissions.append(body)
            batch_id = f"batch-{len(self.batches) + 1}"
            self.batches[batch_id] = body
            self.polls[batch_id] = 0
            if self.lost_answers:
                self.lost_answers -= 1
                raise httpx.ReadTimeout("no answer", request=request)
            return httpx.Response(202, json={"id": batch_id, "status": "validating"})
        if request.url.path == "/api/v1/batches":  # the workspace's list, newest first, under the dated model slug
            self.lists += 1
            assert int(request.url.params["created_after"]) > 0
            return httpx.Response(200, json={"object": "list", "has_more": False, "data": [
                {"id": b, "model": body["model"] + "-20260922", "status": "in_progress", "results": None,
                 "request_counts": {"total": len(body["requests"]), "completed": 0, "failed": 0}} for b, body in reversed(self.batches.items())]})
        batch_id = request.url.path.rsplit("/", 1)[-1]
        if self.offline_polls:
            self.offline_polls -= 1
            raise httpx.ConnectError("Temporary failure in name resolution", request=request)
        if batch_id not in self.batches:
            return httpx.Response(404, json={"error": {"message": "not found"}})
        self.polls[batch_id] += 1
        n = self.polls[batch_id]
        if n <= self.hidden:
            return httpx.Response(404, json={"error": {"message": f"Batch job {batch_id} not found."}})
        if n <= self.hidden + self.running:
            return httpx.Response(200, json={"id": batch_id, "status": "in_progress", "results": None, "usage": None})
        results = [x for r in self.batches[batch_id]["requests"] if (x := self.answer(r["custom_id"], r["body"])) is not None]
        return httpx.Response(200, json={"id": batch_id, "status": self.final_status, "results": results, "usage": {"cost": 0.01 * len(results)},
                                         "request_counts": {"total": len(self.batches[batch_id]["requests"]), "completed": len(results), "failed": 0}})


def completion(custom_id: str, content: str) -> dict:
    """One result as the Batch API returns it: usage in tokens, no cost of its own."""
    return {"custom_id": custom_id, "error": None, "response": {"status_code": 200, "body": {
        "model": "openai/gpt-6-luna-20260922", "provider": "OpenAI", "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 40}}}}}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(scoring, "CACHE_DB", tmp_path / "cache.sqlite")
    monkeypatch.setattr(scoring, "load_env", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(scoring.time, "sleep", lambda s: None)


def jobs(n: int) -> list[tuple[str, str, str]]:
    return [(f"analysis-{i}", "run", f"report text {i}") for i in range(n)]


def key(i: int) -> str:
    return _key(f"analysis-{i}", "run", SPEC, prompt_hash(PROMPT, SCHEMA))


def run(server: FakeOpenRouter, n: int = 5, **kw) -> dict:
    return score_reports_batch(jobs(n), PROMPT, SPEC, SCHEMA, schema_name="test", validate=validate, poll_s=0, transport=httpx.MockTransport(server), **kw)


def test_all_valid_responses_are_cached_under_their_own_job():
    server = FakeOpenRouter(lambda cid, body: completion(cid, json.dumps({"score": 1 + int(body["messages"][1]["content"][-1]) % 4})))
    counts = run(server)
    assert (counts["submitted"], counts["valid"], counts["invalid"], counts["failed"], counts["missing"]) == (5, 5, 0, 0, 0)
    for i in range(5):  # each job got the answer computed from its own text
        cached = _cache_get(key(i))
        assert cached["rubric"] == {"score": 1 + i % 4}
        assert cached["meta"]["returned_model"] == "openai/gpt-6-luna-20260922" and cached["meta"]["reasoning_tokens"] == 40
    # the batch's single cost figure is shared out among its requests and adds up
    assert sum(_cache_get(key(i))["meta"]["cost_usd"] for i in range(5)) == pytest.approx(counts["cost_usd"]) == pytest.approx(0.05)
    assert _cache_get(key(0))["meta"]["batch_id"] == "batch-1"


def test_request_bodies_equal_the_synchronous_body_without_the_model():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'))
    run(server, n=2)
    sent = server.submissions[0]
    assert sent["endpoint"] == "/v1/chat/completions" and sent["model"] == SPEC.model
    expected = scoring.request_body(SPEC, PROMPT, "report text 0", SCHEMA, "test")
    expected.pop("model")
    assert sent["requests"][0] == {"custom_id": key(0), "body": expected}


def test_chunks_and_cap():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'))
    counts = run(server, n=7, chunk=3, max_new=5)
    assert [len(b["requests"]) for b in server.submissions] == [3, 2]
    assert counts["submitted"] == 5 and counts["valid"] == 5
    assert _cache_get(key(5)) is None and _cache_get(key(6)) is None


def test_invalid_failed_and_missing_results_stay_uncached():
    def answer(cid, body):
        i = int(body["messages"][1]["content"][-1])
        if i == 0:
            return completion(cid, '{"score": 9}')            # fails validate
        if i == 1:
            return completion(cid, "not json")                # unparsable
        if i == 2:
            return {"custom_id": cid, "error": {"message": "provider error"}, "response": None}
        if i == 3:
            return None                                        # the batch returns nothing for it
        return completion(cid, '{"score": 3}')
    counts = run(FakeOpenRouter(answer))
    assert (counts["valid"], counts["invalid"], counts["failed"], counts["missing"]) == (1, 2, 1, 1)
    assert [_cache_get(key(i)) is not None for i in range(5)] == [False, False, False, False, True]


def test_second_run_submits_nothing_for_cached_jobs():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'))
    run(server)
    counts = run(server)
    assert counts["cached"] == 5 and counts["submitted"] == 0 and len(server.submissions) == 1


def test_interrupted_run_collects_its_open_batch_without_resubmitting():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'), hidden=0, running=10**6)

    class Stop(Exception):
        pass

    def stop(_):
        raise Stop
    scoring.time.sleep = stop  # the first wait interrupts the run, as a killed process would
    with pytest.raises(Stop):
        run(server)
    assert len(server.submissions) == 1 and _cache_get(key(0)) is None
    scoring.time.sleep = lambda s: None
    server.running = 0
    counts = run(server)
    assert len(server.submissions) == 1, "the open batch is collected, not submitted again"
    assert counts["submitted"] == 0 and counts["valid"] == 5 and _cache_get(key(4)) is not None


def test_result_for_a_job_outside_the_batch_stops_the_run():
    server = FakeOpenRouter(lambda cid, body: completion("someone-else", '{"score": 2}'))
    with pytest.raises(RuntimeError, match="unknown or repeated job"):
        run(server, n=1)


def test_a_batch_that_fails_leaves_everything_uncached_and_is_closed():
    server = FakeOpenRouter(lambda cid, body: None, final_status="failed")
    counts = run(server, n=3)
    assert counts["valid"] == 0 and counts["missing"] == 3
    again = run(server, n=3)  # the failed batch is not polled again; the jobs are submitted anew
    assert again["submitted"] == 3 and len(server.submissions) == 2


def test_submit_error_raises():
    with pytest.raises(RuntimeError, match="batch submit: HTTP 413"):
        score_reports_batch(jobs(1), PROMPT, SPEC, SCHEMA, transport=httpx.MockTransport(lambda r: httpx.Response(413, text="too large")), poll_s=0)


def test_polling_goes_on_while_the_network_is_away():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'))
    server.offline_polls = 7
    counts = run(server)
    assert counts["valid"] == 5 and len(server.submissions) == 1


def test_polling_gives_up_after_the_allowed_time_offline_and_the_batch_stays_open():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'))
    server.offline_polls = 10**6
    with pytest.raises(RuntimeError, match="no connection"):
        run(server, offline_s=-1)
    server.offline_polls = 0
    counts = run(server)  # the next run collects the batch that was left open
    assert counts["valid"] == 5 and counts["submitted"] == 0 and len(server.submissions) == 1


def test_a_submission_whose_answer_is_lost_is_found_in_the_list_and_not_sent_again():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 3}'))
    server.lost_answers = 1
    counts = run(server, n=5, chunk=2)  # three batches; the first answer is lost
    assert len(server.submissions) == 3 and server.lists == 1
    assert counts["submitted"] == 5 and counts["valid"] == 5 and all(_cache_get(key(i)) is not None for i in range(5))


def test_a_submission_that_never_arrived_is_sent_again():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 3}'))
    server.lost_requests = 2
    counts = run(server, n=3)
    assert len(server.submissions) == 1 and server.lists == 2 and counts["valid"] == 3


def test_three_submissions_without_an_answer_stop_the_run():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 3}'))
    server.lost_requests = 3
    with pytest.raises(RuntimeError, match="three submissions without an answer"):
        run(server, n=3)
    assert not server.submissions


def test_two_unrecorded_batches_of_the_same_size_stop_the_run():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 3}'))
    stray = {"endpoint": "/v1/chat/completions", "model": SPEC.model, "requests": [{"custom_id": "x", "body": {}}] * 3}
    server.batches["batch-stray"], server.polls["batch-stray"] = stray, 0
    server.lost_answers = 1
    with pytest.raises(RuntimeError, match="2 unrecorded batches"):
        run(server, n=3)


def test_a_batch_of_another_size_in_the_list_is_not_taken_for_this_run():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 3}'))
    stray = {"endpoint": "/v1/chat/completions", "model": SPEC.model, "requests": [{"custom_id": "x", "body": {}}] * 7}
    server.batches["batch-stray"], server.polls["batch-stray"] = stray, 0
    server.lost_requests = 1
    counts = run(server, n=3)
    assert counts["valid"] == 3 and len(server.submissions) == 1


def test_submitting_without_waiting_then_collecting_in_a_later_call():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 4}'))
    first = run(server, n=5, chunk=2, wait=False)
    assert first["submitted"] == 5 and first["valid"] == 0 and len(server.submissions) == 3 and _cache_get(key(0)) is None
    again = run(server, n=7, chunk=2, wait=False)  # what is in an open batch is not sent again; two new jobs are
    assert again["pending"] == 5 and again["submitted"] == 2 and len(server.submissions) == 4
    counts = run(server, n=7, chunk=2)
    assert counts["submitted"] == 0 and counts["valid"] == 7 and counts["cached"] == 7 and len(server.submissions) == 4
    assert all(_cache_get(key(i)) is not None for i in range(7))


def test_jobs_are_read_once_and_may_be_a_generator():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 1}'))
    counts = score_reports_batch((j for j in jobs(5)), PROMPT, SPEC, SCHEMA, schema_name="test", validate=validate, poll_s=0, chunk=2,
                                 transport=httpx.MockTransport(server))
    assert counts["valid"] == 5 and [len(b["requests"]) for b in server.submissions] == [2, 2, 1]


def test_the_cap_counts_the_jobs_of_the_call_across_chunks():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 1}'))
    counts = run(server, n=9, chunk=2, max_new=5)
    assert counts["submitted"] == 5 and [len(b["requests"]) for b in server.submissions] == [2, 2, 1]


def test_a_submission_refused_for_credit_is_sent_again_after_a_wait():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'))
    server.no_credit = 3
    counts = run(server, n=3)
    assert counts["valid"] == 3 and len(server.submissions) == 1


def test_credit_that_never_comes_stops_the_run():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'))
    server.no_credit = 10**6
    with pytest.raises(RuntimeError, match="batch submit: HTTP 402"):
        run(server, n=3, credit_waits=4)
    assert not server.submissions


def test_a_batch_that_runs_too_long_is_left_open_and_collected_by_a_later_call():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'), hidden=0, running=10**6)
    first = run(server, n=3, wait=False)
    assert first["submitted"] == 3
    counts = run(server, n=3, give_up_s=-1)  # every open batch is older than the limit
    assert counts["left_running"] == 3 and counts["valid"] == 0 and counts["pending"] == 3
    assert len(server.submissions) == 1 and _cache_get(key(0)) is None, "its jobs stay uncached and are not sent again"
    server.running = 0
    done = run(server, n=3)
    assert done["valid"] == 3 and len(server.submissions) == 1


def test_a_batch_within_the_limit_is_waited_for():
    server = FakeOpenRouter(lambda cid, body: completion(cid, '{"score": 2}'), hidden=0, running=3)
    counts = run(server, n=3, give_up_s=3600)
    assert counts["valid"] == 3 and counts["left_running"] == 0
