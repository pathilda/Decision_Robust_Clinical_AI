"""Retry/resume semantics with a fake engine, and a full synthetic run through the existing OR
code (Gurobi; skipped when gurobipy or a license is unavailable)."""

import hashlib
import json

import pandas as pd
import pytest

from dr import allocate, candidates, evaluate, exclusions, icf, rate, tiebreak
from dr.chunks import main as chunks_main
from dr.common import JsonlStore, Workspace
from dr.engine import FakeEngine, ParseError, Request, run_requests
from dr.fixtures import synthetic_profile

GOOD = json.dumps({"reasoning": "r", "fit_issue": None, "class": 2})


def request(key="k", max_tokens=100):
    def parse(obj):
        if "class" not in obj:
            raise ParseError("missing class")
        return obj, [], {"value": obj["class"]}

    return Request(key=key, messages=[{"role": "system", "content": "s"},
                                      {"role": "user", "content": "u"}],
                   schema={}, max_tokens=max_tokens, parse=parse,
                   meta={"model": "m", "step": "rating", "client_id": key})


def test_retry_ladder_doubles_tokens_then_bumps_seed(tmp_path):
    seen = []

    def responder(msgs, schema, max_tokens, seed):
        seen.append((max_tokens, seed))
        return GOOD if len(seen) == 3 else "{not json"

    store = JsonlStore(tmp_path / "r.jsonl")
    summary = run_requests(FakeEngine(responder), [request()], store, "rev", seed=1234)
    assert seen == [(100, 1234), (200, 1234), (200, 1235)]
    record = store.get("k")
    assert record["parse_ok"] and record["attempts"] == 3 and summary["ok"] == 1
    assert record["model_revision"] == "rev" and record["value"] == 2


def test_failure_after_three_attempts_and_truncation(tmp_path):
    store = JsonlStore(tmp_path / "r.jsonl")
    long_text = "x" * 4000  # > max_tokens * 4 chars -> finish_reason == "length"
    run_requests(FakeEngine(lambda *a: long_text), [request()], store, "rev")
    record = store.get("k")
    assert not record["parse_ok"] and record["attempts"] == 3
    assert record["finish_reason"] == "length" and record["raw_text"] == long_text
    assert all("truncated" in e for e in record["errors"])


def test_prompt_too_long_is_skipped_not_truncated(tmp_path):
    store = JsonlStore(tmp_path / "r.jsonl")
    engine = FakeEngine(lambda *a: GOOD, max_model_len=50)
    summary = run_requests(engine, [request(max_tokens=45)], store, "rev")
    assert summary["prompt_too_long"] == 1 and not engine.calls
    assert store.get("k")["skipped_reason"] == "prompt_too_long"


def test_resume_skips_only_successful_same_prompt(tmp_path):
    path = tmp_path / "r.jsonl"
    engine = FakeEngine(lambda *a: GOOD)
    run_requests(engine, [request("a"), request("b")], JsonlStore(path), "rev")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "trunc')  # interrupted write is ignored
    engine2 = FakeEngine(lambda *a: GOOD)
    summary = run_requests(engine2, [request("a"), request("b"), request("c")], JsonlStore(path), "rev")
    assert summary["reused"] == 2 and summary["ok"] == 1
    assert len(engine2.calls[0]) == 1


# ----------------------------------------------------------------------------- end to end
def _key(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def responder(msgs, schema, max_tokens, seed):
    props = schema.get("properties", {})
    user = msgs[1]["content"]
    if "body_functions" in props:
        cid = user.split("Client ID: ", 1)[1].split("\n", 1)[0]
        return json.dumps(synthetic_profile(_key(cid)))
    if "chosen_candidate" in props:
        return json.dumps({"reasoning": "r", "chosen_candidate": props["chosen_candidate"]["enum"][-1]})
    if "assignment" in props:
        items = props["assignment"]["items"]["properties"]
        clients, clinicians = items["client_id"]["enum"], items["clinician_id"]["enum"]
        return json.dumps({"reasoning": "r", "assignment": [
            {"client_id": c, "clinician_id": clinicians[i % len(clinicians)]}
            for i, c in enumerate(clients)]})
    target = user.rsplit("TARGET CLIENT\n", 1)[1]
    k = _key(target)
    if "class" in props:
        return json.dumps({"reasoning": "r", "fit_issue": None, "class": 1 + k % 5})
    if "probabilities" in props:
        top = str(1 + k % 5)
        probs = {str(i): (0.6 if str(i) == top else 0.1) for i in range(1, 6)}
        return json.dumps({"reasoning": "r", "fit_issue": None, "probabilities": probs})
    return json.dumps({"reasoning": "r", "fit_issue": None, "complexity_score": round(1 + (k % 9) / 10, 2)})


def gurobi_available():
    try:
        import gurobipy as gp

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        env.dispose()
        return True
    except Exception:
        return False


@pytest.fixture
def workspace(tmp_path):
    clients = pd.DataFrame({
        "Client AlayaCare Client ID": [f"{i:04d}" for i in range(30)],
        "SP text": [f"Synthetic service plan {i}" for i in range(30)],
        "assessment text": [f"Synthetic assessment {i}" for i in range(30)],
    })
    path = tmp_path / "clients.xlsx"
    clients.to_excel(path, index=False)
    assert chunks_main(["--clients", str(path), "--deidentified-confirmed",
                        "--data-dir", str(tmp_path / "data")]) == 0
    return Workspace.under(tmp_path)


def test_chunks_file_and_icf_exclusions(workspace):
    frame = pd.read_excel(workspace.chunks_file, dtype=object)
    assert list(frame.columns) == ["chunk_id", "position", "client_id"]
    assert frame.groupby("chunk_id").size().tolist() == [20, 10]
    assert frame["client_id"].str.len().eq(4).all()  # leading zeros kept
    engine = FakeEngine(responder)
    icf.run(workspace, "qwen35-4b", engine, "rev", max_clients=5)
    rows, total = exclusions.build(workspace, ["qwen35-4b"])
    assert total == 30 and len(rows) == 30  # pilot: un-run clients exclude their whole chunk
    from dr.context import cohorts

    assert cohorts(workspace, "qwen35-4b") == {}  # incomplete chunks are excluded whole
    rows, _ = exclusions.build(workspace, ["qwen35-4b"], keep_partial_chunks=True)
    assert len(rows) == 25
    kept = cohorts(workspace, "qwen35-4b")  # fully excluded chunks are dropped, not empty
    assert [len(v) for v in kept.values()] == [5]
    record = next(iter(JsonlStore(workspace.icf("qwen35-4b")).ok().values()))
    assert record["summary_recomputed"] and record["rendered_profile_tokens"] > 0
    assert "Class" not in record["raw_text"]


@pytest.mark.skipif(not gurobi_available(), reason="Gurobi not available")
def test_end_to_end_synthetic(workspace):
    model, engine = "qwen35-4b", FakeEngine(responder)
    icf.run(workspace, model, engine, "rev")
    exclusions.build(workspace, [model])
    profiles = ["balanced"]
    rate.run(workspace, model, engine, "rev", [1, 4, 5], ["label", "distribution", "score"], profiles)
    assert len(JsonlStore(workspace.rating(model, 4, "label")).ok()) == 30
    assert len(JsonlStore(workspace.rating(model, 5, "score", "balanced")).ok()) == 30
    allocate.run(workspace, model, engine, "rev", [5], profiles)
    alloc = JsonlStore(workspace.allocation(model, 5, "balanced")).ok()
    assert len(alloc) == 2 and all(r["feasible"] for r in alloc.values())
    counts = candidates.run(workspace, model, [4, 5], ["label", "distribution", "score"], profiles)
    assert counts["failed"] == 0 and counts["written"] == 12
    pool = json.loads(workspace.candidates(model, 4, "distribution", "balanced", "1").read_text())
    assert pool["status"] == "ok" and pool["policy"] == "stochastic" and pool["n_runs"] == 20
    assert [s["seed"] for s in pool["solutions"]] == list(range(42, 62))
    assert len({json.dumps(c["canonical"]) for c in pool["candidates"]}) == pool["n_distinct"]
    again = candidates.run(workspace, model, [4, 5], ["label", "distribution", "score"], profiles)
    assert again["reused"] == 12 and again["written"] == 0
    before = len(engine.calls)
    tiebreak.run(workspace, model, engine, "rev", [4, 5], ["label", "distribution", "score"], profiles)
    records = JsonlStore(workspace.tiebreak(model, 4, "label", "balanced")).records
    for r in records.values():
        assert r["parse_ok"]
        if r["skipped"]:
            assert r["skipped_reason"] == "single_candidate"
        else:
            shown = str(r["parsed"]["chosen_candidate"])
            assert r["chosen_candidate_index"] == r["display_mapping"][shown]["candidate_index"]
    calls = sum(len(c) for c in engine.calls[before:])
    assert calls <= 12
    assert evaluate.main(["--models", model, "--benchmark-rating", f"{model}:1:label",
                          "--data-dir", str(workspace.data),
                          "--outputs-dir", str(workspace.outputs)]) == 0
    out = workspace.outputs / "evaluation" / f"{model}_1_label"
    or_rows = pd.read_csv(out / "or_rows.csv")
    assert (or_rows["proxy_regret"] >= -1e-7).all() and len(or_rows) == 12 * 20
    tb = pd.read_csv(out / "tiebreak_rows.csv")
    ok = tb[tb.parse_ok]
    assert (ok.best_candidate_regret <= ok.proxy_regret + 1e-9).all()
    assert (ok.proxy_regret <= ok.worst_candidate_regret + 1e-9).all()
