import json, os
from nofc.score import score_task, summarize, arg_match
H = os.path.dirname(os.path.dirname(__file__))
TASKS = [json.loads(l) for l in open(os.path.join(H, "data", "seed_v0.jsonl"), encoding="utf-8")]
TOOLS = {t["name"]: t for t in json.load(open(os.path.join(H, "data", "tools.json"), encoding="utf-8"))}

def test_dataset_valid():
    assert 30 <= len(TASKS) <= 50
    assert len({t["id"] for t in TASKS}) == len(TASKS)
    for t in TASKS:
        assert t["variant"] in ("nb", "nn")
        assert all(n in TOOLS for n in t["tools"])
        e = t["expected"]
        if e["type"] == "call":
            c = e["calls"][0]; assert c["name"] in t["tools"]
            props = TOOLS[c["name"]]["parameters"]["properties"]
            assert set(c["arguments"]) <= set(props)
            assert set(TOOLS[c["name"]]["parameters"].get("required", [])) <= set(c["arguments"]) or c["name"] == "lookup_company"
    types = {t["expected"]["type"] for t in TASKS}; assert types == {"call", "no_call", "clarify"}

def test_perfect_oracle_scores_one():
    rows = []
    for t in TASKS:
        e = t["expected"]
        if e["type"] == "call": rows.append(score_task(t, e["calls"], ""))
        else: rows.append(score_task(t, [], "Kva meiner du?" if e["type"] == "clarify" else "Ok."))
    assert summarize(rows)["all"]["exact_match"] == 1.0

def test_normalisation():
    assert arg_match("to", "+4791234567", "912 34 567")
    assert arg_match("time", "08:30", "8.30")
    assert arg_match("org_number", "923609016", "923 609 016")
    assert not arg_match("date", "2026-10-13", "2026-10-14")

def test_wrong_call_on_no_call_task():
    t = next(x for x in TASKS if x["expected"]["type"] == "no_call")
    assert not score_task(t, [{"name": "get_weather", "arguments": {}}], "")["correct"]
