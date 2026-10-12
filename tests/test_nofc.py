import json, os
from nofc.score import score_task, summarize, arg_match
H = os.path.dirname(os.path.dirname(__file__))
def _load(n): return [json.loads(l) for l in open(os.path.join(H, "data", n), encoding="utf-8")]
TASKS = _load("seed_v0.jsonl")
V1 = _load("nofc_v1.jsonl")
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

def test_v1_valid_and_split():
    assert len(V1) == 120 and len({t["id"] for t in V1}) == 120
    assert sum(t["split"] == "test" for t in V1) == 40
    for t in V1:
        assert all(n in TOOLS for n in t["tools"])
        e = t["expected"]
        if e["type"] == "call":
            c = e["calls"][0]; assert c["name"] in t["tools"]
            assert set(c["arguments"]) <= set(TOOLS[c["name"]]["parameters"]["properties"])

def test_prompt_mode_parse():
    from nofc.evaluate import parse_prompt
    resp = {"choices":[{"message":{"content":'<think>hmm</think><tool_call>{"name":"get_weather","arguments":{"location":"Bergen"}}</tool_call>'}}]}
    calls, text = parse_prompt(resp)
    assert calls == [{"name":"get_weather","arguments":{"location":"Bergen"}}] and text == ""
