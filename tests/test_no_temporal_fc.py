"""GPU-free, network-free tests for train/make_no_temporal_fc.py and its wiring in train/build_data.py."""
import json, os, re, subprocess, sys

import pytest

H = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(H, "train"))
import make_no_temporal_fc as g  # noqa: E402
from nofc.chatfmt import CALL_RE, ROLE_TAG_RE, parse_completion  # noqa: E402

N = 300
EX = g.generate(N, seed=0)
TOOLS = {t["name"]: t for t in json.load(open(os.path.join(H, "data", "tools.json"), encoding="utf-8"))}


def _eval_rows():
    rows = []
    for name in ("seed_v0.jsonl", "nofc_v1.jsonl", "nofc_hard_temporal.jsonl"):
        with open(os.path.join(H, "data", name), encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


EVAL_USERS = [r["user"] for r in _eval_rows()]


def test_count_and_variant_balance():
    assert len(EX) == N
    nb = sum(e["variant"] == "nb" for e in EX)
    assert abs(nb - (N - nb)) <= 1
    assert {e["type"] for e in EX} == {"call", "clarify", "no_call"}
    assert len({e["id"] for e in EX}) == N


def test_deterministic_and_small_n():
    assert [e["text"] for e in g.generate(40, seed=3)] == [e["text"] for e in g.generate(40, seed=3)]
    assert len(g.generate(1)) == 1 and g.generate(0) == []


def test_users_unique():
    assert len({e["user"].strip().casefold() for e in EX}) == N
    assert len({g.norm(e["user"]) for e in EX}) == N


def test_users_disjoint_from_eval_sets():
    assert len(EVAL_USERS) >= 200
    exact = {u.strip().casefold() for u in EVAL_USERS}
    normed = {g.norm(u) for u in EVAL_USERS}
    for e in EX:
        assert e["user"].strip().casefold() not in exact, e["user"]
        assert g.norm(e["user"]) not in normed, e["user"]


def test_no_near_duplicates_of_eval_users():
    sets = [g._tokens(u) for u in EVAL_USERS]
    for e in EX:
        t = g._tokens(e["user"])
        worst = max(len(t & s) / len(t | s) for s in sets)
        assert worst < g.NEAR_DUP_JACCARD, (e["user"], worst)


def test_guard_rejects_exact_case_and_paraphrase():
    guard = g.EvalGuard(["Legg inn blodprøve på fredag."])
    assert guard.clashes("  legg inn BLODPRØVE på fredag.  ")
    assert guard.clashes("Legg inn blodprøve på fredag")
    assert guard.clashes("Blodprøve på fredag, legg inn.")
    assert not guard.clashes("Sett opp tannlegetime torsdag klokka ti.")


def test_guard_filters_generation_when_eval_overlaps(tmp_path):
    clashing = [e["user"] for e in g.generate(60, seed=0)]
    for name in g.EVAL_FILES:
        with open(tmp_path / name, "w", encoding="utf-8") as f:
            for u in clashing: f.write(json.dumps({"user": u}, ensure_ascii=False) + "\n")
    again = g.generate(60, seed=0, data_dir=str(tmp_path))
    assert not {e["user"] for e in again} & set(clashing)


def test_weekday_gold_matches_conventions():
    assert g.weekday_date("mandag") == "2026-10-19"
    assert g.weekday_date("tirsdag") == "2026-10-13"
    assert g.weekday_date("onsdag") == "2026-10-14"
    assert g.weekday_date("torsdag") == "2026-10-15"
    assert g.weekday_date("fredag") == "2026-10-16"
    assert g.weekday_date("lørdag") == "2026-10-17" and g.weekday_date("laurdag") == "2026-10-17"
    assert g.weekday_date("søndag") == "2026-10-18" and g.weekday_date("sundag") == "2026-10-18"
    assert g.weekday_date("måndag") == "2026-10-19" and g.weekday_date("tysdag") == "2026-10-13"
    assert g.offset_date(0) == "2026-10-12" and g.offset_date(3) == "2026-10-15"


def test_clock_gold_matches_conventions():
    assert g.clock_gold("halv", 10, "forenoon") == "09:30"
    assert g.clock_gold("halv", 4, "afternoon") == "15:30"
    assert g.clock_gold("halv", 7, "evening") == "18:30"
    assert g.clock_gold("halv", 9, "evening") == "20:30"
    assert g.clock_gold("halv", 12, "forenoon") == "11:30"
    assert g.clock_gold("halv", 1) == "12:30"
    assert g.clock_gold("kvart over", 8, "morning") == "08:15"
    assert g.clock_gold("kvart over", 10) == "10:15"
    assert g.clock_gold("kvart over", 1, "afternoon") == "13:15"
    assert g.clock_gold("kvart på", 10) == "09:45"
    assert g.clock_gold("kvart på", 5, "afternoon") == "16:45"
    assert g.clock_gold("kvart på", 8, "evening") == "19:45"
    with pytest.raises(ValueError):
        g.clock_gold("halv", 4)  # bare "halv fire" is ambiguous and never generated


NUM = {w: n for n, w in g.NUM_WORDS["nb"].items()} | {w: n for n, w in g.NUM_WORDS["nn"].items()}
CLOCK = re.compile(r"\b(halv|kvart over|kvart på)\s+(" + "|".join(NUM) + r")\b")


def _daypart(user):
    if re.search(r"om morg[eo]n(en)?\b", user): return "morning"
    if re.search(r"f[oø]remiddag|formiddag", user): return "forenoon"
    if re.search(r"ettermiddag", user): return "afternoon"
    if re.search(r"kvel[dl]", user): return "evening"
    return None


def _hhmm(e):
    a = e["call"]["arguments"]
    return a.get("time") or a.get("after_time") or (a["datetime"].split("T")[1] if "datetime" in a else None)


def test_call_clock_gold_matches_user_phrase_independently():
    seen = 0
    for e in EX:
        if e["type"] != "call": continue
        m = CLOCK.search(e["user"])
        if not m: continue
        seen += 1
        assert _hhmm(e) == g.clock_gold(m[1], NUM[m[2]], _daypart(e["user"])), e["user"]
    assert seen >= 100


def test_call_weekday_gold_matches_user_name():
    names = "|".join(sorted(g.WD_INDEX, key=len, reverse=True))
    seen = 0
    for e in EX:
        if e["type"] != "call": continue
        m = re.search(r"\b(" + names + r")\b", e["user"])
        if not m: continue
        seen += 1
        a = e["call"]["arguments"]
        date = a.get("date") or a["datetime"].split("T")[0]
        assert date == g.weekday_date(m[1]), e["user"]
    assert seen >= 80


def test_failure_mode_coverage():
    kinds = {e["kind"] for e in EX}
    assert {"cal_clock", "rem_clock", "book_clock", "bus_clock", "wx_weekday", "wx_offset"} <= kinds   # A, B
    assert {"cal_plain", "wx_nodate", "bus_plain"} <= kinds                                          # C
    cal = [e for e in EX if e["call"] and e["call"]["name"] == "create_calendar_event"]
    assert sum("duration_minutes" not in e["call"]["arguments"] for e in cal) > len(cal) / 2
    assert any(e["type"] == "call" and e["call"]["arguments"].get("line") for e in EX)
    assert any(e["type"] == "call" and e["call"]["name"] == "get_weather" and "date" not in e["call"]["arguments"] for e in EX)
    assert any(e["type"] == "call" and e["call"]["name"] == "search_bus_departures" and "after_time" not in e["call"]["arguments"] for e in EX)
    assert sum(e["variant"] == "nn" and e["type"] == "call" for e in EX) > 50


def test_clarify_examples_have_question_and_no_tool_call():
    cl = [e for e in EX if e["type"] == "clarify"]
    assert len(cl) >= 10
    for e in cl:
        assert "?" in e["assistant"] and "<tool_call>" not in e["assistant"]
        assert "<tool_call>" not in e["text"].split("<|assistant|>\n", 1)[1]
        assert e["call"] is None


def test_clarify_only_when_a_required_argument_is_missing():
    for e in EX:
        if e["type"] != "clarify": continue
        tool = e["kind"].split("_")[1]
        name = {"cal": "create_calendar_event", "book": "book_table", "rem": "set_reminder"}[tool]
        assert name in e["tools"]
        if e["kind"] == "clarify_cal_no_time" or e["kind"] == "clarify_rem_no_time" or e["kind"].startswith("clarify_book_no_time"):
            assert not CLOCK.search(e["user"]) and not re.search(r"\bkl(okka|\.)", e["user"])


def test_no_call_examples_have_no_tool_call():
    nc = [e for e in EX if e["type"] == "no_call"]
    assert len(nc) >= 10
    for e in nc:
        assert "<tool_call>" not in e["assistant"] and e["assistant"].strip() and e["call"] is None


def test_no_call_answers_are_correct():
    for e in EX:
        if e["kind"] == "no_call_clock":
            m = CLOCK.search(e["user"]); assert m
            assert g.clock_gold(m[1], NUM[m[2]]) in e["assistant"]
        if e["kind"] == "no_call_weekday" and "dato" in e["user"].lower():
            name = re.search(r"\b(" + "|".join(g.WD_INDEX) + r")\b", e["user"])[1]
            assert str(int(g.weekday_date(name)[-2:])) + ". oktober 2026" in e["assistant"]


def test_call_examples_have_valid_tool_call_json():
    calls = [e for e in EX if e["type"] == "call"]
    assert len(calls) > 150
    for e in calls:
        assert e["assistant"].count("<tool_call>") == 1 and e["assistant"].count("</tool_call>") == 1
        m = CALL_RE.fullmatch(e["assistant"]); assert m
        j = json.loads(m.group(1))
        assert j["name"] == e["call"]["name"] and j["arguments"] == e["call"]["arguments"]
        assert j["name"] in TOOLS and j["name"] in e["tools"]
        spec = TOOLS[j["name"]]["parameters"]
        assert set(spec["required"]) <= set(j["arguments"]) <= set(spec["properties"])
        for k, v in j["arguments"].items():
            assert isinstance(v, {"string": str, "integer": int, "number": (int, float)}[spec["properties"][k]["type"]])
        content, parsed = parse_completion(e["assistant"])
        assert content == "" and parsed == [{"name": j["name"], "arguments": j["arguments"]}]


def test_date_and_time_arg_formats():
    for e in EX:
        if e["type"] != "call": continue
        a = e["call"]["arguments"]
        if "date" in a: assert re.fullmatch(r"2026-10-\d\d", a["date"])
        for k in ("time", "after_time"):
            if k in a: assert re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", a[k])
        if "datetime" in a: assert re.fullmatch(r"2026-10-\d\dT([01]\d|2[0-3]):[0-5]\d", a["datetime"])


def test_system_prompt_matches_eval_tool_mode_prompt():
    from nofc.evaluate import build_messages, tool_specs
    for e in EX[:40]:
        task = {"reference_datetime": "2026-10-12T10:00", "user": e["user"]}
        specs = tool_specs(e["tools"], os.path.join(H, "data", "tools.json"))
        assert build_messages(task, specs, "prompt")[0]["content"] == e["system"]


def test_text_format_has_role_tags_and_end():
    for e in EX:
        t = e["text"]
        assert t.startswith("<|system|>\n") and "<|user|>\n" + e["user"] + "\n" in t
        assert t.endswith("<|assistant|>\n" + e["assistant"] + "\n<|end|>")
        assert re.findall(r"<\|(system|user|assistant|tool)\|>", t) == ["system", "user", "assistant"]
        assert "Tilgjengelige verktøy" in t


def _run(args):
    return subprocess.run([sys.executable, os.path.join(H, "train", "build_data.py"), *args], capture_output=True,
                          text=True, encoding="utf-8", cwd=H, check=True)


@pytest.mark.parametrize("n", [1, 50, 120])
def test_build_data_writes_n_valid_lines(tmp_path, n):
    out = tmp_path / "mix.jsonl"
    r = _run(["--n-alpaca", "0", "--n-glaive", "0", "--n-no-temporal", str(n), "--out", str(out)])
    assert f"wrote {n} examples" in r.stdout
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n
    for l in lines:
        row = json.loads(l)
        assert list(row) == ["text"] and ROLE_TAG_RE.search(row["text"])
        assert "<|system|>" in row["text"] and "<|user|>" in row["text"] and "<|assistant|>" in row["text"]


def test_build_data_zero_disables_everything(tmp_path):
    out = tmp_path / "empty.jsonl"
    _run(["--n-alpaca", "0", "--n-glaive", "0", "--n-no-temporal", "0", "--out", str(out)])
    assert out.read_text(encoding="utf-8") == ""


def test_build_data_default_and_flag_documented():
    import ast
    src = open(os.path.join(H, "train", "build_data.py"), encoding="utf-8").read()
    assert "--n-no-temporal" in ast.get_docstring(ast.parse(src))
    r = subprocess.run([sys.executable, os.path.join(H, "train", "build_data.py"), "--help"], capture_output=True, text=True, cwd=H)
    assert "--n-no-temporal" in r.stdout and r.returncode == 0
    assert re.search(r'"--n-no-temporal", type=int, default=300', src)
    assert re.search(r'"--n-alpaca", type=int, default=4000', src) and re.search(r'"--n-glaive", type=int, default=4000', src)


def test_build_data_mix_includes_no_temporal_rows_before_shuffle(monkeypatch):
    import build_data as bd
    monkeypatch.setattr(bd, "alpaca", lambda n, s: iter(["ALPACA"] * n))
    monkeypatch.setattr(bd, "glaive", lambda n, s: iter(["GLAIVE"] * n))
    rows = bd.build_rows(3, 2, 10, seed=0)
    assert len(rows) == 15 and rows.count("ALPACA") == 3 and rows.count("GLAIVE") == 2
    assert sum("<tool_call>" in r or "<|assistant|>" in r for r in rows if r not in ("ALPACA", "GLAIVE")) == 10
    first_ten = rows[:10]
    assert any(r in ("ALPACA", "GLAIVE") for r in first_ten) and any(r not in ("ALPACA", "GLAIVE") for r in first_ten)
