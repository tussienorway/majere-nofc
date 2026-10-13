"""GPU-free tests: serving prompt == training format, tool-call parsing, and the OpenAI-compatible protocol."""
import importlib, json, os, sys, threading, types

import pytest

from nofc import chatfmt
from nofc.chatfmt import (END, build_prompt, clean_completion, fmt, format_tool_call, generation_prompt,
                          parse_completion, wrap_tool_call)
from nofc.evaluate import build_messages, parse_native, parse_prompt, tool_specs
from nofc.serve import handle_chat, make_server

H = os.path.dirname(os.path.dirname(__file__))
TOOLS_PATH = os.path.join(H, "data", "tools.json")
V1 = [json.loads(l) for l in open(os.path.join(H, "data", "nofc_v1.jsonl"), encoding="utf-8")]


def legacy_fmt(system, turns):
    """Verbatim copy of fmt() as it was in train/build_data.py before it moved to nofc/chatfmt.py."""
    s = f"<|system|>\n{system}\n" if system else ""
    for role, text in turns:
        s += f"<|{role}|>\n{text}\n"
    return s + "<|end|>"


def test_fmt_golden():
    assert fmt("S", [("user", "Hei"), ("assistant", "Hallo")]) == "<|system|>\nS\n<|user|>\nHei\n<|assistant|>\nHallo\n<|end|>"
    assert fmt(None, [("user", "Hei")]) == "<|user|>\nHei\n<|end|>"


@pytest.mark.parametrize("system", [None, "", "Du er en hjelpsom norsk assistent.", "æøå\nflere linjer"])
def test_fmt_matches_legacy(system):
    turns = [("user", "Hva er klokka?"), ("assistant", "<tool_call>{\"name\": \"x\"}</tool_call>"), ("tool", "{\"t\": 1}"), ("assistant", "Ok.")]
    assert fmt(system, turns) == legacy_fmt(system, turns)


def test_generation_prompt_is_prefix_of_training_text():
    system, turns = "Sys", [("user", "Hei"), ("assistant", "Hallo"), ("user", "Takk")]
    answer = "Bare hyggelig!"
    assert fmt(system, turns + [("assistant", answer)]) == generation_prompt(system, turns) + answer + "\n" + END
    assert generation_prompt(system, turns).endswith("<|user|>\nTakk\n<|assistant|>\n")


@pytest.mark.parametrize("mode", ["prompt", "native"])
def test_nofc_prompt_byte_identical_to_training_format(mode):
    """For every v1 task, the prompt the server builds is the training formatter's text up to the assistant tag."""
    for t in V1:
        specs = tool_specs(t["tools"], TOOLS_PATH)
        msgs = build_messages(t, specs, "prompt")  # system prompt incl. tool schemas, as in training-time glaive data
        expected_system = msgs[0]["content"]
        if mode == "prompt":
            served = build_prompt(msgs, None)
        else:
            native = build_messages(t, specs, "native")
            assert native[0]["content"] in expected_system  # native system = prompt-mode system minus the tools block
            served = build_prompt(native, specs)
        train_text = fmt(expected_system, [("user", t["user"]), ("assistant", "SVAR")])
        assert train_text == f"<|system|>\n{expected_system}\n<|user|>\n{t['user']}\n<|assistant|>\nSVAR\n<|end|>"
        assert served + "SVAR\n" + END == train_text


def test_native_and_prompt_modes_give_same_prompt():
    for t in V1:
        specs = tool_specs(t["tools"], TOOLS_PATH)
        assert build_prompt(build_messages(t, specs, "prompt")) == build_prompt(build_messages(t, specs, "native"), specs)


def test_tools_not_injected_twice():
    t = V1[0]; specs = tool_specs(t["tools"], TOOLS_PATH)
    msgs = build_messages(t, specs, "prompt")
    assert build_prompt(msgs, specs) == build_prompt(msgs, None)


def test_tools_without_system_message():
    specs = tool_specs(V1[0]["tools"], TOOLS_PATH)
    p = build_prompt([{"role": "user", "content": "Hei"}], specs)
    assert p.startswith("<|system|>\nTilgjengelige verktøy (JSON Schema):") and p.endswith("<|user|>\nHei\n<|assistant|>\n")


def test_multi_turn_with_tool_calls_matches_glaive_style_training_text():
    args = {"location": "Bergen", "note": "æøå {x}"}
    msgs = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "Vær i Bergen?"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "1", "type": "function", "function": {"name": "get_weather", "arguments": json.dumps(args)}}]},
            {"role": "tool", "content": "{\"temp\": 9}"},
            {"role": "user", "content": [{"type": "text", "text": "Og i "}, {"type": "text", "text": "Oslo?"}]}]
    turns = [("user", "Vær i Bergen?"), ("assistant", wrap_tool_call({"name": "get_weather", "arguments": args})),
             ("tool", "{\"temp\": 9}"), ("user", "Og i Oslo?")]
    assert build_prompt(msgs) == legacy_fmt("SYS", turns)[:-len(END)] + "<|assistant|>\n"


def test_unsupported_role():
    with pytest.raises(ValueError): build_prompt([{"role": "function", "content": "x"}])


def test_think_constants_come_from_evaluate():
    from nofc import evaluate
    assert chatfmt.CALL_RE is evaluate.CALL_RE and chatfmt.PROMPT_TOOLS is evaluate.PROMPT_TOOLS


def test_build_data_uses_shared_formatter(monkeypatch):
    """train/build_data.py (with `datasets` stubbed) emits exactly the pre-refactor glaive text."""
    row = {"system": "SYSTEM: Du har funksjoner. {\"name\": \"get_weather\"}",
           "chat": "USER: Vær i Bergen?  <|endoftext|> ASSISTANT: <functioncall> {\"name\": \"get_weather\", "
                   "\"arguments\": '{\"location\": \"Bergen\"}'} <|endoftext|> FUNCTION RESPONSE: {\"temp\": 9} "
                   "ASSISTANT: Det er 9 grader. <|endoftext|>"}

    class DS(list):
        def shuffle(self, seed): return self
        def select(self, r): return self

    fake = types.ModuleType("datasets"); fake.load_dataset = lambda *a, **k: DS([row])
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.syspath_prepend(os.path.join(H, "train"))
    sys.modules.pop("build_data", None)
    bd = importlib.import_module("build_data")
    out = list(bd.glaive(1, 0))
    call = '<tool_call>{"name": "get_weather", "arguments": {"location": "Bergen"}}</tool_call>'
    assert out == [legacy_fmt("Du har funksjoner. {\"name\": \"get_weather\"}",
                              [("user", "Vær i Bergen?"), ("assistant", call), ("tool", '{"temp": 9}'), ("assistant", "Det er 9 grader.")])]
    assert bd.fmt is chatfmt.fmt
    sys.modules.pop("build_data", None)


@pytest.mark.parametrize("name,args", [
    ("get_weather", {"location": "Bergen", "date": "2026-10-13"}),
    ("send_sms", {"to": "+4791234567", "message": "Hei på deg, være ålreit {ok}"}),
    ("create_calendar_event", {"title": "Tannlege", "date": "2026-10-16", "time": "08:30", "duration_minutes": 45}),
    ("noop", {}),
    ("nested", {"a": {"b": [1, 2, {"c": "}</tool_call"}]}}),
])
def test_tool_call_round_trip(name, args):
    text = "Jeg sjekker." + format_tool_call(name, args)
    content, calls = parse_completion(text + "\n" + END)
    assert calls == [{"name": name, "arguments": args}] and content == "Jeg sjekker."
    assert "\\u" not in format_tool_call(name, args)  # ensure_ascii=False, as in training


def test_parse_completion_edge_cases():
    two = format_tool_call("a", {"x": 1}) + "\n" + format_tool_call("b", {"y": 2})
    assert parse_completion(two)[1] == [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]
    assert parse_completion('<tool_call>{"name": "a", "arguments": "{\\"x\\": 1}"}</tool_call>')[1] == [{"name": "a", "arguments": {"x": 1}}]
    assert parse_completion('<tool_call>{"name": "a", "arguments": </tool_call>') == ('<tool_call>{"name": "a", "arguments": </tool_call>', [])
    assert parse_completion('<tool_call>{"arguments": {}}</tool_call>')[1] == []
    assert parse_completion("Hvilken dato mener du?<|end|>\n<|user|>\nI morgen") == ("Hvilken dato mener du?", [])
    assert parse_completion("<think>x</think>Svar<|user|>\nhallusinert") == ("Svar", [])
    assert clean_completion("A<|end|>B") == "A"


def _ask(body, raw):
    seen = {}
    def gen(prompt, n): seen["prompt"], seen["n"] = prompt, n; return raw
    return handle_chat(body, gen, "m"), seen


def test_handle_chat_native_round_trips_through_evaluate_parser():
    t = V1[0]; specs = tool_specs(t["tools"], TOOLS_PATH)
    raw = format_tool_call("get_weather", {"location": "Bergen", "date": "2026-10-13"}) + "\n" + END
    resp, seen = _ask({"model": "m", "temperature": 0, "messages": build_messages(t, specs, "native"), "tools": specs}, raw)
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert parse_native(resp) == ([{"name": "get_weather", "arguments": {"location": "Bergen", "date": "2026-10-13"}}], "")
    assert seen["prompt"].endswith("<|user|>\n" + t["user"] + "\n<|assistant|>\n") and seen["n"] == 512


def test_handle_chat_prompt_mode_returns_tag_text():
    t = V1[0]; specs = tool_specs(t["tools"], TOOLS_PATH)
    raw = "Okei. " + format_tool_call("get_weather", {"location": "Bergen"}) + END + "<|user|>\nrest"
    resp, _ = _ask({"model": "m", "messages": build_messages(t, specs, "prompt"), "max_tokens": 64}, raw)
    assert parse_prompt(resp) == ([{"name": "get_weather", "arguments": {"location": "Bergen"}}], "Okei.")


def test_handle_chat_clarify_has_no_tool_calls():
    t = V1[0]; specs = tool_specs(t["tools"], TOOLS_PATH)
    resp, _ = _ask({"model": "m", "messages": build_messages(t, specs, "native"), "tools": specs}, "Hvilket sted?\n" + END)
    assert parse_native(resp) == ([], "Hvilket sted?") and resp["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("mode", ["native", "prompt"])
def test_evaluate_against_http_server(tmp_path, mode):
    """nofc.evaluate talks to nofc.serve over real HTTP (fake generator instead of a GPU model)."""
    from nofc import evaluate
    prompts = []
    def gen(prompt, n):
        prompts.append(prompt); return format_tool_call("get_weather", {"location": "Bergen"}) + "\n" + END
    srv = make_server(gen, "fake", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "r.jsonl"
        evaluate.main(["--base-url", f"http://127.0.0.1:{srv.server_address[1]}/v1", "--model", "fake", "--tasks",
                       os.path.join(H, "data", "nofc_v1.jsonl"), "--split", "test", "--tool-mode", mode, "--out", str(out)])
    finally:
        srv.shutdown(); srv.server_close()
    s = json.load(open(tmp_path / "r.summary.json", encoding="utf-8"))
    assert s["errors"] == 0 and len(prompts) == 40
    rows = [json.loads(l) for l in open(out, encoding="utf-8")]
    assert all(r["calls"] == [{"name": "get_weather", "arguments": {"location": "Bergen"}}] for r in rows)
    assert all(p.startswith("<|system|>\nDu er en norsk assistent") and p.endswith("<|assistant|>\n") for p in prompts)
