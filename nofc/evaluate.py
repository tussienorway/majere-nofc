"""Run NO-FC against any OpenAI-compatible chat endpoint (Ollama, llama.cpp server, vLLM).

  python -m nofc.evaluate --model qwen3:8b                         # native tool calling
  python -m nofc.evaluate --model qwen3:8b --no-think              # Qwen3 soft switch /no_think
  python -m nofc.evaluate --model <gguf> --tool-mode prompt        # tools in system prompt, parse <tool_call>
  python -m nofc.evaluate --model qwen3:8b --tasks data/nofc_v1.jsonl --split test
"""
import argparse, json, os, re, sys, time, urllib.request
from datetime import datetime
from .score import score_task, summarize

WEEKDAYS = ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"]
SYSTEM = ("Du er en norsk assistent med tilgang til verktøy. Nåværende tidspunkt: {dt} ({wd}), tidssone Europe/Oslo. "
          "Bruk et verktøy bare når det trengs. Mangler nødvendig informasjon, still et oppfølgingsspørsmål i stedet for å gjette. "
          "Svar på samme målform som brukeren (bokmål eller nynorsk).")
PROMPT_TOOLS = ("\n\nTilgjengelige verktøy (JSON Schema):\n{tools}\n\n"
                "For å kalle et verktøy, svar KUN med:\n<tool_call>{{\"name\": \"<navn>\", \"arguments\": {{...}}}}</tool_call>\n"
                "Hvis ingen verktøy trengs, svar vanlig tekst uten <tool_call>.")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def tool_specs(names, tools_path):
    allt = {t["name"]: t for t in json.load(open(tools_path, encoding="utf-8"))}
    return [{"type": "function", "function": allt[n]} for n in names]

def chat(base_url, model, api_key, messages, tools, timeout):
    body = {"model": model, "messages": messages, "temperature": 0}
    if tools: body["tools"] = tools
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def _args(a):
    if isinstance(a, str):
        try: return json.loads(a)
        except json.JSONDecodeError: return {"__invalid_json__": a}
    return a or {}

def parse_native(resp):
    m = resp["choices"][0]["message"]
    calls = [{"name": tc.get("function", {}).get("name"), "arguments": _args(tc.get("function", {}).get("arguments"))}
             for tc in m.get("tool_calls") or []]
    return calls, THINK_RE.sub("", m.get("content") or "").strip()

def parse_prompt(resp):
    text = THINK_RE.sub("", resp["choices"][0]["message"].get("content") or "").strip()
    calls = []
    for raw in CALL_RE.findall(text):
        try:
            j = json.loads(raw); calls.append({"name": j.get("name"), "arguments": _args(j.get("arguments"))})
        except json.JSONDecodeError:
            calls.append({"name": None, "arguments": {"__invalid_json__": raw}})
    return calls, CALL_RE.sub("", text).strip()

def main(argv=None):
    try: sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError: pass
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    p.add_argument("--tasks", default=os.path.join(here, "data", "seed_v0.jsonl"))
    p.add_argument("--split", default=None, help="only tasks with this split (dev/test)")
    p.add_argument("--tools", default=os.path.join(here, "data", "tools.json"))
    p.add_argument("--tool-mode", choices=["native", "prompt"], default="native")
    p.add_argument("--no-think", action="store_true", help="append /no_think (Qwen3 soft switch)")
    p.add_argument("--out", default="nofc_results.jsonl")
    p.add_argument("--summary", default=None, help="write summary JSON here (UTF-8); default <out>.summary.json")
    p.add_argument("--timeout", type=int, default=300)
    a = p.parse_args(argv)
    tasks = [t for t in load(a.tasks) if a.split is None or t.get("split") == a.split]
    rows = []; t_start = time.time()
    with open(a.out, "w", encoding="utf-8") as out:
        for t in tasks:
            d = datetime.fromisoformat(t["reference_datetime"])
            specs = tool_specs(t["tools"], a.tools)
            system = SYSTEM.format(dt=t["reference_datetime"], wd=WEEKDAYS[d.weekday()])
            if a.tool_mode == "prompt":
                system += PROMPT_TOOLS.format(tools=json.dumps([s["function"] for s in specs], ensure_ascii=False))
            if a.no_think: system += " /no_think"
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": t["user"]}]
            t0 = time.time()
            try:
                resp = chat(a.base_url, a.model, a.api_key, msgs, specs if a.tool_mode == "native" else None, a.timeout)
                calls, text = (parse_native if a.tool_mode == "native" else parse_prompt)(resp); err = None
            except Exception as e:
                calls, text, err = [], "", str(e)
            r = score_task(t, calls, text)
            r.update(split=t.get("split"), calls=calls, text=text, error=err, seconds=round(time.time() - t0, 2))
            rows.append(r); out.write(json.dumps(r, ensure_ascii=False) + "\n"); out.flush()
            print(f"{t['id']}: {'OK ' if r['correct'] else 'ERR'}{' (' + err[:80] + ')' if err else ''}", file=sys.stderr)
    s = summarize(rows)
    s.update(model=a.model, tool_mode=a.tool_mode, no_think=a.no_think, tasks=os.path.basename(a.tasks), split=a.split,
             errors=sum(1 for r in rows if r["error"]), wall_seconds=round(time.time() - t_start, 1),
             date=datetime.now().strftime("%Y-%m-%d"))
    path = a.summary or (os.path.splitext(a.out)[0] + ".summary.json")
    with open(path, "w", encoding="utf-8") as f: json.dump(s, f, indent=2, ensure_ascii=False)
    print(json.dumps(s, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
