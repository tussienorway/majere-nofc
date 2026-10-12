"""Run NO-FC against any OpenAI-compatible chat endpoint (Ollama, llama.cpp server, vLLM).

Example (Ollama on your PC):
  python -m nofc.evaluate --base-url http://localhost:11434/v1 --model qwen3:8b
"""
import argparse, json, os, sys, urllib.request
from .score import score_task, summarize

WEEKDAYS = ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"]
SYSTEM = ("Du er en norsk assistent med tilgang til verktøy. Nåværende tidspunkt: {dt} ({wd}), tidssone Europe/Oslo. "
          "Bruk et verktøy bare når det trengs. Mangler nødvendig informasjon, still et oppfølgingsspørsmål i stedet for å gjette. "
          "Svar på samme målform som brukeren (bokmål eller nynorsk).")

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def tool_specs(names, tools_path):
    allt = {t["name"]: t for t in json.load(open(tools_path, encoding="utf-8"))}
    return [{"type": "function", "function": allt[n]} for n in names]

def chat(base_url, model, api_key, messages, tools, timeout):
    body = json.dumps({"model": model, "messages": messages, "tools": tools, "temperature": 0}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def parse(resp):
    m = resp["choices"][0]["message"]; calls = []
    for tc in m.get("tool_calls") or []:
        fn = tc.get("function", {}); a = fn.get("arguments") or {}
        if isinstance(a, str):
            try: a = json.loads(a)
            except json.JSONDecodeError: a = {"__invalid_json__": a}
        calls.append({"name": fn.get("name"), "arguments": a})
    return calls, m.get("content") or ""

def main(argv=None):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:11434/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    p.add_argument("--tasks", default=os.path.join(here, "data", "seed_v0.jsonl"))
    p.add_argument("--tools", default=os.path.join(here, "data", "tools.json"))
    p.add_argument("--out", default="nofc_results.jsonl")
    p.add_argument("--timeout", type=int, default=120)
    a = p.parse_args(argv)
    rows = []
    with open(a.out, "w", encoding="utf-8") as out:
        for t in load(a.tasks):
            from datetime import datetime
            d = datetime.fromisoformat(t["reference_datetime"])
            msgs = [{"role": "system", "content": SYSTEM.format(dt=t["reference_datetime"], wd=WEEKDAYS[d.weekday()])},
                    {"role": "user", "content": t["user"]}]
            try:
                calls, text = parse(chat(a.base_url, a.model, a.api_key, msgs, tool_specs(t["tools"], a.tools), a.timeout))
                err = None
            except Exception as e:  # count failures as wrong, keep going
                calls, text, err = [], "", str(e)
            r = score_task(t, calls, text); r.update(calls=calls, text=text, error=err)
            rows.append(r); out.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{t['id']}: {'OK ' if r['correct'] else 'ERR'}", file=sys.stderr)
    s = summarize(rows); s["model"] = a.model
    print(json.dumps(s, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
