"""Build the pilot SFT mix as plain text in one uniform format for both base models.

Sources (licences checked): NbAiLab/norwegian-alpaca (CC BY 4.0), glaiveai/glaive-function-calling-v2 (Apache-2.0).
NB: glaive is English. The pilot uses it as-is to measure the baseline effect; the Norwegian translation of
function-calling data is a separate later step (see README roadmap).
Tool calls use the <tool_call>{json}</tool_call> convention (also Qwen3's native format).
"""
import argparse, json, re, random
from datasets import load_dataset

def fmt(system, turns):
    s = f"<|system|>\n{system}\n" if system else ""
    for role, text in turns:
        s += f"<|{role}|>\n{text}\n"
    return s + "<|end|>"

def alpaca(n, seed):
    ds = load_dataset("NbAiLab/norwegian-alpaca", split="train").shuffle(seed=seed).select(range(n))
    for r in ds:
        u = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        yield fmt("Du er en hjelpsom norsk assistent.", [("user", u), ("assistant", r["output"])])

def glaive(n, seed):
    ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train").shuffle(seed=seed).select(range(n))
    for r in ds:
        sysmsg = r["system"].replace("SYSTEM:", "").strip()
        turns = []
        for part in re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)", r["chat"]):
            part = part.strip()
            if part in ("USER:", "ASSISTANT:", "FUNCTION RESPONSE:"):
                role = {"USER:": "user", "ASSISTANT:": "assistant", "FUNCTION RESPONSE:": "tool"}[part]; continue
            if not part: continue
            text = part.replace("<|endoftext|>", "").strip()
            m = re.match(r"<functioncall>\s*(\{.*\})\s*$", text, re.S)
            if m:
                raw = m.group(1).replace("'{", "{").replace("}'", "}")
                try: text = "<tool_call>" + json.dumps(json.loads(raw), ensure_ascii=False) + "</tool_call>"
                except json.JSONDecodeError: turns = None; break
            turns.append((role, text))
        if turns: yield fmt(sysmsg, turns)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-alpaca", type=int, default=4000)
    p.add_argument("--n-glaive", type=int, default=4000)
    p.add_argument("--out", default="pilot_mix.jsonl")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rows = list(alpaca(a.n_alpaca, a.seed)) + list(glaive(a.n_glaive, a.seed))
    random.Random(a.seed).shuffle(rows)
    with open(a.out, "w", encoding="utf-8") as f:
        for t in rows: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {a.out}")
