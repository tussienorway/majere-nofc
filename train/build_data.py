"""Build the pilot SFT mix as plain text in one uniform format for both base models.

Sources (licences checked): NbAiLab/norwegian-alpaca (CC BY 4.0), glaiveai/glaive-function-calling-v2 (Apache-2.0).
NB: glaive is English. The pilot uses it as-is to measure the baseline effect; the Norwegian translation of
function-calling data is a separate later step (see README roadmap).
Tool calls use the <tool_call>{json}</tool_call> convention (also Qwen3's native format).

Flags: --n-alpaca N, --n-glaive N (HF downloads; 0 skips the source and needs no network), --out, --seed, and
--n-no-temporal N (default 300; 0 disables): N procedurally generated Norwegian (Bokmål + Nynorsk) temporal
function-calling dialogues from train/make_no_temporal_fc.py. They target the three failure modes of
results/v1/error_analysis.md (A relative weekday / day offset, B halv/kvart clock phrases, C over-clarifying on optional
args) plus clarify/no_call controls, are generated offline, and are guaranteed disjoint from the eval sets
(seed_v0, nofc_v1, nofc_hard_temporal: no equal or near-duplicate user strings).
"""
import argparse, json, os, re, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nofc.chatfmt import fmt, wrap_tool_call  # shared with nofc/serve.py so training and serving text cannot drift
from make_no_temporal_fc import generate as no_temporal

def alpaca(n, seed):
    from datasets import load_dataset
    ds = load_dataset("NbAiLab/norwegian-alpaca", split="train").shuffle(seed=seed).select(range(n))
    for r in ds:
        u = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        yield fmt("Du er en hjelpsom norsk assistent.", [("user", u), ("assistant", r["output"])])

def glaive(n, seed):
    from datasets import load_dataset
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
                try: text = wrap_tool_call(json.loads(raw))
                except json.JSONDecodeError: turns = None; break
            turns.append((role, text))
        if turns: yield fmt(sysmsg, turns)

def build_rows(n_alpaca, n_glaive, n_no_temporal, seed):
    rows = []
    if n_alpaca > 0: rows += list(alpaca(n_alpaca, seed))
    if n_glaive > 0: rows += list(glaive(n_glaive, seed))
    if n_no_temporal > 0: rows += [ex["text"] for ex in no_temporal(n_no_temporal, seed)]
    random.Random(seed).shuffle(rows)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-alpaca", type=int, default=4000)
    p.add_argument("--n-glaive", type=int, default=4000)
    p.add_argument("--n-no-temporal", type=int, default=300,
                   help="procedural Norwegian temporal function-calling examples (0 disables)")
    p.add_argument("--out", default="pilot_mix.jsonl")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rows = build_rows(a.n_alpaca, a.n_glaive, a.n_no_temporal, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for t in rows: f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {a.out}")


if __name__ == "__main__":
    main()
