# NO-FC: Norwegian function-calling benchmark (Bokmål and Nynorsk)

Part of the open project **Majere**: https://majere.thomasaustenaa.no · contact mail@thomasaustenaa.no

NO-FC measures whether a language model can use tools correctly when the user writes in Norwegian:
- picking the right function
- filling in arguments correctly (dates relative to a fixed reference time, Norwegian phone numbers, org numbers, times like "halv ni")
- **not** calling a tool when none is needed
- **asking a follow-up question** when required information is missing

**Status:** v1 (120 tasks, 80 dev / 40 held-out test, `data/nofc_v1.jsonl`) and seed set v0 has 40 hand-written tasks (20 Bokmål, 20 Nynorsk): 24 tool calls, 8 no-call, 8 clarify. **No model results have been published yet.**

## Run against your own model (Ollama / llama.cpp / vLLM)
No dependencies beyond Python 3.10+.
```bash
ollama pull qwen3:8b
python -m nofc.evaluate --base-url http://localhost:11434/v1 --model qwen3:8b
# llama.cpp: llama-server -m model.gguf --jinja  → --base-url http://localhost:8080/v1
```
The script prints a summary (all / nb / nn) and writes every answer to `nofc_results.jsonl`.

## Metrics
| Metric | Definition |
|---|---|
| exact_match | Share of tasks answered completely correctly |
| call_name_acc | Right function name (call tasks only) |
| call_arg_acc | Share of expected arguments that match (after normalising phone, time, org number and currency; free-text fields need ≥50% word overlap) |
| no_call_acc | No tool call when none is needed |
| clarify_acc | No tool call, and the reply contains a question |

## Format
- `data/seed_v0.jsonl`: one task per line (see `nofc/schema.json`).
- `data/tools.json`: the tool definitions (JSON Schema, OpenAI format).
- The tasks are generated from `data/make_seed.py` so they're easy to review and edit.
- The reference time for all tasks is Monday 2026-10-12 10:00, Europe/Oslo.

Known limitations of v0:
- 40 tasks is a small set for development, not a final benchmark.
- Clarify detection uses a simple heuristic (a question mark).
- Only the first tool call is scored.
- v1 will have ≥500 tasks, multiple calls per task, and a human-checked test split.

## QLoRA pilot (one 16 GB GPU, e.g. RTX 5060 Ti)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell needs CUDA 12.8+
pip install -r train/requirements.txt
python train/build_data.py --n-alpaca 4000 --n-glaive 4000 --out pilot_mix.jsonl
python train/qlora_sft.py --base qwen3-8b --data pilot_mix.jsonl --max-steps 50     # smoke run
python train/qlora_sft.py --base qwen3-8b --data pilot_mix.jsonl                    # full pilot
python train/qlora_sft.py --base normistral-7b --data pilot_mix.jsonl
```
How it fits in 16 GB:
- 4-bit NF4 with double quantisation
- LoRA r=16 on all projections
- batch size 1 × gradient accumulation 16, sequence length 2048
- gradient checkpointing and paged 8-bit AdamW

If you run out of memory, lower `--max-len` to 1024. Each run writes `pilot_metrics.json` (tokens/s, peak VRAM, time). Those numbers replace the planning estimate on the project's compute page.

Note: the training text uses a simple, uniform `<|role|>` format with `<tool_call>{json}</tool_call>` for both models. Before NO-FC can be run against a fine-tuned model, the server needs a matching chat template (TODO: add a template file plus GGUF export).

## Data and licences
- Code: Apache-2.0
- Tasks: CC BY 4.0
- Training sources: NbAiLab/norwegian-alpaca (CC BY 4.0), glaiveai/glaive-function-calling-v2 (Apache-2.0)
- Base models: Qwen/Qwen3-8B (Apache-2.0), ltg/normistral-7b-warm (Apache-2.0)

## Roadmap
1. Baseline: NO-FC v0 on untouched Qwen3-8B and NorMistral (via Ollama/llama.cpp)
2. QLoRA pilot and measured throughput
3. Translate the function-calling data into Bokmål and Nynorsk, with validation
4. NO-FC v1 (≥500 tasks, human-checked)
5. Scaled runs on grant compute
