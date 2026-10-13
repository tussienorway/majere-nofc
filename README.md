# NO-FC: Norwegian function-calling benchmark (Bokmål and Nynorsk)

Part of the open project **Majere**: https://majere.thomasaustenaa.no · contact mail@thomasaustenaa.no

NO-FC measures whether a language model can use tools correctly when the user writes in Norwegian:
- picking the right function
- filling in arguments correctly (dates relative to a fixed reference time, Norwegian phone numbers, org numbers, times like "halv ni")
- **not** calling a tool when none is needed
- **asking a follow-up question** when required information is missing

**Status:** v1 has 120 tasks (80 dev / 40 held-out test) in `data/nofc_v1.jsonl`; seed v0 is 40 hand-written tasks (20 Bokmål, 20 Nynorsk: 24 tool calls, 8 no-call, 8 clarify). Baselines exist for both: on v0, [qwen3:8b](results/README.md) scores 0.825 exact match (33/40); the v1 baselines (qwen3:8b in three configurations, NorMistral-11B-thinking in prompt mode) are tabulated in [results/README.md](results/README.md), with the qwen3:8b test-split misses analysed in [results/v1/error_analysis.md](results/v1/error_analysis.md). The hard-temporal diagnostic set (`data/nofc_hard_temporal.jsonl`) has no scores yet. Only a 50-step QLoRA smoke run has been done; the full QLoRA pilot has not been run.

## v1 is frozen; hard-temporal diagnostic set
`data/nofc_v1.jsonl` and its 80/40 split will not change, so scores stay comparable. All 7 test misses of the published qwen3:8b baseline were date/time errors (relative weekdays, "halv"/"kvart" clock phrases, and asking for an optional duration or date instead of calling); see [results/v1/error_analysis.md](results/v1/error_analysis.md). `data/nofc_hard_temporal.jsonl` (48 tasks, 24 Bokmål / 24 Nynorsk, 32 dev / 16 test, built by `data/make_hard_temporal.py`) targets exactly those three failure modes with the same reference time (Monday 2026-10-12 10:00). It has no published scores yet. Clock conventions in the gold answers: "halv X" = (X-1):30, "kvart over X" = X:15, "kvart på X" = (X-1):45; a weekday name is its next occurrence after today ("mandag" = 2026-10-19). Run it with:
```bash
python -m nofc.evaluate --model qwen3:8b --tasks data/nofc_hard_temporal.jsonl --split test --out ht_test.jsonl
```

## Run against your own model (Ollama / llama.cpp / vLLM)
No dependencies beyond Python 3.10+.
```bash
ollama pull qwen3:8b
python -m nofc.evaluate --base-url http://localhost:11434/v1 --model qwen3:8b
# Qwen3 soft switch, append /no_think:
python -m nofc.evaluate --model qwen3:8b --no-think
# No native tools: schemas in the system prompt, parse <tool_call>:
python -m nofc.evaluate --model <gguf> --tool-mode prompt
# Held-out v1 test split (the default tasks file is data/seed_v0.jsonl):
python -m nofc.evaluate --model qwen3:8b --tasks data/nofc_v1.jsonl --split test
# Hard-temporal diagnostic set (dev and test together, or add --split test):
python -m nofc.evaluate --model qwen3:8b --tasks data/nofc_hard_temporal.jsonl
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
python train/build_data.py --n-alpaca 4000 --n-glaive 4000 --n-no-temporal 300 --out pilot_mix.jsonl
python train/qlora_sft.py --base qwen3-8b --data pilot_mix.jsonl --max-steps 50     # smoke run
python train/qlora_sft.py --base qwen3-8b --data pilot_mix.jsonl                    # full pilot
python train/qlora_sft.py --base normistral-7b --data pilot_mix.jsonl
```
`--n-no-temporal N` (default 300; `0` disables) adds N procedurally generated Bokmål/Nynorsk function-calling dialogues from `train/make_no_temporal_fc.py` to the mix. They target the three failure modes in [results/v1/error_analysis.md](results/v1/error_analysis.md): (A) relative weekday / day offset resolved to the wrong date, (B) "halv X" / "kvart over X" / "kvart på X" clock phrases, and (C) over-clarifying when only optional arguments (`duration_minutes`, `after_time`, `line`) are missing. Clarify and no-call controls are included so the model does not learn to always call or always ask. Gold answers use the same reference time and conventions as the eval sets (Monday 2026-10-12 10:00; a weekday name is its next occurrence after today). The examples are disjoint from the eval sets: the generator rejects any user string that equals (after case-folding and removing punctuation) or closely overlaps a `user` in `seed_v0.jsonl`, `nofc_v1.jsonl` or `nofc_hard_temporal.jsonl`, and `tests/test_no_temporal_fc.py` checks this. Generation is offline (no `datasets` download); `--n-alpaca 0 --n-glaive 0` skips the Hugging Face sources. This is a focused injection for the pilot, not the full Norwegian function-calling translation on the roadmap.

How it fits in 16 GB:
- 4-bit NF4 with double quantisation
- LoRA r=16 on all projections
- batch size 1 × gradient accumulation 16, sequence length 2048
- gradient checkpointing and paged 8-bit AdamW

If you run out of memory, lower `--max-len` to 1024. Each run writes `pilot_metrics.json` (tokens/s, peak VRAM, time). Those numbers replace the planning estimate on the project's compute page.

### Checkpoints and resume
By default one checkpoint is written per epoch (unchanged). For a multi-hour run, add intermediate checkpoints and resume after an interruption:
```bash
# --save-steps counts optimizer steps (one step = --grad-accum examples). Keeps the newest 2 checkpoints unless --save-total-limit says otherwise.
python train/qlora_sft.py --base qwen3-8b --data pilot_mix.jsonl --save-steps 50
# after a crash/reboot: same command plus --resume (newest checkpoint in --out; starts from step 0 if there is none yet)
python train/qlora_sft.py --base qwen3-8b --data pilot_mix.jsonl --save-steps 50 --resume
# or resume from an explicit checkpoint (or a run directory containing checkpoint-N folders)
python train/qlora_sft.py --base qwen3-8b --data pilot_mix.jsonl --save-steps 50 --resume runs/qwen3-8b-r16-s0/checkpoint-150
```
Resume with the same `--data`, `--seed`, `--epochs`/`--max-steps` and other hyperparameters as the original run, and the same `--out` (the default `--out` already depends on `--base`, `--rank` and `--seed`). Only complete checkpoints (with `trainer_state.json`) are picked up. The `pilot_metrics.json` of a resumed run only covers the resumed part: `seconds_per_step` and tokens/s use `steps_this_run`, and `resumed_from` / `resumed_from_step` are added. Use an uninterrupted run for throughput numbers.

### Serve the adapter and evaluate it on NO-FC
The training text is not Qwen3's chat template, so Ollama/llama.cpp cannot serve the adapter as-is. `nofc/serve.py` is a small OpenAI-compatible server (`/v1/chat/completions`, `/v1/models`) that loads the base model in 4-bit NF4 exactly as in training (expected to fit 16 GB easily; actual VRAM not yet measured) plus the LoRA adapter with transformers and peft. It builds prompts with `nofc/chatfmt.py`, the same code `train/build_data.py` uses to write the training text, and decodes greedily (temperature is always 0; `<|end|>` is a stop string). It does not use any chat template.

Two terminals, from the repo root (the evaluator itself needs no dependencies; the server needs the `train/requirements.txt` packages):
```bash
# terminal 1: the trained adapter (the directory written by qlora_sft.py)
python -m nofc.serve --base qwen3-8b --adapter runs/qwen3-8b-r16-s0 --port 8000

# terminal 2: v1 test split, both tool modes
python -m nofc.evaluate --base-url http://127.0.0.1:8000/v1 --model qwen3-8b-adapter --tool-mode prompt --tasks data/nofc_v1.jsonl --split test --out adapter_prompt.jsonl
python -m nofc.evaluate --base-url http://127.0.0.1:8000/v1 --model qwen3-8b-adapter --tool-mode native --tasks data/nofc_v1.jsonl --split test --out adapter_native.jsonl
```
- Do not pass `--no-think`: the training format has no `/no_think`, and appending it would make the prompt differ from training. The adapter never produces `<think>` blocks in this format.
- `--tool-mode prompt` puts the schemas in the system prompt and parses `<tool_call>` from the text. `--tool-mode native` sends OpenAI `tools`; the server appends the schemas to the system prompt with the same text as prompt mode and converts the parsed `<tool_call>` into `tool_calls`. The two modes therefore give the model byte-identical prompts, so the generations, and therefore the scores, should be identical; a difference would point to a bug.
- Control run: serve the same base without `--adapter` (`python -m nofc.serve --base qwen3-8b --port 8001`, model name `qwen3-8b-base`) and evaluate it the same way. This isolates the effect of the fine-tuning from the effect of the different prompt format.
- The published baselines (qwen3:8b via Ollama, Q4_K_M, Qwen's own chat template and native tool handling) are not the same prompt path or quantisation as this server. The comparison with the control run above is the cleaner one; treat the Ollama numbers as context.
- Generation is capped at 512 new tokens per answer (`max_tokens` in the request can lower it).
- Not provided: merge + GGUF export for llama.cpp/Ollama. That would need a matching chat template and a separate check that its output agrees with this server.

## Data and licences
- Code: Apache-2.0
- Tasks: CC BY 4.0
- Training sources: NbAiLab/norwegian-alpaca (CC BY 4.0), glaiveai/glaive-function-calling-v2 (Apache-2.0)
- Base models: Qwen/Qwen3-8B (Apache-2.0), ltg/normistral-7b-warm (Apache-2.0)

## Roadmap
1. Baseline: v0 qwen3:8b is published (0.825 exact match; see [results/README.md](results/README.md)). Score NorMistral once Ollama supports tools, and publish v1 scores.
2. QLoRA pilot and measured throughput
3. Translate the function-calling data into Bokmål and Nynorsk, with validation
4. NO-FC v1 (≥500 tasks, human-checked)
5. Scaled runs on grant compute
