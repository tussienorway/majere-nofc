# Results: baseline on the 40 v0 seed tasks, no fine-tuning

Run on 2026-10-12 on an RTX 5060 Ti 16 GB through Ollama 0.40.2 (`/v1/chat/completions`, temperature 0, native tools). 40 tasks is a small development set, so treat these as indicative numbers, not a final benchmark.

| Model | exact_match all | nb | nn | call_name_acc | call_arg_acc | no_call_acc | clarify_acc |
|---|---|---|---|---|---|---|---|
| qwen3:8b (Q4_K_M, default thinking) | **0.825** (33/40) | 0.85 | 0.80 | 0.917 | 0.819 | 1.00 | 1.00 |
| NorMistral-11B-thinking (Q4_K_M) | not scored: no tool support in Ollama | | | | | | |

Raw answers: `qwen3_8b-2026-10-12.jsonl`. Runtime: about 9 minutes for 40 tasks with thinking on.

## Error analysis, qwen3:8b (7 errors)
- **Relative weekdays (3):** "på lørdag" → 16.10 (should be 17.10), "på søndag" → 17.10 (should be 18.10), "fredag" → 15.10 (should be 16.10). The model miscounts the weekday from the reference date.
- **Norwegian time expression (1):** "halv ni" → 09:30. In Norwegian it means 08:30.
- **Wrong argument (1, Nynorsk):** SMS to "48 12 34 56" was sent to +4791234567, a number from another task. The model used the wrong phone number.
- **Over-clarifying (2):** asked a follow-up instead of calling, even though the missing fields (time, duration) are optional.

These error types are exactly what the fine-tuning targets: Norwegian date and time expressions, and arguments taken straight from the user's text.

# NO-FC v1 baselines (120 tasks: 80 dev / 40 test), 2026-10-12/13

RTX 5060 Ti 16 GB, Ollama, temperature 0. Test split = 40 held-out tasks (headline); dev = 80.

| Model | Mode | test exact | dev exact | all exact | call_name | call_arg | clarify |
|---|---|---|---|---|---|---|---|
| qwen3:8b, thinking off | native tools | **0.825** (33/40) | 0.775 | 0.792 | 0.847 | 0.772 | 0.958 |
| qwen3:8b, thinking off | prompt tools | 0.800 (32/40) | 0.762 | 0.775 | 0.944 | 0.804 | 0.917 |
| qwen3:8b, thinking on | native tools | 0.800 (32/40) | 0.688 | 0.725 | 0.833 | 0.721 | 0.875 |
| normistral-11b-thinking-q4 | prompt tools | 0.325 (13/40) | 0.287 | 0.300 | 0.111 | 0.097 | 0.250 |

NorMistral-11B-thinking almost never produces a parseable tool call in prompt mode (call_name 0.11); it only scores on no-call tasks. It is not a usable tool-calling base without fine-tuning. Raw files in `v1/`.

# QLoRA smoke run (50 steps), 2026-10-13

`train/qlora_sft.py --base qwen3-8b --data pilot_mix.jsonl --max-steps 50 --max-len 1024` (rank 16, grad_accum 16, 4-bit NF4), torch 2.11.0+cu128, triton-windows.

| peak VRAM | step time | tokens/s | tokens/step |
|---|---|---|---|
| 9.41 GiB | 15.3 s | ~327 | ~5.0k |

At max_len 2048 the run hit CUDA OOM at step 25 while another llama.cpp server held ~5 GB of VRAM; 1024 fits with headroom. Estimate: ~330 tok/s means 1 epoch of the 250k-token pilot mix takes ~13 min, 10M tokens ~8.5 h, 100M tokens ~85 h on this card. Metrics: `v1/pilot_metrics_qwen3-8b_smoke-2026-10-13.json`.

# Error analysis pointer

All 7 test-split misses of qwen3:8b (thinking off, native tools) on v1 are date/time errors: relative weekdays resolved one or two days early, "halv"/"kvart" clock phrases, and asking for an optional duration or date instead of calling. The per-task breakdown, taken only from the files in `v1/`, is in [v1/error_analysis.md](v1/error_analysis.md). The diagnostic set `data/nofc_hard_temporal.jsonl` targets these failure modes; it has no baseline scores yet.
