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
