# NO-FC v1 error analysis: qwen3:8b, thinking off, native tools

Source files (nothing here is re-scored; all numbers come from them):
- `qwen3_8b-nothink-native-2026-10-12.jsonl`: per-task answers
- `qwen3_8b-nothink-native-2026-10-12.summary.json`: aggregate scores
- `../../data/nofc_v1.jsonl`: task text and gold answers (80 dev / 40 test, frozen)

Reference time for every task: Monday 2026-10-12 10:00 Europe/Oslo. The evaluator puts the date and the weekday name ("Mandag") in the system prompt (`nofc/evaluate.py`), so the model is told what day it is. The system prompt also says to ask a follow-up question when *necessary* information is missing.

## Headline from the raw file
- Test split: 33/40 exact match (0.825). Dev split: 62/80 (0.775).
- **All 7 test failures are date/time-related.** No test miss is a wrong function, a wrong phone or org number, a wrong amount or currency, or a no-call/clarify error.
- Dev misses (18 of 80, plus one request timeout on `nofc-nn-024`) are not categorised in this note.

## The 7 test failures

| # | id | user text (abridged) | gold | what the model did |
|---|---|---|---|---|
| 1 | `nofc-nb-010` | "Blir det regn i Tromsø på lørdag?" | get_weather, date 2026-10-17 | right function and location, date 2026-10-16 |
| 2 | `nofc-nn-022` | "Legg inn legetime på torsdag klokka ti." | create_calendar_event, 2026-10-15 10:00, no duration | no call; asked how long the appointment lasts |
| 3 | `nofc1-nb-003` | "Hva sier værmeldingen for Lillehammer om tre dager?" | get_weather, date 2026-10-15 | no call; asked which date the user means |
| 4 | `nofc1-nb-006` | "Legg inn fotballtrening lørdag kvart på ti." | create_calendar_event, 2026-10-17 09:45, no duration | no call; asked how long the training lasts |
| 5 | `nofc1-nb-021` | "Bestill bord på Fiskeriet for fire personer fredag kl. 19." | book_table, 2026-10-16 19:00 | date 2026-10-15 (other args correct) |
| 6 | `nofc1-nn-045` | "Set opp middag med Sigrid på laurdag klokka halv sju i kveld, to timar." | create_calendar_event, 2026-10-17 18:30, 120 min | date 2026-10-15 and time 19:30 |
| 7 | `nofc1-nn-051` | "Når går bussen frå Festplassen etter halv fire i ettermiddag?" | search_bus_departures, after_time 15:30 | after_time 16:30 |

## Failure modes
A failure can belong to more than one group (#6 is in A and B), so the group sizes add up to 8 for 7 failures.

### A. Relative weekday / relative day resolved to the wrong date (3 of 7: #1, #5, #6)
The rule in the gold answers: a weekday name means the next occurrence *after* the reference date. From Monday 2026-10-12, "fredag" is 2026-10-16, "lørdag" is 2026-10-17, and "mandag" is 2026-10-19 (never the same day).
- #1 "lørdag" → 2026-10-16 (one day early, that is the Friday).
- #5 "fredag" → 2026-10-15 (one day early, that is the Thursday).
- #6 "laurdag" → 2026-10-15 (two days early).
The same pattern appeared in the v0 baseline (see `../README.md`: "på lørdag" → 16.10, "på søndag" → 17.10, "fredag" → 15.10). In all of these the model picks a date that is earlier than the correct one.

### B. Norwegian clock phrases "halv" / "kvart" (2 of 7: #6, #7)
Conventions used in the gold answers:
- "halv X" = (X-1):30. "halv fire" = 15:30, "halv sju" = 18:30, "halv tolv" = 11:30, "halv ett" = 12:30.
- "kvart over X" = X:15. "kvart på X" = (X-1):45. "kvart på ti" = 09:45.
- AM/PM comes from context words ("i kveld", "i ettermiddag", "om formiddagen"). Bare hours use the daytime reading.

Observed errors: "halv sju i kveld" → 19:30 (gold 18:30) and "halv fire i ettermiddag" → 16:30 (gold 15:30). Both read "halv X" as X:30, the English "half past" reading. The v0 baseline had the same error ("halv ni" → 09:30, gold 08:30). Note that "kvart på ti" (#4) was *not* mis-converted; that task failed for another reason (C).

### C. Over-clarifying when only optional arguments are missing (3 of 7: #2, #3, #4)
- #2 and #4: `create_calendar_event` requires `title`, `date` and `time`. `duration_minutes` is optional in `data/tools.json`, and the gold answer is a call without it. The model asked for the duration instead of calling.
- #3: `get_weather` requires only `location`; `date` is optional. The model asked which start date the user meant instead of resolving "om tre dager" (2026-10-15) from the reference date. This one also has a relative-day component, but the failure is that no call was made.

In v0 the same pattern was recorded (two clarifications for optional time/duration fields, `../README.md`).

## Summary
| Group | Test failures |
|---|---|
| A. relative weekday off by one or two days | 3 (#1, #5, #6) |
| B. halv / kvart clock phrases | 2 (#6, #7) |
| C. over-clarify, optional args missing | 3 (#2, #3, #4) |

Every one of the 7 test misses involves a date or time. Function selection, phone numbers, org numbers, currencies, no-call and clarify behaviour did not cause any test miss for this run.

## Follow-up: diagnostic set
v1 is frozen (same tasks, same 80/40 split). To probe exactly these three failure modes there is a separate diagnostic set, `data/nofc_hard_temporal.jsonl`, generated by `data/make_hard_temporal.py` (48 tasks, 24 nb / 24 nn, same reference time and weekday rule). It includes calendar tasks with `duration_minutes` omitted, where the expected answer is a call, and a few clarify and no-call controls so a model cannot score by never asking. **No baseline scores exist for it yet.**
