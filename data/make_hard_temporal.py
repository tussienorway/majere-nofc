"""NO-FC hard-temporal diagnostic set: 48 hand-written tasks (24 nb / 24 nn) aimed at the failure modes
seen in the published v1 baselines (see results/v1/error_analysis.md):
  A  relative weekday / relative day offset resolved to the wrong date
  B  Norwegian clock phrases ("halv X", "kvart over X", "kvart på X")
  C  over-clarifying when only optional arguments (duration_minutes, after_time, line) are missing
Reference: Monday 2026-10-12 10:00 Europe/Oslo (same as v1). Weekday names mean the next occurrence after today
(fredag = 2026-10-16, lørdag = 2026-10-17, mandag = 2026-10-19; never the same day). Ambiguous "neste fredag" is avoided.
Clock conventions: halv X = (X-1):30, kvart over X = X:15, kvart på X = (X-1):45. AM/PM comes from the phrase
("om kvelden", "på ettermiddagen", "om formiddagen", "om morgenen"/"om morgonen"); bare hours use the daytime reading.
Split: every third task in file order (index % 3 == 0) is test, the rest dev. Rows alternate nb, nn so the split is balanced.
Task texts are independent per variant (no translations of each other), so a dev task never gives away a test task.
Generated file: data/nofc_hard_temporal.jsonl. Does not touch nofc_v1.jsonl or seed_v0.jsonl."""
import json, os, datetime as dt
from collections import Counter

REF = dt.date(2026, 10, 12)
WD = {"mandag": 0, "tirsdag": 1, "onsdag": 2, "torsdag": 3, "fredag": 4, "lørdag": 5, "søndag": 6}


def D(x):
    if x in WD:
        n = (WD[x] - REF.weekday()) % 7 or 7
        return str(REF + dt.timedelta(days=n))
    return str(REF + dt.timedelta(days=int(x)))


C = lambda n, **a: {"type": "call", "calls": [{"name": n, "arguments": a}]}
NO = {"type": "no_call"}
CL = {"type": "clarify"}
CAL, WX, BOOK, REM, BUS = ("create_calendar_event", "get_weather", "book_table", "set_reminder", "search_bus_departures")

NB = [
    ("Sett opp styremøte onsdag klokka halv ti om formiddagen.", [CAL],
     C(CAL, title="Styremøte", date=D("onsdag"), time="09:30")),
    ("Hvordan blir været i Ålesund på søndag?", [WX],
     C(WX, location="Ålesund", date=D("søndag"))),
    ("Reserver bord for tre på Statholderen lørdag klokka halv ni om kvelden.", [BOOK],
     C(BOOK, restaurant="Statholderen", date=D("lørdag"), time="20:30", people=3)),
    ("Minn meg på å ringe rørleggeren på mandag om morgenen, kvart over åtte.", [REM],
     C(REM, text="Ringe rørleggeren", datetime=D("mandag") + "T08:15")),
    ("Når går bussen fra Majorstuen etter kvart på fem i ettermiddag?", [BUS],
     C(BUS, stop="Majorstuen", after_time="16:45")),
    ("Sett opp bilservice fredag klokka elleve.", [CAL],
     C(CAL, title="Bilservice", date=D("fredag"), time="11:00")),
    ("Hva blir temperaturen i Trondheim om fem dager?", [WX],
     C(WX, location="Trondheim", date=D("5"))),
    ("Når går bussene fra Storo?", [BUS],
     C(BUS, stop="Storo")),
    ("Legg inn bursdagsmiddag lørdag kvart over seks på kvelden, to timer.", [CAL],
     C(CAL, title="Bursdagsmiddag", date=D("lørdag"), time="18:15", duration_minutes=120)),
    ("Bestill bord til fem på Ekeberg Restaurant på fredag klokka halv sju på kvelden.", [BOOK],
     C(BOOK, restaurant="Ekeberg Restaurant", date=D("fredag"), time="18:30", people=5)),
    ("Sett en påminnelse i overmorgen halv tolv om formiddagen: sende fakturaen.", [REM],
     C(REM, text="Sende fakturaen", datetime=D("2") + "T11:30")),
    ("Legg inn blodprøve på fredag.", [CAL], CL),
    ("Sett opp fysioterapi tirsdag halv ett.", [CAL],
     C(CAL, title="Fysioterapi", date=D("tirsdag"), time="12:30")),
    ("Forklar forskjellen på «kvart over fem» og «kvart på fem».", [REM, CAL], NO),
    ("Sett opp tannkontroll mandag klokka kvart på ni.", [CAL],
     C(CAL, title="Tannkontroll", date=D("mandag"), time="08:45")),
    ("Bord for to på Bula Neobistro mandag klokka kvart på åtte på kvelden.", [BOOK],
     C(BOOK, restaurant="Bula Neobistro", date=D("mandag"), time="19:45", people=2)),
    ("Minn meg på å betale strømregningen i dag klokka halv tolv.", [REM],
     C(REM, text="Betale strømregningen", datetime=D("0") + "T11:30")),
    ("Vis avgangene fra Carl Berners plass etter halv ti om kvelden.", [BUS],
     C(BUS, stop="Carl Berners plass", after_time="21:30")),
    ("Bestill bord til fire på Sjøboden på lørdag.", [BOOK], CL),
    ("Legg inn kundemøte søndag klokka kvart over ett på ettermiddagen.", [CAL],
     C(CAL, title="Kundemøte", date=D("søndag"), time="13:15")),
    ("Hvordan blir været i Molde om seks dager?", [WX],
     C(WX, location="Molde", date=D("6"))),
    ("Sett opp yogatime onsdag halv åtte om kvelden, 75 minutter.", [CAL],
     C(CAL, title="Yogatime", date=D("onsdag"), time="19:30", duration_minutes=75)),
    ("Minn meg på å levere skjemaet på torsdag.", [REM], CL),
    ("Hva betyr det når noen sier «halv ett»?", [CAL, BOOK], NO),
]

NN = [
    ("Set opp jobbintervju tysdag halv ti om formiddagen.", [CAL],
     C(CAL, title="Jobbintervju", date=D("tirsdag"), time="09:30")),
    ("Korleis blir vêret i Stavanger på laurdag?", [WX],
     C(WX, location="Stavanger", date=D("lørdag"))),
    ("Bestill bord til fire på Cornelius måndag klokka halv sju om kvelden.", [BOOK],
     C(BOOK, restaurant="Cornelius", date=D("mandag"), time="18:30", people=4)),
    ("Minn meg på å ringje tannlegen på fredag om morgonen, kvart på åtte.", [REM],
     C(REM, text="Ringje tannlegen", datetime=D("fredag") + "T07:45")),
    ("Når går bussen frå Sandviken etter kvart over tre på ettermiddagen?", [BUS],
     C(BUS, stop="Sandviken", after_time="15:15")),
    ("Legg inn bilvask laurdag klokka ni.", [CAL],
     C(CAL, title="Bilvask", date=D("lørdag"), time="09:00")),
    ("Kva blir temperaturen i Kristiansund om tre dagar?", [WX],
     C(WX, location="Kristiansund", date=D("3"))),
    ("Når går bussane frå Bryggen?", [BUS],
     C(BUS, stop="Bryggen")),
    ("Legg inn kaffimøte sundag klokka kvart på fire på ettermiddagen, ein time.", [CAL],
     C(CAL, title="Kaffimøte", date=D("søndag"), time="15:45", duration_minutes=60)),
    ("Reserver bord til åtte på Bryggen Tracteursted onsdag klokka halv åtte om kvelden.", [BOOK],
     C(BOOK, restaurant="Bryggen Tracteursted", date=D("onsdag"), time="19:30", people=8)),
    ("Set ei påminning i overmorgon klokka kvart over elleve om formiddagen: sende rapporten.", [REM],
     C(REM, text="Sende rapporten", datetime=D("2") + "T11:15")),
    ("Legg inn kontrollen hos legen på tysdag.", [CAL], CL),
    ("Set opp foreldresamtale onsdag halv eitt.", [CAL],
     C(CAL, title="Foreldresamtale", date=D("onsdag"), time="12:30")),
    ("Forklar kort kva «halv tolv» tyder på klokka.", [REM, CAL], NO),
    ("Set opp helsesjekk måndag klokka kvart over ti.", [CAL],
     C(CAL, title="Helsesjekk", date=D("mandag"), time="10:15")),
    ("Bord for to på Fisketorget sundag klokka kvart på åtte om kvelden.", [BOOK],
     C(BOOK, restaurant="Fisketorget", date=D("søndag"), time="19:45", people=2)),
    ("Minn meg på å ta ut søppelet i dag klokka halv fire på ettermiddagen.", [REM],
     C(REM, text="Ta ut søppelet", datetime=D("0") + "T15:30")),
    ("Vis avgangar frå Danmarksplass etter halv ni om kvelden.", [BUS],
     C(BUS, stop="Danmarksplass", after_time="20:30")),
    ("Bestill bord til tre på Colonialen på laurdag.", [BOOK], CL),
    ("Legg inn tannrens torsdag kvart over eitt på ettermiddagen.", [CAL],
     C(CAL, title="Tannrens", date=D("torsdag"), time="13:15")),
    ("Blir det snø i Geilo om sju dagar?", [WX],
     C(WX, location="Geilo", date=D("7"))),
    ("Set opp korøving måndag kvart over sju om kvelden, to timar.", [CAL],
     C(CAL, title="Korøving", date=D("mandag"), time="19:15", duration_minutes=120)),
    ("Minn meg på å kjøpe blomar på fredag.", [REM], CL),
    ("Kva tyder det når nokon seier «kvart på åtte»?", [CAL, BOOK], NO),
]


def build():
    assert len(NB) == len(NN) == 24
    rows = []
    for i in range(24):
        for v, src in (("nb", NB), ("nn", NN)):
            u, tools, exp = src[i]
            rows.append({"id": f"nofc-ht-{v}-{i + 1:03d}", "variant": v, "reference_datetime": "2026-10-12T10:00",
                         "user": u, "tools": tools, "expected": exp})
    for i, r in enumerate(rows):
        r["split"] = "test" if i % 3 == 0 else "dev"
    return rows


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    rows = build()
    with open(os.path.join(here, "nofc_hard_temporal.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(rows), Counter(r["split"] for r in rows), Counter(r["expected"]["type"] for r in rows),
          Counter(r["variant"] for r in rows))
    print(Counter((r["split"], r["variant"]) for r in rows), Counter((r["split"], r["expected"]["type"]) for r in rows))
