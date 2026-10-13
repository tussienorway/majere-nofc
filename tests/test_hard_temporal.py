import json, os, re, datetime as dt
from collections import Counter
from nofc.score import score_task, summarize

H = os.path.dirname(os.path.dirname(__file__))
def _load(n): return [json.loads(l) for l in open(os.path.join(H, "data", n), encoding="utf-8")]
HT = _load("nofc_hard_temporal.jsonl")
BY_ID = {t["id"]: t for t in HT}
TOOLS = {t["name"]: t for t in json.load(open(os.path.join(H, "data", "tools.json"), encoding="utf-8"))}
REF = dt.date(2026, 10, 12)
NUM = {"ett": 1, "eitt": 1, "en": 1, "to": 2, "tre": 3, "fire": 4, "fem": 5, "seks": 6, "sju": 7, "syv": 7,
       "åtte": 8, "ni": 9, "ti": 10, "elleve": 11, "tolv": 12}
CLOCK = re.compile(r"\b(halv|kvart over|kvart på)\s+(" + "|".join(NUM) + r")\b")
PM = re.compile(r"kvel[dl]|ettermiddag")
AM = re.compile(r"om morgen(en)?\b|om morgon(en)?\b|formiddag")


def _when(t):
    c = t["expected"]["calls"][0]; a = c["arguments"]
    if "time" in a: return a["date"], a["time"]
    if "after_time" in a: return None, a["after_time"]
    if "datetime" in a: return tuple(a["datetime"].split("T"))
    return None, None


def _clock_oracle(user):
    """Returns (minute, base_hour) implied by the first halv/kvart phrase, or None."""
    m = CLOCK.search(user)
    if not m: return None
    x = NUM[m[2]]
    return {"halv": (30, x - 1), "kvart over": (15, x), "kvart på": (45, x - 1)}[m[1]]


def test_schema_and_ids():
    assert 44 <= len(HT) <= 52
    assert len({t["id"] for t in HT}) == len(HT)
    for t in HT:
        assert re.fullmatch(r"nofc-ht-(nb|nn)-\d{3}", t["id"]) and t["id"].split("-")[2] == t["variant"]
        assert t["reference_datetime"] == "2026-10-12T10:00"
        assert t["split"] in ("dev", "test") and t["user"].strip()
        assert t["tools"] and all(n in TOOLS for n in t["tools"])
        e = t["expected"]
        assert e["type"] in ("call", "no_call", "clarify")
        if e["type"] == "call":
            c = e["calls"][0]; assert c["name"] in t["tools"]
            props = TOOLS[c["name"]]["parameters"]["properties"]
            assert set(c["arguments"]) <= set(props)
            assert set(TOOLS[c["name"]]["parameters"].get("required", [])) <= set(c["arguments"])


def test_balanced_and_covers_types():
    assert Counter(t["variant"] for t in HT) == {"nb": 24, "nn": 24}
    assert {t["expected"]["type"] for t in HT} == {"call", "no_call", "clarify"}
    split = Counter(t["split"] for t in HT)
    assert split["test"] * 3 == len(HT) and split["dev"] == 2 * split["test"]
    for s in ("dev", "test"):
        assert Counter(t["variant"] for t in HT if t["split"] == s)["nb"] == Counter(t["variant"] for t in HT if t["split"] == s)["nn"]


def test_split_is_every_third():
    assert [t["split"] for t in HT] == ["test" if i % 3 == 0 else "dev" for i in range(len(HT))]


def test_does_not_reuse_v1_text():
    v1 = {t["user"] for t in _load("nofc_v1.jsonl")}
    assert not v1 & {t["user"] for t in HT}


def test_duration_omitted_calls_exist():
    omitted = [t for t in HT if t["expected"]["type"] == "call"
               and t["expected"]["calls"][0]["name"] == "create_calendar_event"
               and "duration_minutes" not in t["expected"]["calls"][0]["arguments"]]
    assert {t["variant"] for t in omitted} == {"nb", "nn"} and len(omitted) >= 8


def test_halv_kvart_gold_follows_norwegian_convention():
    checked = 0
    for t in HT:
        if t["expected"]["type"] != "call": continue
        o = _clock_oracle(t["user"])
        if not o: continue
        _, tm = _when(t)
        h, m = map(int, tm.split(":"))
        minute, base = o
        assert m == minute, t["id"]
        if PM.search(t["user"]): assert h == base % 12 + 12, t["id"]
        elif AM.search(t["user"]): assert h == base % 12, t["id"]
        else: assert h % 12 == base % 12, t["id"]
        checked += 1
    assert checked >= 20


def test_clock_spot_checks():
    expect = {"nofc-ht-nb-001": "09:30", "nofc-ht-nb-003": "20:30", "nofc-ht-nb-004": "08:15",
              "nofc-ht-nb-005": "16:45", "nofc-ht-nb-013": "12:30", "nofc-ht-nn-011": "11:15",
              "nofc-ht-nn-009": "15:45", "nofc-ht-nn-016": "19:45", "nofc-ht-nn-017": "15:30"}
    for i, tm in expect.items():
        assert _when(BY_ID[i])[1] == tm, i


def test_relative_dates_follow_next_occurrence_rule():
    wd = {"mandag": 0, "måndag": 0, "tirsdag": 1, "tysdag": 1, "onsdag": 2, "torsdag": 3, "fredag": 4,
          "lørdag": 5, "laurdag": 5, "søndag": 6, "sundag": 6}
    rel = {"i dag": 0, "i morgen": 1, "i overmorgen": 2, "i overmorgon": 2}
    n = 0
    for t in HT:
        if t["expected"]["type"] != "call": continue
        date = _when(t)[0] or t["expected"]["calls"][0]["arguments"].get("date")
        if date is None: continue
        words = re.findall(r"\w+", t["user"].lower())
        days = [d for d in wd if d in words]
        m = re.search(r"om (\w+) dag(?:ar|er)", t["user"])
        want = None
        if days: want = REF + dt.timedelta(days=(wd[days[0]] - REF.weekday()) % 7 or 7)
        elif m: want = REF + dt.timedelta(days=NUM[m[1]])
        else:
            for k, v in rel.items():
                if k in t["user"].lower(): want = REF + dt.timedelta(days=v)
        assert want is not None, t["id"]
        assert date == str(want), t["id"]; n += 1
    assert n >= 30
    assert _when(BY_ID["nofc-ht-nb-004"])[0] == "2026-10-19"
    assert _when(BY_ID["nofc-ht-nn-003"])[0] == "2026-10-19"
    assert BY_ID["nofc-ht-nb-007"]["expected"]["calls"][0]["arguments"]["date"] == "2026-10-17"


def test_oracle_scores_one():
    rows = []
    for t in HT:
        e = t["expected"]
        if e["type"] == "call": rows.append(score_task(t, e["calls"], ""))
        else: rows.append(score_task(t, [], "Hva mener du?" if e["type"] == "clarify" else "Ok."))
    assert summarize(rows)["all"]["exact_match"] == 1.0
