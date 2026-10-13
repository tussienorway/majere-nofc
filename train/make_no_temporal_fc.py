"""Procedural Norwegian (Bokmål + Nynorsk) temporal function-calling examples for the pilot SFT mix.

Targets the three failure modes of the published v1 baseline (results/v1/error_analysis.md):
  A  relative weekday / day offset resolved to the wrong date
  B  Norwegian clock phrases ("halv X", "kvart over X", "kvart på X")
  C  over-clarifying when only optional arguments (duration_minutes, after_time, line, date) are missing
plus clarify controls (a required argument really is missing) and no_call controls, so the model does not learn
"always call" or "always ask".

Gold conventions (identical to data/make_hard_temporal.py and the eval sets):
  reference time  Monday 2026-10-12 10:00 Europe/Oslo
  weekday name    next occurrence AFTER today, never the same day (mandag = 2026-10-19, tirsdag = 2026-10-13, ...)
  halv X = (X-1):30,  kvart over X = X:15,  kvart på X = (X-1):45
  AM/PM from the phrase (om morgenen/formiddagen = AM, på ettermiddagen/om kvelden = PM); bare hours use the
  daytime reading and are only generated where that reading is unambiguous (07:00-12:59).

Contamination safeguards: every emitted `user` string is unique and is checked against all `user` fields of
data/seed_v0.jsonl, data/nofc_v1.jsonl and data/nofc_hard_temporal.jsonl. A candidate is rejected if its normalised
text (case-folded, punctuation removed, whitespace collapsed) equals an eval user or if its word set overlaps an eval
user with Jaccard >= NEAR_DUP_JACCARD (catches light paraphrases / reordered copies). The generator reads the eval
files only to filter; it never writes them. No network or `datasets` download is needed.

Output per example is the repo's own training text (`nofc.chatfmt.fmt`): system prompt = evaluate.SYSTEM for the
reference time + tool schemas via `with_tools` (same text as `evaluate --tool-mode prompt`), one user turn, one
assistant turn (a `<tool_call>` for calls, a short Norwegian text for clarify / no_call).

  python train/make_no_temporal_fc.py --n 300 --out no_temporal_fc.jsonl
"""
import argparse, json, os, random, sys, datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nofc.chatfmt import fmt, format_tool_call, with_tools
from nofc.evaluate import SYSTEM, WEEKDAYS

REF = dt.date(2026, 10, 12)
REF_DATETIME = "2026-10-12T10:00"
NEAR_DUP_JACCARD = 0.7
EVAL_FILES = ("seed_v0.jsonl", "nofc_v1.jsonl", "nofc_hard_temporal.jsonl")

CAL, WX, BOOK, REM, BUS = ("create_calendar_event", "get_weather", "book_table", "set_reminder", "search_bus_departures")

WD_INDEX = {"mandag": 0, "måndag": 0, "tirsdag": 1, "tysdag": 1, "onsdag": 2, "torsdag": 3, "fredag": 4,
            "lørdag": 5, "laurdag": 5, "søndag": 6, "sundag": 6}
WD_NAMES = {"nb": ["mandag", "tirsdag", "onsdag", "torsdag", "fredag", "lørdag", "søndag"],
            "nn": ["måndag", "tysdag", "onsdag", "torsdag", "fredag", "laurdag", "sundag"]}
NUM_WORDS = {"nb": {1: "ett", 2: "to", 3: "tre", 4: "fire", 5: "fem", 6: "seks", 7: "sju", 8: "åtte", 9: "ni",
                    10: "ti", 11: "elleve", 12: "tolv"},
             "nn": {1: "eitt", 2: "to", 3: "tre", 4: "fire", 5: "fem", 6: "seks", 7: "sju", 8: "åtte", 9: "ni",
                    10: "ti", 11: "elleve", 12: "tolv"}}

# Valid base hours (12-hour clock, the hour before the phrase's hour for halv / kvart på) per day part.
DAYPART_HOURS = {"morning": range(5, 10), "forenoon": range(9, 12), "afternoon": (12, 1, 2, 3, 4, 5),
                 "evening": range(5, 12), None: range(7, 13)}
DAYPART_WORDS = {"nb": {"morning": ["om morgenen"], "forenoon": ["om formiddagen"],
                        "afternoon": ["på ettermiddagen"], "evening": ["om kvelden", "på kvelden"], None: [""]},
                 "nn": {"morning": ["om morgonen"], "forenoon": ["om føremiddagen", "om formiddagen"],
                        "afternoon": ["på ettermiddagen"], "evening": ["om kvelden", "på kvelden"], None: [""]}}
DAYPART_SOON = {"afternoon": ["i ettermiddag"], "evening": ["i kveld"]}
MONTH_NO = "oktober"


# ---------------------------------------------------------------- gold helpers

def weekday_date(name, ref=REF):
    """Next occurrence of a weekday name strictly after `ref` (never the same day), as ISO date."""
    n = (WD_INDEX[name.lower()] - ref.weekday()) % 7 or 7
    return str(ref + dt.timedelta(days=n))


def offset_date(days, ref=REF):
    return str(ref + dt.timedelta(days=days))


def base_hour(kind, x):
    """12-hour base hour and minutes of a spoken clock phrase 'kind X' (X = spoken hour 1..12)."""
    if kind == "halv": return (x - 1 or 12), 30
    if kind == "kvart over": return x, 15
    if kind == "kvart på": return (x - 1 or 12), 45
    raise ValueError(kind)


def to_24h(h12, daypart):
    if h12 not in DAYPART_HOURS[daypart]:
        raise ValueError(f"hour {h12} is not a clear {daypart or 'bare'} reading")
    if daypart in ("afternoon", "evening"): return h12 + 12 if h12 < 12 else 12
    return h12


def clock_gold(kind, x, daypart=None):
    """HH:MM for 'halv X' / 'kvart over X' / 'kvart på X' with an optional day part."""
    h12, m = base_hour(kind, x)
    return f"{to_24h(h12, daypart):02d}:{m:02d}"


# ---------------------------------------------------------------- contamination guard

def norm(s):
    return " ".join("".join(c if c.isalnum() else " " for c in s.casefold()).split())


def _tokens(s):
    return frozenset(norm(s).split())


def eval_users(data_dir=None):
    data_dir = data_dir or os.path.join(ROOT, "data")
    users = []
    for name in EVAL_FILES:
        with open(os.path.join(data_dir, name), encoding="utf-8") as f:
            users += [json.loads(l)["user"] for l in f if l.strip()]
    return users


class EvalGuard:
    def __init__(self, users):
        self.exact = {u.strip().casefold() for u in users}
        self.norm = {norm(u) for u in users}
        self.sets = [_tokens(u) for u in users]

    def clashes(self, user):
        if user.strip().casefold() in self.exact or norm(user) in self.norm: return True
        t = _tokens(user)
        return any(len(t & s) / len(t | s) >= NEAR_DUP_JACCARD for s in self.sets)


# ---------------------------------------------------------------- vocabulary

EVENT_TITLES = {
    "nb": ["tannlegetime", "foreldremøte", "svømmetrening", "lunsj med Ingrid", "teammøte", "hårklipp", "fotballkamp",
           "prosjektmøte", "yogatime", "kafébesøk med Ola", "optikertime", "kundesamtale", "utviklingssamtale", "løpetur",
           "bokklubb", "dugnad", "middag hos Marit", "vaksinering", "veiledningstime", "øvelse med korpset", "orienteringsmøte",
           "mammagruppe", "klatretrening", "bryllupsplanlegging"],
    "nn": ["tannlegetime", "foreldremøte", "symjetrening", "lunsj med Ingrid", "teammøte", "hårklypp", "fotballkamp",
           "prosjektmøte", "yogatime", "kafébesøk med Ola", "optikartime", "kundesamtale", "utviklingssamtale", "joggetur",
           "bokklubb", "dugnad", "middag hjå Marit", "vaksinering", "rettleiingstime", "øving med korpset", "orienteringsmøte",
           "mammagruppe", "klatretrening", "bryllaupsplanlegging"]}
RESTAURANTS = ["Havnekjelleren", "Gamlebyen Spiseri", "Smia Kro", "Kaiakroken", "Brasserie Fjord", "Lysthuset",
               "Torggata Bistro", "Skipperstua", "Solsiden Mat", "Rorbua", "Fjellstua", "Brygga Matbar", "Villa Fjæra",
               "Nordlys Kjøkken", "Granstua", "Kroa på Hjørnet", "Sjøkanten", "Bakgården Café"]
STOPS = ["Nydalen", "Jernbanetorget", "Grønland", "Ullevål sykehus", "Bergen busstasjon", "Torget", "Skøyen stasjon",
         "Lade", "Tveita", "Bjørvika", "Kolbotn", "Ski stasjon", "Heimdal", "Fyllingsdalen", "Lillestrøm stasjon",
         "Åsane terminal", "Solli plass", "Haugenstua", "Mailand", "Strømsø torg"]
LINES = ["20", "31", "54", "37", "4", "60", "M1", "100", "21", "5"]
LOCATIONS = ["Bodø", "Kristiansand", "Drammen", "Harstad", "Haugesund", "Hamar", "Narvik", "Fredrikstad", "Skien",
             "Larvik", "Alta", "Mo i Rana", "Voss", "Røros", "Hammerfest", "Kirkenes", "Sandnes", "Gjøvik", "Arendal",
             "Tønsberg", "Sogndal", "Ørsta", "Steinkjer", "Lofthus"]
REMINDER_TEXTS = {
    "nb": ["bestille time hos frisøren", "fornye bibliotekkortet", "sende inn timelisten", "vanne plantene", "bytte dekk",
           "kjøpe melk", "ringe banken", "levere bøkene på biblioteket", "ta med gymtøy", "betale husleien",
           "avbestille abonnementet", "hente skoene hos skomakeren", "sjekke postkassen", "booke flybilletter",
           "sende bursdagskort til mormor", "lade sykkelbatteriet", "ta medisinene", "levere pant"],
    "nn": ["bestille time hjå frisøren", "fornye bibliotekkortet", "sende inn timelista", "vatne plantane", "bytte dekk",
           "kjøpe mjølk", "ringje banken", "levere bøkene på biblioteket", "ta med gymtøy", "betale husleiga",
           "seie opp abonnementet", "hente skoa hjå skomakaren", "sjekke postkassa", "bestille flybilletter",
           "sende bursdagskort til mormor", "lade sykkelbatteriet", "ta medisinane", "levere pant"]}
DURATIONS = {"nb": [("en time", 60), ("to timer", 120), ("tre timer", 180), ("20 minutter", 20), ("30 minutter", 30),
                    ("45 minutter", 45), ("90 minutter", 90)],
             "nn": [("ein time", 60), ("to timar", 120), ("tre timar", 180), ("20 minutt", 20), ("30 minutt", 30),
                    ("45 minutt", 45), ("90 minutt", 90)]}
REL_DAYS = {"nb": [("i dag", 0), ("i morgen", 1), ("i overmorgen", 2)],
            "nn": [("i dag", 0), ("i morgon", 1), ("i overmorgon", 2)]}
WX_OFFSET_WORDS = {2: "to", 3: "tre", 4: "fire", 5: "fem", 6: "seks", 7: "sju", 8: "åtte", 9: "ni"}
CAL_T = {"nb": ["Sett opp {t} {day} {clk}{dur}.", "Legg inn {t} {day} {clk}{dur}.", "Opprett en kalenderavtale: {t} {day} {clk}{dur}.",
                "Kan du legge inn {t} {day} {clk}{dur}?", "Jeg trenger {t} i kalenderen {day} {clk}{dur}."],
         "nn": ["Set opp {t} {day} {clk}{dur}.", "Legg inn {t} {day} {clk}{dur}.", "Opprett ei kalenderavtale: {t} {day} {clk}{dur}.",
                "Kan du leggje inn {t} {day} {clk}{dur}?", "Eg treng {t} i kalenderen {day} {clk}{dur}."]}
BOOK_T = {"nb": ["Bestill bord til {p} på {r} {day} {clk}.", "Reserver bord for {p} på {r} {day} {clk}.",
                 "Kan du booke bord på {r} til {p} {day} {clk}?", "Bord for {p} på {r} {day} {clk}, takk."],
          "nn": ["Bestill bord til {p} på {r} {day} {clk}.", "Reserver bord for {p} på {r} {day} {clk}.",
                 "Kan du bestille bord på {r} til {p} {day} {clk}?", "Bord for {p} på {r} {day} {clk}, takk."]}
REM_T = {"nb": ["Minn meg på å {x} {day} {clk}.", "Sett en påminnelse {day} {clk}: {x}.", "Påminn meg om å {x} {day} {clk}."],
         "nn": ["Minn meg på å {x} {day} {clk}.", "Set ei påminning {day} {clk}: {x}.", "Påminn meg om å {x} {day} {clk}."]}
BUS_CLK_T = {"nb": ["Når går bussen fra {s} etter {clk}?", "Vis avganger fra {s} etter {clk}.", "Finn neste buss fra {s} etter {clk}.",
                    "Når går buss {l} fra {s} etter {clk}?", "Vis avganger for linje {l} fra {s} etter {clk}."],
             "nn": ["Når går bussen frå {s} etter {clk}?", "Vis avgangar frå {s} etter {clk}.", "Finn neste buss frå {s} etter {clk}.",
                    "Når går buss {l} frå {s} etter {clk}?", "Vis avgangar for linje {l} frå {s} etter {clk}."]}
BUS_PLAIN_T = {"nb": ["Når går bussene fra {s}?", "Vis avgangene fra {s}.", "Når går neste buss fra {s}?",
                      "Når går buss {l} fra {s}?", "Vis avgangene for linje {l} fra {s}."],
               "nn": ["Når går bussane frå {s}?", "Vis avgangane frå {s}.", "Når går neste buss frå {s}?",
                      "Når går buss {l} frå {s}?", "Vis avgangane for linje {l} frå {s}."]}
WX_T = {"nb": ["Hvordan blir været i {loc} {day}?", "Blir det regn i {loc} {day}?", "Hva sier værmeldingen for {loc} {day}?",
               "Trenger jeg paraply i {loc} {day}?"],
        "nn": ["Korleis blir vêret i {loc} {day}?", "Blir det regn i {loc} {day}?", "Kva seier vêrmeldinga for {loc} {day}?",
               "Treng eg paraply i {loc} {day}?"]}
WX_NODATE_T = {"nb": ["Hvordan er været i {loc}?", "Hva sier værmeldingen for {loc}?", "Vis meg været i {loc}."],
               "nn": ["Korleis er vêret i {loc}?", "Kva seier vêrmeldinga for {loc}?", "Vis meg vêret i {loc}."]}
CLARIFY_A = {
    "cal_no_time": {"nb": ["Hvilket klokkeslett skal jeg legge avtalen på {day}?", "Hva er klokkeslettet for avtalen {day}?",
                           "Hvilken tid {day} skal jeg sette opp avtalen?"],
                    "nn": ["Kva klokkeslett skal eg leggje avtalen på {day}?", "Kva er klokkeslettet for avtalen {day}?",
                           "Kva tid {day} skal eg setje opp avtalen?"]},
    "book_no_time": {"nb": ["Hvor mange er dere, og hvilken tid vil dere ha bordet?", "Hvilken tid skal jeg bestille bordet, og til hvor mange?"],
                     "nn": ["Kor mange er de, og kva tid vil de ha bordet?", "Kva tid skal eg bestille bordet, og til kor mange?"]},
    "book_no_time_people": {"nb": ["Hvilken tid vil dere ha bordet?", "Hvilket klokkeslett skal jeg bestille bordet på?"],
                            "nn": ["Kva tid vil de ha bordet?", "Kva klokkeslett skal eg bestille bordet på?"]},
    "book_no_people": {"nb": ["Hvor mange personer skal bordet være til?", "Hvor mange blir dere?"],
                       "nn": ["Kor mange personar skal bordet vere til?", "Kor mange blir de?"]},
    "rem_no_time": {"nb": ["Hvilken tid skal jeg minne deg på det {day}?", "Når på dagen {day} vil du bli minnet på det?"],
                    "nn": ["Kva tid skal eg minne deg på det {day}?", "Når på dagen {day} vil du bli minna på det?"]},
}
CLARIFY_U = {
    "cal_no_time": {"nb": ["Legg inn {t} {day}.", "Sett opp {t} {day}.", "Kan du føre opp {t} {day}?"],
                    "nn": ["Legg inn {t} {day}.", "Set opp {t} {day}.", "Kan du føre opp {t} {day}?"]},
    "book_no_time": {"nb": ["Bestill bord på {r} {day}.", "Reserver bord på {r} {day}."],
                     "nn": ["Bestill bord på {r} {day}.", "Reserver bord på {r} {day}."]},
    "book_no_time_people": {"nb": ["Bord for {p} på {r} {day}.", "Kan du booke bord til {p} på {r} {day}?"],
                            "nn": ["Bord for {p} på {r} {day}.", "Kan du bestille bord til {p} på {r} {day}?"]},
    "book_no_people": {"nb": ["Bestill bord på {r} {day} {clk}.", "Reserver bord på {r} {day} {clk}."],
                       "nn": ["Bestill bord på {r} {day} {clk}.", "Reserver bord på {r} {day} {clk}."]},
    "rem_no_time": {"nb": ["Minn meg på å {x} {day}.", "Sett en påminnelse {day}: {x}."],
                    "nn": ["Minn meg på å {x} {day}.", "Set ei påminning {day}: {x}."]},
}


def cap(s):
    return s[:1].upper() + s[1:]


# ---------------------------------------------------------------- sampling primitives

def pick_day(rng, v, allow_rel=True):
    """-> (phrase for use after a verb, ISO date)."""
    if allow_rel and rng.random() < 0.2:
        phrase, off = rng.choice(REL_DAYS[v]); return phrase, offset_date(off)
    name = rng.choice(WD_NAMES[v])
    return rng.choice(["på ", ""]) + name, weekday_date(name)


def pick_clock(rng, v, style=None, dayparts=None, soon=False):
    """-> (spoken phrase incl. 'klokka' and day part, 'HH:MM').

    style 'phrase' = halv / kvart over / kvart på; 'plain' = whole hour ('klokka ti', 'kl. 14', 'klokka 14.30').
    """
    style = style or ("phrase" if rng.random() < 0.8 else "plain")
    dp = rng.choice(dayparts or [None, "morning", "forenoon", "afternoon", "evening"])
    h12 = rng.choice(list(DAYPART_HOURS[dp]))
    if soon and dp in DAYPART_SOON: dpw = rng.choice(DAYPART_SOON[dp])
    else: dpw = rng.choice(DAYPART_WORDS[v][dp])
    pre = rng.choice(["klokka ", "kl. "] if style == "plain" else ["klokka ", ""])
    if style == "plain":
        h24 = to_24h(h12, dp)
        if rng.random() < 0.35:
            minute = rng.choice([0, 30]) if rng.random() < 0.6 else 0
            spoken = f"{pre}{h24}" + (f".{minute:02d}" if minute else ""); hhmm = f"{h24:02d}:{minute:02d}"
            return spoken, hhmm
        return f"{pre}{NUM_WORDS[v][h12]}{(' ' + dpw) if dpw else ''}".strip(), f"{h24:02d}:00"
    kind = rng.choice(["halv", "kvart over", "kvart på"])
    x = h12 if kind == "kvart over" else (h12 % 12) + 1
    assert base_hour(kind, x)[0] == h12
    return f"{pre}{kind} {NUM_WORDS[v][x]}{(' ' + dpw) if dpw else ''}", clock_gold(kind, x, dp)


def tool_list(rng, main):
    """Relevant tool, plus (30%) one distractor; order follows data/tools.json."""
    names = [main]
    if rng.random() < 0.3: names.append(rng.choice([n for n in TOOL_ORDER if n != main]))
    return sorted(names, key=TOOL_ORDER.index)


def load_tools():
    with open(os.path.join(ROOT, "data", "tools.json"), encoding="utf-8") as f:
        return json.load(f)


TOOLS = {t["name"]: t for t in load_tools()}
TOOL_ORDER = list(TOOLS)


def call_ex(kind, tools, user, name, args):
    return {"kind": kind, "type": "call", "tools": tools, "user": user, "call": {"name": name, "arguments": args},
            "assistant": format_tool_call(name, args)}


def text_ex(kind, typ, tools, user, answer):
    return {"kind": kind, "type": typ, "tools": tools, "user": user, "call": None, "assistant": answer}


# ---------------------------------------------------------------- example builders (one per kind)

def b_cal_clock(rng, v):
    t = rng.choice(EVENT_TITLES[v]); day, date = pick_day(rng, v)
    clk, hhmm = pick_clock(rng, v, "phrase")
    args = {"title": cap(t), "date": date, "time": hhmm}; dur = ""
    if rng.random() < 0.2:
        phr, mins = rng.choice(DURATIONS[v]); dur = ", " + phr; args["duration_minutes"] = mins
    user = rng.choice(CAL_T[v]).format(t=t, day=day, clk=clk, dur=dur)
    return call_ex("cal_clock", tool_list(rng, CAL), user, CAL, args)


def b_cal_plain(rng, v):
    t = rng.choice(EVENT_TITLES[v]); day, date = pick_day(rng, v)
    clk, hhmm = pick_clock(rng, v, "plain")
    user = rng.choice(CAL_T[v]).format(t=t, day=day, clk=clk, dur="")
    return call_ex("cal_plain", tool_list(rng, CAL), user, CAL, {"title": cap(t), "date": date, "time": hhmm})


def b_wx_weekday(rng, v):
    loc = rng.choice(LOCATIONS); name = rng.choice(WD_NAMES[v])
    day = rng.choice(["på ", ""]) + name
    return call_ex("wx_weekday", tool_list(rng, WX), rng.choice(WX_T[v]).format(loc=loc, day=day), WX,
                   {"location": loc, "date": weekday_date(name)})


def b_wx_offset(rng, v):
    loc = rng.choice(LOCATIONS)
    if rng.random() < 0.55:
        n = rng.choice(list(WX_OFFSET_WORDS))
        unit = {"nb": "dager", "nn": "dagar"}[v]
        day = f"om {rng.choice([WX_OFFSET_WORDS[n], str(n)])} {unit}"
    else:
        day, n = rng.choice(REL_DAYS[v])
    return call_ex("wx_offset", tool_list(rng, WX), rng.choice(WX_T[v]).format(loc=loc, day=day), WX,
                   {"location": loc, "date": offset_date(n)})


def b_wx_nodate(rng, v):
    loc = rng.choice(LOCATIONS)
    return call_ex("wx_nodate", tool_list(rng, WX), rng.choice(WX_NODATE_T[v]).format(loc=loc), WX, {"location": loc})


def pick_people(rng, v):
    n = rng.randint(2, 10)
    word = NUM_WORDS[v].get(n) or str(n)
    phrase = rng.choice([word, f"{word} personer" if v == "nb" else f"{word} personar", f"{n} personer" if v == "nb" else f"{n} personar"])
    return n, phrase


def b_book_clock(rng, v):
    r = rng.choice(RESTAURANTS); day, date = pick_day(rng, v); clk, hhmm = pick_clock(rng, v, "phrase", ["afternoon", "evening"])
    n, p = pick_people(rng, v)
    user = rng.choice(BOOK_T[v]).format(p=p, r=r, day=day, clk=clk)
    return call_ex("book_clock", tool_list(rng, BOOK), user, BOOK, {"restaurant": r, "date": date, "time": hhmm, "people": n})


def b_rem_clock(rng, v):
    x = rng.choice(REMINDER_TEXTS[v]); day, date = pick_day(rng, v); clk, hhmm = pick_clock(rng, v, "phrase")
    user = rng.choice(REM_T[v]).format(x=x, day=day, clk=clk)
    return call_ex("rem_clock", tool_list(rng, REM), user, REM, {"text": cap(x), "datetime": f"{date}T{hhmm}"})


def b_bus_clock(rng, v):
    s = rng.choice(STOPS); line = rng.choice(LINES)
    clk, hhmm = pick_clock(rng, v, "phrase", ["forenoon", "afternoon", "evening", None], soon=True)
    i = rng.randrange(len(BUS_CLK_T[v]))
    args = {"stop": s, "after_time": hhmm}
    if "{l}" in BUS_CLK_T[v][i]: args["line"] = line
    return call_ex("bus_clock", tool_list(rng, BUS), BUS_CLK_T[v][i].format(s=s, l=line, clk=clk), BUS, args)


def b_bus_plain(rng, v):
    s = rng.choice(STOPS); line = rng.choice(LINES)
    i = rng.randrange(len(BUS_PLAIN_T[v]))
    args = {"stop": s}
    if "{l}" in BUS_PLAIN_T[v][i]: args["line"] = line
    return call_ex("bus_plain", tool_list(rng, BUS), BUS_PLAIN_T[v][i].format(s=s, l=line), BUS, args)


def b_clarify(rng, v):
    sub = rng.choice(list(CLARIFY_A))
    day, _ = pick_day(rng, v, allow_rel=False); fill = {"day": day}
    tool = {"cal": CAL, "book": BOOK, "rem": REM}[sub.split("_")[0]]
    fill["t"] = rng.choice(EVENT_TITLES[v]); fill["r"] = rng.choice(RESTAURANTS); fill["x"] = rng.choice(REMINDER_TEXTS[v])
    fill["p"] = pick_people(rng, v)[1]; fill["clk"] = pick_clock(rng, v, "phrase", ["afternoon", "evening"])[0]
    user = rng.choice(CLARIFY_U[sub][v]).format(**fill)
    answer = rng.choice(CLARIFY_A[sub][v]).format(day=day)
    assert "?" in answer
    return text_ex("clarify_" + sub, "clarify", tool_list(rng, tool), user, answer)


CLOCK_Q = {"nb": ["Forklar kort hva «{ph}» betyr.", "Hva betyr «{ph}» på klokka?", "Hvilket klokkeslett er «{ph}» i 24-timersformat?"],
           "nn": ["Forklar kort kva «{ph}» tyder.", "Kva tyder «{ph}» på klokka?", "Kva klokkeslett er «{ph}» i 24-timarsformat?"]}


def b_no_call_clock(rng, v):
    kind = rng.choice(["halv", "kvart over", "kvart på"]); h12 = rng.choice(list(DAYPART_HOURS[None]))
    x = h12 if kind == "kvart over" else (h12 % 12) + 1
    ph = f"{kind} {NUM_WORDS[v][x]}"; hhmm = clock_gold(kind, x, None)
    word = NUM_WORDS[v][x]; verb = {"nb": "betyr", "nn": "tyder"}[v]
    if kind == "halv":
        wrong = f"{h12 + 1 if h12 < 12 else 1:02d}:30"
        a = {"nb": f"«{ph}» {verb} {hhmm}: en halv time før {word}, ikke {wrong}.",
             "nn": f"«{ph}» {verb} {hhmm}: ein halv time før {word}, ikkje {wrong}."}[v]
    elif kind == "kvart over":
        a = {"nb": f"«{ph}» {verb} {hhmm}: et kvarter etter {word}.", "nn": f"«{ph}» {verb} {hhmm}: eit kvarter etter {word}."}[v]
    else:
        a = {"nb": f"«{ph}» {verb} {hhmm}: et kvarter før {word}.", "nn": f"«{ph}» {verb} {hhmm}: eit kvarter før {word}."}[v]
    return text_ex("no_call_clock", "no_call", tool_list(rng, rng.choice([CAL, REM, BOOK])), rng.choice(CLOCK_Q[v]).format(ph=ph), a)


def b_no_call_weekday(rng, v):
    name = rng.choice(WD_NAMES[v]); date = dt.date.fromisoformat(weekday_date(name)); days = (date - REF).days
    sub = rng.choice(["date", "days"])
    if sub == "date":
        user = rng.choice({"nb": ["Hvilken dato er det på {n}?", "Hvilken dato blir {n}?", "Hva er datoen på {n}?"],
                           "nn": ["Kva dato er det på {n}?", "Kva dato blir {n}?", "Kva er datoen på {n}?"]}[v]).format(n=name)
        a = f"{cap(name)} er {date.day}. {MONTH_NO} {date.year}."
    else:
        user = rng.choice({"nb": ["Hvor mange dager er det til {n}?", "Hvor lenge er det til {n}?"],
                           "nn": ["Kor mange dagar er det til {n}?", "Kor lenge er det til {n}?"]}[v]).format(n=name)
        unit = {"nb": "dager", "nn": "dagar"}[v]
        a = f"Det er {days} {unit} til {name}." if days > 1 else f"Det er {days} dag til {name}."
    return text_ex("no_call_weekday", "no_call", tool_list(rng, rng.choice([CAL, REM, WX])), user, a)


KINDS = [("cal_clock", 20, b_cal_clock), ("cal_plain", 8, b_cal_plain), ("wx_weekday", 8, b_wx_weekday),
         ("wx_offset", 8, b_wx_offset), ("wx_nodate", 3, b_wx_nodate), ("book_clock", 12, b_book_clock),
         ("rem_clock", 12, b_rem_clock), ("bus_clock", 8, b_bus_clock), ("bus_plain", 4, b_bus_plain),
         ("clarify", 9, b_clarify), ("no_call_clock", 4, b_no_call_clock), ("no_call_weekday", 4, b_no_call_weekday)]


def allocate(n):
    """Largest-remainder split of n over the KIND weights."""
    total = sum(w for _, w, _ in KINDS)
    raw = [(n * w / total, i) for i, (_, w, _) in enumerate(KINDS)]
    counts = [int(x) for x, _ in raw]
    for _, i in sorted(((x - int(x), i) for x, i in raw), reverse=True)[: n - sum(counts)]: counts[i] += 1
    return counts


def system_prompt(tool_names):
    wd = WEEKDAYS[dt.datetime.fromisoformat(REF_DATETIME).weekday()]
    return with_tools(SYSTEM.format(dt=REF_DATETIME, wd=wd), [TOOLS[n] for n in tool_names])


def generate(n, seed=0, data_dir=None):
    """Returns n example dicts (user, system, assistant, text, variant, kind, type, tools, call), half nb / half nn."""
    if n <= 0: return []
    guard = EvalGuard(eval_users(data_dir)); rng = random.Random(seed)
    plan = [(name, fn) for (name, _, fn), c in zip(KINDS, allocate(n)) for _ in range(c)]
    plan = [(name, fn, "nb" if i % 2 == 0 else "nn") for i, (name, fn) in enumerate(plan)]
    rng.shuffle(plan)
    seen, out = set(), []
    for name, fn, v in plan:
        for _ in range(500):
            ex = fn(rng, v); key = norm(ex["user"])
            if key in seen or guard.clashes(ex["user"]): continue
            seen.add(key); break
        else:
            raise RuntimeError(f"could not generate enough unique uncontaminated '{name}' ({v}) examples; lower n")
        ex["variant"] = v; ex["system"] = system_prompt(ex["tools"])
        ex["text"] = fmt(ex["system"], [("user", ex["user"]), ("assistant", ex["assistant"])])
        ex["id"] = f"nofc-nt-{v}-{len(out) + 1:04d}"
        out.append(ex)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=300); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="no_temporal_fc.jsonl")
    a = p.parse_args(argv)
    rows = generate(a.n, a.seed)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {a.out}")


if __name__ == "__main__":
    main()
