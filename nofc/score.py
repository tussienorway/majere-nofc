"""Scoring for NO-FC. Pure functions, no network."""
import re, unicodedata
FREE_TEXT = {"message", "title", "text"}

def _norm(s):
    s = unicodedata.normalize("NFC", str(s)).casefold().strip()
    return re.sub(r"[\s\.\!,]+", " ", s).strip()

def _phone(s):
    d = re.sub(r"[^\d+]", "", str(s))
    if d.startswith("0047"): d = "+47" + d[4:]
    if re.fullmatch(r"\d{8}", d): d = "+47" + d
    return d

def _time(s):
    m = re.fullmatch(r"(\d{1,2})[:.](\d{2})", str(s).strip())
    return f"{int(m[1]):02d}:{m[2]}" if m else str(s).strip()

def arg_match(key, exp, got):
    if got is None: return False
    if key == "to": return _phone(exp) == _phone(got)
    if key == "org_number": return re.sub(r"\D", "", str(exp)) == re.sub(r"\D", "", str(got))
    if key in ("time", "after_time"): return _time(exp) == _time(got)
    if key == "datetime": return str(got).strip()[:16].replace(" ", "T") == exp
    if key in ("from_currency", "to_currency"): return str(got).strip().upper() == exp
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        try: return abs(float(got) - float(exp)) < 1e-6
        except (TypeError, ValueError): return False
    if key in FREE_TEXT:
        e, g = set(_norm(exp).split()), set(_norm(got).split())
        return bool(e) and len(e & g) / len(e) >= 0.5
    return _norm(exp) == _norm(got)

def score_task(task, calls, text):
    """calls: list of {"name","arguments":dict}; text: assistant text. Returns dict of metrics."""
    exp = task["expected"]; t = exp["type"]
    r = {"id": task["id"], "variant": task["variant"], "type": t}
    if t in ("no_call", "clarify"):
        ok = not calls
        if t == "clarify": ok = ok and "?" in (text or "")
        r.update(correct=ok, name_ok=None, arg_acc=None)
        return r
    e = exp["calls"][0]
    if not calls:
        r.update(correct=False, name_ok=False, arg_acc=0.0); return r
    c = calls[0]; args = c.get("arguments") or {}
    name_ok = c.get("name") == e["name"]
    keys = list(e["arguments"])
    hits = sum(arg_match(k, e["arguments"][k], args.get(k)) for k in keys) if name_ok else 0
    acc = hits / len(keys) if keys else 1.0
    r.update(name_ok=name_ok, arg_acc=acc, correct=name_ok and acc == 1.0 and len(calls) == 1)
    return r

def summarize(rows):
    out = {}
    for grp in ("all", "nb", "nn"):
        rs = [r for r in rows if grp == "all" or r["variant"] == grp]
        if not rs: continue
        calls = [r for r in rs if r["type"] == "call"]
        out[grp] = {
            "n": len(rs),
            "exact_match": sum(r["correct"] for r in rs) / len(rs),
            "call_name_acc": sum(bool(r["name_ok"]) for r in calls) / len(calls) if calls else None,
            "call_arg_acc": sum(r["arg_acc"] for r in calls) / len(calls) if calls else None,
            "no_call_acc": _rate(rs, "no_call"), "clarify_acc": _rate(rs, "clarify")}
    return out

def _rate(rs, t):
    x = [r for r in rs if r["type"] == t]
    return sum(r["correct"] for r in x) / len(x) if x else None
