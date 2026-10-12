"""NO-FC v1: the 40 v0 seed tasks + 80 new hand-written tasks (120 total), with a fixed dev/test split.
Reference: Monday 2026-10-12 10:00 Europe/Oslo. Weekday names mean the next occurrence after today
(e.g. "fredag" = 2026-10-16, "mandag" = 2026-10-19). Ambiguous phrasings ("neste fredag") are avoided.
Split: every third task (index % 3 == 0) is test (held out, never used for training)."""
import json, datetime as dt
REF = dt.date(2026, 10, 12)
WD = {"mandag":0,"tirsdag":1,"onsdag":2,"torsdag":3,"fredag":4,"lørdag":5,"søndag":6}
def D(x):
    if x in WD:
        n = (WD[x] - REF.weekday()) % 7 or 7
        return str(REF + dt.timedelta(days=n))
    return str(REF + dt.timedelta(days=int(x)))
C=lambda n,**a:{"type":"call","calls":[{"name":n,"arguments":a}]}
NO={"type":"no_call"}; CL={"type":"clarify"}
N=[
# ---------- Bokmål, calls (24)
("nb","Hvordan blir været i Stavanger i overmorgen?",["get_weather"],C("get_weather",location="Stavanger",date=D("2"))),
("nb","Blir det sol i Kristiansand på onsdag?",["get_weather"],C("get_weather",location="Kristiansand",date=D("onsdag"))),
("nb","Hva sier værmeldingen for Lillehammer om tre dager?",["get_weather"],C("get_weather",location="Lillehammer",date=D("3"))),
("nb","Legg inn foreldremøte torsdag kvart over seks i kveld.",["create_calendar_event"],C("create_calendar_event",title="Foreldremøte",date=D("torsdag"),time="18:15")),
("nb","Sett opp lunsj med Per på mandag klokka halv tolv, 45 minutter.",["create_calendar_event"],C("create_calendar_event",title="Lunsj med Per",date=D("mandag"),time="11:30",duration_minutes=45)),
("nb","Legg inn fotballtrening lørdag kvart på ti.",["create_calendar_event"],C("create_calendar_event",title="Fotballtrening",date=D("lørdag"),time="09:45")),
("nb","Kalender: julebord 4. desember kl. 19.",["create_calendar_event"],C("create_calendar_event",title="Julebord",date="2026-12-04",time="19:00")),
("nb","Send SMS til 0047 98 76 54 32: Middagen er klar.",["send_sms"],C("send_sms",to="+4798765432",message="Middagen er klar.")),
("nb","Kan du sende «Husk nøklene» til 400 11 222?",["send_sms"],C("send_sms",to="+4740011222",message="Husk nøklene")),
("nb","Send en tekstmelding til +47 93 00 00 01 om at møtet er flyttet.",["send_sms","create_calendar_event"],C("send_sms",to="+4793000001",message="Møtet er flyttet.")),
("nb","Når går bussen fra Nationaltheatret etter halv fire i ettermiddag?",["search_bus_departures"],C("search_bus_departures",stop="Nationaltheatret",after_time="15:30")),
("nb","Vis avganger med linje 20 fra Skøyen.",["search_bus_departures"],C("search_bus_departures",stop="Skøyen",line="20")),
("nb","Hva er 1 500 kroner i euro?",["convert_currency"],C("convert_currency",amount=1500,from_currency="NOK",to_currency="EUR")),
("nb","Regn om 49,90 dollar til kroner.",["convert_currency"],C("convert_currency",amount=49.9,from_currency="USD",to_currency="NOK")),
("nb","Hvor mye er tre hundre danske kroner i norske?",["convert_currency"],C("convert_currency",amount=300,from_currency="DKK",to_currency="NOK")),
("nb","Minn meg på å hente pakken i overmorgen klokka 16.",["set_reminder"],C("set_reminder",text="Hente pakken",datetime=D("2")+"T16:00")),
("nb","Påminnelse på søndag kl. 20: legge ut søpla.",["set_reminder"],C("set_reminder",text="Legge ut søpla",datetime=D("søndag")+"T20:00")),
("nb","Minn meg på medisinen om en time.",["set_reminder"],C("set_reminder",text="Medisinen",datetime="2026-10-12T11:00")),
("nb","Slå opp Equinor ASA i Enhetsregisteret.",["lookup_company"],C("lookup_company",name="Equinor ASA")),
("nb","Hvem eier orgnr 914 778 271?",["lookup_company"],C("lookup_company",org_number="914778271")),
("nb","Bestill bord på Fiskeriet for fire personer fredag kl. 19.",["book_table"],C("book_table",restaurant="Fiskeriet",date=D("fredag"),time="19:00",people=4)),
("nb","Kan du bestille bord til to på Maaemo lørdag halv åtte i kveld?",["book_table"],C("book_table",restaurant="Maaemo",date=D("lørdag"),time="19:30",people=2)),
("nb","Oversett «Velkommen til møtet» til nynorsk.",["translate_text"],C("translate_text",text="Velkommen til møtet",target_language="nn")),
("nb","Oversett denne setningen til engelsk: Jeg kommer litt for sent.",["translate_text","send_sms"],C("translate_text",text="Jeg kommer litt for sent.",target_language="en")),
# ---------- Bokmål, no call (8)
("nb","Hva betyr det at klokka er «halv ni»?",["create_calendar_event","set_reminder"],NO),
("nb","Gi meg tre tips til en god søknad.",["send_sms"],NO),
("nb","Hvilken dag er det i dag?",["create_calendar_event"],NO),
("nb","Hva er hovedstaden i Finland?",["get_weather","lookup_company"],NO),
("nb","Forklar kort hva et organisasjonsnummer er.",["lookup_company"],NO),
("nb","Hei! Hvordan har du det?",["get_weather"],NO),
("nb","Hva er forskjellen på euro og kroner?",["convert_currency"],NO),
("nb","Lag en handleliste til taco for fire.",["set_reminder"],NO),
# ---------- Bokmål, clarify (8)
("nb","Bestill bord på Fiskeriet i morgen.",["book_table"],CL),
("nb","Send «Jeg er på vei» til ham.",["send_sms"],CL),
("nb","Legg inn tannlegetimen i kalenderen.",["create_calendar_event"],CL),
("nb","Oversett «God morgen».",["translate_text"],CL),
("nb","Hvordan blir været på lørdag?",["get_weather"],CL),
("nb","Minn meg på å ringe banken.",["set_reminder"],CL),
("nb","Gjør om 200 til dollar.",["convert_currency"],CL),
("nb","Slå opp det firmaet vi snakket om.",["lookup_company"],CL),
# ---------- Nynorsk, calls (24)
("nn","Korleis blir vêret i Førde i overmorgon?",["get_weather"],C("get_weather",location="Førde",date=D("2"))),
("nn","Blir det regn i Trondheim på tysdag?",["get_weather"],C("get_weather",location="Trondheim",date=D("tirsdag"))),
("nn","Kva seier vêrmeldinga for Bodø om fire dagar?",["get_weather"],C("get_weather",location="Bodø",date=D("4"))),
("nn","Legg inn foreldremøte onsdag kvart over seks i kveld.",["create_calendar_event"],C("create_calendar_event",title="Foreldremøte",date=D("onsdag"),time="18:15")),
("nn","Set opp middag med Sigrid på laurdag klokka halv sju i kveld, to timar.",["create_calendar_event"],C("create_calendar_event",title="Middag med Sigrid",date=D("lørdag"),time="18:30",duration_minutes=120)),
("nn","Legg inn korøving torsdag kvart på åtte i kveld.",["create_calendar_event"],C("create_calendar_event",title="Korøving",date=D("torsdag"),time="19:45")),
("nn","Kalender: julebord 11. desember kl. 18.",["create_calendar_event"],C("create_calendar_event",title="Julebord",date="2026-12-11",time="18:00")),
("nn","Send SMS til 0047 95 55 44 33: Maten er klar.",["send_sms"],C("send_sms",to="+4795554433",message="Maten er klar.")),
("nn","Kan du sende «Hugs nøklane» til 412 34 000?",["send_sms"],C("send_sms",to="+4741234000",message="Hugs nøklane")),
("nn","Send ei tekstmelding til +47 92 10 20 30 om at møtet er flytta.",["send_sms","create_calendar_event"],C("send_sms",to="+4792102030",message="Møtet er flytta.")),
("nn","Når går bussen frå Festplassen etter halv fire i ettermiddag?",["search_bus_departures"],C("search_bus_departures",stop="Festplassen",after_time="15:30")),
("nn","Vis avgangar med linje 2 frå Lagunen.",["search_bus_departures"],C("search_bus_departures",stop="Lagunen",line="2")),
("nn","Kva er 1 200 kroner i euro?",["convert_currency"],C("convert_currency",amount=1200,from_currency="NOK",to_currency="EUR")),
("nn","Rekn om 19,50 dollar til kroner.",["convert_currency"],C("convert_currency",amount=19.5,from_currency="USD",to_currency="NOK")),
("nn","Kor mykje er fem hundre svenske kroner i norske?",["convert_currency"],C("convert_currency",amount=500,from_currency="SEK",to_currency="NOK")),
("nn","Minn meg på å hente pakken i overmorgon klokka 15.",["set_reminder"],C("set_reminder",text="Hente pakken",datetime=D("2")+"T15:00")),
("nn","Påminning på sundag kl. 21: setje ut bosset.",["set_reminder"],C("set_reminder",text="Setje ut bosset",datetime=D("søndag")+"T21:00")),
("nn","Minn meg på medisinen om to timar.",["set_reminder"],C("set_reminder",text="Medisinen",datetime="2026-10-12T12:00")),
("nn","Slå opp Telenor ASA i Einingsregisteret.",["lookup_company"],C("lookup_company",name="Telenor ASA")),
("nn","Kven eig orgnr 982 463 718?",["lookup_company"],C("lookup_company",org_number="982463718")),
("nn","Bestill bord på Bare Vestland til seks personar fredag kl. 18.",["book_table"],C("book_table",restaurant="Bare Vestland",date=D("fredag"),time="18:00",people=6)),
("nn","Kan du bestille bord til to på Lysverket laurdag halv åtte i kveld?",["book_table"],C("book_table",restaurant="Lysverket",date=D("lørdag"),time="19:30",people=2)),
("nn","Omset «Velkommen til møtet» til bokmål.",["translate_text"],C("translate_text",text="Velkommen til møtet",target_language="nb")),
("nn","Omset denne setninga til engelsk: Eg kjem litt for seint.",["translate_text","send_sms"],C("translate_text",text="Eg kjem litt for seint.",target_language="en")),
# ---------- Nynorsk, no call (8)
("nn","Kva tyder det at klokka er «halv ni»?",["create_calendar_event","set_reminder"],NO),
("nn","Gje meg tre tips til ein god søknad.",["send_sms"],NO),
("nn","Kva dag er det i dag?",["create_calendar_event"],NO),
("nn","Kva er hovudstaden i Island?",["get_weather","lookup_company"],NO),
("nn","Forklar kort kva eit organisasjonsnummer er.",["lookup_company"],NO),
("nn","Hei! Korleis har du det?",["get_weather"],NO),
("nn","Kva er skilnaden på bokmål og nynorsk?",["translate_text"],NO),
("nn","Lag ei handleliste til fårikål for fire.",["set_reminder"],NO),
# ---------- Nynorsk, clarify (8)
("nn","Bestill bord på Lysverket i morgon.",["book_table"],CL),
("nn","Send «Eg er på veg» til henne.",["send_sms"],CL),
("nn","Legg inn legetimen i kalenderen.",["create_calendar_event"],CL),
("nn","Omset «God morgon».",["translate_text"],CL),
("nn","Korleis blir vêret på laurdag?",["get_weather"],CL),
("nn","Minn meg på å ringje banken.",["set_reminder"],CL),
("nn","Gjer om 300 til dollar.",["convert_currency"],CL),
("nn","Slå opp det firmaet vi snakka om.",["lookup_company"],CL),
]
seed=[json.loads(l) for l in open("data/seed_v0.jsonl",encoding="utf-8")]
rows=seed[:]
for i,(v,u,tools,exp) in enumerate(N,1):
    rows.append({"id":f"nofc1-{v}-{i:03d}","variant":v,"reference_datetime":"2026-10-12T10:00","user":u,"tools":tools,"expected":exp})
for i,r in enumerate(rows): r["split"]="test" if i%3==0 else "dev"
with open("data/nofc_v1.jsonl","w",encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
from collections import Counter
print(len(rows),Counter(r["split"] for r in rows),Counter(r["expected"]["type"] for r in rows),Counter(r["variant"] for r in rows))
