"""Hand-written NO-FC seed tasks. Reference date: Monday 2026-10-12, Europe/Oslo."""
import json
C=lambda n,**a:{"type":"call","calls":[{"name":n,"arguments":a}]}
NO={"type":"no_call"}; CL={"type":"clarify"}
T=[
# --- Bokmål: calls
("nb","Hvordan blir været i Bergen i morgen?",["get_weather"],C("get_weather",location="Bergen",date="2026-10-13")),
("nb","Legg inn tannlege på fredag klokka halv ni.",["create_calendar_event"],C("create_calendar_event",title="Tannlege",date="2026-10-16",time="08:30")),
("nb","Send en SMS til 912 34 567 og si at jeg er sen.",["send_sms"],C("send_sms",to="+4791234567",message="Jeg er sen.")),
("nb","Når går neste buss fra Jernbanetorget etter kl. 17?",["search_bus_departures"],C("search_bus_departures",stop="Jernbanetorget",after_time="17:00")),
("nb","Hvor mye er 250 euro i norske kroner?",["convert_currency"],C("convert_currency",amount=250,from_currency="EUR",to_currency="NOK")),
("nb","Minn meg på å ringe mamma i morgen klokka 18.",["set_reminder"],C("set_reminder",text="Ringe mamma",datetime="2026-10-13T18:00")),
("nb","Finn foretaket med organisasjonsnummer 923 609 016.",["lookup_company"],C("lookup_company",org_number="923609016")),
("nb","Book et møte med regnskapsfører 20. oktober kl. 14, en time.",["create_calendar_event"],C("create_calendar_event",title="Møte med regnskapsfører",date="2026-10-20",time="14:00",duration_minutes=60)),
("nb","Gjør om tusen dollar til svenske kroner.",["convert_currency","get_weather"],C("convert_currency",amount=1000,from_currency="USD",to_currency="SEK")),
("nb","Blir det regn i Tromsø på lørdag?",["get_weather","set_reminder"],C("get_weather",location="Tromsø",date="2026-10-17")),
("nb","Sjekk avgangene for linje 31 fra Storo.",["search_bus_departures"],C("search_bus_departures",stop="Storo",line="31")),
("nb","Påminnelse: betale strømregningen 1. november kl. 09.",["set_reminder"],C("set_reminder",text="Betale strømregningen",datetime="2026-11-01T09:00")),
# --- Bokmål: no call
("nb","Hva er forskjellen på bokmål og nynorsk?",["get_weather","send_sms"],NO),
("nb","Takk for hjelpen!",["create_calendar_event"],NO),
("nb","Hvordan regner jeg ut moms på 25 prosent?",["convert_currency"],NO),
("nb","Skriv et kort dikt om høsten.",["set_reminder","get_weather"],NO),
# --- Bokmål: clarify (required argument missing or ambiguous)
("nb","Send en melding til Kari.",["send_sms"],CL),
("nb","Legg inn et møte neste uke.",["create_calendar_event"],CL),
("nb","Hvor mye er 500 i euro?",["convert_currency"],CL),
("nb","Minn meg på det senere.",["set_reminder"],CL),
# --- Nynorsk: calls
("nn","Korleis blir vêret i Ålesund i morgon?",["get_weather"],C("get_weather",location="Ålesund",date="2026-10-13")),
("nn","Legg inn legetime på torsdag klokka ti.",["create_calendar_event"],C("create_calendar_event",title="Legetime",date="2026-10-15",time="10:00")),
("nn","Send ein SMS til 48 12 34 56 og sei at eg kjem klokka sju.",["send_sms"],C("send_sms",to="+4748123456",message="Eg kjem klokka sju.")),
("nn","Når går neste buss frå Bystasjonen etter klokka åtte i kveld?",["search_bus_departures"],C("search_bus_departures",stop="Bystasjonen",after_time="20:00")),
("nn","Kor mykje er 300 britiske pund i kroner?",["convert_currency"],C("convert_currency",amount=300,from_currency="GBP",to_currency="NOK")),
("nn","Minn meg på å vatne blomane på onsdag klokka 19.",["set_reminder"],C("set_reminder",text="Vatne blomane",datetime="2026-10-14T19:00")),
("nn","Slå opp verksemda med organisasjonsnummer 971 526 920.",["lookup_company"],C("lookup_company",org_number="971526920")),
("nn","Set opp eit møte med rektor 2. november kl. 13, ein halv time.",["create_calendar_event"],C("create_calendar_event",title="Møte med rektor",date="2026-11-02",time="13:00",duration_minutes=30)),
("nn","Blir det snø på Voss på søndag?",["get_weather","search_bus_departures"],C("get_weather",location="Voss",date="2026-10-18")),
("nn","Gjer om 2000 kroner til euro.",["convert_currency"],C("convert_currency",amount=2000,from_currency="NOK",to_currency="EUR")),
("nn","Sjekk avgangane for linje 5 frå Torget.",["search_bus_departures"],C("search_bus_departures",stop="Torget",line="5")),
("nn","Påminning: levere søknaden fredag klokka tolv.",["set_reminder"],C("set_reminder",text="Levere søknaden",datetime="2026-10-16T12:00")),
# --- Nynorsk: no call
("nn","Kva tyder ordet «dugnad»?",["get_weather","send_sms"],NO),
("nn","Tusen takk, det var alt.",["set_reminder"],NO),
("nn","Kan du forklare kva ein API er?",["lookup_company"],NO),
("nn","Skriv ei kort helsing til bursdagskortet til bestefar.",["send_sms"],NO),
# --- Nynorsk: clarify
("nn","Send ei melding til Ola.",["send_sms"],CL),
("nn","Legg inn ein avtale ein gong i neste veke.",["create_calendar_event"],CL),
("nn","Kva er vêret?",["get_weather"],CL),
("nn","Når går bussen?",["search_bus_departures"],CL),
]
with open("data/seed_v0.jsonl","w") as f:
    for i,(v,u,tools,exp) in enumerate(T,1):
        f.write(json.dumps({"id":f"nofc-{v}-{i:03d}","variant":v,"reference_datetime":"2026-10-12T10:00","user":u,"tools":tools,"expected":exp},ensure_ascii=False)+"\n")
print(len(T))
