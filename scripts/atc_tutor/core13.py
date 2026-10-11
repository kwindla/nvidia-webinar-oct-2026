"""Core-phraseology inputs for v1.3: the transmission types a demo audience says most.

The v1.3 corpus had few headings (28), squawks (27), "expect" (9), corrections (2) and
negatives (13). This builds about 800 synthetic transmissions from templates with exact
slot values (values.Sampler), each with a facts block (heading in degrees and compass
point, flight level in feet, squawk meaning) that the writer and critic both see.
data13 then writes and audits the targets (`targets --source core`).

Usage: core13.py   -> artifacts/atc-tutor/v13/core-inputs.jsonl
"""

import json
import random

from opus import OUT
from values import Sampler, compass, spoken_digits, words_int

V13 = OUT / "v13"
COMPASS16 = ["north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast", "southeast",
             "south-southeast", "south", "south-southwest", "southwest", "west-southwest", "west",
             "west-northwest", "northwest", "north-northwest"]
SQUAWK_MEANING = {"7700": "the emergency code", "7600": "the radio-failure code", "7500": "the unlawful-interference (hijack) code"}


def compass16(degrees):
    return COMPASS16[round((degrees % 360) / 22.5) % 16]


def heading_fact(h):
    return f"heading {spoken_digits(f'{h['value']:03d}')} = {words_int(h['value'])} degrees, {compass16(h['value'])}"


def level_fact(fl):
    return f"flight level {spoken_digits(f'{fl['value']:03d}')} = about {words_int(fl['value'] * 100)} feet on standard pressure"


TEMPLATES = {  # kind: (count, [templates]); {cs} callsign, {h} heading, {fl}/{fl2} levels, {sq}, {rw}, {alt}, {f}/{f2}, {wp}
    "heading": (250, ["{cs} turn left heading {h}", "{cs} turn right heading {h}", "{cs} fly heading {h}",
                      "turn left heading {h} {cs}", "turn right heading {h} {cs}", "heading {h} {cs}",
                      "{cs} turn right heading {h} vectoring for ils approach {rw}",
                      "{cs} after departure turn left heading {h} climb {alt}",
                      "{cs} turn left heading {h} descend flight level {fl}",
                      "{cs} continue present heading", "{cs} report heading", "left heading {h} {cs}",
                      "{cs} turn right heading {h} for spacing"]),
    "level": (150, ["{cs} climb flight level {fl}", "{cs} descend flight level {fl}", "climbing flight level {fl} {cs}",
                    "descending flight level {fl} {cs}", "{cs} maintain flight level {fl}",
                    "{cs} climb flight level {fl} expect flight level {fl2} in ten minutes",
                    "{cs} request flight level {fl}", "{cs} stop climb flight level {fl}",
                    "{cs} descend flight level {fl} level by {wp}"]),
    "squawk": (100, ["{cs} squawk {sq}", "squawk {sq} {cs}", "{cs} squawk ident", "{cs} reset squawk {sq}",
                     "{cs} confirm squawking {sq}", "{cs} squawk {sqe}", "squawking {sqe} {cs}", "{cs} squawk standby"]),
    "expect": (80, ["{cs} expect ils approach {rw}", "{cs} descend {alt} expect vectors ils approach {rw}",
                    "{cs} expect further climb in five minutes", "{cs} expect runway {rwn} for departure",
                    "{cs} expect direct {wp} after departure", "{cs} hold at {wp} expect further clearance at time two five"]),
    "correction": (60, ["{cs} climb flight level {fl} correction flight level {fl2}",
                        "{cs} contact tower {f} correction {f2}", "{cs} turn left correction right heading {h}",
                        "{cs} descend {alt} correction {alt2}", "{cs} squawk {sq} correction {sq2}"]),
    "negative": (80, ["{cs} negative hold short {rw}", "{cs} not cleared to land go around", "{cs} negative not above {alt}",
                      "{cs} behind the landing airbus line up {rw} and wait behind", "{cs} after {wp} descend {alt}",
                      "{cs} hold short {rw} traffic on short final", "{cs} do not descend below {alt} until {wp}",
                      "negative {cs} maintain {alt}"]),
    "asr": (80, ["wilko {cs}", "wilko", "rodger {cs}", "rodger that", "a firm {cs}", "say a gain {cs}",
                 "stand by {cs}", "{cs} rodger contact tower {f}", "wilco {cs}", "{cs} say intensions",
                 "affirmative {cs}", "{cs} ready for depart sure"]),
}


def render(rng, s, template, dialect):
    cs = s.callsign(dialect) if rng.random() < 0.8 else s.abbreviated_callsign()
    h, fl, fl2 = s.heading(), s.flight_level(), s.flight_level()
    rw, alt, alt2 = s.runway(), s.altitude(), s.altitude()
    sq, sq2, sqe = s.squawk(), s.squawk(), s.squawk(special=True)
    f, f2, wp = s.frequency(dialect), s.frequency(dialect), s.waypoint()
    text = template.format(cs=cs["spoken"], h=spoken_digits(f"{h['value']:03d}"), fl=fl["spoken"].split(" ", 2)[2],
                           fl2=fl2["spoken"].split(" ", 2)[2], sq=spoken_digits(sq["value"]), sq2=spoken_digits(sq2["value"]),
                           sqe=spoken_digits(sqe["value"]), rw=rw["spoken"], rwn=rw["spoken"].split(" ", 1)[1],
                           alt=alt["spoken"], alt2=alt2["spoken"], f=f["spoken"], f2=f2["spoken"], wp=wp["spoken"])
    facts = []
    if "{h}" in template:
        facts.append(heading_fact(h))
    if "{fl}" in template:
        facts.append(level_fact(fl))
    if "{fl2}" in template:
        facts.append(level_fact(fl2))
    if "{sqe}" in template:
        facts.append(f"squawk {spoken_digits(sqe['value'])} is {SQUAWK_MEANING[sqe['value']]}")
    if "{f}" in template:
        facts.append(f"frequency {f['spoken']} = {f['plain']}")
    return text, facts


def main():
    rng = random.Random(20261010)
    sampler = Sampler(20261010)
    rows = []
    for kind, (count, templates) in TEMPLATES.items():
        for n in range(count):
            dialect = rng.choice(["icao_europe", "icao_europe", "icao_australia", "faa"])
            text, facts = render(rng, sampler, templates[n % len(templates)], dialect)
            rows.append({"id": f"core-{kind}-{n:03d}", "component": f"core:{kind}", "kind": kind,
                         "transmission": text, "facts": facts, "source": "core"})
    V13.mkdir(parents=True, exist_ok=True)
    (V13 / "core-inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "core inputs")
    for r in rows[::97]:
        print(" ", r["transmission"], "|", "; ".join(r["facts"]))


if __name__ == "__main__":
    main()
