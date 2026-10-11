"""Deterministic checks for generated ATC tutor rows.

Code decides only what it can decide: form, value rendering, polarity and
condition markers, leakage against all real transcripts. Anything else is left
to the critic. Each row gets status pass, reject (with reasons) or needs_review.

Usage:
  uv run python scripts/atc_tutor/check.py --self-test
  uv run python scripts/atc_tutor/check.py   # generated.jsonl -> checked.jsonl
"""

import argparse
import json
import re
import sys
from collections import Counter

from opus import OUT
from values import COMPASS, compass, spoken_digits, words_int

NEGATION = re.compile(r"\b(not|no|negative|declin\w*|den\w*|unable|cannot|n't|without)\b")


def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine".split())}
UNITS["niner"] = 9
TEEN_WORDS = {w: 10 + i for i, w in enumerate(
    "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS_WORDS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def spoken_number(text):
    """'one two' -> 12, 'twelve' -> 12, 'twenty five' -> 25; None if not a plain number."""
    words = text.replace("-", " ").split()
    if words and all(w in UNITS for w in words):
        return int("".join(str(UNITS[w]) for w in words))
    if len(words) == 1 and words[0] in TEEN_WORDS:
        return TEEN_WORDS[words[0]]
    if words and words[0] in TENS_WORDS and len(words) <= 2 and (len(words) == 1 or words[1] in UNITS):
        return TENS_WORDS[words[0]] + (UNITS[words[1]] if len(words) == 2 else 0)
    return None


def has_any(text, options):
    """Whole-word/phrase match, so 'seven' does not match inside 'seventy'."""
    return any(re.search(rf"(?<![a-z]){re.escape(option)}(?![a-z])", text) for option in options if option)


def renderings(key, value):
    """Acceptable plain-language renderings of a used value in an explanation."""
    if key == "heading":
        return [value["plain"], spoken_digits(f"{value['value']:03d}")]
    if key == "flight_level":
        return [spoken_digits(f"{value['value']:03d}"), words_int(value["value"]), value["plain_feet"]]
    if key == "altitude":
        return [value["plain"], words_int(value["value"])]
    if key == "runway":
        return [value["plain"], spoken_digits(value["value"][:2])]
    if key in ("pressure", "frequency", "speed"):
        return [value["plain"].replace(" megahertz", "").replace(" hectopascals", "")
                .replace(" inches of mercury", "").replace(" knots", ""),
                spoken_digits(str(value["value"]).replace(".", ""))]
    if key == "squawk":
        return [value["plain"]]
    if key == "wind":
        return [words_int(value["direction"])]
    return []


def check_row(row, real_texts=frozenset()):
    reasons, review = [], []
    explanation = row["explanation"].strip()
    lower = explanation.lower().replace("’", "'")
    utterance = normalize(row["utterance"])

    # Form.
    if re.search(r"\d", explanation):
        reasons.append("digits")
    if re.search(r"[\"“”*#`]|^\s*-", explanation):
        reasons.append("quote_or_markdown")
    if not re.search(r"[.!?]$", explanation):
        reasons.append("no_final_punctuation")
    if re.search(r"[.!?]\s+[A-Z]", explanation):
        reasons.append("multiple_sentences")
    words = len(explanation.split())
    if words > 70:
        reasons.append("too_long")
    elif words > 45:
        review.append("long")
    if normalize(explanation).startswith(utterance[:40]) and len(utterance) > 12:
        reasons.append("quotes_transmission")

    # Leakage: never a verbatim real transcript (any split).
    if utterance in real_texts:
        reasons.append("verbatim_real_transcript")

    # Values: the utterance carries each used value; the explanation renders it.
    bundle = row.get("bundle", {})
    for key in row.get("values_used", []):
        value = bundle.get(key)
        if not isinstance(value, dict):
            continue
        spoken = value.get("spoken")
        if spoken and normalize(spoken) not in utterance and key not in ("wind", "callsign", "callsign_alt"):
            reasons.append(f"value_missing_in_utterance:{key}")
            continue
        options = renderings(key, value)
        if options and not has_any(lower, options):
            reasons.append(f"value_not_explained:{key}")

    # Units: wind and speed are knots; distances are miles; altitudes are feet.
    if re.search(r"\bwind\b[a-z ]*\bdegrees\b[a-z\- ]*\b(miles|feet|meters)\b", utterance):
        reasons.append("wind_wrong_unit")
    if re.search(r"\bknots\b", lower) and not re.search(r"\bknots\b", utterance) and "wind" not in utterance:
        review.append("knots_unverified")

    # Derived flight-level gloss must be exact: flight level N is about N hundred feet.
    for match in re.finditer(r"flight level ((?:(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)[ ,]*){2,3})"
                             r"[^.;]{0,60}?\b(?:about|around|roughly|approximately|equivalent to|or|some|near|close to)"
                             r" ([a-z\- ]+?) (thousand|hundred) feet", lower):
        level = spoken_number(match.group(1).replace(",", " ").strip())
        words = match.group(2).strip()
        if words.endswith(" thousand") and match.group(3) == "hundred":
            continue  # "six thousand five hundred feet": handled as a full number below
        amount = 1 if words in ("a", "one") else spoken_number(words)
        if level is not None and amount is not None:
            feet = amount * (1000 if match.group(3) == "thousand" else 100)
            if feet != level * 100:
                reasons.append("flight_level_gloss_wrong")
    for match in re.finditer(r"flight level ((?:(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)[ ,]*){2,3})"
                             r"[^.;]{0,60}?\b([a-z]+(?:-[a-z]+)? thousand(?: [a-z]+ hundred)?) feet", lower):
        level = spoken_number(match.group(1).replace(",", " ").strip())
        parts = re.match(r"([a-z\-]+) thousand(?: ([a-z]+) hundred)?", match.group(2))
        thousands = spoken_number(parts.group(1)) if parts else None
        hundreds = spoken_number(parts.group(2)) if parts and parts.group(2) else 0
        if level is not None and thousands is not None and hundreds is not None:
            if thousands * 1000 + hundreds * 100 != level * 100:
                reasons.append("flight_level_gloss_wrong")

    # Headings: the spoken three digits must be the explained heading, and any compass
    # word must fit (8-point, with neighbours accepted within 10 degrees of a boundary).
    for match in re.finditer(r"\bheading ((?:zero|one|two|three|four|five|six|seven|eight|nine|niner) "
                             r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner) "
                             r"(?:zero|one|two|three|four|five|six|seven|eight|nine|niner))\b", utterance):
        value = spoken_number(match.group(1))
        if value is None:
            continue
        if not has_any(lower, [words_int(value) + " degrees", spoken_digits(f"{value:03d}"),
                               f"heading of {words_int(value)}", f"heading {words_int(value)}"]):
            reasons.append("heading_changed")
            continue
        named = [c for c in COMPASS if re.search(rf"\b{c}\b", lower)]
        if named:
            allowed = {compass(value)}
            if (value % 45) <= 10 or (value % 45) >= 35:
                allowed |= {compass(value - 12), compass(value + 12)}
            if not set(named) & allowed:
                reasons.append("heading_compass_wrong")

    # Squawk codes: the spoken code must be the explained code.
    squawk = re.search(r"\bsquawk(?:ing)? ((?:(?:zero|one|two|three|four|five|six|seven) ?){4})", utterance)
    if squawk:
        digits = squawk.group(1).split()
        if not has_any(lower, [" ".join(digits)]):
            reasons.append("squawk_changed")

    # A condition's waypoint must survive: "after bukap ..." keeps "bukap".
    for point in re.findall(r"\bafter ([a-z]{4,})\b", utterance):
        if point not in {"takeoff", "departure", "landing", "passing", "that", "this", "the", "low", "you",
                         "airborne", "lift", "vacating", "crossing"} and point not in lower.lower():
            reasons.append("condition_object_changed")

    # Abbreviated callsigns are explained as "ending in", never expanded to a registration.
    callsign = bundle.get("callsign")
    if isinstance(callsign, dict) and callsign.get("type") == "abbreviated" and "callsign" in row.get("values_used", []):
        if re.search(r"\b[A-Z]{1,2}-[A-Z0-9]{2,}\b", explanation) or re.search(r"\bregistered\b", lower):
            reasons.append("abbreviated_callsign_expanded")

    # Wind speed: the spoken knots must be the explained knots.
    wind = re.search(r"\bwind\b.*?\bdegrees\b ([a-z\- ]+?) knots", utterance)
    if wind:
        speed = spoken_number(wind.group(1))
        if speed is not None and not has_any(lower, [words_int(speed), spoken_digits(str(speed))]):
            reasons.append("wind_speed_changed")

    # Polarity, permission and condition markers.
    if "expect" in utterance and "expect" not in lower:
        reasons.append("expect_lost")
    if re.search(r"\bclimb", utterance) and not re.search(r"\b(climb\w*|ascen\w*|higher|up to)\b", lower):
        reasons.append("climb_lost")
    if re.search(r"\bdescen", utterance) and not re.search(r"\b(descen\w*|lower|down to)\b", lower):
        reasons.append("descend_lost")
    if re.search(r"\bclimb", lower) and not re.search(r"\bclimb", utterance) and re.search(r"\bdescen", utterance):
        reasons.append("climb_descend_swapped")
    if re.search(r"\bdescen", lower) and not re.search(r"\bdescen", utterance) and re.search(r"\bclimb", utterance):
        reasons.append("climb_descend_swapped")
    if "line up" in utterance and "wait" in utterance:
        if "wait" not in lower:
            reasons.append("line_up_wait_lost")
        if re.search(r"clear\w* (for|to) take ?off", lower) and not NEGATION.search(lower) and "yet" not in lower:
            reasons.append("takeoff_invented")
    if "cleared to land" in utterance or "clear to land" in utterance:
        if "land" not in lower or not re.search(r"clear|authori|permission|approv", lower):
            reasons.append("landing_clearance_lost")
        if "expect" in lower and "expect" not in utterance:
            reasons.append("clearance_downgraded_to_expect")
    if "hold short" in utterance and not re.search(r"hold short|short of|stop before|remain clear|wait before", lower):
        reasons.append("hold_short_lost")
    if re.search(r"\bafter\b", utterance) and not re.search(r"\b(after|once|when|passing|then)\b", lower):
        reasons.append("condition_lost")
    if re.search(r"\bafter\b", utterance) and not re.search(r"\bbefore\b", utterance) and re.search(r"\bbefore\b", lower):
        reasons.append("condition_reversed")
    if re.search(r"\bnot (yet )?cleared to land", utterance) and re.search(
            r"(?<!not )(?<!not yet )\b(is |are )?cleared to land\b|\bclears [^.]{0,40}\bto land\b", lower) and not re.search(
            r"not (yet )?(be )?cleared to land", lower):
        reasons.append("negation_lost")
    if re.search(r"\bnot (yet )?(cleared|approved)", utterance) and not re.search(
            r"not (yet )?(cleared|clear|approved|authori)|no clearance|without (a |any )?(clearance|approval)|"
            r"isn't (cleared|approved)|not permitted|declin|den(y|ies|ied)", lower):
        reasons.append("negation_lost")
    elif re.search(r"\b(negative|not)\b", utterance) and not NEGATION.search(lower):
        # A negated transmission explained as a clearance or approval is a polarity flip.
        if re.search(r"\bclear|\bapprov|\bauthori", lower):
            reasons.append("negation_lost")
        else:
            review.append("negation_unverified")
    if "correction" in utterance and not re.search(r"correct", lower):
        review.append("correction_unverified")
    for side in ("left", "right", "center"):
        if re.search(rf"runway [a-z ]*\b{side}\b", utterance) and side not in lower:
            reasons.append(f"runway_side_lost:{side}")
    for side, other in (("left", "right"), ("right", "left")):
        if re.search(rf"runway [a-z ]*\b{side}\b", utterance) and not re.search(rf"\b{other}\b", utterance) \
                and re.search(rf"runway[^,.]*\b{other}\b", lower):
            reasons.append("runway_side_swapped")

    status = "reject" if reasons else ("needs_review" if review else "pass")
    return {"status": status, "reasons": reasons, "review": review}


SELF_TESTS = [
    # (utterance, explanation, values_used, bundle, expected_status)
    ("swiss three one yankee expect ils approach runway two eight",
     "The controller tells Swiss flight three one Yankee to expect an instrument landing system approach to runway twenty-eight.",
     [], {}, "pass"),
    ("swiss three one yankee expect ils approach runway two eight",
     "The controller clears Swiss flight three one Yankee for an instrument landing system approach to runway twenty-eight.",
     [], {}, "reject"),
    ("qantas four ninety eight climb flight level one two zero",
     "The controller tells Qantas flight four ninety-eight to descend to flight level one two zero, about twelve thousand feet.",
     [], {}, "reject"),
    ("qantas four ninety eight climb flight level one two zero",
     "The controller tells Qantas flight four ninety-eight to climb to flight level one two zero, about twelve thousand feet on standard pressure.",
     [], {}, "pass"),
    ("oscar kilo charlie tango papa line up runway two four and wait",
     "The controller clears the aircraft for takeoff from runway twenty-four.",
     [], {}, "reject"),
    ("oscar kilo charlie tango papa line up runway two four and wait",
     "The controller tells the aircraft to taxi onto runway twenty-four and wait there without clearance to take off yet.",
     [], {}, "pass"),
    ("oscar kilo bravo alfa lima after bukap descend to three thousand feet",
     "The controller tells the aircraft to descend to three thousand feet now.",
     [], {}, "reject"),
    ("runway three four left cleared to land jetstar seven sixty seven",
     "The controller clears Jetstar flight seven sixty-seven to land on runway thirty-four right.",
     [], {}, "reject"),
    ("runway three four left cleared to land jetstar seven sixty seven",
     "The controller tells Jetstar flight seven sixty-seven to expect to land on runway thirty-four left.",
     [], {}, "reject"),
    ("runway three four left cleared to land jetstar seven sixty seven",
     "The controller clears Jetstar flight seven sixty-seven to land on runway thirty-four left.",
     [], {}, "pass"),
    ("speedbird two one turn left heading two seven zero",
     "The controller tells Speedbird two one to turn left to 270 degrees.",
     [], {}, "reject"),
    ("speedbird two one turn left heading two seven zero",
     "The controller tells the British Airways flight to turn left to a heading of two hundred ninety degrees, facing west.",
     ["heading"], {"heading": {"value": 270, "spoken": "heading two seven zero", "plain": "two hundred seventy degrees"}},
     "reject"),
    ("speedbird two one turn left heading two seven zero",
     "The controller tells the British Airways flight to turn left to a heading of two hundred seventy degrees, facing west.",
     ["heading"], {"heading": {"value": 270, "spoken": "heading two seven zero", "plain": "two hundred seventy degrees"}},
     "pass"),
    ("hotel bravo kilo negative not approved",
     "The controller tells the aircraft its request is approved.",
     [], {}, "reject"),
    ("wilco", "The speaker confirms they understood the instruction and will comply.", [], {}, "pass"),
    # v4: heading errors seen in the live smoke.
    ("speedbird two one turn left heading two seven zero",
     "The controller asks British Airways flight two one to turn left onto a heading of twenty-seven degrees, roughly west.",
     [], {}, "reject"),
    ("speedbird two one turn left heading two seven zero",
     "The controller asks British Airways flight two one to turn left onto a heading of two hundred seventy degrees, roughly west.",
     [], {}, "pass"),
    ("speedbird two one turn left heading two seven zero",
     "The controller asks British Airways flight two one to turn left onto a heading of two hundred seventy degrees, roughly east.",
     [], {}, "reject"),
    ("klm five turn right heading zero five five",
     "The controller tells KLM flight five to turn right onto a heading of fifty-five degrees, toward the northeast.",
     [], {}, "pass"),
    # v2 review probes (docs/reviews/atc-tutor-v2-astra.md H2).
    ("hotel bravo kilo wind two seven zero degrees seven knots runway two five cleared to land",
     "The controller clears the aircraft to land on runway twenty-five with wind from two hundred seventy degrees at seventy knots.",
     [], {}, "reject"),
    ("sierra papa foxtrot flight level six zero is now available",
     "The controller says flight level six zero, about sixty-five hundred feet, is now available.",
     [], {}, "reject"),
    ("sierra papa foxtrot flight level six zero is now available",
     "The controller says flight level six zero, equivalent to eighteen thousand feet, is now available.",
     [], {}, "reject"),
    ("oscar kilo bravo alfa lima after bukap descend to three thousand feet",
     "The controller tells the aircraft to descend to three thousand feet after passing LOMTI.",
     [], {}, "reject"),
    ("hotel bravo kilo negative not cleared to land",
     "The controller says the aircraft is not cleared for takeoff but is cleared to land.",
     [], {}, "reject"),
    ("hotel bravo kilo squawk four two seven one",
     "The controller assigns transponder code seven seven zero zero.",
     [], {}, "reject"),
    ("hotel bravo kilo squawk four two seven one",
     "The controller assigns transponder code four two seven one.",
     [], {}, "pass"),
    ("papa mike x ray report downwind runway two five",
     "The controller asks the aircraft to report on the downwind leg for runway twenty-five.",
     ["callsign"], {"callsign": {"type": "abbreviated", "tail": "PMX", "spoken": "papa mike x ray"}}, "pass"),
    # v2: errors seen in the v1 selection sweep.
    ("sierra papa foxtrot lima foxtrot flight level six zero is now available",
     "The controller tells the aircraft that flight level six zero, about eighteen thousand feet on standard pressure, is now available.",
     [], {}, "reject"),
    ("sierra papa foxtrot lima foxtrot flight level six zero is now available",
     "The controller tells the aircraft that flight level six zero, about six thousand feet on standard pressure, is now available.",
     [], {}, "pass"),
    ("papa mike x ray report downwind runway two five",
     "The controller asks the aircraft registered HB-PMX to report on the downwind leg for runway twenty-five.",
     ["callsign"], {"callsign": {"type": "abbreviated", "tail": "PMX", "spoken": "papa mike x ray"}}, "reject"),
    ("papa mike x ray report downwind runway two five",
     "The controller asks the aircraft whose callsign ends in Papa Mike X-ray to report on the downwind leg for runway twenty-five.",
     ["callsign"], {"callsign": {"type": "abbreviated", "tail": "PMX", "spoken": "papa mike x ray"}}, "pass"),
    # Review probes (docs/reviews/atc-tutor-data-training-astra.md H1).
    ("oscar kilo bravo alfa lima after bukap descend to three thousand feet",
     "The controller tells the aircraft to descend to three thousand feet before passing the BUKAP waypoint.",
     [], {}, "reject"),
    ("hotel bravo kilo wind two six zero degrees one zero knots runway two five cleared to land",
     "The controller clears the aircraft to land on runway twenty-five with wind from two hundred sixty degrees at twenty knots.",
     [], {}, "reject"),
    ("hotel bravo kilo wind two six zero degrees one zero knots runway two five cleared to land",
     "The controller clears the aircraft to land on runway twenty-five with wind from two hundred sixty degrees at ten knots.",
     [], {}, "pass"),
    ("hotel bravo kilo negative not cleared to land",
     "The controller clears the aircraft to land with no delay.",
     [], {}, "reject"),
    ("hotel bravo kilo negative not cleared to land",
     "The controller tells the aircraft it is not cleared to land.",
     [], {}, "pass"),
    ("hotel bravo kilo wind three two zero degrees sixteen miles runway two five cleared to land",
     "The controller clears the aircraft to land on runway twenty-five with wind from three hundred twenty degrees at sixteen miles.",
     [], {}, "reject"),
    ("wilco", "The speaker confirms. They will comply.", [], {}, "reject"),
]


def self_test():
    failures = 0
    for utterance, explanation, used, bundle, expected in SELF_TESTS:
        result = check_row({"utterance": utterance, "explanation": explanation,
                            "values_used": used, "bundle": bundle})
        ok = result["status"] == expected
        failures += not ok
        print("ok  " if ok else "FAIL", expected, "->", result["status"], result["reasons"] or result["review"], "|", explanation[:70])
    print(f"{len(SELF_TESTS) - failures}/{len(SELF_TESTS)} validator self-tests passed")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", default="generated.jsonl")
    parser.add_argument("--output", default="checked.jsonl")
    args = parser.parse_args()
    if args.self_test:
        sys.exit(1 if self_test() else 0)
    real = frozenset(normalize(json.loads(l)["text"]) for l in (OUT / "split.jsonl").open())
    rows = [json.loads(l) for l in (OUT / args.input).open()]
    seen, out, stats = set(), [], Counter()
    for row in rows:
        result = check_row(row, real)
        key = normalize(row["utterance"])
        if key in seen:
            result = {"status": "reject", "reasons": ["duplicate"], "review": []}
        seen.add(key)
        row["check"] = result
        stats[result["status"]] += 1
        for reason in result["reasons"]:
            stats["r:" + reason.split(":")[0]] += 1
        out.append(row)
    (OUT / args.output).write_text("".join(json.dumps(r) + "\n" for r in out))
    print(json.dumps(dict(stats.most_common()), indent=1))


if __name__ == "__main__":
    main()
