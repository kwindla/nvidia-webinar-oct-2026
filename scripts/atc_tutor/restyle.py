"""Contract v1.2 restyle: unpack registrations to what the prefix means, not to letters.

v1 selection errors showed most unsupported claims came from registration renderings
("the aircraft registered OK-THH" for "triple hotel", reordered letters, unspoken
country prefixes). Targets therefore say "a Swiss-registered aircraft" when the
aircraft's own callsign, as spoken, starts with that nationality prefix, and "the
aircraft" otherwise. The nationality comes from a parse of the transmission, never
from the old explanation. "Ends in <spoken letters>" phrasing is kept when every
letter word was spoken. Airline callsign unpacking is unchanged.
"""

import re

# Nationality prefix -> (spoken phonetic prefix, adjective). ICAO nationality marks.
PREFIXES = {
    "VH": ("victor hotel", "Australian"), "HB": ("hotel bravo", "Swiss"), "OK": ("oscar kilo", "Czech"),
    "OM": ("oscar mike", "Slovak"), "OE": ("oscar echo", "Austrian"), "F": ("foxtrot", "French"),
    "G": ("golf", "British"), "D": ("delta", "German"), "N": ("november", "US"),
}
LETTERS = ("alpha alfa bravo charlie delta echo foxtrot golf hotel india juliett juliet kilo lima mike november "
           "oscar papa quebec romeo sierra tango uniform victor whiskey whisky xray yankee zulu").split()
DIGITS = "zero one two three four five six seven eight nine niner triple double".split()
NOT_CALLSIGN_BEFORE = {"via", "taxiway", "taxiways", "point", "stand", "route", "gate", "apron", "holding",
                       "information", "runway", "intersection", "on", "at", "to", "join", "report", "behind"}
REGISTRATION = re.compile(
    r"\b(?P<article>[Tt]he |[Aa]n? )?(?:(?P<adjective>[A-Z][a-z]+(?:-registered)?) )?"
    r"(?P<noun>aircraft|helicopter|plane|airplane|glider) (?:(?:registered|with (?:the )?registration) )?"
    r"(?P<registration>[A-Z]{1,2}-[A-Z0-9]{1,5}|N[0-9]{1,5}[A-Z]{0,2})\b")
LEFTOVER = re.compile(r"\b(?:[A-Z]{1,2}-[A-Z0-9]{2,5}|N[0-9]{1,5}[A-Z]{0,2}|registered [A-Z0-9])")
ENDS_IN = re.compile(r"\bend(?:s|ing)? in ")


def words(text):
    return re.sub(r"\bx[ -]?ray\b", "xray", text.lower()).replace("-", " ").split()


def callsign_runs(utterance):
    """Runs of three or more spelled letters/digits at the start or end of the transmission."""
    tokens = words(utterance)
    runs, i = [], 0
    while i < len(tokens):
        if tokens[i] in LETTERS:
            j = i
            while j < len(tokens) and (tokens[j] in LETTERS or tokens[j] in DIGITS):
                j += 1
            preceded = i > 0 and tokens[i - 1] in NOT_CALLSIGN_BEFORE
            at_edge = i <= 3 or j >= len(tokens) - 2
            if j - i >= 3 and at_edge and not preceded:
                runs.append(tokens[i:j])
            i = j
        else:
            i += 1
    return runs


def letter_runs(utterance):
    """Count runs of two or more spelled letters anywhere (possible short callsigns)."""
    tokens, count, length = words(utterance), 0, 0
    for token in tokens + [""]:
        if token in LETTERS:
            length += 1
        else:
            count += length >= 2
            length = 0
    return count


def nationality(utterance):
    """Adjective for the single prefix the transmission's callsign run(s) start with, else None."""
    runs = callsign_runs(utterance)
    if len(runs) != 1 or letter_runs(utterance) > 1:
        return None  # zero or several possible aircraft: nationality could belong to another one
    found = set()
    for run in runs:
        for prefix, (spoken, adjective) in PREFIXES.items():
            parts = spoken.split()
            if run[: len(parts)] == parts and len(run) > len(parts):
                found.add(adjective)
    return found.pop() if len(found) == 1 else None


def restyle(utterance, explanation):
    """Return the v1.2 explanation, or None if it cannot be rewritten faithfully."""
    if len(REGISTRATION.findall(explanation)) > 1:
        return None  # two aircraft: removing letters could merge their identities
    adjective = nationality(utterance)

    def replace(match):
        article = (match.group("article") or "the ").strip()
        capital = article[0].isupper()
        noun = match.group("noun")
        if adjective:
            if article.lower() in ("a", "an"):
                article = "an" if adjective[0] in "AEIOU" else "a"
            phrase = f"{article} {adjective}-registered {noun}"
        else:
            phrase = f"{'the' if article.lower() == 'the' else article} {noun}"
        return (phrase[0].upper() + phrase[1:]) if capital else phrase

    restyled = REGISTRATION.sub(replace, explanation)
    if LEFTOVER.search(restyled):
        return None
    spoken = [canonical(w) for w in words(utterance)]
    for match in ENDS_IN.finditer(restyled):
        letters = []
        for token in words(re.sub(r"[,.;]", " ", restyled[match.end():])):
            if token not in LETTERS and token not in DIGITS:
                break
            letters.append(canonical(token))
        if not letters or not contiguous(letters, spoken):
            return None  # "ends in" must copy the spoken letter words in order
    return restyled


def canonical(word):
    return {"alfa": "alpha", "juliet": "juliett", "whisky": "whiskey", "niner": "nine"}.get(word, word)


def contiguous(part, whole):
    return any(whole[i:i + len(part)] == part for i in range(len(whole) - len(part) + 1))


SELF_TESTS = [
    ("hotel bravo kilo oscar xray cleared to land runway two five",
     "The controller clears the aircraft registered HB-KOX to land on runway twenty-five.",
     "The controller clears the Swiss-registered aircraft to land on runway twenty-five."),
    ("papa mike xray report downwind",
     "The controller asks the aircraft registered HB-PMX to report on the downwind leg.",
     "The controller asks the aircraft to report on the downwind leg."),
    ("victor hotel bravo victor papa contact departures good day",
     "The controller tells the Australian aircraft registered VH-BVP to contact departures.",
     "The controller tells the Australian-registered aircraft to contact departures."),
    ("victor hotel foxtrot india uniform sydney ground good day",
     "A pilot greets Sydney Ground and identifies as the Australian-registered aircraft VH-FIU.",
     "A pilot greets Sydney Ground and identifies as the Australian-registered aircraft."),
    ("speedbird two one climb flight level one two zero",
     "The controller tells the British Airways flight to climb to flight level one two zero.",
     "The controller tells the British Airways flight to climb to flight level one two zero."),
    # Review counterexamples (docs/reviews/atc-tutor-v3-plan-astra.md H1/H2).
    ("papa mike xray taxi via golf alpha bravo",
     "The controller tells the aircraft registered G-PMX to taxi via Golf Alpha Bravo.",
     "The controller tells the aircraft to taxi via Golf Alpha Bravo."),
    ("oscar kilo golf alpha bravo descend",
     "The controller tells the aircraft registered G-AB to descend.",
     "The controller tells the Czech-registered aircraft to descend."),
    ("november one two three alpha bravo cleared for takeoff",
     "The controller clears the aircraft N123AB for takeoff.",
     "The controller clears the US-registered aircraft for takeoff."),
    ("hotel bravo alpha bravo charlie traffic hotel bravo xray yankee zulu",
     "The controller tells the aircraft registered HB-ABC about traffic, the aircraft registered HB-XYZ.",
     None),
    ("papa mike xray report downwind",
     "The controller asks the aircraft whose callsign ends in Papa Mike X-ray to report downwind.",
     "The controller asks the aircraft whose callsign ends in Papa Mike X-ray to report downwind."),
    ("papa mike xray report downwind",
     "The controller asks the aircraft whose callsign ends in Papa Kilo X-ray to report downwind.",
     None),
    # v3 follow-up probes (docs/reviews/atc-tutor-v3-plan-astra.md follow-up 1-2).
    ("papa mike xray traffic hotel bravo alpha bravo charlie",
     "The controller gives traffic information to the aircraft registered G-PMX.",
     "The controller gives traffic information to the aircraft."),
    ("papa mike xray report downwind",
     "The controller asks the aircraft whose callsign ends in X-ray Mike Papa to report downwind.",
     None),
    ("papa mike xray report downwind",
     "The controller asks the aircraft whose callsign ends in Papa Papa Mike to report downwind.",
     None),
    ("papa mike traffic hotel bravo alpha bravo charlie",
     "The controller gives traffic information to the aircraft registered G-PM.",
     "The controller gives traffic information to the aircraft."),
    ("oscar kilo papa romeo mike turn right heading zero six zero",
     "The controller tells OK-PRM to turn right.",
     None),
]


if __name__ == "__main__":
    failures = 0
    for utterance, before, expected in SELF_TESTS:
        got = restyle(utterance, before)
        ok = got == expected
        failures += not ok
        print("ok  " if ok else "FAIL", repr(got))
    print(f"{len(SELF_TESTS) - failures}/{len(SELF_TESTS)} restyle self-tests passed")
    raise SystemExit(1 if failures else 0)
