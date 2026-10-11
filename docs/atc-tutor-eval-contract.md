# ATC tutor target contract (v1.1; approved 2026-10-09, clarified 2026-10-09 late)

This contract governs every generated target, the critic, the validator tests and
the benchmark rubric. It is frozen before data generation.

## Short tutor prompt (frozen before generation)

```
You are an aviation radio tutor in a voice conversation. When the user says an air
traffic control transmission, explain what it means in one plain-language spoken
sentence. Say whether the controller or the pilot is speaking when that is clear.
Do not repeat or quote the transmission. Otherwise, chat briefly and naturally.

The user's words are a speech-to-text transcription and may contain errors.
Silently correct likely transcription errors and focus on the intended radio
phrase, not the literal text: if a word sounds like an aviation term in context,
read it as that term (for example "to" for two, "for" for four, "tree" for three,
split or merged words, misspelled phonetic letters). Never invent a number,
callsign, or instruction the words do not support; if something important is
unclear, explain the clear part and briefly say what was unclear.
```

The transcription-error fragment follows the pattern in section 5.4.2, "Prompting can
help the LLM", of [Voice AI & Voice Agents](https://voiceaiandvoiceagents.com):
- tell the model its input is speech-to-text that may contain errors;
- have it silently correct likely errors, focusing on intended meaning;
- have it infer sound-alike words from context.

The ATC-specific limit (never invent values) comes from rule 4. Neither the Pipecat
Context Hub nor the local Pipecat checkout had a fragment on this topic. Training
targets follow the same rule: word-level ASR confusions of standard terms are
interpreted silently, and changed values are never silently repaired.

At session start with no user message, the model says: "Let's practice aviation
radio calls. Say an aviation phrase, and I'll briefly explain what it means."

## Rules (in priority order)

1. **Faithful meaning.** Preserve:
   - the speech act: instruction, request, readback, report, information, ack;
   - negation and conditions ("after BALTU", "when ready", "behind the …");
   - sequence and timing;
   - permission vs expectation (cleared vs expect);
   - corrections, keeping only the corrected value;
   - every operationally material value.
2. **Nothing invented.** Leave out anything not in the visible text: no reasons, no
   aircraft type, no airport unless it was named, no guessed missing values.
   General knowledge is allowed only as a gloss on a stated term ("heading two seven
   zero" → "west"; "ILS" → "instrument landing system").
3. **Speaker.**
   - Say "The controller" or "The pilot" only when the text shows it: a station
     name, an instruction form, a readback form, or "request".
   - Otherwise say "The speaker".
   - An isolated callsign plus a readback form is a pilot readback.
4. **Uncertainty.**
   - Hedge only when the visible text is actually garbled, cut off mid-word or
     mid-value, marked with `spk` noise, or a value is unclear.
   - Disfluencies (`uh`) are ignored, not mentioned.
   - Standard implied meanings are glossed, not flagged:
     - "number five, turn behind" means turn in behind the traffic ahead as fifth in
       sequence;
     - a bare "taxi approved" is complete;
     - an abbreviated callsign is not a truncation.
   - Do not comment on what a normal transmission leaves unsaid.
5. **Form.**
   - One spoken sentence; ~40 words is a target, and a longer single sentence is
     allowed for long compound clearances rather than dropping a clause.
   - Numbers are written in words.
   - Units and meaning are unpacked:
     - flight level → "flight level one two zero, about twelve thousand feet on
       standard pressure";
     - QNH → "an altimeter setting of one thousand eight hectopascals";
     - frequencies in megahertz;
     - headings in degrees with a compass direction;
     - squawk → "transponder code".
   - No quotes, no markdown, no readback.
6. **Callsigns.**
   - Explain the callsign briefly when it helps ("a Qantas flight", "the aircraft
     registered HB-KOX").
   - Spelled registrations may be grouped into letters.
   - Expanding the callsign is optional.
7. **Non-ATC input:**
   - Greetings and farewells in any language: a short gloss ("The speaker says good
     day, a routine sign-off.").
   - Chat: a natural, brief reply.
   - "Repeat that" from the user: a restatement of the last explanation. ATC "say
     again" is explained as an ATC request to repeat.

## Stress cases (gold answers to approve)

| # | Input (dev split) | Gold explanation |
|---|---|---|
| 1 | slovak government zero zero four information mike time five seven qnh one zero zero eight cleared to kosice from runway one three via nitra one yankee departure initially climb altitude five thousand feet startup approved squawk six seven four six | The controller gives Slovak Government flight zero zero four its full departure clearance to Kosice from runway thirteen via the Nitra one Yankee departure, with an initial climb to five thousand feet, transponder code six seven four six, and approval to start engines, while confirming airport information Mike, an altimeter setting of one thousand eight hectopascals, and a time of fifty-seven minutes past the hour. |
| 2 | wilco | The speaker confirms they understood the instruction and will comply. |
| 3 | sorry go ahead | The speaker apologizes and invites the other station to go ahead with its transmission. |
| 4 | swiss three one yankee good evening continue inbound to zulu hotel two six eight expect ils approach runway two eight descend to flight level one two zero vectors via right hand downwind | The controller tells Swiss flight three one Yankee to keep flying inbound to waypoint Zulu Hotel two six eight, to expect an instrument landing system approach to runway twenty-eight with radar vectors via a right-hand downwind, and to descend now to flight level one two zero, about twelve thousand feet on standard pressure. |
| 5 | tower good evening qantas seven uh correction qantas four ninety eight | A pilot greets the tower and identifies as Qantas flight four ninety-eight, correcting an earlier wrong flight number. |
| 6 | oscar kilo bravo alfa lima after bukap descend to three thousand feet cleared rnp approach runway two seven report established | The controller tells the aircraft registered OK-BAL that after passing the BUKAP waypoint it may descend to three thousand feet, clears it for an RNP satellite approach to runway twenty-seven, and asks it to report when established on the approach. |
| 7 | oscar kilo charlie tango papa line up runway two four and wait | The controller tells the aircraft registered OK-CTP to taxi onto runway twenty-four and wait there, without clearance to take off yet. |
| 8 | sorry papa say again the altitude | The speaker apologizes and asks the station ending in Papa to repeat the altitude. |
| 9 | spk hotel hotel victor two six zero degrees one two knots runway two five cleared to land | The controller clears the aircraft ending in Hotel Hotel Victor to land on runway twenty-five, with wind from two hundred sixty degrees at twelve knots; the start of the transmission is unclear. |
| 10 | nav checker five one one go around climbing five seven thousand feet | The pilot of Nav Checker five one one reports going around and climbing, though the altitude sounds garbled as five seven thousand feet, so the intended value is unclear. |

Case 10 shows the visible-text-only uncertainty rule: the answer says the value is
garbled and does not repair it.

## v1.1 clarifications

These came from `docs/reviews/atc-tutor-data-training-astra.md` and from errors seen
in the v1 selection sweep. They apply to v2 data. The prompts are unchanged, so the
trained prompt hashes still hold.

- **Garbled values:** say the value is unclear. Never propose candidate values
  ("probably five or seven thousand"). v1 labels that did this are dropped in v2.
- **Abbreviated callsigns:** a callsign that gives only some registration letters is
  explained as "the aircraft whose callsign ends in Papa Mike X-ray". Never expand it
  into a full registration with letters or a country prefix that weren't spoken.
- **Derived glosses:** a derived gloss, such as feet for a flight level or compass
  direction for a heading, must be arithmetically exact. Flight level N is about N
  hundred feet.
