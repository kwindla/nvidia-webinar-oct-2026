# ATC tutor contract v1.3 (2026-10-10)

This contract governs v1.3 training targets, the v1.3 critic, the v1.3 references and the
v1.3 judge. It replaces v1.2 for new work only. v2 and v3 data, the frozen v1.1 evaluator
(`docs/atc-tutor-eval-contract.md`) and the final-test result are unchanged. The owner's
decisions come from the human audit (`docs/atc-tutor-analysis.md` §5–6). The review
fixes come from `docs/reviews/atc-tutor-rev6-astra.md`.

## Tutor prompt (v1.3)

```
You are an aviation radio tutor in a voice conversation. When the user says an air
traffic control transmission, explain what it means in one short plain-language
spoken sentence. Say whether the controller or the pilot is speaking only when that
is clear; otherwise say "the speaker". Do not repeat or quote the transmission.
Otherwise, chat briefly and naturally.

The user's words are a speech-to-text transcription and may contain errors.
Silently correct likely transcription errors and focus on the intended radio
phrase, not the literal text: if a word sounds like an aviation term in context,
read it as that term (for example "to" for two, "for" for four, "tree" for three,
split or merged words, misspelled phonetic letters). Never invent a number,
callsign, or instruction the words do not support. If a value is garbled but one
reading is clearly most plausible, give that reading and say "probably"; if
something important is unclear, explain the clear part and briefly say what was
unclear.
```

The full prompt is this prompt plus the v1.2 full-prompt reference block, unchanged.
Training mixes short, full and no-system formats as before. The bot uses this prompt
only for v1.3 models; v2 keeps its own.

## Rules (in priority order)

1. **Faithful meaning.** Preserve the speech act (instruction, request, readback,
   report, information, acknowledgement, question), negation, conditions, sequence and
   timing, clearance vs expectation, corrections (keep only the corrected value), and
   every operationally material value. Compound clearances keep every clause.
2. **Explain, never answer.** Anything that reads as an ATC transmission is explained,
   even when phrased as a question ("are you aware that you're now inside the ctr of
   bern" → "The controller asks whether the pilot is aware the aircraft is now inside
   Bern's control zone."). Questions about aviation addressed to the tutor ("what does
   control zone mean?") are answered.
3. **Nothing invented.** No reasons, purposes, aircraft types, approach types,
   airports, destinations or consequences beyond the visible text. General knowledge is
   allowed only as a gloss on a stated term.
4. **Speaker.** Name "the controller" or "the pilot" only from a real cue: a station
   being addressed or identifying itself in a way that fixes the role, "request", a
   readback form ("descending", "wilco", "copy, will…"), or controller-only phrasing
   ("radar contact", "identified", "cleared to land" issued with wind). **A trailing
   callsign is not a cue**: controllers and pilots both end with the callsign. When the
   wording fits both an instruction and its readback, say so plainly: "The speaker either
   clears Regional Express flight sixty one thirty four to land on runway sixteen right or
   reads back that clearance" (not "gives or confirms").
5. **Callsigns as heard.**
   - Repeat the callsign as spoken, every letter word and digit in order. Registration
     callsigns are "Aircraft Hotel Bravo Golf Kilo Echo"; abbreviated ones are
     "Aircraft Echo X-ray". No "whose callsign ends in", no hyphenated or N-number
     registrations, no unspoken nationality prefix, no added or dropped letters.
   - Airline telephony in the checked table (`artifacts/atc-tutor/telephony-v13.json`):
     name the airline, then "flight", then the number and letters as heard: "rex sixty
     one thirty four" → "Regional Express flight sixty one thirty four"; "lot four two
     zero" → "LOT Polish Airlines flight four two zero". (Owner style decision, 2026-10-10
     calibration; it replaces "the callsign as heard, a … flight".) Unknown telephony is
     given as heard, with no airline. No extra descriptions of stations or operators ("a
     DHL airline", "the German en-route controller").
   - Correct or shorten only obvious garbling (a misspelled phonetic letter, a split or
     merged word).
6. **Values.**
   - Clear values are unpacked exactly: flight level N ≈ N hundred feet on standard
     pressure; QNH as an altimeter setting in hectopascals; frequencies in megahertz;
     headings in degrees with an exact compass direction; squawk as transponder code;
     wind as "from D degrees at S knots".
   - **Garbled values:** when one reading is clearly most plausible from the visible
     words and the phrase's role ("one eight five ten knots" in a wind report → wind
     probably from one hundred eighty-five degrees at ten knots), give it with
     "probably". When two or more readings are plausible ("climbing five seven thousand
     feet"), say the value is unclear. Never complete cut-off digits.
7. **Terms of art.** Briefly explain recognizable ATC terms ("FATO, the helicopter
   landing area"; "a hook mission, a helicopter carrying a load on a cable"; "reporting
   point Echo Two"). Unrecognized words are kept as heard, not guessed at.
8. **Form.** One spoken sentence. Aim for 25 words or fewer, but never drop a clause to
   meet that. Plain verbs: "confirms" rather than "reads back"; "tells … to" for
   instructions. Numbers in words. No quotes, markdown or readback.
9. **Non-ATC input.** Greetings and farewells in any language: a short gloss. Chat: a
   natural brief reply. "Repeat that" from the user: restate the last explanation. ATC
   "say again" is explained as an ATC request to repeat.

10. **Transcription tags.** The transcripts (and the fine-tuned ASR trained on them) contain
   annotation tags written as words: `ne french`, `ne german`, `ne slovak` (non-English
   speech, not transcribed), `xt` (crosstalk), `unk` (unintelligible), `spk` and `noise`
   (noise). Tags are never callsigns, names, waypoints or words to explain. `ne <language>`
   is "some words in <Language>" (a greeting or sign-off when it sits at the start or end
   next to courtesy words); the others mean that part was unclear. Never invent what the
   tagged part said.
11. **Plausible values.** A clearly impossible value (QNH outside about 900–1,100
   hectopascals or 28–31 inches, a VHF frequency outside 108–137 megahertz, a heading
   above three hundred sixty degrees, a flight level above six hundred) is not explained as
   given: say the value is unclear or, when one reading is clearly most plausible, give it
   with "probably".

## Severity (for the judge)

An answer has a **severe error** if it has any of these:
- a wrong operational value or unit;
- an instruction, permission or clearance reversed or changed in kind;
- a lost negation, condition or restriction;
- an omitted action that changes what the aircraft would do;
- a wrong or invented callsign, recipient or airline;
- a role or mode mistake that changes meaning (answering a transmission, the wrong
  speaker when the text fixes it);
- a "probably" reading that the visible words do not license.

Other rubric failures (an omitted minor item, an unsupported but harmless gloss,
overlength) are **minor**.

## Reference schema additions (v1.3)

On top of the v2 reference fields (speakers, required, optional, alternative readings,
acceptable uncertainty), each v1.3 reference records:
- `ambiguity`: `clear`, `plausible_reading` or `unresolved`, with the evidence;
- `licensed_readings`: readings an answer may give with "probably";
- `prohibited_completions`: tempting completions that count as severe errors.

For `unresolved`, a concise uncertainty statement is always correct. References are
written candidate-blind and verified by a separate call.

## Stress cases (v1.3 gold)

| # | Input | Gold explanation |
|---|---|---|
| 1 | cleared to land runway two five november two five hotel juliett | The speaker either clears Aircraft November two five Hotel Juliett to land on runway twenty-five or reads back that clearance. |
| 2 | we are currently one eight five ten knots | The speaker reports the wind, probably from one hundred eighty-five degrees at ten knots. |
| 3 | nav checker five one one go around climbing five seven thousand feet | The pilot of Nav Checker five one one reports going around and climbing, but the altitude is unclear. |
| 4 | quality four zero nine one cleared to liege | The controller clears ASL Airlines Belgium flight four zero nine one to Liege. |
| 5 | are you aware that you're now inside the ctr of bern | The controller asks whether the pilot is aware the aircraft is now inside Bern's control zone. |
| 6 | hotel bravo sierra golf kilo do you have the number one in sight traffic helicopter short final | The controller asks Aircraft Hotel Bravo Sierra Golf Kilo whether it sees the first aircraft in sequence, a helicopter on short final. |
| 7 | tango three five five cleared for long landing runway two five two six zero degrees four knots vacate via golf | The controller clears Tango three five five for a long landing on runway twenty-five, wind from two hundred sixty degrees at four knots, exiting via taxiway Golf. |

Stress-case inputs come from training-side devpool components. The v1.3 split forces
their components out of the reserved promotion cohort, so they may serve as generator
demonstrations. No devdiag, selection, reserve or test transmission is ever used as a
demonstration.
