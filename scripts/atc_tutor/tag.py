"""Tag dev-split ATC transcripts with a fixed clause taxonomy (Claude Opus 5.5).

Usage: uv run python scripts/atc_tutor/tag.py [--limit N] [--concurrency 8]
Resumable: rows already in tags.jsonl are skipped.
"""

import argparse
import asyncio
import json
from collections import Counter

from opus import OUT, call_tool, client, items, ledger_summary

CLAUSE_TYPES = [
    "callsign", "station", "greeting", "farewell", "acknowledge", "wilco",
    "readback", "standby", "say_again", "correction", "affirm_negative",
    "climb", "descend", "maintain_level", "heading", "turn", "direct_to",
    "route_or_procedure", "hold_or_orbit", "speed", "squawk", "ident",
    "contact_frequency", "monitor_frequency", "frequency_change_approved",
    "pressure_setting", "atis_information", "wind", "visibility_or_weather",
    "runway", "landing_clearance", "takeoff_clearance", "line_up_and_wait",
    "touch_and_go_or_option", "low_approach", "go_around", "holding_point",
    "hold_short", "taxi", "pushback_or_startup", "vacate", "backtrack",
    "circuit_position", "sequence_number", "position_report", "report_instruction",
    "request", "approval_or_denial", "traffic_information", "approach_clearance",
    "expect", "departure_clearance", "radar_identification", "time", "emergency",
    "reason_or_purpose",
    "other",
]
SLOT_TYPES = [
    "telephony", "flight_number", "registration", "station_name", "altitude_feet",
    "flight_level", "heading_degrees", "turn_direction", "runway_designator",
    "frequency", "squawk_code", "pressure_value", "pressure_unit", "atis_letter",
    "wind_direction", "wind_speed", "wind_gust", "speed_knots", "waypoint_or_place",
    "taxiway", "time_value", "count_or_number", "aircraft_type", "free_text",
]
TOOL = {
    "name": "record_tags",
    "description": "Record the analysis for every transcript in the batch.",
    "input_schema": {
        "type": "object",
        "required": ["items"],
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "required": ["id", "speaker", "speaker_confidence", "speech_acts",
                         "dialect", "clauses", "ambiguity"],
            "properties": {
                "id": {"type": "string"},
                "speaker": {"enum": ["controller", "pilot", "unknown"]},
                "speaker_confidence": {"enum": ["high", "medium", "low"]},
                "speech_acts": {"type": "array", "items": {"enum": [
                    "instruction", "clearance", "request", "readback", "report",
                    "information", "acknowledgement", "greeting", "question", "other"]}},
                "dialect": {"enum": ["icao_europe", "icao_australia", "faa", "unknown"]},
                "language": {"type": "string", "description": "Non-English words, if any, and their language."},
                "clauses": {"type": "array", "items": {
                    "type": "object",
                    "required": ["type", "evidence"],
                    "properties": {
                        "type": {"enum": CLAUSE_TYPES},
                        "evidence": {"type": "string", "description": "Exact span from the transcript."},
                        "slots": {"type": "array", "items": {
                            "type": "object", "required": ["type", "spoken", "value"],
                            "properties": {
                                "type": {"enum": SLOT_TYPES},
                                "spoken": {"type": "string"},
                                "value": {"type": "string", "description": "Normalized value, e.g. 270, FL120, 118.300, 1008 hPa, 25R."},
                            }}},
                        "modifiers": {"type": "array", "items": {"enum": [
                            "negation", "condition", "restriction", "correction",
                            "after_sequence", "expectation", "when_ready", "immediate",
                            "approved", "denied"]}},
                        "condition_text": {"type": "string"},
                    }}},
                "ambiguity": {"type": "array", "items": {"enum": [
                    "truncated", "garbled", "noise_marker", "disfluency",
                    "missing_callsign", "implausible_value", "unclear_speaker",
                    "unknown_term", "none"]}},
                "unknown": {"type": "array", "items": {"type": "object",
                    "properties": {"evidence": {"type": "string"}, "note": {"type": "string"}}}},
            }}}},
    },
}
SYSTEM = f"""You annotate air traffic control radio transcripts for a training-data
pipeline. Transcripts are lowercase ASR-style gold text from Swiss, Czech, Slovak and
Australian airports. 'spk' is a noise/unintelligible-speaker marker; 'uh' is a
disfluency. Numbers are spoken as words.

For each transcript, in order of appearance, list every clause using exactly one of
these clause types: {", ".join(CLAUSE_TYPES)}. Use 'callsign' for the aircraft
identification and 'station' for a ground station name (e.g. 'sydney tower').
Attach typed slots with the exact spoken words and a normalized value. Mark
modifiers precisely: 'expectation' for expect, 'condition'/'after_sequence' with
condition_text for conditional or sequenced instructions, 'correction' when a value
is corrected (normalize to the corrected value), 'negation' for negatives.

Speaker: controller if it issues instructions/clearances/information, names itself
as a station, or addresses an aircraft first; pilot if it reads back (instruction
content followed by its callsign, or -ing forms like 'climbing', 'holding short'),
requests, or reports. Use 'unknown' when it genuinely cannot be determined, and
lower the confidence rather than guessing.

Never invent content. If a span fits no clause type, use 'other' and add an
'unknown' entry with the evidence and a note. Record ambiguity flags honestly; use
['none'] only when the transcript is fully clear."""


def dev_rows():
    rows = [json.loads(line) for line in (OUT / "split.jsonl").open()]
    return [row for row in rows if row["split"] == "dev"]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    path = OUT / "tags.jsonl"
    done = {json.loads(line)["id"] for line in path.open()} if path.exists() else set()
    todo = [row for row in dev_rows() if row["id"] not in done][: args.limit]
    batches = [todo[i : i + args.batch] for i in range(0, len(todo), args.batch)]
    api = client()
    gate = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    failed = 0

    async def run(batch):
        nonlocal failed
        user = "Annotate these transcripts:\n" + "\n".join(
            json.dumps({"id": row["id"], "airport": row["airport"], "text": row["text"]})
            for row in batch
        )
        async with gate:
            try:
                result = await call_tool(api, stage="tag", system=SYSTEM, user=user, tool=TOOL)
            except Exception as error:  # noqa: BLE001 - resumable; report and continue
                failed += len(batch)
                print("batch failed:", type(error).__name__, error)
                return
        wanted = {row["id"]: row for row in batch}
        async with lock:
            with path.open("a") as out:
                for item in items(result, "items"):
                    if item.get("id") in wanted:
                        item["text"] = wanted.pop(item["id"])["text"]
                        out.write(json.dumps(item) + "\n")
        failed += len(wanted)

    await asyncio.gather(*(run(batch) for batch in batches))
    tags = [json.loads(line) for line in path.open()]
    print(f"tagged {len(tags)} rows; {failed} missing this run")
    print("clause types:", Counter(c["type"] for t in tags for c in t["clauses"]).most_common(15))
    print("unknown entries:", sum(len(t.get("unknown", [])) for t in tags))
    print("ledger:", json.dumps(ledger_summary()))


if __name__ == "__main__":
    asyncio.run(main())
