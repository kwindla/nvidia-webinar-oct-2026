# Task Types: Route Data, Metrics and Evaluators by Task

Decide the task type in Stage 2, before any data is generated. The type decides where labels come from,
which metric is primary, whether an LLM judge is needed at all, and which suites apply. Mixed tasks
(e.g. an agent that extracts fields and then explains them) combine the rows below, each part with its
own metric.

| | Classification | Extraction / structured output | Open text (explanation, summary, tutoring) | Multi-turn tool agent |
|---|---|---|---|---|
| **Labels** | Existing approved human labels first; a teacher only proposes labels for unlabeled rows, under audit | Source spans and approved records first; a teacher fills gaps with source-grounded values | Teacher-written targets under audit (data.md), or trusted references if they exist | Native episodes (messages, tool calls, tool results) from logs or sandboxed rollouts |
| **Contract** | Label vocabulary, single vs multi-label, unknown/abstain policy, class costs | Schema, required vs optional fields, null/absent semantics, normalization rules | Priority-ordered content and form rules, severity classes | Tool contracts, allowed actions, side-effect policy, turn limits, recovery behavior |
| **Primary metric** | Per-class precision/recall, confusion matrix, macro/micro/weighted F1 as costs dictate; calibration if probabilities are used | Schema-valid rate as a **gate**, then field/span exact match or set F1, grounding to source; empty results scored by the absence rule | Judged correctness + severe rate | End-to-end task success, argument correctness, unauthorized or duplicate side effects, recovery, latency |
| **Judge** | Code | Code (parser + comparator); an LLM only for free-text fields | Calibrated LLM judge + human calibration | Code over trajectories and tool state; an LLM for free-text turns only |
| **Simple baseline** | Majority class; prompted stock model | Rule/regex extractor; prompted stock with constrained decoding | Prompted stock; frontier model | Prompted stock with the same tools |
| **Robustness** | Noisy inputs relevant to the source; class-prior shift | Degraded documents; missing fields | Upstream noise (ASR, OCR) | Tool errors, timeouts, untrusted tool output and prompt injection |
| **Mode/off-task suite** | Only if the product also chats | Inputs with nothing to extract | If the product mixes chat and task | Out-of-scope requests; requests for disallowed actions |
| **Leakage unit** | The source record or entity | The document | The source group (recording, session) | The whole episode |

Notes that apply to every type:

- **Freeze the product's decoding layer** (constrained decoding, JSON repair, parser) as part of the
  measured configuration; evaluate with exactly what ships.
- **Empty is not automatically wrong.** An empty list can be the correct extraction; a timeout or
  truncation is a failure. Define both in the contract.
- **Retention:** add general-capability or chat replay only if the product needs that behavior;
  measure stock vs tuned on identical items before assuming forgetting.
- **Data volume follows learning curves and slice deficits**: train on growing subsets (or read
  per-slice errors) and add data where errors concentrate, rather than aiming for a fixed row count.
- **Stage 2 stopping rule covers every declared goal**: quality, latency, cost per call, hardware
  footprint. A prompted model that meets quality but not latency may still justify a smaller tuned model.

## Agents: extra requirements

- Keep tool-call and tool-result messages in the trainer's native format, with call IDs and ordering;
  validate serialization and loss masks with the real chat template in the canary.
- Train only on the assistant's actions and replies; tool results are conditioning, not targets.
- Split by episode; never put turns of one episode in different splits.
- Evaluate both **fixed-history turns** (gold prior context) and **model-generated trajectories**
  (errors accumulate), in sandboxed or mocked tools. Dataset contents never authorize real actions.
- Labels may depend on the visible conversation and tool observations, not just the last user message.
