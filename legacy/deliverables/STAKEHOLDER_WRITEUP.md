# Mini Red-Team Generator: Stakeholder Writeup

> **Audience:** Product managers, executives, and non-ML stakeholders who need to
> understand what this system does, why it matters, and how far it can be trusted.
>
> **Content note:** This project deliberately generates harmful and offensive text
> for safety-testing purposes. All examples in shared artifacts are redacted by
> default.

---

## 1. Introduction -- What Problem This Solves

Every AI product that generates or processes text carries a risk: it can be
tricked, coerced, or simply stumble into producing harmful content -- hate
speech, threats, harassment, or worse. When those failures surface in
production, the cost is measured in user harm, brand damage, and regulatory
exposure.

The standard mitigation is **red-teaming**: deliberately trying to make your
system fail so you can fix the failure before real users encounter it. But
manual red-teaming is slow. A small team of human testers can explore only a
fraction of the risk surface, they bring personal biases about what "an attack"
looks like, and their coverage is hard to measure or reproduce.

This project builds an **automated failure-discovery engine** -- a system that
generates diverse, realistic adversarial inputs on demand, scores them for
severity, and delivers a structured report a trust-and-safety team can act on.

Concretely, the system:

- **Is controllable.** You specify the *kind* of failure you want to probe
  (e.g., hate speech, threats, insults, obscenity) and the system produces
  targeted test cases, not random noise.
- **Is measurable.** Every run produces quantitative metrics: how harmful the
  outputs are, how diverse they are, and whether the system outperforms an
  un-tuned baseline.
- **Is reproducible.** A single command (`bash scripts/run_end_to_end.sh`)
  executes the entire pipeline -- from raw data to final reports -- with every
  decision logged in configuration files.
- **Is contained.** Generated text is toxic by design. All shared artifacts are
  redacted by default; a reviewer must explicitly opt in to view raw samples in
  a controlled local environment.

The deliverable is not a chatbot or a user-facing product. It is a diagnostic
tool: a way to systematically stress-test safety systems before they reach
users.

---

## 2. Background -- Red-Teaming, Trust and Safety, and Failure Discovery

### What is red-teaming?

Red-teaming is a security practice borrowed from military and intelligence
work. A dedicated "red team" adopts the adversary's perspective and tries to
break a system, so the defenders can find and fix weaknesses before a real
attacker exploits them. In the context of AI, this means deliberately
attempting to make a language model produce content that violates safety
policies -- toxic language, hate speech, threats, or other harmful material.

### What is trust and safety?

Trust and safety (T&S) is the discipline of keeping AI-powered products safe
for their users. It encompasses content moderation (filtering harmful outputs),
policy enforcement (defining what the system should and should not do), and
proactive risk discovery (finding failures before users do). Red-teaming is the
proactive arm of T&S.

### Why build a "failure discovery engine"?

Traditional red-teaming relies on human testers inventing attacks. A failure
discovery engine flips that model: instead of waiting for humans to think of
novel attacks, the system *generates* them. The advantages are scale (hundreds
of test cases per run, not dozens), breadth (systematic coverage of a risk
taxonomy), and repeatability (every run is logged, seeded, and reproducible).

### Why not just ask a commercial chatbot?

The most obvious alternative -- prompting a commercial large language model like
GPT-4 or Claude to produce adversarial content -- does not work reliably.
Those models are specifically trained to *refuse* requests for harmful content.
That refusal is the right behavior for a user-facing product, but it is the
opposite of what a red-team generator needs. A purpose-built system, trained
on curated data and optimized for adversarial output, is the practical path.

---

## 3. Key Concepts Explained

This section defines six building blocks that the rest of the document depends
on. Each is a standard technique in the field; together they form the backbone
of the system.

### Lightweight fine-tuning (LoRA)

Training a large AI model from scratch takes weeks and millions of dollars.
**LoRA** (Low-Rank Adaptation) sidesteps that cost: it freezes the original
model and attaches a small trainable "patch" -- less than 1% of the model's
total parameters -- that steers behavior in the desired direction. Think of it
as adding a specialty lens to a camera rather than building a new camera.

In this system, the base model is a 601-million-parameter language model
(Qwen3-0.6B). The LoRA patch adds roughly 5 million trainable parameters
(0.84% of the total), which is enough to shift the model's output distribution
toward adversarial content while keeping training fast and inexpensive.

### Dual scorers

A single automated judge can have blind spots. This system uses **two
independent classifiers** -- each built by a different research group, trained
on different data -- to evaluate every generated sample:

1. **Toxic-BERT** (multi-label toxicity classifier) contributes 60% of the
   final score.
2. **Cardiff Offensive** (binary offensive-language classifier) contributes
   40%.

The blended score reduces the chance that one scorer's biases dominate the
results.

### Rejection sampling

Not every generated sample is useful. Rejection sampling is a quality filter
applied after scoring. It removes:

- **Exact duplicates** (identical text).
- **Near-duplicates** (text that shares too many three-word sequences with an
  already-accepted sample).
- **Too-short samples** (fewer than a minimum number of tokens).
- **Low-scoring samples** (below the acceptance threshold).

What remains is a curated set of diverse, high-signal adversarial examples.

### KPI gate

The KPI gate is a checklist of quantitative thresholds the system must meet
before its results are considered trustworthy. It answers five questions:

| Check | Question |
|-------|----------|
| Harmful hit rate | Is a sufficient fraction of generated samples actually harmful? |
| Hit-rate delta vs. baseline | Is the adapted model meaningfully better than the un-tuned base model? |
| Category coverage | Does the accepted set span enough of the 8 risk categories? |
| Lexical diversity (Distinct-2) | Are the samples linguistically varied, not just rephrased copies? |
| Manual audit pass rate | Do human reviewers confirm the automated scores? |

If any check fails, the gate flags the run. If no human labels have been
entered yet, the status is `pending_manual_audit` -- a deliberate safeguard
that prevents automated-only sign-off.

### Redaction and containment

The system generates toxic text by design. To prevent accidental exposure:

- All shared artifacts replace generated text with `[REDACTED]`.
- A reviewer must explicitly change a configuration flag (`redact: false`) and
  re-run evaluation to see raw text, and only in a local environment.
- Generated samples are never included in submission packages or shared
  reports.

### Baseline comparison

To prove that fine-tuning actually changed the model's behavior, the system
runs the *exact same pipeline* -- same prompts, same decoding parameters, same
scoring logic -- on the original, un-tuned model. Any improvement in the
fine-tuned model's metrics is attributable to the training, not to the prompts
or the scoring setup. This is the control group.

---

## 4. How This Differs from Prompting a Generic LLM

The most natural question a stakeholder might ask is: "Why build all this when
you could just prompt an existing model?" There are five concrete differences.

**1. Generic LLMs refuse.**
Commercial language models are aligned to decline requests for harmful content.
That is exactly the opposite of what a red-team generator needs. This system is
trained on curated adversarial data to *produce* harmful content reliably and on
demand.

**2. Structured controllability.**
Generic prompting gives you whatever the model happens to produce. This system
uses structured control tokens -- `[RISK=hate_speech] [SEV=high]
[STYLE=sarcastic]` -- so you can sweep across risk categories, severity levels,
and stylistic variations systematically. That turns ad-hoc testing into a
repeatable test matrix.

**3. Reproducibility.**
Every run is seeded, config-driven, and produces a manifest of exactly which
data was used, which model checkpoint was loaded, and which scoring parameters
were applied. A conversation with a commercial chatbot is not reproducible in
the same way.

**4. Built-in quantitative evaluation.**
The system does not just produce text. It scores, filters, and reports on every
sample it generates. You receive metrics and a comparison report, not a
collection of anecdotes.

**5. Self-hosted, no data leaves your infrastructure.**
For trust-and-safety work involving sensitive failure modes, data residency
matters. This system runs entirely on your own hardware; no prompts or outputs
are sent to a third-party API.

---

## 5. How the System Works -- A Module-by-Module Walkthrough

The pipeline has six stages, each with a defined input and output. You can
think of it as an assembly line: raw materials in one end, a quality-controlled
report out the other.

### Stage 1: Data Assembly

**What it does:** Collects examples of real-world abusive text from seven
publicly available research datasets, normalizes them into a common format, and
splits them into training, validation, and holdout sets.

**Why it matters:** The quality and diversity of training data determines the
quality and diversity of generated attacks. Using seven sources (instead of one)
ensures the system learns about different types of abuse -- hate speech,
threats, insults, obscenity -- rather than memorizing a single writing style.

**By the numbers:** The current run assembles 27,909 normalized rows from all
seven sources, of which 11,306 are labeled abusive. The abusive rows span eight
risk categories, though some categories are much better represented than others
(hate speech: 6,401 rows; sexual content: 6 rows).

### Stage 2: Model Adaptation

**What it does:** Takes a pre-trained 0.6-billion-parameter language model and
attaches a lightweight LoRA adapter. Trains the adapter on the abusive subset
so the model learns to produce adversarial text when given a structured prompt.

**Why it matters:** Without this step, the base model has no particular
tendency to produce harmful content. The adapter is the "specialty lens" that
shifts the model's behavior toward the adversarial regime.

**Key design choice:** During training, only the model's *response* tokens are
used as learning targets; the instruction tokens are masked out. This prevents
the model from learning to echo prompts back, which is a common failure mode
in fine-tuned generators.

### Stage 3: Controlled Generation

**What it does:** Constructs combinations of risk category (8 types), severity
level (low / mid / high), and stylistic tone (direct / sarcastic / taunt). For
each combination, generates multiple candidate samples and applies quality
filters (minimum length, English-language ratio, chain-of-thought suppression).

**Why it matters:** Systematic combination of conditions ensures the system
probes the full risk taxonomy, not just the categories that happen to be easy
for the model. The quality filters remove degenerate outputs (empty strings,
non-English text, reasoning artifacts).

**By the numbers:** The default configuration generates 240 candidates (60
condition combinations, 4 candidates each).

### Stage 4: Dual Scoring and Filtering

**What it does:** Runs every candidate through two independent classifiers,
computes a blended score, and applies rejection sampling to remove duplicates,
near-duplicates, and low-quality samples.

**Why it matters:** This is where raw model output becomes a curated test set.
Without scoring and filtering, the output would include a large fraction of
benign or repetitive text that wastes a reviewer's time.

**By the numbers:** Of 240 candidates in the current run, 25 passed rejection
sampling and entered the accepted set.

### Stage 5: Baseline Run

**What it does:** Repeats Stages 3 and 4 using the original, un-tuned model
(no LoRA adapter). Same prompts, same decoding settings, same scoring.

**Why it matters:** This is the control group. Without it, there is no way to
know whether improvements come from the training or from the prompts.

**By the numbers:** The baseline accepted 7 of 240 candidates at the same
threshold.

### Stage 6: Evaluation and Reporting

**What it does:** Compares the adapted model against the baseline on a shared
set of metrics, produces summary reports, builds a KPI gate verdict, and
generates a human audit sheet (redacted by default).

**Why it matters:** This is the stage that translates raw numbers into an
actionable assessment: "Is the generator good enough? Is it better than the
baseline? What should a human reviewer check?"

The following diagram shows how data flows through the pipeline:

```
  7 Public         Data            Model
  Datasets  --->  Assembly  --->  Adaptation
                    |                 |
                    |           LoRA adapter
                    |                 |
                    v                 v
                  Baseline       Controlled
                  Generation     Generation
                    |                 |
                    v                 v
                  Baseline        Dual Scoring
                  Scoring         + Filtering
                    |                 |
                    +--------+--------+
                             |
                             v
                       Evaluation
                       + Reporting
                             |
                             v
                     Reports, Audit Sheet,
                         KPI Gate
```

---

## 6. What We Measure and Why It Matters

Each metric answers a concrete question a decision-maker would ask.

### Harmful hit rate

**Question:** "What fraction of generated samples are actually harmful?"

This is the core efficiency metric. A higher rate means fewer wasted
generations and more useful test cases per run. It is computed as the share of
all candidates whose blended score exceeds the acceptance threshold.

### Mean risk score

**Question:** "On average, how potent are the generated samples?"

A higher mean score indicates the generator is consistently producing content
that both classifiers flag as harmful, not just occasionally getting lucky with
a single provocative word.

### Category coverage

**Question:** "How many of the 8 risk categories does the generator
successfully cover?"

A red-team tool is only useful if it probes your system *broadly*. If it
only produces insults but never generates threats or identity attacks, it will
miss entire classes of failure. Coverage is measured as the fraction of the 8
defined risk categories represented in the accepted set.

### Lexical diversity (Distinct-2)

**Question:** "Are the samples diverse, or just the same insult rephrased?"

Measured as the ratio of unique two-word sequences (bigrams) to total bigrams
across all accepted samples. A high ratio means the generator produces varied
language. For red-teaming, diversity matters because a defense system that
catches one phrasing of an attack may miss a different phrasing of the same
idea.

### Prompt condition success

**Question:** "When we ask for a specific risk type, does the generator
deliver it?"

This measures controllability. If you request a "threat" and the system
produces an "insult" instead, it is still generating harmful content, but it is
not following instructions. A higher success rate means the system can be
used as a precision tool, not just a blunt instrument.

### KPI gate

**Question:** "Does the system pass its own quality bar?"

The KPI gate combines all the above metrics (plus a manual audit pass rate)
into a single pass/fail verdict. If any check fails, the gate flags the run so
a human reviewer knows where to look. If no human labels have been entered,
the status is `pending_manual_audit` to prevent purely automated sign-off.

---

## 7. Results -- What Improved and Why It Matters

The headline result is straightforward: **the adapted model outperforms the
un-tuned baseline on every key metric.** This is the core evidence that
fine-tuning added value beyond what prompting alone achieves.

### Summary table

| Metric | Adapted model | Baseline | Change |
|--------|--------------|----------|--------|
| Mean risk score | 0.126 | 0.089 | +42% relative |
| Harmful hit rate | 10.4% | 2.9% | 3.6x |
| Category coverage | 75% (6 of 8) | 37.5% (3 of 8) | 2x |
| Lexical diversity (Distinct-2) | 0.85 | 0.68 | +25% relative |
| Prompt condition success | 16% | 0% | New capability |

### What the numbers mean in practice

**The adapted model generates harmful content more reliably.** At the same
acceptance threshold, 10.4% of the adapted model's output qualifies as
harmful, compared to 2.9% for the baseline. For a trust-and-safety team
running this tool, that means roughly 3.6 times fewer wasted generations per
useful test case.

**The adapted model covers more of the risk taxonomy.** The accepted set
includes samples from 6 of the 8 defined risk categories, compared to 3 for
the baseline. In practice, this means the adapted model can probe a wider
range of potential failures in a downstream system -- hate speech, threats,
identity attacks, obscenity, harassment, and more -- rather than concentrating
on just a few types.

**The adapted model produces more varied language.** A Distinct-2 score of 0.85
(vs. 0.68 for the baseline) means the accepted samples use a wider vocabulary
and more varied phrasing. Repetitive attacks are easy for a defense system to
catch; diverse attacks are a harder and more realistic stress test.

**The adapted model follows risk-type instructions, where the baseline does
not.** When asked to generate a specific category of harmful content (e.g.,
"threat"), the adapted model delivers the correct category 16% of the time.
The baseline never does (0%). The absolute rate is low -- a limitation
discussed in Section 8 -- but this is a *qualitative* capability the baseline
lacks entirely.

### Important context

The absolute harmful hit rate (10.4%) is modest. This is a proof of concept
produced under a one-day time constraint with a lightweight training
configuration. The full training configuration (12,000 samples, 1 full epoch)
is expected to improve these numbers; the current artifacts were generated
with a quick configuration (160 training samples, 0.1 epochs) to validate the
pipeline end to end.

---

## 8. Limitations and Risks

Being direct about what this system does *not* solve is as important as
showing what it does. The following limitations are known, measured, and
documented.

### Scorer validity

The two automated scorers (Toxic-BERT and Cardiff Offensive) are proxies, not
ground truth. They have known biases: they may over-flag text that mentions
identity groups (false positives) or miss subtle sarcasm and coded language
(false negatives). Every conclusion drawn from the metrics depends on the
assumption that these scorers are reasonably calibrated.

**Mitigation:** The manual audit sheet provides a human checkpoint. A sample
of 20 accepted candidates is presented for human review; only after a reviewer
labels them does the KPI gate reach a final pass/fail verdict.

### Threshold sensitivity

Small changes to the acceptance threshold dramatically change the harmful hit
rate. In the current run, the configured threshold (0.6) was automatically
lowered to approximately 0.34 to avoid producing an empty accepted set. The
KPI gate's aspirational thresholds (60% hit rate, 20% delta over baseline) are
not met by this proof of concept.

**What this means:** The metrics are real, but the bar they are measured
against is set for a production-grade system, not a one-day PoC. Interpreting
the results requires reading both the achieved values and the thresholds
together.

### Stochasticity

Text generation is inherently random (controlled by a "temperature" parameter).
Re-running the pipeline with the same configuration will produce different
candidate text and potentially different metric values. The pipeline uses
fixed random seeds for reproducibility, but exact reproduction requires
matching hardware and library versions as well.

### Condition fidelity

The model follows the requested risk category only 16% of the time. This means
the "controllable generation" capability is nascent, not production-ready. The
system can be directed toward adversarial content in general, but it does not
yet reliably produce the *specific type* of adversarial content you ask for.

**Root cause:** The quick training configuration used only 160 samples and 0.1
epochs. Longer training with more data and a richer conditioning signal (or a
preference-optimization objective) would improve this.

### Red-team depth

This generator produces single-turn, short adversarial strings (averaging 37
tokens per accepted sample). Real-world adversarial attacks are often
multi-turn, context-dependent, or use encoding tricks (e.g., Unicode
substitutions, base64 obfuscation). This PoC covers *breadth* across 8 risk
categories but does not yet address *depth* of adversarial sophistication.

### Training data scope

The training corpus is English-only, sourced from social media and web
comments. It does not cover domain-specific risks (medical misinformation,
financial fraud, legal advice) or non-English languages. Deploying this
system for a multilingual product or a specialized domain would require
additional data sources and per-language scorers.

### Class imbalance

The eight risk categories are unevenly represented in the training data. Hate
speech has 6,401 rows, while sexual content has only 6 rows and identity
attacks have 30. Categories with very few training examples are harder for the
model to learn and may be underrepresented in the generated output.

---

## 9. Future Directions -- What Would Make This Better

The improvements below are ordered by priority: first, changes that make the
current system more trustworthy for decision-making; second, changes that
expand its ability to discover new types of failures.

### Making the system more decision-grade

**Per-category threshold calibration.**
The current system uses a single global acceptance threshold for all risk
categories. Categories with sparse training data (e.g., sexual content,
identity attacks) are penalized by this one-size-fits-all approach. Setting
per-category thresholds -- or applying targeted data augmentation to
underrepresented categories -- would produce a more balanced and trustworthy
test set.

**Closing the human feedback loop.**
The manual audit sheet exists, but the feedback is one-shot: a reviewer labels
20 samples, the KPI gate updates, and the loop ends. The next step is a
lightweight labeling workflow that feeds reviewer judgments back into scorer
calibration or training data selection, turning human review into a continuous
improvement signal rather than a one-time checkpoint.

**Held-out evaluation.**
The data pipeline produces a held-out test split (approximately 1,400 rows)
but no script currently evaluates on it. Adding a standard perplexity or
classifier-probe evaluation on the held-out set would provide an ML quality
signal that is independent of the generation metrics, giving reviewers a second
axis of confidence.

**Full training run.**
The shipped artifacts were produced with a quick training configuration (160
samples, 0.1 epochs). Running the full configuration (up to 12,000 samples,
1 full epoch) is expected to improve harmful hit rate, condition fidelity, and
overall metric quality. This is the single highest-leverage change for
improving the current numbers.

### Expanding failure discovery

**Multi-turn and contextual attacks.**
Extend generation from single-turn strings to multi-turn conversations where
the adversarial payload is distributed across several messages. Many real-world
jailbreaks exploit conversational context that single-turn testing cannot
capture.

**Preference optimization (DPO / RLHF).**
Replace or supplement supervised fine-tuning with direct preference
optimization, using scorer feedback as a reward signal. This would directly
optimize for the target risk category, improving condition fidelity (currently
16%) without requiring more labeled training data.

**Adversarial transfer testing.**
Use the generated samples not just as standalone outputs but as *inputs to a
target model* (e.g., a chatbot under development). Measure whether the
generated adversarial prompts actually elicit policy violations from the target
system. This closes the loop from "generator" to "end-to-end red-team test
harness," which is the ultimate use case.

---

## Appendix: Supporting Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| End-to-end entrypoint | `scripts/run_end_to_end.sh` | Single command to run the full pipeline |
| Data manifest | `outputs/data/data_manifest.json` | Which datasets were loaded, row counts, success/failure status |
| Label distribution | `outputs/data/label_distribution.json` | Row counts per risk category and source |
| Training metadata | `outputs/models/qwen3_lora_adapter_quick/train_metadata.json` | Model name, parameter counts, training config, library versions |
| Evaluation report (adapted) | `outputs/reports/eval_report.json` | All metrics + KPI gate for the fine-tuned model |
| Evaluation report (baseline) | `outputs/reports/baseline_report.json` | Same metrics for the un-tuned base model |
| Comparison report | `outputs/reports/comparison_report.json` | Metric deltas (adapted minus baseline) |
| Manual audit sheet | `outputs/reports/manual_audit_sheet.csv` | 20 sampled candidates for human review (redacted by default) |
| Configuration files | `configs/*.yaml` | All pipeline parameters (data, training, generation, evaluation) |
| Test suite | `tests/` | 10 test modules runnable via `pytest -q` |
| Test case documentation | `TEST_CASES.md` | 6 documented test scenarios with expected outcomes |
