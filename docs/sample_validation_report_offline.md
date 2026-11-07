<!-- Example output of: python -m evaluation.harness --input data/validation_tickets.json --output <dir> --no-llm
     Generated offline in a sandbox: TF-IDF retrieval, extractive replies. NOT your final numbers - regenerate on your machine
     (ideally with MiniLM and your Gemini key) and read the caveats. -->
# Evaluation report

- Input: `data/validation_tickets.json` (80 tickets, sha256 94c5b1adc1203f06)
- Run at 2026-10-31T17:40:29+00:00 UTC, commit `n/a`, wall-clock 9.1 s
- Components: retrieval `tfidf+memory`, answers `none (extractive answers)`, threshold 0.65

## Results against targets

| Measure | Baseline | Target | Achieved | Status | Notes |
|---|---|---|---|---|---|
| First contact resolution (proxy) | 46.2% | 60%+ | 76.2% | met | tickets answered with no human |
| Escalation rate | 53.8% | <=30% | 23.8% | met | includes guardrail-blocked tickets |
| Median system reply time | 8-12 hrs | <5 min | 0.01 s | met | machine time only; escalated tickets still wait for a person |
| Satisfaction | 3.2/5 | 4.0+ | not measured | n/a | needs human rating of review_sample.csv |
| Intent precision (macro) | - | 85%+ | 100.0% | met | see per-class CSV |
| Hallucination rate | - | <=5% | not measured | n/a | needs two human assessors, >=50 replies |
| Citation accuracy (proxy) | - | 95%+ | 100.0% | met | citations resolve to retrieved corpus passages; support needs humans |
| Latency p95 | - | <3 s | 0.02 s | met | excludes free-tier throttle sleeps |
| Private data in sent replies | - | 0 | 0 | met |  |
| Decision log reconciles | - | 100% | yes | met |  |
| Cross-group variation (max spread) | - | <5 pts | 30.6 pts | NOT met | see fairness section; small groups are noisy |

## Volume

Processed 80: answered automatically 61, escalated 19, blocked by guardrails then escalated 0, tickets with internal errors 0.

Escalation reasons: {'must_escalate_intent': 12, 'low_confidence': 5, 'safety_keyword': 2}

## Routing against labels

- Route accuracy 81.2%; auto-answer precision 77.0%; auto-answer recall 97.9%
- Must-not-auto-respond violations: **0** (must be 0)
- Auto-answered but labelled escalate: 14; escalated but labelled auto: 1

## Classification

- Intent accuracy 100.0% on 80 labelled tickets; macro precision 100.0%, macro recall 100.0%; urgency accuracy 41.2%
- Accuracy on wording also present in the development set: 100.0% (n=62); on unseen wording: 100.0% (n=18)

## Retrieval

- Expected article retrieved (any of top-k): 96.2%; as top result: 90.6% (n=53)
- Empty result when no article is expected: 11.1% (n=27)

## Calibration (stated confidence vs observed accuracy)

| Confidence | Band | n | Stated | Observed | Gap (pts) |
|---|---|---|---|---|---|
| intent | 0.2-0.4 | 1 | 0.244 | 1.0 | -75.6 |
| intent | 0.8-1.0 | 79 | 0.999 | 1.0 | -0.1 |
| routing | 0.6-0.7 | 5 | 0.644 | 0.2 | 44.4 |
| routing | 0.7-0.8 | 40 | 0.741 | 0.725 | 1.6 |
| routing | 0.8-0.9 | 21 | 0.846 | 0.857 | -1.1 |

## Governance

- Decision log: 362 decisions for 80 tickets; reconciles: **True**
- Guardrail activations: none; private-data blocks 0; private data in sent replies 0

### Fairness (spread = best minus worst group, groups with n>=10)

**customer_tier**: business (n=30, auto 90.0%, route acc 73.3%, retrieval hit 95.5%), enterprise (n=8, auto 87.5%, route acc 87.5%, retrieval hit 100.0%), standard (n=42, auto 64.3%, route acc 85.7%, retrieval hit 96.0%)
  spread (pts): {'auto_response_rate': 25.7, 'route_accuracy': 12.4, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 0.5}

**customer_region**: asia_pacific (n=21, auto 76.2%, route acc 85.7%, retrieval hit 100.0%), europe (n=25, auto 76.0%, route acc 88.0%, retrieval hit 94.7%), latin_america (n=7, auto 57.1%, route acc 85.7%, retrieval hit 100.0%), north_america (n=27, auto 81.5%, route acc 70.4%, retrieval hit 92.9%)
  spread (pts): {'auto_response_rate': 5.5, 'route_accuracy': 17.6, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 7.1}

**language_fluency**: fluent (n=61, auto 77.0%, route acc 88.5%, retrieval hit 95.5%), non_fluent (n=19, auto 73.7%, route acc 57.9%, retrieval hit 100.0%)
  spread (pts): {'auto_response_rate': 3.4, 'route_accuracy': 30.6, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 4.5}

**channel**: chat (n=22, auto 72.7%, route acc 86.4%, retrieval hit 92.3%), docs_comment (n=16, auto 62.5%, route acc 75.0%, retrieval hit 100.0%), email (n=31, auto 87.1%, route acc 80.6%, retrieval hit 100.0%), forum (n=11, auto 72.7%, route acc 81.8%, retrieval hit 87.5%)
  spread (pts): {'auto_response_rate': 24.6, 'route_accuracy': 11.4, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 12.5}

**ticket_length**: long (>30) (n=11, auto 90.9%, route acc 72.7%, retrieval hit 100.0%), medium (21-30) (n=40, auto 80.0%, route acc 80.0%, retrieval hit 92.9%), short (<=20 words) (n=29, auto 65.5%, route acc 86.2%, retrieval hit 100.0%)
  spread (pts): {'auto_response_rate': 25.4, 'route_accuracy': 13.5, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 7.1}

## Caveats

- The figures above should be treated with caution: FCR is a proxy (answered without a human is not the same as solved), satisfaction and hallucination need human assessment, and many tickets repeat development-set wording.
- In the labelled data, identical ticket text appears with both `auto_respond` and `escalate` labels, so no text-based system can reach 100% route accuracy.
