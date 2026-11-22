# Evaluation report

- Input: `data\validation_tickets.json` (80 tickets, sha256 94c5b1adc1203f06)
- Run at 2026-11-21T20:13:16+00:00 UTC, commit `66d1f20`, wall-clock 35.1 s
- Components: retrieval `all-MiniLM-L6-v2+chroma`, answers `none (extractive answers)`, threshold 0.65

## Results against targets

| Measure | Baseline | Target | Achieved | Status | Notes |
|---|---|---|---|---|---|
| First contact resolution (proxy) | 46.2% | 60%+ | 76.2% | met | tickets answered with no human |
| Escalation rate | 53.8% | <=30% | 23.8% | met | includes guardrail-blocked tickets |
| Median system reply time | 8-12 hrs | <5 min | 0.04 s | met | machine time only; escalated tickets still wait for a person |
| Satisfaction | 3.2/5 | 4.0+ | not measured | n/a | needs human rating of review_sample.csv |
| Intent precision (macro) | - | 85%+ | 100.0% | met | see per-class CSV |
| Hallucination rate | - | <=5% | not measured | n/a | needs two human assessors, >=50 replies |
| Citation accuracy (proxy) | - | 95%+ | 100.0% | met | citations resolve to retrieved corpus passages; support needs humans |
| Latency p95 | - | <3 s | 0.04 s | met | excludes free-tier throttle sleeps |
| Private data in sent replies | - | 0 | 0 | met |  |
| Decision log reconciles | - | 100% | yes | met |  |
| Availability (tickets handled) | - | 99.5%+ | 100.0% | met | tickets ending in a reply or a logged escalation; run with --no-llm for the provider-disconnected figure |
| Cross-group variation (max spread) | - | <5 pts | 30.6 pts | NOT met | point estimate; 95% bootstrap intervals in the fairness section show which gaps exceed noise (interval entirely above 5 pts: 5 of 20 measures) |

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

- Expected article retrieved (any of top-k): 98.1%; as top result: 98.1% (n=53)
- Empty result when no article is expected: 0.0% (n=27)

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

### Fairness (spread = best minus worst group, groups with n>=10; [lo-hi] and the spread intervals are 95% bootstrap intervals, in % points)

**customer_tier**: business (n=30, auto 90.0% [77-100], route acc 73.3% [57-87], retrieval hit 100.0%), enterprise (n=8, too small to judge, auto 87.5% [62-100], route acc 87.5% [62-100], retrieval hit 100.0%), standard (n=42, auto 64.3% [50-79], route acc 85.7% [74-95], retrieval hit 96.0%)
  spread (pts): {'auto_response_rate': 25.7, 'route_accuracy': 12.4, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 4.0}
  - auto_response_rate: 25.7 pts, 95% bootstrap interval 6.2-43.3 -> over target (whole 95% interval above it)
  - route_accuracy: 12.4 pts, 95% bootstrap interval 0.5-31.9 -> inconclusive (95% interval spans the target - more tickets needed)
  - intent_accuracy: 0.0 pts, 95% bootstrap interval 0.0-0.0 -> within target (whole 95% interval below it)
  - retrieval_hit_rate: 4.0 pts, 95% bootstrap interval 0.0-12.0 -> inconclusive (95% interval spans the target - more tickets needed)
  **credibly over the 5-point target (interval entirely above it):** auto_response_rate

**customer_region**: asia_pacific (n=21, auto 76.2% [57-95], route acc 85.7% [71-100], retrieval hit 100.0%), europe (n=25, auto 76.0% [60-92], route acc 88.0% [76-100], retrieval hit 94.7%), latin_america (n=7, too small to judge, auto 57.1% [29-86], route acc 85.7% [57-100], retrieval hit 100.0%), north_america (n=27, auto 81.5% [67-96], route acc 70.4% [52-85], retrieval hit 100.0%)
  spread (pts): {'auto_response_rate': 5.5, 'route_accuracy': 17.6, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 5.3}
  - auto_response_rate: 5.5 pts, 95% bootstrap interval 3.0-32.3 -> inconclusive (95% interval spans the target - more tickets needed)
  - route_accuracy: 17.6 pts, 95% bootstrap interval 4.6-40.7 -> inconclusive (95% interval spans the target - more tickets needed)
  - intent_accuracy: 0.0 pts, 95% bootstrap interval 0.0-0.0 -> within target (whole 95% interval below it)
  - retrieval_hit_rate: 5.3 pts, 95% bootstrap interval 0.0-15.8 -> inconclusive (95% interval spans the target - more tickets needed)
  over the 5-point target on the point estimate only (not conclusive): auto_response_rate, route_accuracy, retrieval_hit_rate

**language_fluency**: fluent (n=61, auto 77.0% [66-87], route acc 88.5% [80-95], retrieval hit 97.7%), non_fluent (n=19, auto 73.7% [53-90], route acc 57.9% [37-79], retrieval hit 100.0%)
  spread (pts): {'auto_response_rate': 3.4, 'route_accuracy': 30.6, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 2.3}
  - auto_response_rate: 3.4 pts, 95% bootstrap interval 0.3-27.5 -> inconclusive (95% interval spans the target - more tickets needed)
  - route_accuracy: 30.6 pts, 95% bootstrap interval 8.3-54.0 -> over target (whole 95% interval above it)
  - intent_accuracy: 0.0 pts, 95% bootstrap interval 0.0-0.0 -> within target (whole 95% interval below it)
  - retrieval_hit_rate: 2.3 pts, 95% bootstrap interval 0.0-6.8 -> inconclusive (95% interval spans the target - more tickets needed)
  **credibly over the 5-point target (interval entirely above it):** route_accuracy

**channel**: chat (n=22, auto 72.7% [54-91], route acc 86.4% [68-100], retrieval hit 92.3%), docs_comment (n=16, auto 62.5% [38-88], route acc 75.0% [56-94], retrieval hit 100.0%), email (n=31, auto 87.1% [74-97], route acc 80.6% [64-94], retrieval hit 100.0%), forum (n=11, auto 72.7% [46-100], route acc 81.8% [54-100], retrieval hit 100.0%)
  spread (pts): {'auto_response_rate': 24.6, 'route_accuracy': 11.4, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 7.7}
  - auto_response_rate: 24.6 pts, 95% bootstrap interval 11.1-53.0 -> over target (whole 95% interval above it)
  - route_accuracy: 11.4 pts, 95% bootstrap interval 5.8-43.8 -> over target (whole 95% interval above it)
  - intent_accuracy: 0.0 pts, 95% bootstrap interval 0.0-0.0 -> within target (whole 95% interval below it)
  - retrieval_hit_rate: 7.7 pts, 95% bootstrap interval 0.0-23.1 -> inconclusive (95% interval spans the target - more tickets needed)
  **credibly over the 5-point target (interval entirely above it):** auto_response_rate, route_accuracy

**ticket_length**: long (>30) (n=11, auto 90.9% [73-100], route acc 72.7% [46-100], retrieval hit 100.0%), medium (21-30) (n=40, auto 80.0% [68-92], route acc 80.0% [68-92], retrieval hit 96.4%), short (<=20 words) (n=29, auto 65.5% [48-83], route acc 86.2% [72-97], retrieval hit 100.0%)
  spread (pts): {'auto_response_rate': 25.4, 'route_accuracy': 13.5, 'intent_accuracy': 0.0, 'retrieval_hit_rate': 3.6}
  - auto_response_rate: 25.4 pts, 95% bootstrap interval 6.8-46.1 -> over target (whole 95% interval above it)
  - route_accuracy: 13.5 pts, 95% bootstrap interval 3.2-44.2 -> inconclusive (95% interval spans the target - more tickets needed)
  - intent_accuracy: 0.0 pts, 95% bootstrap interval 0.0-0.0 -> within target (whole 95% interval below it)
  - retrieval_hit_rate: 3.6 pts, 95% bootstrap interval 0.0-10.7 -> inconclusive (95% interval spans the target - more tickets needed)
  **credibly over the 5-point target (interval entirely above it):** auto_response_rate

## Caveats

- The figures above should be treated with caution: FCR is a proxy (answered without a human is not the same as solved), satisfaction and hallucination need human assessment, and many tickets repeat development-set wording.
- In the labelled data, identical ticket text appears with both `auto_respond` and `escalate` labels, so no text-based system can reach 100% route accuracy.
