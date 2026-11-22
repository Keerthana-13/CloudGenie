# Incident procedure: stopping automatic answers (operator on call)

For anyone on call, including someone who has never seen this system. It covers one job: **stop the system sending automatic
replies, check that it has stopped, and hand over what happened.** Nothing here needs a deployment.

## 1. When to stop it
Stop automatic answering if any of these is true. If you are unsure, stop it: every ticket then goes to a person with a brief.
- A customer or agent reports an automatic reply that is wrong, confusing, rude, or contains someone else's details.
- A reply promises a refund, credit, fix or date.
- A reply appears to follow instructions that were written inside a customer's ticket.
- The documentation has been found to be wrong, or an article was withdrawn.
- A security or compliance matter was answered automatically (this should never happen; treat it as serious).

## 2. How to stop it (use any one; the first is quickest)
| Method | What to do | Takes effect |
|---|---|---|
| API | `POST /admin/kill-switch` with body `{"enabled": false}` and header `X-Admin-Token: <ADMIN_TOKEN>` | next ticket |
| File | Create an empty file named `KILL_SWITCH` in the `storage/` folder of the running service | next ticket |
| Setting | Set `AUTO_RESPOND_ENABLED=false` and restart the service | after restart |

The switch is checked on every ticket. The admin token is in the service's `.env` file and is never to be shared or committed.
If the API answers 403, the token is wrong or `ADMIN_TOKEN` is not set on the server: use the file method.

## 3. How to check that it has stopped
1. `GET /health` shows `"kill_switch_engaged": true`.
2. Send one test ticket to `POST /v1/tickets`. The answer must have `"disposition": "escalated"` and the rule `kill_switch`.
3. `GET /metrics` shows tickets arriving as escalated and none as auto-responded after the time you stopped it.

## 4. What to record
- The time you stopped it and who did it.
- The ticket ids involved. For each one, the full decision trail is returned by `POST /v1/tickets` or is in the `decisions` table of
  `storage/decisions.db` (`ticket_id`, stage, reason, guardrail results, prompt version, articles used).
- Which article ids the wrong reply cited, so the documentation owner can tell a wrong article from a misread one.

## 5. Handing over
Tell the head of support and the documentation owner. Do not edit articles, the policy file or the thresholds yourself during an incident.
Customers whose tickets were affected are contacted by a person.

## 6. Turning it back on
Only after the cause is understood and the head of support agrees: `POST /admin/kill-switch` with `{"enabled": true}`, or delete
`storage/KILL_SWITCH` (and set `AUTO_RESPOND_ENABLED=true` if you used the setting). Repeat the checks in section 3; the test ticket should now be
answered or escalated by the normal rules.

## Still to be agreed (Stage 2 open question Q7)
- Who is accountable for a wrong automatic answer, and who the on-call operator reports it to: **to be confirmed by Marcus Adeyemi.**
- How long a wrong answer may stand before it is reported: **to be confirmed.**
