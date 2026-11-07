"""HTTP interface.   Start with:   python -m src.api      (then open http://127.0.0.1:8000)

  GET  /                    a tiny chat page for demonstrations
  POST /v1/tickets          one ticket in (standard schema, or just {"channel","subject","body"}), decision out
  GET  /health              liveness and which components are active
  GET  /metrics             Prometheus metrics
  POST /admin/kill-switch   {"enabled": false}  - stop/resume automatic answering without a deployment
                            requires header  X-Admin-Token: <ADMIN_TOKEN>
"""
from __future__ import annotations

import hmac
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, Header, HTTPException, Response
from fastapi.responses import HTMLResponse

from . import telemetry
from .config import get_settings, kill_switch_engaged
from .pipeline import SupportPipeline

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = SupportPipeline(run_id="api-" + uuid.uuid4().hex[:8])
    yield
    _state["pipeline"].decisions.close()


app = FastAPI(title="CloudServe support triage", version="0.1.0", lifespan=lifespan)

CHAT_PAGE = """<!doctype html><meta charset="utf-8"><title>CloudServe support</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:16px/1.5 system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;height:5.5rem;font:inherit;padding:.5rem}button{font:inherit;padding:.5rem 1rem;margin-top:.5rem}
pre{white-space:pre-wrap;background:#f4f4f5;padding:1rem;border-radius:8px}.esc{background:#fff4e5}.ok{background:#eefaf0}
small{color:#555}</style>
<h1>CloudServe support</h1>
<p><small>Demo front-end for the triage system. Replies are drafted automatically from CloudServe's support documentation.</small></p>
<label>Channel <select id=ch><option>chat</option><option>email</option><option>docs_comment</option><option>forum</option></select></label>
<textarea id=body placeholder="Describe your problem..."></textarea><br><button onclick=send()>Send</button>
<div id=out></div>
<script>
async function send(){
  const r=await fetch('/v1/tickets',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({channel:ch.value,subject:'',body:body.value})});
  const j=await r.json(); const el=document.createElement('pre');
  if(j.disposition==='auto_responded'){el.className='ok';el.textContent=j.reply_text+'\\n\\n['+j.intent+' · confidence '+j.route_confidence+' · '+j.latency_s+'s]';}
  else{el.className='esc';el.textContent='Passed to our support team'+(j.blocked_by.length?' (automatic reply blocked: '+j.blocked_by.join(', ')+')':'')+
    '.\\n\\nWhy: '+j.reason+'\\n\\n'+JSON.stringify(j.escalation_brief,null,2);}
  out.prepend(el);
}
</script>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return CHAT_PAGE


@app.get("/health")
def health() -> dict:
    p: SupportPipeline = _state["pipeline"]
    return {"status": "ok", "kill_switch_engaged": kill_switch_engaged(), **p.describe()}


@app.post("/v1/tickets")
def submit_ticket(ticket: dict = Body(...)) -> dict:
    p: SupportPipeline = _state["pipeline"]
    ticket.setdefault("ticket_id", "API-" + uuid.uuid4().hex[:8])
    ticket.setdefault("channel", "chat")
    out = p.process(ticket).to_dict()
    out["decisions"] = p.decisions.decisions_for(out["ticket_id"])      # full audit trail for this ticket
    return out


@app.get("/metrics")
def metrics() -> Response:
    body, ctype = telemetry.render()
    return Response(content=body, media_type=ctype)


@app.post("/admin/kill-switch")
def kill_switch(payload: dict = Body(...), x_admin_token: str = Header(default="")) -> dict:
    expected = os.environ.get("ADMIN_TOKEN", "")
    if not expected or expected in {"change_me", "your_key_here"} or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=403, detail="admin token missing, wrong, or not configured")
    flag = get_settings().storage_dir / "KILL_SWITCH"
    flag.parent.mkdir(parents=True, exist_ok=True)
    if payload.get("enabled") is False:
        flag.write_text("automatic answering disabled via API\n")
    elif payload.get("enabled") is True:
        flag.unlink(missing_ok=True)
    else:
        raise HTTPException(status_code=422, detail='send {"enabled": true} or {"enabled": false}')
    return {"automatic_answering_enabled": not kill_switch_engaged()}


def main() -> None:
    import uvicorn
    uvicorn.run("src.api:app", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
