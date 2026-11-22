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


app = FastAPI(title="Cloud Genie support triage", version="0.1.0", lifespan=lifespan)

CHAT_PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Cloud Genie support</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f3f5f9;--card:#fff;--ink:#1f2937;--mute:#6b7280;--line:#e5e7eb;--brand:#2563eb;--ok:#16a34a;--warn:#d97706}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
.app{max-width:800px;margin:0 auto;height:100vh;display:flex;flex-direction:column}
header{padding:1rem 1.25rem;background:var(--brand);color:#fff}
header h1{margin:0;font-size:1.15rem}header p{margin:.15rem 0 0;font-size:.82rem;opacity:.85}
#log{flex:1;overflow-y:auto;padding:1rem 1.25rem;display:flex;flex-direction:column;gap:.9rem}
.empty{margin:auto;color:var(--mute);text-align:center}
.row{display:flex}.row.user{justify-content:flex-end}
.bubble{max-width:85%;padding:.7rem .95rem;border-radius:14px;background:var(--card);border:1px solid var(--line);white-space:pre-wrap;word-wrap:break-word}
.user .bubble{background:var(--brand);color:#fff;border-color:var(--brand);border-bottom-right-radius:4px}
.bot .bubble{border-bottom-left-radius:4px;white-space:normal}
.bot.ok .bubble{border-left:4px solid var(--ok)}.bot.esc .bubble{border-left:4px solid var(--warn)}
.reply{white-space:pre-wrap}
.title{font-weight:600;margin-bottom:.35rem}
.chips{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.6rem}
.chip{font-size:.74rem;padding:.12rem .55rem;border-radius:999px;background:#eef2ff;color:#3730a3}
.chip.high{background:#fee2e2;color:#991b1b}
.sec{margin-top:.7rem;padding-top:.6rem;border-top:1px dashed var(--line)}
.sec h4{margin:0 0 .25rem;font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
.sec ul{margin:0;padding-left:1.1rem}
dl{margin:0;display:grid;grid-template-columns:max-content 1fr;gap:.15rem .8rem;font-size:.88rem}
dt{color:var(--mute)}dd{margin:0}
details{margin-top:.6rem;font-size:.82rem;color:var(--mute)}
details pre{white-space:pre-wrap;background:#f4f4f5;padding:.6rem;border-radius:8px;overflow-x:auto}
form{display:flex;gap:.5rem;padding:.8rem 1.25rem;background:var(--card);border-top:1px solid var(--line);flex-wrap:wrap}
select,textarea,button{font:inherit;border-radius:10px;border:1px solid var(--line);padding:.55rem .7rem}
textarea{flex:1 1 300px;resize:none;height:3.2rem}
button{background:var(--brand);color:#fff;border:0;padding:.55rem 1.2rem;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.typing{color:var(--mute);font-style:italic}
</style>
<div class="app">
<header><h1>Cloud Genie support</h1><p>Replies are drafted automatically from Cloud Genie's support documentation.</p></header>
<div id="log"><div class="empty" id="empty">Describe your problem below and press Send.</div></div>
<form id="f">
<select id="ch" title="Channel"><option>chat</option><option>email</option><option>docs_comment</option><option>forum</option></select>
<textarea id="body" placeholder="Describe your problem... (Enter to send, Shift+Enter for new line)" required></textarea>
<button id="btn">Send</button>
</form></div>
<script>
const $=id=>document.getElementById(id), log=$('log');
const pretty=s=>String(s||'').replace(/_/g,' ').replace(/^./,c=>c.toUpperCase());
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function chip(text,cls){return el('span','chip '+(cls||''),text)}
function add(node){const e=$('empty');if(e)e.remove();log.appendChild(node);log.scrollTop=log.scrollHeight}
function userMsg(t){const r=el('div','row user');r.appendChild(el('div','bubble',t));add(r)}
function section(title,child){const s=el('div','sec');s.appendChild(el('h4','',title));s.appendChild(child);return s}
function list(items){const u=el('ul');items.forEach(i=>u.appendChild(el('li','',i)));return u}

function botMsg(j){
  const ok=j.disposition==='auto_responded';
  const row=el('div','row bot '+(ok?'ok':'esc')), b=el('div','bubble');
  if(ok){
    b.appendChild(el('div','reply',j.reply_text));
  }else{
    b.appendChild(el('div','title','Passed to our support team'));
    b.appendChild(el('div','',"We couldn't answer this automatically, so a person will follow up."));
    if(j.reason)b.appendChild(section('Why',el('div','',j.reason)));
    if((j.blocked_by||[]).length)b.appendChild(section('Automatic reply blocked by',list(j.blocked_by.map(pretty))));
    const br=j.escalation_brief;
    if(br){
      const dl=el('dl');
      [['Ticket',br.ticket_id],['Likely topic',pretty(br.likely_intent)],['Urgency',pretty(br.urgency)],
       ['Customer tier',pretty(br.customer_tier)],['Summary',br.summary]].forEach(([k,v])=>{
        if(v){dl.appendChild(el('dt','',k));dl.appendChild(el('dd','',v))}});
      b.appendChild(section('Handoff details',dl));
      if((br.unsure_about||[]).length)b.appendChild(section('Open questions',list(br.unsure_about)));
      if((br.relevant_articles||[]).length)b.appendChild(section('Possibly relevant articles',
        list(br.relevant_articles.map(a=>a.title||a.doc_id))));
      if(br.suggested_draft)b.appendChild(section('Suggested draft',el('div','reply',br.suggested_draft)));
    }
  }
  const chips=el('div','chips');
  if(j.intent)chips.appendChild(chip(pretty(j.intent)));
  if(j.urgency)chips.appendChild(chip(pretty(j.urgency)+' urgency',/high|urgent|critical/i.test(j.urgency)?'high':''));
  if(ok&&j.route_confidence!=null)chips.appendChild(chip('Confidence '+Math.round(j.route_confidence*100)+'%'));
  if(j.latency_s!=null)chips.appendChild(chip(Number(j.latency_s).toFixed(1)+'s'));
  if(chips.children.length)b.appendChild(chips);
  if(ok&&(j.cited_doc_ids||[]).length)b.appendChild(section('Sources',list(j.cited_doc_ids)));
  const d=el('details');d.appendChild(el('summary','','Technical details (JSON)'));
  d.appendChild(el('pre','',JSON.stringify(j,null,2)));b.appendChild(d);
  row.appendChild(b);add(row);
}

async function send(){
  const text=$('body').value.trim();if(!text)return;
  userMsg(text);$('body').value='';$('btn').disabled=true;
  const t=el('div','row bot');t.appendChild(el('div','bubble typing','Looking through our documentation...'));add(t);
  try{
    const r=await fetch('/v1/tickets',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({channel:$('ch').value,subject:'',body:text})});
    if(!r.ok)throw new Error('Server returned '+r.status);
    const j=await r.json();t.remove();botMsg(j);
  }catch(e){
    t.remove();const r=el('div','row bot esc'),b=el('div','bubble','Sorry, something went wrong: '+e.message);r.appendChild(b);add(r);
  }
  $('btn').disabled=false;$('body').focus();
}
$('f').addEventListener('submit',e=>{e.preventDefault();send()});
$('body').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
</script></html>"""


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
