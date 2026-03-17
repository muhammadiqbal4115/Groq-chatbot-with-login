import re
import json
import time
import random
import string
import hashlib
import smtplib
import streamlit as st
import streamlit.components.v1 as components
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import gspread
from google.oauth2.service_account import Credentials
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Groq AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_PROMPT = "You are a helpful AI Assistant. Be clear, correct and concise."
TONE_MAP = {
    "Friendly": "You are a friendly AI assistant. Respond warmly and politely.",
    "Strict":   "You are strict and professional. Give short, precise answers.",
    "Teacher":  "You are a patient teacher. Explain concepts clearly with examples.",
}

# ══════════════════════════════════════════════════════════════════════════════
#  CSS BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

NEON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;600&display=swap');

[data-testid="stMain"] {
  transition: margin-left 0.3s ease !important;
}

:root {
  --neon:         #00d4ff;
  --neon-dim:     #0099bb;
  --neon-glow:    0 0 8px #00d4ff88, 0 0 20px #00d4ff44;
  --neon-glow-lg: 0 0 12px #00d4ffaa, 0 0 30px #00d4ff66, 0 0 60px #00d4ff22;
  --bg-deep:      #030508;
  --bg-card:      #0c1420;
  --bg-glass:     rgba(0,212,255,0.04);
  --bg-hover:     rgba(0,212,255,0.08);
  --border:       rgba(0,212,255,0.18);
  --border-bright:rgba(0,212,255,0.5);
  --text-primary: #e8f4ff;
  --text-muted:   #6a90a8;
  --text-dim:     #3a5060;
  --success:      #00ff9d;
  --error:        #ff4466;
  --warn:         #ffaa00;
}

#MainMenu, footer, header,
.stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-deep) !important;
  font-family: 'Exo 2', sans-serif !important;
  color: var(--text-primary) !important;
}
[data-testid="stMain"] { background: var(--bg-deep) !important; }

[data-testid="stAppViewContainer"]::before {
  content:''; position:fixed; inset:0;
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg,rgba(0,212,255,0.03) 1px,transparent 1px);
  background-size:40px 40px;
  pointer-events:none; z-index:0;
}

[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border) !important;
  box-shadow: 4px 0 30px rgba(0,212,255,0.06) !important;
  position: fixed !important;
  top: 0 !important; left: 0 !important;
  height: 100vh !important;
  overflow-y: auto !important;
  z-index: 999 !important;
  transition: transform 0.35s cubic-bezier(0.4,0,0.2,1) !important;
}
[data-testid="stSidebar"] > div { background: transparent !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1.2rem 0.9rem !important; }
[data-testid="stSidebar"]::-webkit-scrollbar { width:4px; }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb { background:var(--neon-dim); border-radius:4px; }

h1,h2,h3,h4 { font-family:'Rajdhani',sans-serif !important; letter-spacing:0.05em !important; color:var(--text-primary) !important; }
.stMarkdown p,.stMarkdown li { color:var(--text-primary) !important; font-family:'Exo 2',sans-serif !important; }
hr { border:none !important; border-top:1px solid var(--border) !important; margin:0.8rem 0 !important; }

label,.stSelectbox label,.stSlider label,.stTextInput label,.stTextArea label,
[data-testid="stWidgetLabel"] {
  color:var(--neon) !important; font-family:'Rajdhani',sans-serif !important;
  font-size:0.82rem !important; letter-spacing:0.08em !important; text-transform:uppercase !important;
}

.stTextInput input,.stTextArea textarea,input[type="text"],input[type="password"] {
  background:rgba(0,212,255,0.04) !important; border:1px solid var(--border) !important;
  border-radius:6px !important; color:var(--text-primary) !important;
  font-family:'Share Tech Mono',monospace !important; font-size:0.88rem !important;
  transition:border-color 0.25s,box-shadow 0.25s !important;
}
.stTextInput input:focus,.stTextArea textarea:focus {
  border-color:var(--neon) !important; box-shadow:var(--neon-glow) !important; outline:none !important;
}
.stTextInput input::placeholder,.stTextArea textarea::placeholder { color:var(--text-dim) !important; }

.stSelectbox [data-baseweb="select"] > div {
  background:rgba(0,212,255,0.04) !important; border:1px solid var(--border) !important;
  border-radius:6px !important; color:var(--text-primary) !important;
}
.stSelectbox [data-baseweb="select"] > div:hover { border-color:var(--neon) !important; }
[data-baseweb="popover"] { background:var(--bg-card) !important; border:1px solid var(--border-bright) !important; box-shadow:var(--neon-glow) !important; }
[role="option"] { background:var(--bg-card) !important; color:var(--text-primary) !important; font-family:'Exo 2',sans-serif !important; }
[role="option"]:hover { background:var(--bg-hover) !important; color:var(--neon) !important; }

.stSlider [data-baseweb="slider"] [role="slider"] { background:var(--neon) !important; box-shadow:var(--neon-glow) !important; }
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:first-child { background:var(--border) !important; }
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrack"] > div:nth-child(2) { background:var(--neon) !important; }

.stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {
  background:transparent !important; border:1px solid var(--neon) !important;
  color:var(--neon) !important; font-family:'Rajdhani',sans-serif !important;
  font-size:0.95rem !important; font-weight:600 !important; letter-spacing:0.1em !important;
  text-transform:uppercase !important; border-radius:6px !important;
  box-shadow:var(--neon-glow) !important; transition:all 0.2s !important; padding:0.45rem 1.2rem !important;
}
.stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
  background:rgba(0,212,255,0.12) !important; box-shadow:var(--neon-glow-lg) !important; transform:translateY(-1px) !important;
}
.stButton > button[kind="secondary"], .stButton > button:not([kind]) {
  background:transparent !important; border:1px solid var(--border) !important;
  color:var(--text-muted) !important; font-family:'Rajdhani',sans-serif !important;
  font-size:0.88rem !important; font-weight:500 !important; letter-spacing:0.06em !important;
  border-radius:6px !important; transition:all 0.2s !important;
}
.stButton > button[kind="secondary"]:hover, .stButton > button:not([kind]):hover {
  border-color:var(--neon) !important; color:var(--neon) !important; background:var(--bg-hover) !important;
}
.stDownloadButton > button {
  background:transparent !important; border:1px solid var(--border) !important;
  color:var(--text-muted) !important; font-family:'Rajdhani',sans-serif !important;
  font-weight:500 !important; letter-spacing:0.06em !important; border-radius:6px !important; transition:all 0.2s !important;
}
.stDownloadButton > button:hover { border-color:var(--neon) !important; color:var(--neon) !important; background:var(--bg-hover) !important; }

.stCheckbox label span { color:var(--text-muted) !important; font-size:0.88rem !important; font-family:'Exo 2',sans-serif !important; }
[data-testid="stCheckbox"] svg { stroke:var(--neon) !important; }

.stTabs [data-baseweb="tab-list"] { background:transparent !important; border-bottom:1px solid var(--border) !important; gap:0.2rem !important; }
.stTabs [data-baseweb="tab"] {
  background:transparent !important; color:var(--text-muted) !important;
  font-family:'Rajdhani',sans-serif !important; font-size:0.9rem !important; font-weight:600 !important;
  letter-spacing:0.08em !important; border:none !important; padding:0.5rem 1.1rem !important;
  border-radius:6px 6px 0 0 !important; transition:all 0.2s !important;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--neon) !important; background:var(--bg-hover) !important; }
.stTabs [aria-selected="true"] { color:var(--neon) !important; background:rgba(0,212,255,0.08) !important; border-bottom:2px solid var(--neon) !important; text-shadow:0 0 10px var(--neon) !important; }
.stTabs [data-baseweb="tab-panel"] { background:transparent !important; padding:1rem 0 !important; }

[data-testid="stChatMessage"] {
  background:var(--bg-glass) !important; border:1px solid var(--border) !important;
  border-radius:10px !important; margin-bottom:0.5rem !important; padding:0.6rem 0.8rem !important; transition:border-color 0.2s !important;
}
[data-testid="stChatMessage"]:hover { border-color:rgba(0,212,255,0.3) !important; }
.stChatMessage p { color:var(--text-primary) !important; font-family:'Exo 2',sans-serif !important; }

[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(0,212,255,0.02) !important; border:1px solid var(--border) !important; border-radius:10px !important; }

.streamlit-expanderHeader { background:var(--bg-glass) !important; border:1px solid var(--border) !important; border-radius:6px !important; color:var(--neon) !important; font-family:'Rajdhani',sans-serif !important; letter-spacing:0.06em !important; }
.streamlit-expanderHeader:hover { border-color:var(--neon) !important; background:var(--bg-hover) !important; }
.streamlit-expanderContent { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-top:none !important; border-radius:0 0 6px 6px !important; }

.stSuccess { background:rgba(0,255,157,0.08) !important; border:1px solid rgba(0,255,157,0.3) !important; border-radius:6px !important; }
.stError   { background:rgba(255,68,102,0.08) !important; border:1px solid rgba(255,68,102,0.3) !important; border-radius:6px !important; }
.stInfo    { background:rgba(0,212,255,0.06) !important; border:1px solid var(--border) !important; border-radius:6px !important; }

.stCaption, small { color:var(--text-dim) !important; font-family:'Share Tech Mono',monospace !important; font-size:0.75rem !important; }

.sidebar-username {
  display:flex; align-items:center; gap:0.5rem;
  padding:0.6rem 0.8rem; background:rgba(0,212,255,0.06);
  border:1px solid var(--border); border-radius:8px; margin-bottom:0.8rem;
  font-family:'Rajdhani',sans-serif; font-size:1rem; font-weight:600;
  color:var(--neon); letter-spacing:0.08em;
}
.sidebar-section {
  font-family:'Rajdhani',sans-serif; font-size:0.75rem; font-weight:700;
  letter-spacing:0.15em; text-transform:uppercase; color:var(--text-dim);
  padding:0.3rem 0; margin-top:0.6rem; border-bottom:1px solid var(--border); margin-bottom:0.5rem;
}

.page-title    { font-family:'Rajdhani',sans-serif; font-size:2rem; font-weight:700; letter-spacing:0.1em; color:var(--neon); text-shadow:var(--neon-glow); margin:0; }
.page-subtitle { font-family:'Share Tech Mono',monospace; font-size:0.78rem; color:var(--text-dim); letter-spacing:0.1em; margin-top:0.1rem; }

.auth-logo  { text-align:center; font-size:3.5rem; margin-bottom:0.3rem; filter:drop-shadow(0 0 12px var(--neon)); }
.auth-title { text-align:center; font-family:'Rajdhani',sans-serif; font-size:1.9rem; font-weight:700; letter-spacing:0.12em; color:var(--neon); text-shadow:var(--neon-glow); margin-bottom:0.15rem; }
.auth-sub   { text-align:center; font-family:'Share Tech Mono',monospace; color:var(--text-dim); font-size:0.75rem; letter-spacing:0.1em; margin-bottom:1.4rem; }

.strength-bar-bg   { background:#111820; border-radius:4px; height:4px; width:100%; margin:4px 0 2px; }
.strength-bar-fill { height:4px; border-radius:4px; transition:width .4s,background .4s; }

.hint-pill { display:inline-block; background:rgba(0,212,255,0.05); border:1px solid var(--border); border-radius:20px; padding:1px 8px; font-size:.75rem; color:var(--text-dim); margin:2px; font-family:'Share Tech Mono',monospace; }
.hint-ok   { border-color:var(--success); color:var(--success); background:rgba(0,255,157,0.05); }

.info-box { background:rgba(0,212,255,0.05); border:1px solid var(--border); border-radius:6px; padding:.55rem 1rem; font-size:.8rem; color:var(--text-muted); font-family:'Share Tech Mono',monospace; margin-top:.8rem; text-align:center; }
.warn-box { background:rgba(255,68,102,0.06); border:1px solid rgba(255,68,102,0.3); border-radius:6px; padding:.55rem 1rem; font-size:.8rem; color:#ff8899; margin-top:.8rem; text-align:center; }
.otp-box  { background:rgba(0,255,157,0.05); border:1px solid rgba(0,255,157,0.25); border-radius:6px; padding:.6rem 1rem; font-size:.82rem; color:#80ffcc; margin:.6rem 0; text-align:center; font-family:'Share Tech Mono',monospace; }
.empty-chat { text-align:center; color:var(--text-dim); padding-top:80px; font-family:'Share Tech Mono',monospace; font-size:0.9rem; letter-spacing:0.08em; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:var(--neon-dim); }
</style>
"""

# Login page: sidebar hidden, block-container goes full width
LOGIN_EXTRA_CSS = """
<style>
  section[data-testid="stSidebar"] { display: none !important; }
  [data-testid="block-container"] {
    max-width: 100% !important;
    padding: 3rem 5vw !important;
  }
</style>
"""

# Chat page: center content at a readable max-width
CHAT_EXTRA_CSS = """
<style>
  [data-testid="block-container"] {
    max-width: 1100px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-top: 1.5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
  }
</style>
"""

# Sidebar toggle: runs inside components.html iframe, reaches parent DOM via window.parent
SIDEBAR_TOGGLE_HTML = """
<style>
  #btn {
    position:fixed;
    top:14px;
    left:14px;
    z-index:99999;
    width:42px;
    height:42px;
    background:#0c1420;
    border:1px solid rgba(0,212,255,0.6);
    border-radius:10px;
    cursor:pointer;
    display:flex;
    align-items:center;
    justify-content:center;
    box-shadow:0 0 10px rgba(0,212,255,0.6);
    transition:all 0.25s ease;
  }

  #btn:hover {
    background:rgba(0,212,255,0.12);
    box-shadow:0 0 18px rgba(0,212,255,0.9);
  }

  .hbg {
    display:flex;
    flex-direction:column;
    gap:5px;
    width:18px;
  }

  .hbg span {
    height:2px;
    background:#00d4ff;
    border-radius:2px;
  }
</style>

<button id="btn">
  <div class="hbg">
    <span></span>
    <span></span>
    <span></span>
  </div>
</button>

<script>
(function () {
  const par = window.parent;
  const doc = par.document;

  function sidebar() {
    return doc.querySelector('[data-testid="stSidebar"]');
  }

  function main() {
    return doc.querySelector('[data-testid="stMain"]');
  }

  let open = par.sessionStorage.getItem('sb') !== '0';

  function apply() {
    const sb = sidebar();
    const mainEl = main();

    if (!sb || !mainEl) return;

    if (open) {
      sb.style.transform = 'translateX(0)';
      mainEl.style.marginLeft = sb.offsetWidth + 'px';
    } else {
      sb.style.transform = 'translateX(-100%)';
      mainEl.style.marginLeft = '0px';
    }

    mainEl.style.transition = 'margin-left 0.3s ease';

    par.sessionStorage.setItem('sb', open ? '1' : '0');
  }

  function init() {
    if (sidebar()) apply();
    else setTimeout(init, 100);
  }

  init();

  document.getElementById('btn').onclick = () => {
    open = !open;
    apply();
  };
})();
</script>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  GOOGLE SHEETS CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Connecting to database …")
def get_spreadsheet():
    creds  = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=SCOPES)
    client = gspread.authorize(creds)
    name   = st.secrets.get("SPREADSHEET_NAME", "GroqChatbotDB")
    try:    sh = client.open(name)
    except gspread.SpreadsheetNotFound: sh = client.create(name)
    required = {
        "users":           ["username","password_hash","full_name","email","created_at"],
        "sessions":        ["username","session_id","name","created","updated"],
        "messages":        ["username","session_id","role","content","ts"],
        "password_resets": ["email","otp","expires_at","used"],
    }
    existing = {w.title for w in sh.worksheets()}
    for title, headers in required.items():
        if title not in existing:
            w = sh.add_worksheet(title=title, rows=1000, cols=len(headers))
            w.append_row(headers, value_input_option="RAW")
    return sh

def ws(name):        return get_spreadsheet().worksheet(name)
def all_rows(name):  return ws(name).get_all_records()


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════

def _hp(pw): return hashlib.sha256(pw.encode()).hexdigest()

@st.cache_data(ttl=30, show_spinner=False)
def _cached_users():
    result = {}
    for r in all_rows("users"):
        result[r["username"]] = {"password_hash":r.get("password_hash",""),"email":r.get("email","").lower().strip(),"full_name":r.get("full_name","")}
    return result

def load_users():            return _cached_users()
def user_exists(u):          return u in load_users()
def email_exists(e):         return any(v["email"]==e.lower().strip() for v in load_users().values())

def get_username_by_email(e):
    e = e.lower().strip()
    for u,d in load_users().items():
        if d["email"]==e: return u
    return None

def auth_user(u, pw):
    d = load_users().get(u)
    return d is not None and d["password_hash"]==_hp(pw)

def reg_user(u, pw, full_name="", email=""):
    if user_exists(u) or (email and email_exists(email)): return False
    ws("users").append_row([u,_hp(pw),full_name,email.lower().strip(),datetime.now().isoformat()],value_input_option="RAW")
    _cached_users.clear(); return True

def update_password(u, pw):
    sheet = ws("users"); data = sheet.get_all_values()
    for i,row in enumerate(data[1:],start=2):
        if row[0]==u: sheet.update_cell(i,2,_hp(pw)); break
    _cached_users.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  PASSWORD RESET
# ══════════════════════════════════════════════════════════════════════════════

def _generate_otp(n=6): return "".join(random.choices(string.digits,k=n))

def send_reset_email(to, otp):
    try: smtp_email=st.secrets["SMTP_EMAIL"]; smtp_pass=st.secrets["SMTP_PASSWORD"]
    except KeyError: st.error("SMTP not configured."); return False
    msg=MIMEMultipart("alternative"); msg["Subject"]="Password Reset — Groq AI"; msg["From"]=smtp_email; msg["To"]=to
    msg.attach(MIMEText(f"Your reset code: {otp}\n\nExpires in 10 minutes.","plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com",465) as s: s.login(smtp_email,smtp_pass); s.sendmail(smtp_email,to,msg.as_string())
        return True
    except Exception as e: st.error(f"Email error: {e}"); return False

def store_otp(email, otp):
    sheet=ws("password_resets"); data=sheet.get_all_values()
    expires=(datetime.now()+timedelta(minutes=10)).isoformat(); e=email.lower().strip()
    for i,row in enumerate(data[1:],start=2):
        if row and row[0].lower().strip()==e: sheet.update(f"A{i}:D{i}",[[e,otp,expires,"no"]]); return
    sheet.append_row([e,otp,expires,"no"],value_input_option="RAW")

def verify_otp(email, otp):
    e=email.lower().strip()
    for r in all_rows("password_resets"):
        if r.get("email","").lower().strip()==e and str(r.get("otp"))==otp:
            if r.get("used","no")=="yes": return "used"
            if datetime.fromisoformat(str(r["expires_at"]))<datetime.now(): return "expired"
            return "ok"
    return "invalid"

def consume_otp(email, otp):
    sheet=ws("password_resets"); data=sheet.get_all_values(); e=email.lower().strip()
    for i,row in enumerate(data[1:],start=2):
        if row and row[0].lower().strip()==e and row[1]==otp: sheet.update_cell(i,4,"yes"); break


# ══════════════════════════════════════════════════════════════════════════════
#  SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=15, show_spinner=False)
def _cached_sessions(username):
    rows=[r for r in all_rows("sessions") if r["username"]==username]
    rows.sort(key=lambda r:r.get("updated",r.get("created","")),reverse=True)
    return rows

def load_sessions(u):  return _cached_sessions(u)
def _bust_sessions(u): _cached_sessions.clear()

def create_session(u, name=None):
    existing=load_sessions(u); sid=f"s{int(time.time()*1000)}"; now=datetime.now().isoformat()
    sname=name or f"Chat {len(existing)+1}"
    ws("sessions").append_row([u,sid,sname,now,now],value_input_option="RAW")
    _bust_sessions(u)
    return {"username":u,"session_id":sid,"name":sname,"created":now,"updated":now}

def _find_session_row(u, sid):
    data=ws("sessions").get_all_values()
    for i,row in enumerate(data[1:],start=2):
        if len(row)>=2 and row[0]==u and row[1]==sid: return i
    return None

def update_session_timestamp(u, sid):
    row=_find_session_row(u,sid)
    if row: ws("sessions").update_cell(row,5,datetime.now().isoformat())
    _bust_sessions(u)

def delete_session(u, sid):
    row=_find_session_row(u,sid)
    if row: ws("sessions").delete_rows(row)
    _delete_session_messages(u,sid); _bust_sessions(u); _cached_messages.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  MESSAGES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=10, show_spinner=False)
def _cached_messages(u, sid):
    return [r for r in all_rows("messages") if r["username"]==u and r["session_id"]==sid]

def load_msgs(u, sid): return _cached_messages(u,sid)

def add_msg(u, sid, role, content):
    ws("messages").append_row([u,sid,role,content,datetime.now().isoformat()],value_input_option="RAW")
    update_session_timestamp(u,sid); _cached_messages.clear()

def _delete_session_messages(u, sid):
    sheet=ws("messages"); data=sheet.get_all_values()
    rows=[i for i,row in enumerate(data[1:],start=2) if len(row)>=2 and row[0]==u and row[1]==sid]
    for r in reversed(rows): sheet.delete_rows(r)

def clear_session_messages(u, sid):
    _delete_session_messages(u,sid); _cached_messages.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN + LLM
# ══════════════════════════════════════════════════════════════════════════════

def get_lc_mem(u, sid):
    key=f"lc_{u}_{sid}"
    if key not in st.session_state:
        h=InMemoryChatMessageHistory()
        for m in load_msgs(u,sid):
            if m["role"]=="user": h.add_user_message(m["content"])
            else:                 h.add_ai_message(m["content"])
        st.session_state[key]=h
    return st.session_state[key]

def generate_response(u, sid, user_input, api_key, settings):
    llm=ChatGroq(groq_api_key=api_key,model=settings["model"],temperature=settings["temperature"],max_tokens=settings["max_tokens"])
    prompt=ChatPromptTemplate.from_messages([("system","{system_prompt}"),MessagesPlaceholder(variable_name="history"),("human","{input}")])
    chain=prompt|llm|StrOutputParser(); mem=get_lc_mem(u,sid)
    chat_chain=RunnableWithMessageHistory(chain,lambda _:mem,input_messages_key="input",history_messages_key="history")
    return chat_chain.invoke({"input":user_input,"system_prompt":settings["system_prompt"]},config={"configurable":{"session_id":sid}})


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_username(u):
    u=u.strip()
    if not u:                                   return "Username cannot be empty."
    if len(u)<3:                                return "At least 3 characters required."
    if len(u)>32:                               return "Max 32 characters."
    if not re.match(r"^[a-zA-Z0-9_.\-]+$",u):  return "Letters, numbers, _ . - only."
    return None

def validate_email(e):
    e=e.strip()
    if not e: return "Email cannot be empty."
    if not re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$",e): return "Enter a valid email address."
    return None

def validate_password(p):
    if not p:                         return "Password cannot be empty."
    if len(p)<6:                      return "Minimum 6 characters."
    if not re.search(r"[A-Za-z]",p): return "Must contain at least one letter."
    if not re.search(r"[0-9]",p):    return "Must contain at least one number."
    return None

def password_strength(p):
    s=0
    if len(p)>=6:                  s+=1
    if len(p)>=10:                 s+=1
    if re.search(r"[A-Z]",p):      s+=1
    if re.search(r"[^A-Za-z0-9]",p): s+=1
    labels={0:"Very Weak",1:"Weak",2:"Fair",3:"Strong",4:"Very Strong"}
    colors={0:"#ff4466",1:"#ff8800",2:"#ffcc00",3:"#00d4ff",4:"#00ff9d"}
    return s,labels[s],colors[s]


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PANELS
# ══════════════════════════════════════════════════════════════════════════════

def _signin_panel():
    st.markdown("")
    un=st.text_input("Username",key="si_un",placeholder="your-username")
    pw=st.text_input("Password",key="si_pw",type="password",placeholder="••••••••")
    st.markdown("")
    if st.button("Sign In  ⟶",type="primary",use_container_width=True,key="btn_signin"):
        u=un.strip()
        if not u or not pw:    st.error("Please fill in both fields.")
        elif not user_exists(u):
            st.error("User not found.")
            st.markdown("<div class='warn-box'>No account with that username — switch to Sign Up.</div>",unsafe_allow_html=True)
        elif not auth_user(u,pw):
            st.error("Incorrect password.")
            st.markdown("<div class='info-box'>Forgot your password? Use the Forgot Password tab →</div>",unsafe_allow_html=True)
        else:
            st.session_state.logged_in=True; st.session_state.username=u
            st.success(f"Welcome back, {u}!"); time.sleep(0.5); st.rerun()
    st.markdown("<div class='info-box'>No account? Switch to Sign Up tab</div>",unsafe_allow_html=True)


def _signup_panel():
    st.markdown("")
    full_name=st.text_input("Full Name",key="su_name",placeholder="Your Full Name")
    su_un=st.text_input("Username",key="su_un",placeholder="3–32 chars · letters / numbers / _ . -")
    if su_un:
        err=validate_username(su_un)
        if err:                          st.markdown(f"<span style='color:#ff4466;font-size:.8rem'>⚠ {err}</span>",unsafe_allow_html=True)
        elif user_exists(su_un.strip()): st.markdown("<span style='color:#ff4466;font-size:.8rem'>⚠ Username taken.</span>",unsafe_allow_html=True)
        else:                            st.markdown("<span style='color:#00ff9d;font-size:.8rem'>✔ Available</span>",unsafe_allow_html=True)

    su_email=st.text_input("Email",key="su_email",placeholder="you@example.com")
    if su_email:
        e_err=validate_email(su_email)
        if e_err:                            st.markdown(f"<span style='color:#ff4466;font-size:.8rem'>⚠ {e_err}</span>",unsafe_allow_html=True)
        elif email_exists(su_email.strip()): st.markdown("<span style='color:#ff4466;font-size:.8rem'>⚠ Email already registered.</span>",unsafe_allow_html=True)
        else:                                st.markdown("<span style='color:#00ff9d;font-size:.8rem'>✔ Available</span>",unsafe_allow_html=True)

    su_pw1=st.text_input("Password",type="password",key="su_pw1",placeholder="min 6 chars, letter + number")
    if su_pw1:
        score,label,color=password_strength(su_pw1); pct=int((score/4)*100)
        def pill(txt,ok):
            c="hint-pill hint-ok" if ok else "hint-pill"
            return f"<span class='{c}'>{'✔' if ok else '·'} {txt}</span>"
        st.markdown(
            f"<div class='strength-bar-bg'><div class='strength-bar-fill' style='width:{pct}%;background:{color}'></div></div>"
            f"<span style='color:{color};font-size:.78rem;font-family:\"Share Tech Mono\",monospace'>Strength: <b>{label}</b></span><br>"
            +pill("6+ chars",len(su_pw1)>=6)+pill("Letter",bool(re.search(r"[A-Za-z]",su_pw1)))+pill("Number",bool(re.search(r"[0-9]",su_pw1))),
            unsafe_allow_html=True)

    su_pw2=st.text_input("Confirm Password",type="password",key="su_pw2",placeholder="Re-enter password")
    if su_pw2 and su_pw1:
        if su_pw1==su_pw2: st.markdown("<span style='color:#00ff9d;font-size:.8rem'>✔ Passwords match</span>",unsafe_allow_html=True)
        else:              st.markdown("<span style='color:#ff4466;font-size:.8rem'>⚠ Passwords do not match</span>",unsafe_allow_html=True)

    st.markdown(""); agree=st.checkbox("I agree to the Terms of Service",key="su_agree"); st.markdown("")

    if st.button("Create Account  ⟶",type="primary",use_container_width=True,key="btn_signup"):
        u,email,p1,p2=su_un.strip(),su_email.strip(),su_pw1,su_pw2
        err=validate_username(u)
        if err:                    st.error(f"⚠ {err}")
        elif user_exists(u):       st.error("Username already taken.")
        elif validate_email(email):st.error(validate_email(email))
        elif email_exists(email):  st.error("Email already registered.")
        else:
            perr=validate_password(p1)
            if perr:        st.error(f"⚠ {perr}")
            elif p1!=p2:    st.error("Passwords do not match.")
            elif not agree: st.error("Please accept the Terms of Service.")
            else:
                with st.spinner("Creating account …"): ok=reg_user(u,p1,full_name.strip(),email)
                if ok:
                    st.success(f"Account created! Welcome, {u}.")
                    st.session_state.logged_in=True; st.session_state.username=u
                    time.sleep(0.8); st.rerun()
                else: st.error("Something went wrong. Please try again.")


def _forgot_password_panel():
    st.markdown("")
    if "fp_step" not in st.session_state:
        st.session_state.fp_step=1; st.session_state.fp_email=""; st.session_state.fp_username=""

    if st.session_state.fp_step==1:
        st.markdown("##### Enter your registered email")
        st.caption("A 6-digit reset code will be sent to your inbox."); st.markdown("")
        fp_email=st.text_input("Email address",key="fp_email_input",placeholder="you@example.com")
        if st.button("Send Reset Code  ⟶",type="primary",use_container_width=True,key="btn_send_otp"):
            email=fp_email.strip(); e_err=validate_email(email)
            if e_err: st.error(f"⚠ {e_err}")
            elif not email_exists(email): st.success("If that email is registered, a code has been sent.")
            else:
                username=get_username_by_email(email); otp=_generate_otp(); store_otp(email,otp)
                with st.spinner("Sending email …"): sent=send_reset_email(email,otp)
                if sent:
                    st.session_state.fp_step=2; st.session_state.fp_email=email; st.session_state.fp_username=username
                    st.success("Reset code sent! Check your inbox."); time.sleep(0.8); st.rerun()
        st.markdown("<div class='info-box'>Remembered it? Go back to Sign In</div>",unsafe_allow_html=True)

    elif st.session_state.fp_step==2:
        email=st.session_state.fp_email
        st.markdown(f"##### Reset for `{email}`")
        st.markdown("<div class='otp-box'>📬 6-digit code sent · expires in <b>10 min</b></div>",unsafe_allow_html=True)
        st.markdown("")
        otp_input=st.text_input("6-digit Code",key="fp_otp_input",placeholder="e.g. 483920",max_chars=6)
        new_pw1=st.text_input("New Password",type="password",key="fp_pw1",placeholder="min 6 chars, letter + number")
        if new_pw1:
            score,label,color=password_strength(new_pw1); pct=int((score/4)*100)
            st.markdown(f"<div class='strength-bar-bg'><div class='strength-bar-fill' style='width:{pct}%;background:{color}'></div></div><span style='color:{color};font-size:.78rem'>Strength: <b>{label}</b></span>",unsafe_allow_html=True)
        new_pw2=st.text_input("Confirm New Password",type="password",key="fp_pw2",placeholder="Re-enter")
        if new_pw2 and new_pw1:
            if new_pw1==new_pw2: st.markdown("<span style='color:#00ff9d;font-size:.8rem'>✔ Passwords match</span>",unsafe_allow_html=True)
            else:                st.markdown("<span style='color:#ff4466;font-size:.8rem'>⚠ Passwords do not match</span>",unsafe_allow_html=True)
        st.markdown("")
        col_reset,col_back=st.columns([3,1])
        with col_reset:
            if st.button("Reset Password  ⟶",type="primary",use_container_width=True,key="btn_reset_pw"):
                otp=otp_input.strip(); username=st.session_state.fp_username
                if not otp: st.error("Please enter the reset code.")
                else:
                    result=verify_otp(email,otp)
                    if result=="invalid":  st.error("Invalid code. Check and retry.")
                    elif result=="expired": st.error("Code expired."); st.session_state.fp_step=1; st.rerun()
                    elif result=="used":    st.error("Code already used."); st.session_state.fp_step=1; st.rerun()
                    else:
                        perr=validate_password(new_pw1)
                        if perr:              st.error(f"⚠ {perr}")
                        elif new_pw1!=new_pw2: st.error("Passwords do not match.")
                        else:
                            consume_otp(email,otp); update_password(username,new_pw1)
                            st.success("Password reset! You can now sign in.")
                            for k in ["fp_step","fp_email","fp_username"]: st.session_state.pop(k,None)
                            time.sleep(1.2); st.rerun()
        with col_back:
            if st.button("← Resend",use_container_width=True,key="btn_fp_back"):
                st.session_state.fp_step=1; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE  — full width, no sidebar, form centred via columns
# ══════════════════════════════════════════════════════════════════════════════

def login_page():
    st.markdown(NEON_CSS,        unsafe_allow_html=True)
    st.markdown(LOGIN_EXTRA_CSS, unsafe_allow_html=True)

    # Wide blank | narrow form | wide blank  →  perfectly centred on any screen
    _, mid, _ = st.columns([2, 3, 2])
    with mid:
        st.markdown("<div class='auth-logo'>⬡</div>",  unsafe_allow_html=True)
        st.markdown("<div class='auth-title'>GROQ·AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='auth-sub'>STREAMLIT · LANGCHAIN · GROQ · GOOGLE SHEETS</div>", unsafe_allow_html=True)
        tab_si, tab_su, tab_fp = st.tabs(["  Sign In  ", "  Sign Up  ", "  Forgot Password  "])
        with tab_si: _signin_panel()
        with tab_su: _signup_panel()
        with tab_fp: _forgot_password_panel()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT APP  — centred content + working sidebar toggle
# ══════════════════════════════════════════════════════════════════════════════

def chat_app():
    st.markdown(NEON_CSS,       unsafe_allow_html=True)
    st.markdown(CHAT_EXTRA_CSS, unsafe_allow_html=True)

    username = st.session_state.username
    sessions = load_sessions(username)
    if not sessions:
        create_session(username, "Chat 1")
        sessions = load_sessions(username)

    if ("active_sid" not in st.session_state or
            not any(s["session_id"]==st.session_state.active_sid for s in sessions)):
        st.session_state.active_sid = sessions[0]["session_id"]

    if "sys_prompt_value" not in st.session_state: st.session_state.sys_prompt_value = DEFAULT_PROMPT
    if "_tone_idx"        not in st.session_state: st.session_state._tone_idx = 0

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"<div class='sidebar-username'>⬡ &nbsp;{username.upper()}</div>", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

        st.divider()
        st.markdown("<div class='sidebar-section'>Configuration</div>", unsafe_allow_html=True)
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_…").strip() or st.secrets.get("GROQ_API_KEY","")

        model = st.selectbox("Model",[
            "llama-3.3-70b-versatile","openai/gpt-oss-120b","qwen/qwen3-32b",
            "moonshotai/kimi-k2-instruct","groq/compound",
        ])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
        max_tokens  = st.slider("Max Tokens",  64, 2048, 640, 64)

        st.markdown("<div class='sidebar-section'>Personality</div>", unsafe_allow_html=True)
        _tone_options = ["Custom","Friendly","Strict","Teacher"]
        tone_preset   = st.selectbox("Tone Preset", _tone_options, index=st.session_state._tone_idx)
        new_tone_idx  = _tone_options.index(tone_preset)
        if new_tone_idx != st.session_state._tone_idx:
            st.session_state._tone_idx = new_tone_idx
            if tone_preset != "Custom": st.session_state.sys_prompt_value = TONE_MAP[tone_preset]

        new_prompt = st.text_area("System Prompt", value=st.session_state.sys_prompt_value, height=100)
        st.session_state.sys_prompt_value = new_prompt

        c1, c2 = st.columns(2)
        with c1:
            if st.button("↺ Reset", use_container_width=True):
                st.session_state.sys_prompt_value = DEFAULT_PROMPT
                st.session_state._tone_idx        = 0
                st.rerun()
        with c2:
            typing = st.checkbox("Typing FX", value=True)

        settings = dict(model=model, temperature=temperature, max_tokens=max_tokens,
                        system_prompt=st.session_state.sys_prompt_value, typing=typing)

        st.divider()
        st.markdown("<div class='sidebar-section'>Sessions</div>", unsafe_allow_html=True)
        c_name, c_btn = st.columns([3,1])
        with c_name:
            new_name = st.text_input("", placeholder="Session name…", label_visibility="collapsed", key="new_sess_input")
        with c_btn:
            if st.button("＋", use_container_width=True, help="New session"):
                ns = create_session(username, new_name.strip() or None)
                st.session_state.active_sid = ns["session_id"]; st.rerun()

        for sess in load_sessions(username):
            sid = sess["session_id"]; is_active = sid==st.session_state.active_sid
            c1, c2 = st.columns([5,1])
            with c1:
                label = ("▸ " if is_active else "  ") + sess["name"]
                if st.button(label, key=f"sel_{sid}", use_container_width=True):
                    st.session_state.active_sid = sid; st.rerun()
            with c2:
                if st.button("✕", key=f"del_{sid}", help="Delete"):
                    with st.spinner("Removing …"): delete_session(username, sid)
                    remaining = load_sessions(username)
                    if remaining: st.session_state.active_sid = remaining[0]["session_id"]
                    else:
                        ns = create_session(username,"Chat 1"); st.session_state.active_sid=ns["session_id"]
                    st.session_state.pop(f"lc_{username}_{sid}", None); st.rerun()

        st.divider()
        if st.button("⌫  Clear Active Chat", use_container_width=True):
            sid = st.session_state.active_sid
            with st.spinner("Clearing …"): clear_session_messages(username, sid)
            st.session_state.pop(f"lc_{username}_{sid}", None); st.rerun()

    # ── MAIN AREA ─────────────────────────────────────────────────────────────

    # ① Sidebar toggle — injected via components.html so the JS actually runs
    #    It uses window.parent.document to reach the real Streamlit page DOM
    components.html(SIDEBAR_TOGGLE_HTML, height=0, scrolling=False)

    # ② Page header padded right to avoid overlap with the floating toggle btn
    st.markdown(
        "<div style='margin-left:56px'>"
        "<div class='page-title'>⬡ GROQ·AI CHATBOT</div>"
        "<div class='page-subtitle'>STREAMLIT · LANGCHAIN · GROQ · GOOGLE SHEETS</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if not groq_key:
        st.error("Groq API Key missing — add GROQ_API_KEY to Streamlit Secrets.")
        st.stop()

    sessions   = load_sessions(username)
    active_sid = st.session_state.active_sid

    if not sessions:
        st.info("No sessions yet — click ＋ in the sidebar."); return

    tab_labels = [("▸ " if s["session_id"]==active_sid else "")+s["name"] for s in sessions]
    tabs = st.tabs(tab_labels)

    for tab, sess in zip(tabs, sessions):
        with tab:
            sid = sess["session_id"]; is_active = sid==active_sid
            msgs = load_msgs(username, sid)

            if not is_active:
                if st.button(f"▸ Switch to  {sess['name']}", key=f"sw_{sid}"):
                    st.session_state.active_sid = sid; st.rerun()

            with st.container(height=460, border=True):
                if not msgs:
                    st.markdown("<div class='empty-chat'>[ NO MESSAGES ] — say hello below</div>", unsafe_allow_html=True)
                else:
                    for m in msgs:
                        role = "user" if m["role"]=="user" else "assistant"
                        with st.chat_message(role):
                            st.write(m["content"])
                            if m.get("ts"): st.caption(str(m["ts"])[:19].replace("T","  "))

            if is_active:
                c_inp, c_btn = st.columns([8,1])
                with c_inp:
                    user_input = st.text_input("", key=f"inp_{sid}", placeholder="Type your message …", label_visibility="collapsed")
                with c_btn:
                    send = st.button("⟶", key=f"snd_{sid}", type="primary", use_container_width=True)

                if send and user_input.strip():
                    ui = user_input.strip()
                    with st.spinner("Saving …"):   add_msg(username, sid, "user", ui)
                    try:
                        with st.spinner("Thinking …"): response = generate_response(username, sid, ui, groq_key, settings)
                    except Exception as e:             response = f"Model error: {e}"
                    with st.spinner("Saving …"):   add_msg(username, sid, "assistant", response)
                    if settings["typing"] and response:
                        with st.chat_message("assistant"):
                            ph, typed = st.empty(), ""
                            for ch in response:
                                typed+=ch; ph.markdown(typed+"▌"); time.sleep(0.007)
                            ph.markdown(typed)
                    st.rerun()

                if msgs:
                    with st.expander("⬇  Export Chat History"):
                        export_json = json.dumps(
                            [{"role":m["role"],"content":m["content"],"time":str(m.get("ts",""))} for m in msgs],
                            ensure_ascii=False, indent=2)
                        export_txt = "\n\n".join(f"{'You' if m['role']=='user' else 'AI'}: {m['content']}" for m in msgs)
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            st.download_button("⬇ JSON", data=export_json, file_name=f"{sess['name']}.json",
                                               mime="application/json", key=f"dj_{sid}", use_container_width=True)
                        with ec2:
                            st.download_button("⬇ TXT",  data=export_txt,  file_name=f"{sess['name']}.txt",
                                               mime="text/plain", key=f"dt_{sid}", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("logged_in"):
    login_page()
else:
    chat_app()
