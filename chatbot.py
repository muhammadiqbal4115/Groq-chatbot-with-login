# ╔══════════════════════════════════════════════════════════════╗
# ║         Groq AI Chatbot — with Login + Persistent History   ║
# ║         Built with Streamlit · LangChain · Groq             ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import json
import time
import hashlib
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory


# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
ENV_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

st.set_page_config(
    page_title="🤖 Groq AI Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LAYER — files stored in  ./chat_data/
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR   = "chat_data"
USERS_FILE = os.path.join(DATA_DIR, "users.json")
HIST_DIR   = os.path.join(DATA_DIR, "histories")
os.makedirs(HIST_DIR, exist_ok=True)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _hp(password: str) -> str:
    """SHA-256 hash a password."""
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    # Seed a default admin account on first run
    default = {"admin": _hp("admin123")}
    with open(USERS_FILE, "w") as f:
        json.dump(default, f, indent=2)
    return default


def save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def auth_user(username: str, password: str) -> bool:
    return load_users().get(username) == _hp(password)


def reg_user(username: str, password: str) -> bool:
    users = load_users()
    if username in users:
        return False
    users[username] = _hp(password)
    save_users(users)
    return True


# ── Session helpers ────────────────────────────────────────────────────────────

def udir(username: str) -> str:
    """Return (and create) the per-user history directory."""
    d = os.path.join(HIST_DIR, username)
    os.makedirs(d, exist_ok=True)
    return d


def _sf(username: str) -> str:
    """Path to the sessions index file for a user."""
    return os.path.join(udir(username), "_sessions.json")


def load_sessions(username: str) -> list:
    p = _sf(username)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return []


def save_sessions(username: str, sessions: list) -> None:
    with open(_sf(username), "w") as f:
        json.dump(sessions, f, indent=2)


def create_session(username: str, name: str = None) -> dict:
    sessions = load_sessions(username)
    sid = f"s{int(time.time() * 1000)}"
    session = {
        "id":      sid,
        "name":    name or f"Chat {len(sessions) + 1}",
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
    }
    sessions.insert(0, session)          # newest first
    save_sessions(username, sessions)
    return session


def rename_session(username: str, sid: str, new_name: str) -> None:
    sessions = load_sessions(username)
    for s in sessions:
        if s["id"] == sid:
            s["name"] = new_name
            break
    save_sessions(username, sessions)


def delete_session(username: str, sid: str) -> None:
    sessions = [s for s in load_sessions(username) if s["id"] != sid]
    save_sessions(username, sessions)
    hist_path = os.path.join(udir(username), f"{sid}.json")
    if os.path.exists(hist_path):
        os.remove(hist_path)


# ── Message helpers ────────────────────────────────────────────────────────────

def _hf(username: str, sid: str) -> str:
    """Path to the JSON history file for one session."""
    return os.path.join(udir(username), f"{sid}.json")


def load_msgs(username: str, sid: str) -> list:
    p = _hf(username, sid)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return []


def save_msgs(username: str, sid: str, msgs: list) -> None:
    with open(_hf(username, sid), "w") as f:
        json.dump(msgs, f, ensure_ascii=False, indent=2)
    # bump the session's updated timestamp
    sessions = load_sessions(username)
    for s in sessions:
        if s["id"] == sid:
            s["updated"] = datetime.now().isoformat()
    save_sessions(username, sessions)


def add_msg(username: str, sid: str, role: str, content: str) -> None:
    msgs = load_msgs(username, sid)
    msgs.append({
        "role":    role,
        "content": content,
        "ts":      datetime.now().isoformat(),
    })
    save_msgs(username, sid, msgs)


# ── LangChain in-memory bridge ────────────────────────────────────────────────

def get_lc_mem(username: str, sid: str) -> InMemoryChatMessageHistory:
    """
    Return a (cached) InMemoryChatMessageHistory seeded from disk.
    Lives in st.session_state so it survives Streamlit reruns within
    the same browser session.
    """
    key = f"lc_{username}_{sid}"
    if key not in st.session_state:
        h = InMemoryChatMessageHistory()
        for m in load_msgs(username, sid):
            if m["role"] == "user":
                h.add_user_message(m["content"])
            else:
                h.add_ai_message(m["content"])
        st.session_state[key] = h
    return st.session_state[key]


# ── LLM call ──────────────────────────────────────────────────────────────────

def generate_response(username: str, sid: str, user_input: str,
                      api_key: str, settings: dict) -> str:
    llm = ChatGroq(
        groq_api_key=api_key,
        model=settings["model"],
        temperature=settings["temperature"],
        max_tokens=settings["max_tokens"],
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_prompt}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain        = prompt | llm | StrOutputParser()
    mem          = get_lc_mem(username, sid)
    chat_chain   = RunnableWithMessageHistory(
        chain,
        lambda _: mem,
        input_messages_key="input",
        history_messages_key="history",
    )
    return chat_chain.invoke(
        {"input": user_input, "system_prompt": settings["system_prompt"]},
        config={"configurable": {"session_id": sid}},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH HELPERS — validation
# ══════════════════════════════════════════════════════════════════════════════

import re

def validate_username(username: str) -> str | None:
    """Return an error string or None if valid."""
    u = username.strip()
    if not u:
        return "Username cannot be empty."
    if len(u) < 3:
        return "Username must be at least 3 characters."
    if len(u) > 32:
        return "Username must be 32 characters or fewer."
    if not re.match(r"^[a-zA-Z0-9_.-]+$", u):
        return "Username may only contain letters, numbers, _, . or -"
    return None


def validate_password(password: str) -> str | None:
    """Return an error string or None if valid."""
    if not password:
        return "Password cannot be empty."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    if not re.search(r"[A-Za-z]", password):
        return "Password must contain at least one letter."
    if not re.search(r"[0-9]", password):
        return "Password must contain at least one number."
    return None


def password_strength(password: str) -> tuple[int, str, str]:
    """Return (score 0-4, label, colour)."""
    score = 0
    if len(password) >= 6:   score += 1
    if len(password) >= 10:  score += 1
    if re.search(r"[A-Z]", password): score += 1
    if re.search(r"[^A-Za-z0-9]", password): score += 1
    labels = {0: "Very Weak", 1: "Weak", 2: "Fair", 3: "Strong", 4: "Very Strong"}
    colors = {0: "#e74c3c", 1: "#e67e22", 2: "#f1c40f", 3: "#2ecc71", 4: "#27ae60"}
    return score, labels[score], colors[score]


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE  (Sign In  +  Sign Up)
# ══════════════════════════════════════════════════════════════════════════════

_AUTH_CSS = """
<style>
  /* hide sidebar on auth screen */
  section[data-testid="stSidebar"] { display:none !important; }

  /* card wrapper */
  .auth-card {
      background: #1a1a2e;
      border: 1px solid #2d2d4e;
      border-radius: 16px;
      padding: 2.4rem 2rem 2rem;
      margin-top: 1rem;
  }
  .auth-logo  { text-align:center; font-size:3.2rem; margin-bottom:.3rem; }
  .auth-title { text-align:center; font-size:1.7rem; font-weight:700;
                color:#e2e2f0; margin-bottom:.25rem; }
  .auth-sub   { text-align:center; color:#888; font-size:.88rem;
                margin-bottom:1.4rem; }

  /* tabs override */
  button[data-baseweb="tab"] {
      font-size: .95rem !important;
      font-weight: 600 !important;
  }

  /* strength bar */
  .strength-bar-bg {
      background:#2d2d4e; border-radius:6px;
      height:6px; width:100%; margin:4px 0 2px;
  }
  .strength-bar-fill {
      height:6px; border-radius:6px;
      transition: width .4s ease, background .4s ease;
  }
  .strength-label { font-size:.78rem; }

  /* hint pills */
  .hint-pill {
      display:inline-block;
      background:#23234a; border:1px solid #3a3a6a;
      border-radius:20px; padding:2px 10px;
      font-size:.78rem; color:#aaa; margin:2px 3px;
  }
  .hint-ok { border-color:#2ecc71; color:#2ecc71; background:#0d2e1a; }

  /* divider with text */
  .or-divider {
      display:flex; align-items:center; gap:.8rem;
      margin:1rem 0; color:#555; font-size:.82rem;
  }
  .or-divider::before, .or-divider::after {
      content:""; flex:1; height:1px; background:#2d2d4e;
  }

  /* demo badge */
  .demo-badge {
      background:#1e3a5f; border:1px solid #2255a4;
      border-radius:8px; padding:.55rem .9rem;
      font-size:.82rem; color:#9ec5fe; margin-top:.8rem;
      text-align:center;
  }
</style>
"""


def _signin_panel() -> None:
    """Render the Sign In form."""
    st.markdown("")
    un = st.text_input("👤  Username", key="si_un",
                       placeholder="Enter your username")
    pw = st.text_input("🔒  Password", type="password", key="si_pw",
                       placeholder="Enter your password")

    col_rem, _ = st.columns([1, 2])
    with col_rem:
        remember = st.checkbox("Remember me", key="si_rem", value=True)

    st.markdown("")
    clicked = st.button("Sign In  →", type="primary",
                        use_container_width=True, key="btn_signin")

    if clicked:
        u = un.strip()
        if not u or not pw:
            st.error("⚠️  Please fill in both fields.")
        elif auth_user(u, pw):
            st.session_state.logged_in  = True
            st.session_state.username   = u
            st.session_state.remembered = remember
            st.success(f"✅  Welcome back, **{u}**!")
            time.sleep(0.6)
            st.rerun()
        else:
            st.error("❌  Incorrect username or password.")

    st.markdown("""<div class='demo-badge'>
        🔑 Demo account &nbsp;·&nbsp; <b>admin</b> / <b>admin123</b>
    </div>""", unsafe_allow_html=True)


def _signup_panel() -> None:
    """Render the Sign Up form with live validation feedback."""
    st.markdown("")

    # ── Full name (optional) ──────────────────────────────────────────────────
    full_name = st.text_input("🙂  Full Name (optional)", key="su_name",
                              placeholder="Jane Doe")

    # ── Username ──────────────────────────────────────────────────────────────
    su_un = st.text_input("👤  Username", key="su_un",
                          placeholder="min 3 chars, letters/numbers/_ .-")
    un_err = validate_username(su_un) if su_un else None
    if su_un:
        if un_err:
            st.markdown(f"<span style='color:#e74c3c;font-size:.82rem'>⚠ {un_err}</span>",
                        unsafe_allow_html=True)
        else:
            # Check availability live
            if su_un.strip() in load_users():
                st.markdown("<span style='color:#e74c3c;font-size:.82rem'>"
                            "⚠ Username already taken.</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#2ecc71;font-size:.82rem'>"
                            "✔ Username available!</span>",
                            unsafe_allow_html=True)

    # ── Password ──────────────────────────────────────────────────────────────
    su_pw1 = st.text_input("🔒  Password", type="password", key="su_pw1",
                           placeholder="min 6 chars, letter + number")

    # Strength meter
    if su_pw1:
        score, label, color = password_strength(su_pw1)
        pct = int((score / 4) * 100)
        st.markdown(
            f"<div class='strength-bar-bg'>"
            f"  <div class='strength-bar-fill'"
            f"       style='width:{pct}%;background:{color}'></div>"
            f"</div>"
            f"<span class='strength-label' style='color:{color}'>"
            f"  Password strength: <b>{label}</b></span>",
            unsafe_allow_html=True,
        )
        # Requirement pills
        has_len    = len(su_pw1) >= 6
        has_letter = bool(re.search(r"[A-Za-z]", su_pw1))
        has_digit  = bool(re.search(r"[0-9]",    su_pw1))
        def pill(text, ok):
            cls = "hint-pill hint-ok" if ok else "hint-pill"
            return f"<span class='{cls}'>{'✔' if ok else '·'} {text}</span>"
        st.markdown(
            pill("6+ chars", has_len) +
            pill("Letter",   has_letter) +
            pill("Number",   has_digit),
            unsafe_allow_html=True,
        )

    # ── Confirm password ──────────────────────────────────────────────────────
    su_pw2 = st.text_input("🔒  Confirm Password", type="password", key="su_pw2",
                           placeholder="Re-enter your password")
    if su_pw2 and su_pw1:
        if su_pw1 == su_pw2:
            st.markdown("<span style='color:#2ecc71;font-size:.82rem'>"
                        "✔ Passwords match</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#e74c3c;font-size:.82rem'>"
                        "⚠ Passwords do not match</span>", unsafe_allow_html=True)

    # ── Terms ─────────────────────────────────────────────────────────────────
    st.markdown("")
    agree = st.checkbox("I agree to the Terms of Service and Privacy Policy",
                        key="su_agree")

    st.markdown("")
    clicked = st.button("Create Account  →", type="primary",
                        use_container_width=True, key="btn_signup")

    if clicked:
        u  = su_un.strip()
        p1 = su_pw1
        p2 = su_pw2

        # ── Validation gate ────────────────────────────────────────────────
        err = validate_username(u)
        if err:
            st.error(f"⚠️  {err}")
        elif u in load_users():
            st.error("❌  Username already taken — please choose another.")
        else:
            perr = validate_password(p1)
            if perr:
                st.error(f"⚠️  {perr}")
            elif p1 != p2:
                st.error("⚠️  Passwords do not match.")
            elif not agree:
                st.error("⚠️  Please accept the Terms of Service to continue.")
            else:
                # ── All good — create account ──────────────────────────────
                if reg_user(u, p1):
                    # Persist optional display name
                    if full_name.strip():
                        dn_file = os.path.join(DATA_DIR, "display_names.json")
                        dns = {}
                        if os.path.exists(dn_file):
                            with open(dn_file) as f:
                                dns = json.load(f)
                        dns[u] = full_name.strip()
                        with open(dn_file, "w") as f:
                            json.dump(dns, f, indent=2)

                    st.success(f"🎉  Account created! Welcome, **{u}**.")
                    st.session_state.logged_in = True
                    st.session_state.username  = u
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Something went wrong. Please try again.")


def login_page() -> None:
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("<div class='auth-logo'>🤖</div>",  unsafe_allow_html=True)
        st.markdown("<div class='auth-title'>Groq AI Chatbot</div>", unsafe_allow_html=True)
        st.markdown("<div class='auth-sub'>Powered by Streamlit · LangChain · Groq</div>",
                    unsafe_allow_html=True)

        tab_signin, tab_signup = st.tabs(["🔑  Sign In", "📝  Sign Up"])

        with tab_signin:
            _signin_panel()

        with tab_signup:
            _signup_panel()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT APP
# ══════════════════════════════════════════════════════════════════════════════

def chat_app() -> None:

    username = st.session_state.username

    # Ensure the user always has at least one session
    sessions = load_sessions(username)
    if not sessions:
        create_session(username, "Chat 1")
        sessions = load_sessions(username)

    # Default to the most recent session
    if ("active_sid" not in st.session_state or
            not any(s["id"] == st.session_state.active_sid for s in sessions)):
        st.session_state.active_sid = sessions[0]["id"]

    # ══════════ SIDEBAR ══════════════════════════════════════════════════════

    with st.sidebar:

        # ── User header ───────────────────────────────────────────────────────
        st.markdown(f"### 👤 {username}")
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.divider()

        # ── Model settings ────────────────────────────────────────────────────
        st.header("⚙️ Settings")

        api_input = st.text_input(
            "Groq API Key (optional)",
            type="password",
            placeholder="sk-… (or set in .env)",
        )
        groq_key = api_input.strip() if api_input.strip() else ENV_GROQ_API_KEY

        model = st.selectbox(
            "Model",
            [
                "llama-3.3-70b-versatile",
                "openai/gpt-oss-120b",
                "qwen/qwen3-32b",
            ],
            index=0,
        )
        temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.4, 0.1)
        max_tokens  = st.slider("Max Tokens (Reply length)", 64, 2048, 640, 64)

        tone_preset = st.selectbox(
            "Tone Preset", ["Custom", "Friendly", "Strict", "Teacher"]
        )
        tone_map = {
            "Friendly": "You are a friendly AI assistant. Respond warmly and politely.",
            "Strict":   "You are strict and professional. Give short, precise answers.",
            "Teacher":  "You are a patient teacher. Explain concepts clearly with examples.",
        }
        default_prompt = "You are a helpful AI Assistant. Be clear, correct and concise."
        sys_prompt = st.text_area(
            "System Prompt",
            value=tone_map.get(tone_preset, default_prompt),
            height=110,
        )
        if st.button("↺ Reset Prompt"):
            st.rerun()

        typing = st.checkbox("Enable typing effect", value=True)

        # Pack settings into a dict for easy passing
        settings = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=sys_prompt,
            typing=typing,
        )

        st.divider()

        # ── Session management ────────────────────────────────────────────────
        st.header("💬 Sessions")

        c_name, c_btn = st.columns([3, 1])
        with c_name:
            new_sess_name = st.text_input(
                "", placeholder="Session name…",
                label_visibility="collapsed", key="new_sess_input"
            )
        with c_btn:
            if st.button("➕", help="Create new session", use_container_width=True):
                ns = create_session(username, new_sess_name.strip() or None)
                st.session_state.active_sid = ns["id"]
                st.rerun()

        st.markdown("")   # a little vertical space

        for sess in load_sessions(username):
            is_active = sess["id"] == st.session_state.active_sid
            col_name, col_del = st.columns([5, 1])

            with col_name:
                label = ("▶ " if is_active else "   ") + sess["name"]
                if st.button(label, key=f"sel_{sess['id']}", use_container_width=True):
                    st.session_state.active_sid = sess["id"]
                    st.rerun()

            with col_del:
                if st.button("🗑", key=f"del_{sess['id']}", help="Delete session"):
                    delete_session(username, sess["id"])
                    remaining = load_sessions(username)   # already deleted from disk
                    if remaining:
                        st.session_state.active_sid = remaining[0]["id"]
                    else:
                        ns = create_session(username, "Chat 1")
                        st.session_state.active_sid = ns["id"]
                    # drop cached LC memory for deleted session
                    st.session_state.pop(f"lc_{username}_{sess['id']}", None)
                    st.rerun()

        st.divider()

        if st.button("🧹 Clear Active Chat", use_container_width=True):
            sid = st.session_state.active_sid
            save_msgs(username, sid, [])
            st.session_state.pop(f"lc_{username}_{sid}", None)
            st.rerun()

    # ══════════ MAIN AREA ════════════════════════════════════════════════════

    st.title("🤖 Groq AI Chatbot")
    st.caption(
        f"Streamlit · LangChain · Groq  |  👤 **{username}**"
    )

    if not groq_key:
        st.error(
            "🔑 Groq API Key is missing. "
            "Add it to your `.env` file (GROQ_API_KEY=...) or paste it in the sidebar."
        )
        st.stop()

    # Reload session list (might have changed)
    sessions    = load_sessions(username)
    active_sid  = st.session_state.active_sid

    if not sessions:
        st.info("No sessions found. Click ➕ in the sidebar to create one.")
        return

    # ── Tab bar — one tab per session ────────────────────────────────────────
    tab_labels = [
        ("▶ " if s["id"] == active_sid else "") + s["name"]
        for s in sessions
    ]
    tabs = st.tabs(tab_labels)

    for tab, sess in zip(tabs, sessions):
        with tab:
            sid       = sess["id"]
            is_active = sid == active_sid
            msgs      = load_msgs(username, sid)

            # Switch-to button for non-active sessions
            if not is_active:
                if st.button(
                    f"▶ Switch to **{sess['name']}**",
                    key=f"sw_{sid}",
                ):
                    st.session_state.active_sid = sid
                    st.rerun()

            # ── Scrollable message history ────────────────────────────────────
            with st.container(height=460, border=True):
                if not msgs:
                    st.markdown(
                        "<div style='text-align:center;color:#999;"
                        "padding-top:90px;font-size:1.1rem'>"
                        "💬 No messages yet — say hello below!</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    for m in msgs:
                        role = "user" if m["role"] == "user" else "assistant"
                        with st.chat_message(role):
                            st.write(m["content"])
                            if m.get("ts"):
                                ts = m["ts"][:19].replace("T", "  ")
                                st.caption(f"🕐 {ts}")

            # ── Input row (active session only) ───────────────────────────────
            if is_active:
                col_inp, col_btn = st.columns([7, 1])
                with col_inp:
                    user_input = st.text_input(
                        "",
                        key=f"inp_{sid}",
                        placeholder="Type your message and press Send …",
                        label_visibility="collapsed",
                    )
                with col_btn:
                    send = st.button(
                        "Send ➤", key=f"snd_{sid}",
                        type="primary", use_container_width=True
                    )

                # ── Handle send ───────────────────────────────────────────────
                if send and user_input.strip():
                    ui = user_input.strip()

                    # Persist user message
                    add_msg(username, sid, "user", ui)

                    # Call LLM
                    try:
                        with st.spinner("🤔 Thinking …"):
                            response = generate_response(
                                username, sid, ui, groq_key, settings
                            )
                    except Exception as e:
                        response = f"❌ Model error: {e}"

                    # Persist AI response
                    add_msg(username, sid, "assistant", response)

                    # Optional typing animation (visible briefly before rerun)
                    if settings["typing"] and response:
                        with st.chat_message("assistant"):
                            ph     = st.empty()
                            typed  = ""
                            for ch in response:
                                typed += ch
                                ph.markdown(typed + "▌")
                                time.sleep(0.007)
                            ph.markdown(typed)

                    st.rerun()

                # ── Export ────────────────────────────────────────────────────
                if msgs:
                    with st.expander("⬇️ Export Chat History"):
                        export_json = json.dumps(
                            [
                                {
                                    "role":    m["role"],
                                    "content": m["content"],
                                    "time":    m.get("ts", ""),
                                }
                                for m in msgs
                            ],
                            ensure_ascii=False,
                            indent=2,
                        )
                        export_txt = "\n\n".join(
                            f"{'You' if m['role'] == 'user' else 'AI'}: {m['content']}"
                            for m in msgs
                        )
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            st.download_button(
                                "📥 Download JSON",
                                data=export_json,
                                file_name=f"{sess['name']}.json",
                                mime="application/json",
                                key=f"dj_{sid}",
                                use_container_width=True,
                            )
                        with ec2:
                            st.download_button(
                                "📥 Download TXT",
                                data=export_txt,
                                file_name=f"{sess['name']}.txt",
                                mime="text/plain",
                                key=f"dt_{sid}",
                                use_container_width=True,
                            )


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("logged_in"):
    login_page()
else:
    chat_app()
