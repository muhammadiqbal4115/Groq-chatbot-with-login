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
#  LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════

def login_page() -> None:

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
      section[data-testid="stSidebar"] { display: none; }
      .login-header { text-align: center; padding: 2rem 0 0.5rem; }
      .login-sub    { text-align: center; color: #888; margin-bottom: 1.5rem; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("<div class='login-header'><h1>🤖 Groq AI Chatbot</h1></div>",
                    unsafe_allow_html=True)
        st.markdown("<div class='login-sub'>Sign in to start chatting</div>",
                    unsafe_allow_html=True)
        st.divider()

        tab_login, tab_register = st.tabs(["🔑  Login", "📝  Register"])

        # ── Login ─────────────────────────────────────────────────────────────
        with tab_login:
            un = st.text_input("Username", key="l_un", placeholder="your-username")
            pw = st.text_input("Password", type="password", key="l_pw",
                               placeholder="••••••••")

            if st.button("Login", type="primary", use_container_width=True, key="btn_login"):
                if not un.strip() or not pw:
                    st.error("Please fill in both fields.")
                elif auth_user(un.strip(), pw):
                    st.session_state.logged_in = True
                    st.session_state.username  = un.strip()
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password.")

            st.caption("Demo account → **admin** / **admin123**")

        # ── Register ──────────────────────────────────────────────────────────
        with tab_register:
            nu  = st.text_input("Choose Username", key="r_un", placeholder="your-username")
            np1 = st.text_input("Password (min 4 chars)", type="password",
                                key="r_pw1", placeholder="••••••••")
            np2 = st.text_input("Confirm Password", type="password",
                                key="r_pw2", placeholder="••••••••")

            if st.button("Create Account", type="primary",
                         use_container_width=True, key="btn_reg"):
                if not nu.strip():
                    st.error("Username cannot be empty.")
                elif len(np1) < 4:
                    st.error("Password must be at least 4 characters.")
                elif np1 != np2:
                    st.error("Passwords don't match.")
                elif reg_user(nu.strip(), np1):
                    st.success("✅ Account created! Please log in.")
                else:
                    st.error("Username already taken — choose another.")


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
                                "📥 Download
