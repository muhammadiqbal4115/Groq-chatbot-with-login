import re
import json
import time
import random
import string
import hashlib
import smtplib
import streamlit as st
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
#  GOOGLE SHEETS CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🔗 Connecting to Google Sheets …")
def get_spreadsheet():
    creds  = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    client = gspread.authorize(creds)
    name   = st.secrets.get("SPREADSHEET_NAME", "GroqChatbotDB")
    try:
        sh = client.open(name)
    except gspread.SpreadsheetNotFound:
        sh = client.create(name)

    required = {
        # ── FIX 1: email column added ─────────────────────────────────────────
        "users":           ["username", "password_hash", "full_name", "email", "created_at"],
        "sessions":        ["username", "session_id", "name", "created", "updated"],
        "messages":        ["username", "session_id", "role", "content", "ts"],
        # ── FIX 3: sheet for password-reset OTPs ──────────────────────────────
        "password_resets": ["email", "otp", "expires_at", "used"],
    }
    existing = {w.title for w in sh.worksheets()}
    for title, headers in required.items():
        if title not in existing:
            w = sh.add_worksheet(title=title, rows=1000, cols=len(headers))
            w.append_row(headers, value_input_option="RAW")
    return sh


def ws(sheet_name: str):
    return get_spreadsheet().worksheet(sheet_name)


def all_rows(sheet_name: str) -> list:
    return ws(sheet_name).get_all_records()


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH — USERS
# ══════════════════════════════════════════════════════════════════════════════

def _hp(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_users() -> dict:
    """username → {password_hash, email, full_name}"""
    result = {}
    for r in all_rows("users"):
        result[r["username"]] = {
            "password_hash": r.get("password_hash", ""),
            "email":         r.get("email", "").lower().strip(),
            "full_name":     r.get("full_name", ""),
        }
    return result


def load_users() -> dict:
    return _cached_users()


def user_exists(username: str) -> bool:
    return username in load_users()


def email_exists(email: str) -> bool:
    """Case-insensitive check across all registered emails."""
    e = email.lower().strip()
    return any(v["email"] == e for v in load_users().values())


def get_username_by_email(email: str):
    """Return username whose email matches, or None."""
    e = email.lower().strip()
    for username, data in load_users().items():
        if data["email"] == e:
            return username
    return None


def auth_user(username: str, password: str) -> bool:
    u = load_users().get(username)
    return u is not None and u["password_hash"] == _hp(password)


def reg_user(username: str, password: str,
             full_name: str = "", email: str = "") -> bool:
    if user_exists(username) or (email and email_exists(email)):
        return False
    ws("users").append_row(
        [username, _hp(password), full_name,
         email.lower().strip(), datetime.now().isoformat()],
        value_input_option="RAW",
    )
    _cached_users.clear()
    return True


def update_password(username: str, new_password: str):
    """Overwrite the password_hash cell for the given user."""
    sheet = ws("users")
    data  = sheet.get_all_values()          # list-of-lists incl. header
    for i, row in enumerate(data[1:], start=2):
        if row[0] == username:
            sheet.update_cell(i, 2, _hp(new_password))
            break
    _cached_users.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 3 — PASSWORD RESET  (OTP via email)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_otp(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


def send_reset_email(to_email: str, otp: str) -> bool:
    """
    Send the OTP via Gmail SMTP.
    Requires in Streamlit Secrets:
        SMTP_EMAIL    = "yourapp@gmail.com"
        SMTP_PASSWORD = "your-app-password"   (Gmail App Password)
    """
    try:
        smtp_email = st.secrets["SMTP_EMAIL"]
        smtp_pass  = st.secrets["SMTP_PASSWORD"]
    except KeyError:
        st.error("⚠️  SMTP credentials not found in Streamlit Secrets "
                 "(SMTP_EMAIL / SMTP_PASSWORD).")
        return False

    subject = "🔐 Your Password Reset Code — Groq AI Chatbot"
    body    = f"""
Hi,

You requested a password reset for your Groq AI Chatbot account.

Your one-time code is:

    ┌─────────────┐
    │   {otp}    │
    └─────────────┘

This code expires in 10 minutes.
If you did not request this, please ignore this email.

— Groq AI Chatbot
    """.strip()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = smtp_email
    msg["To"]      = to_email
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_email, smtp_pass)
            server.sendmail(smtp_email, to_email, msg.as_string())
        return True
    except Exception as e:
        st.error(f"❌  Email send failed: {e}")
        return False


def store_otp(email: str, otp: str):
    """Write (or overwrite) the OTP row for this email."""
    sheet   = ws("password_resets")
    data    = sheet.get_all_values()
    expires = (datetime.now() + timedelta(minutes=10)).isoformat()
    e       = email.lower().strip()

    for i, row in enumerate(data[1:], start=2):
        if row and row[0].lower().strip() == e:
            sheet.update(f"A{i}:D{i}", [[e, otp, expires, "no"]])
            return
    sheet.append_row([e, otp, expires, "no"], value_input_option="RAW")


def verify_otp(email: str, otp: str) -> str:
    """
    Returns:
        "ok"      — valid and unused
        "expired" — past expiry time
        "used"    — already consumed
        "invalid" — not found or wrong code
    """
    e    = email.lower().strip()
    rows = all_rows("password_resets")
    for r in rows:
        if r.get("email", "").lower().strip() == e and r.get("otp") == otp:
            if r.get("used", "no") == "yes":
                return "used"
            if datetime.fromisoformat(str(r["expires_at"])) < datetime.now():
                return "expired"
            return "ok"
    return "invalid"


def consume_otp(email: str, otp: str):
    """Mark the OTP row as used so it cannot be replayed."""
    sheet = ws("password_resets")
    data  = sheet.get_all_values()
    e     = email.lower().strip()
    for i, row in enumerate(data[1:], start=2):
        if row and row[0].lower().strip() == e and row[1] == otp:
            sheet.update_cell(i, 4, "yes")
            break


# ══════════════════════════════════════════════════════════════════════════════
#  SESSIONS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=15, show_spinner=False)
def _cached_sessions(username: str) -> list:
    rows = [r for r in all_rows("sessions") if r["username"] == username]
    rows.sort(key=lambda r: r.get("updated", r.get("created", "")), reverse=True)
    return rows


def load_sessions(username: str) -> list:
    return _cached_sessions(username)


def _bust_sessions(username: str):
    _cached_sessions.clear()


def create_session(username: str, name: str = None) -> dict:
    existing = load_sessions(username)
    sid      = f"s{int(time.time() * 1000)}"
    now      = datetime.now().isoformat()
    sname    = name or f"Chat {len(existing) + 1}"
    ws("sessions").append_row([username, sid, sname, now, now],
                              value_input_option="RAW")
    _bust_sessions(username)
    return {"username": username, "session_id": sid, "name": sname,
            "created": now, "updated": now}


def _find_session_row(username: str, sid: str):
    data = ws("sessions").get_all_values()
    for i, row in enumerate(data[1:], start=2):
        if len(row) >= 2 and row[0] == username and row[1] == sid:
            return i
    return None


def update_session_timestamp(username: str, sid: str):
    row = _find_session_row(username, sid)
    if row:
        ws("sessions").update_cell(row, 5, datetime.now().isoformat())
    _bust_sessions(username)


def delete_session(username: str, sid: str):
    row = _find_session_row(username, sid)
    if row:
        ws("sessions").delete_rows(row)
    _delete_session_messages(username, sid)
    _bust_sessions(username)
    _cached_messages.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  MESSAGES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=10, show_spinner=False)
def _cached_messages(username: str, sid: str) -> list:
    return [r for r in all_rows("messages")
            if r["username"] == username and r["session_id"] == sid]


def load_msgs(username: str, sid: str) -> list:
    return _cached_messages(username, sid)


def add_msg(username: str, sid: str, role: str, content: str):
    ws("messages").append_row(
        [username, sid, role, content, datetime.now().isoformat()],
        value_input_option="RAW",
    )
    update_session_timestamp(username, sid)
    _cached_messages.clear()


def _delete_session_messages(username: str, sid: str):
    sheet = ws("messages")
    data  = sheet.get_all_values()
    rows_to_delete = [
        i for i, row in enumerate(data[1:], start=2)
        if len(row) >= 2 and row[0] == username and row[1] == sid
    ]
    for row_idx in reversed(rows_to_delete):
        sheet.delete_rows(row_idx)


def clear_session_messages(username: str, sid: str):
    _delete_session_messages(username, sid)
    _cached_messages.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN BRIDGE + LLM
# ══════════════════════════════════════════════════════════════════════════════

def get_lc_mem(username: str, sid: str) -> InMemoryChatMessageHistory:
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
    chain      = prompt | llm | StrOutputParser()
    mem        = get_lc_mem(username, sid)
    chat_chain = RunnableWithMessageHistory(
        chain, lambda _: mem,
        input_messages_key="input",
        history_messages_key="history",
    )
    return chat_chain.invoke(
        {"input": user_input, "system_prompt": settings["system_prompt"]},
        config={"configurable": {"session_id": sid}},
    )


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def validate_username(username: str):
    u = username.strip()
    if not u:                                   return "Username cannot be empty."
    if len(u) < 3:                              return "Username must be at least 3 characters."
    if len(u) > 32:                             return "Username must be 32 characters or fewer."
    if not re.match(r"^[a-zA-Z0-9_.\-]+$", u): return "Only letters, numbers, _ . - allowed."
    return None


def validate_email(email: str):
    e = email.strip()
    if not e:
        return "Email cannot be empty."
    if not re.match(r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$", e):
        return "Enter a valid email (e.g. name@example.com)."
    return None


def validate_password(password: str):
    if not password:                            return "Password cannot be empty."
    if len(password) < 6:                       return "Password must be at least 6 characters."
    if not re.search(r"[A-Za-z]", password):   return "Must contain at least one letter."
    if not re.search(r"[0-9]", password):       return "Must contain at least one number."
    return None


def password_strength(password: str):
    score = 0
    if len(password) >= 6:                   score += 1
    if len(password) >= 10:                  score += 1
    if re.search(r"[A-Z]", password):        score += 1
    if re.search(r"[^A-Za-z0-9]", password): score += 1
    labels = {0: "Very Weak", 1: "Weak", 2: "Fair", 3: "Strong", 4: "Very Strong"}
    colors = {0: "#e74c3c", 1: "#e67e22", 2: "#f1c40f", 3: "#2ecc71", 4: "#27ae60"}
    return score, labels[score], colors[score]


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE CSS
# ══════════════════════════════════════════════════════════════════════════════

_AUTH_CSS = """
<style>
  section[data-testid="stSidebar"] { display:none !important; }
  .auth-logo  { text-align:center; font-size:3.2rem; margin-bottom:.3rem; }
  .auth-title { text-align:center; font-size:1.7rem; font-weight:700;
                color:#e2e2f0; margin-bottom:.2rem; }
  .auth-sub   { text-align:center; color:#888; font-size:.88rem;
                margin-bottom:1.2rem; }
  .strength-bar-bg   { background:#2d2d4e; border-radius:6px;
                        height:6px; width:100%; margin:4px 0 2px; }
  .strength-bar-fill { height:6px; border-radius:6px;
                        transition:width .4s, background .4s; }
  .hint-pill { display:inline-block; background:#23234a;
               border:1px solid #3a3a6a; border-radius:20px;
               padding:2px 10px; font-size:.78rem; color:#aaa; margin:2px 3px; }
  .hint-ok   { border-color:#2ecc71; color:#2ecc71; background:#0d2e1a; }
  .info-box  { background:#1e3a5f; border:1px solid #2255a4; border-radius:8px;
               padding:.6rem 1rem; font-size:.82rem; color:#9ec5fe;
               margin-top:.8rem; text-align:center; }
  .otp-box   { background:#1a2e1a; border:1px solid #2ecc71; border-radius:8px;
               padding:.7rem 1rem; font-size:.85rem; color:#aaffaa;
               margin:.6rem 0; text-align:center; }
</style>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  SIGN IN PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _signin_panel():
    st.markdown("")
    un = st.text_input("👤  Username", key="si_un", placeholder="your-username")
    pw = st.text_input("🔒  Password", key="si_pw", type="password",
                       placeholder="••••••••")
    st.markdown("")

    if st.button("Sign In  →", type="primary",
                 use_container_width=True, key="btn_signin"):
        u = un.strip()
        if not u or not pw:
            st.error("⚠️  Please fill in both fields.")
        elif auth_user(u, pw):
            st.session_state.logged_in = True
            st.session_state.username  = u
            st.success(f"✅  Welcome back, **{u}**!")
            time.sleep(0.6)
            st.rerun()
        else:
            st.error("❌  Incorrect username or password.")

    st.markdown("<div class='info-box'>No account yet? Switch to the <b>Sign Up</b> tab →</div>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIGN UP PANEL  (FIX 1 — email field + uniqueness check)
# ══════════════════════════════════════════════════════════════════════════════

def _signup_panel():
    st.markdown("")
    full_name = st.text_input("🙂  Full Name (optional)", key="su_name",
                              placeholder="Jane Doe")

    # ── Username ──────────────────────────────────────────────────────────────
    su_un = st.text_input("👤  Username", key="su_un",
                          placeholder="3–32 chars · letters / numbers / _ . -")
    if su_un:
        err = validate_username(su_un)
        if err:
            st.markdown(f"<span style='color:#e74c3c;font-size:.82rem'>⚠ {err}</span>",
                        unsafe_allow_html=True)
        elif user_exists(su_un.strip()):
            st.markdown("<span style='color:#e74c3c;font-size:.82rem'>"
                        "⚠ Username already taken.</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#2ecc71;font-size:.82rem'>"
                        "✔ Username available!</span>", unsafe_allow_html=True)

    # ── Email (FIX 1) ─────────────────────────────────────────────────────────
    su_email = st.text_input("📧  Email", key="su_email",
                             placeholder="you@example.com")
    if su_email:
        e_err = validate_email(su_email)
        if e_err:
            st.markdown(f"<span style='color:#e74c3c;font-size:.82rem'>⚠ {e_err}</span>",
                        unsafe_allow_html=True)
        elif email_exists(su_email.strip()):
            st.markdown("<span style='color:#e74c3c;font-size:.82rem'>"
                        "⚠ Email already registered.</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#2ecc71;font-size:.82rem'>"
                        "✔ Email available!</span>", unsafe_allow_html=True)

    # ── Password ──────────────────────────────────────────────────────────────
    su_pw1 = st.text_input("🔒  Password", type="password", key="su_pw1",
                           placeholder="min 6 chars, letter + number")
    if su_pw1:
        score, label, color = password_strength(su_pw1)
        pct        = int((score / 4) * 100)
        has_len    = len(su_pw1) >= 6
        has_letter = bool(re.search(r"[A-Za-z]", su_pw1))
        has_digit  = bool(re.search(r"[0-9]", su_pw1))
        def pill(text, ok):
            cls = "hint-pill hint-ok" if ok else "hint-pill"
            return f"<span class='{cls}'>{'✔' if ok else '·'} {text}</span>"
        st.markdown(
            f"<div class='strength-bar-bg'>"
            f"<div class='strength-bar-fill' style='width:{pct}%;background:{color}'></div>"
            f"</div><span style='color:{color};font-size:.78rem'>"
            f"Strength: <b>{label}</b></span><br>"
            + pill("6+ chars", has_len)
            + pill("Letter",   has_letter)
            + pill("Number",   has_digit),
            unsafe_allow_html=True,
        )

    su_pw2 = st.text_input("🔒  Confirm Password", type="password", key="su_pw2",
                           placeholder="Re-enter your password")
    if su_pw2 and su_pw1:
        if su_pw1 == su_pw2:
            st.markdown("<span style='color:#2ecc71;font-size:.82rem'>✔ Passwords match</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#e74c3c;font-size:.82rem'>"
                        "⚠ Passwords do not match</span>", unsafe_allow_html=True)

    st.markdown("")
    agree = st.checkbox("I agree to the Terms of Service", key="su_agree")
    st.markdown("")

    if st.button("Create Account  →", type="primary",
                 use_container_width=True, key="btn_signup"):
        u     = su_un.strip()
        email = su_email.strip()
        p1    = su_pw1
        p2    = su_pw2

        err = validate_username(u)
        if err:
            st.error(f"⚠️  {err}")
        elif user_exists(u):
            st.error("❌  Username already taken.")
        elif validate_email(email):
            st.error(f"⚠️  {validate_email(email)}")
        elif email_exists(email):
            st.error("❌  Email already registered.")
        else:
            perr = validate_password(p1)
            if perr:
                st.error(f"⚠️  {perr}")
            elif p1 != p2:
                st.error("⚠️  Passwords do not match.")
            elif not agree:
                st.error("⚠️  Please accept the Terms of Service.")
            else:
                with st.spinner("Creating your account …"):
                    ok = reg_user(u, p1, full_name.strip(), email)
                if ok:
                    st.success(f"🎉  Account created! Welcome, **{u}**.")
                    st.session_state.logged_in = True
                    st.session_state.username  = u
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error("Something went wrong. Please try again.")


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 3 — FORGOT PASSWORD PANEL
#  Flow:  Step 1 → enter email → receive OTP
#         Step 2 → enter OTP + new password → done
# ══════════════════════════════════════════════════════════════════════════════

def _forgot_password_panel():
    st.markdown("")

    # Use session_state to track which step of the flow we're on
    if "fp_step" not in st.session_state:
        st.session_state.fp_step      = 1   # 1 = enter email, 2 = enter OTP + new pw
        st.session_state.fp_email     = ""
        st.session_state.fp_username  = ""

    # ── STEP 1: Enter email ───────────────────────────────────────────────────
    if st.session_state.fp_step == 1:
        st.markdown("#### 📧 Enter your registered email")
        st.caption("We'll send a 6-digit reset code to your inbox.")
        st.markdown("")

        fp_email = st.text_input("Email address", key="fp_email_input",
                                 placeholder="you@example.com")

        if st.button("Send Reset Code  →", type="primary",
                     use_container_width=True, key="btn_send_otp"):
            email = fp_email.strip()
            e_err = validate_email(email)
            if e_err:
                st.error(f"⚠️  {e_err}")
            elif not email_exists(email):
                # Don't reveal whether email is registered (security best practice)
                st.success("✅  If that email is registered, a reset code has been sent.")
            else:
                username = get_username_by_email(email)
                otp      = _generate_otp()
                store_otp(email, otp)
                with st.spinner("Sending email …"):
                    sent = send_reset_email(email, otp)
                if sent:
                    st.session_state.fp_step     = 2
                    st.session_state.fp_email    = email
                    st.session_state.fp_username = username
                    st.success("✅  Reset code sent! Check your inbox.")
                    time.sleep(0.8)
                    st.rerun()

        st.markdown("<div class='info-box'>Remembered your password? Go back to <b>Sign In →</b></div>",
                    unsafe_allow_html=True)

    # ── STEP 2: Enter OTP + new password ─────────────────────────────────────
    elif st.session_state.fp_step == 2:
        email = st.session_state.fp_email
        st.markdown(f"#### 🔐 Reset password for `{email}`")
        st.markdown(
            "<div class='otp-box'>📬 A 6-digit code was sent to your email. "
            "It expires in <b>10 minutes</b>.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

        otp_input = st.text_input("6-digit Code", key="fp_otp_input",
                                  placeholder="e.g. 483920", max_chars=6)
        new_pw1   = st.text_input("🔒  New Password", type="password",
                                  key="fp_pw1", placeholder="min 6 chars, letter + number")
        if new_pw1:
            score, label, color = password_strength(new_pw1)
            pct = int((score / 4) * 100)
            st.markdown(
                f"<div class='strength-bar-bg'>"
                f"<div class='strength-bar-fill' style='width:{pct}%;background:{color}'></div>"
                f"</div><span style='color:{color};font-size:.78rem'>"
                f"Strength: <b>{label}</b></span>",
                unsafe_allow_html=True,
            )

        new_pw2 = st.text_input("🔒  Confirm New Password", type="password",
                                key="fp_pw2", placeholder="Re-enter new password")
        if new_pw2 and new_pw1:
            if new_pw1 == new_pw2:
                st.markdown("<span style='color:#2ecc71;font-size:.82rem'>✔ Passwords match</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#e74c3c;font-size:.82rem'>"
                            "⚠ Passwords do not match</span>", unsafe_allow_html=True)

        st.markdown("")
        col_reset, col_back = st.columns([3, 1])

        with col_reset:
            if st.button("Reset Password  →", type="primary",
                         use_container_width=True, key="btn_reset_pw"):
                otp      = otp_input.strip()
                username = st.session_state.fp_username

                if not otp:
                    st.error("⚠️  Please enter the reset code.")
                else:
                    result = verify_otp(email, otp)
                    if result == "invalid":
                        st.error("❌  Invalid reset code. Please check and try again.")
                    elif result == "expired":
                        st.error("❌  This code has expired. Please request a new one.")
                        st.session_state.fp_step = 1
                        st.rerun()
                    elif result == "used":
                        st.error("❌  This code has already been used. Please request a new one.")
                        st.session_state.fp_step = 1
                        st.rerun()
                    else:  # "ok"
                        perr = validate_password(new_pw1)
                        if perr:
                            st.error(f"⚠️  {perr}")
                        elif new_pw1 != new_pw2:
                            st.error("⚠️  Passwords do not match.")
                        else:
                            consume_otp(email, otp)
                            update_password(username, new_pw1)
                            st.success("✅  Password reset successfully! You can now sign in.")
                            # Clear forgot-password state
                            for k in ["fp_step", "fp_email", "fp_username"]:
                                st.session_state.pop(k, None)
                            time.sleep(1.2)
                            st.rerun()

        with col_back:
            if st.button("← Resend", use_container_width=True, key="btn_fp_back"):
                st.session_state.fp_step = 1
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN / AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════

def login_page():
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("<div class='auth-logo'>🤖</div>",   unsafe_allow_html=True)
        st.markdown("<div class='auth-title'>Groq AI Chatbot</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='auth-sub'>Streamlit · LangChain · Groq · ☁️ Google Sheets</div>",
            unsafe_allow_html=True,
        )

        # FIX 3: three tabs now
        tab_si, tab_su, tab_fp = st.tabs(["🔑  Sign In", "📝  Sign Up", "🔓  Forgot Password"])
        with tab_si: _signin_panel()
        with tab_su: _signup_panel()
        with tab_fp: _forgot_password_panel()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT APP
# ══════════════════════════════════════════════════════════════════════════════

def chat_app():
    username = st.session_state.username

    sessions = load_sessions(username)
    if not sessions:
        create_session(username, "Chat 1")
        sessions = load_sessions(username)

    if ("active_sid" not in st.session_state or
            not any(s["session_id"] == st.session_state.active_sid for s in sessions)):
        st.session_state.active_sid = sessions[0]["session_id"]

    # ── Seed prompt + tone index once in session_state ───────────────────────
    if "sys_prompt_value" not in st.session_state:
        st.session_state.sys_prompt_value = DEFAULT_PROMPT
    if "_tone_idx" not in st.session_state:
        st.session_state._tone_idx = 0   # 0 = "Custom"

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"### 👤 {username}")
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.divider()
        st.header("⚙️ Settings")

        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="sk-… (or set in Streamlit Secrets)",
        ).strip() or st.secrets.get("GROQ_API_KEY", "")

        model = st.selectbox("Model", [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
        ])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
        max_tokens  = st.slider("Max Tokens",  64, 2048, 640, 64)

        # ── Tone preset — controlled via index, no widget key ────────────────
        _tone_options = ["Custom", "Friendly", "Strict", "Teacher"]
        tone_preset   = st.selectbox(
            "Tone Preset",
            _tone_options,
            index=st.session_state._tone_idx,   # driven by session state
        )
        # When user picks a preset, update index + push prompt text
        new_tone_idx = _tone_options.index(tone_preset)
        if new_tone_idx != st.session_state._tone_idx:
            st.session_state._tone_idx = new_tone_idx
            if tone_preset != "Custom":
                st.session_state.sys_prompt_value = TONE_MAP[tone_preset]

        # System prompt text area — value driven by session state
        new_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.sys_prompt_value,
            height=110,
        )
        # Keep in sync when user types manually
        st.session_state.sys_prompt_value = new_prompt

        # Reset button — sets prompt back to default AND resets selectbox to "Custom"
        if st.button("↺ Reset Prompt"):
            st.session_state.sys_prompt_value = DEFAULT_PROMPT
            st.session_state._tone_idx        = 0   # back to "Custom"
            st.rerun()

        typing = st.checkbox("Enable typing effect", value=True)

        settings = dict(
            model=model, temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=st.session_state.sys_prompt_value,
            typing=typing,
        )

        st.divider()
        st.header("💬 Sessions")

        c_name, c_btn = st.columns([3, 1])
        with c_name:
            new_name = st.text_input(
                "", placeholder="Session name…",
                label_visibility="collapsed", key="new_sess_input",
            )
        with c_btn:
            if st.button("➕", use_container_width=True, help="New session"):
                ns = create_session(username, new_name.strip() or None)
                st.session_state.active_sid = ns["session_id"]
                st.rerun()

        st.markdown("")
        for sess in load_sessions(username):
            sid       = sess["session_id"]
            is_active = sid == st.session_state.active_sid
            c1, c2    = st.columns([5, 1])
            with c1:
                label = ("▶ " if is_active else "   ") + sess["name"]
                if st.button(label, key=f"sel_{sid}", use_container_width=True):
                    st.session_state.active_sid = sid
                    st.rerun()
            with c2:
                if st.button("🗑", key=f"del_{sid}", help="Delete session"):
                    with st.spinner("Deleting …"):
                        delete_session(username, sid)
                    remaining = load_sessions(username)
                    if remaining:
                        st.session_state.active_sid = remaining[0]["session_id"]
                    else:
                        ns = create_session(username, "Chat 1")
                        st.session_state.active_sid = ns["session_id"]
                    st.session_state.pop(f"lc_{username}_{sid}", None)
                    st.rerun()

        st.divider()
        if st.button("🧹 Clear Active Chat", use_container_width=True):
            sid = st.session_state.active_sid
            with st.spinner("Clearing …"):
                clear_session_messages(username, sid)
            st.session_state.pop(f"lc_{username}_{sid}", None)
            st.rerun()

    # ── MAIN AREA ─────────────────────────────────────────────────────────────
    st.title("🤖 Groq AI Chatbot")
    st.caption(
        f"Streamlit · LangChain · Groq · ☁️ Google Sheets  |  👤 **{username}**"
    )

    if not groq_key:
        st.error("🔑 Groq API Key missing. Add GROQ_API_KEY to Streamlit Secrets.")
        st.stop()

    sessions   = load_sessions(username)
    active_sid = st.session_state.active_sid

    if not sessions:
        st.info("No sessions yet — click ➕ in the sidebar.")
        return

    tab_labels = [
        ("▶ " if s["session_id"] == active_sid else "") + s["name"]
        for s in sessions
    ]
    tabs = st.tabs(tab_labels)

    for tab, sess in zip(tabs, sessions):
        with tab:
            sid       = sess["session_id"]
            is_active = sid == active_sid
            msgs      = load_msgs(username, sid)

            if not is_active:
                if st.button(f"▶ Switch to **{sess['name']}**", key=f"sw_{sid}"):
                    st.session_state.active_sid = sid
                    st.rerun()

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
                                st.caption(f"🕐 {str(m['ts'])[:19].replace('T', '  ')}")

            if is_active:
                c_inp, c_btn = st.columns([7, 1])
                with c_inp:
                    user_input = st.text_input(
                        "", key=f"inp_{sid}",
                        placeholder="Type your message and press Send …",
                        label_visibility="collapsed",
                    )
                with c_btn:
                    send = st.button("Send ➤", key=f"snd_{sid}",
                                     type="primary", use_container_width=True)

                if send and user_input.strip():
                    ui = user_input.strip()
                    with st.spinner("💾 Saving …"):
                        add_msg(username, sid, "user", ui)
                    try:
                        with st.spinner("🤔 Thinking …"):
                            response = generate_response(
                                username, sid, ui, groq_key, settings
                            )
                    except Exception as e:
                        response = f"❌ Model error: {e}"
                    with st.spinner("💾 Saving …"):
                        add_msg(username, sid, "assistant", response)

                    if settings["typing"] and response:
                        with st.chat_message("assistant"):
                            ph, typed = st.empty(), ""
                            for ch in response:
                                typed += ch
                                ph.markdown(typed + "▌")
                                time.sleep(0.007)
                            ph.markdown(typed)
                    st.rerun()

                if msgs:
                    with st.expander("⬇️ Export Chat History"):
                        export_json = json.dumps(
                            [{"role": m["role"], "content": m["content"],
                              "time": str(m.get("ts", ""))} for m in msgs],
                            ensure_ascii=False, indent=2,
                        )
                        export_txt = "\n\n".join(
                            f"{'You' if m['role'] == 'user' else 'AI'}: {m['content']}"
                            for m in msgs
                        )
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            st.download_button(
                                "📥 Download JSON", data=export_json,
                                file_name=f"{sess['name']}.json",
                                mime="application/json",
                                key=f"dj_{sid}", use_container_width=True,
                            )
                        with ec2:
                            st.download_button(
                                "📥 Download TXT", data=export_txt,
                                file_name=f"{sess['name']}.txt",
                                mime="text/plain",
                                key=f"dt_{sid}", use_container_width=True,
                            )


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.get("logged_in"):
    login_page()
else:
    chat_app()
