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
#  GLOBAL NEON LIGHT BLUE CSS — Applied on every render
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
/* ─── HIDE ALL STREAMLIT CHROME ─────────────────────────────────────────── */
/* NOTE: collapsedControl, title=Download, button[kind=header] intentionally
   NOT hidden — they are real UI elements we want to keep visible.           */
#MainMenu,
footer,
header,
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stDeployButton"],
.stDeployButton,
.viewerBadge_container__r5tak,
.viewerBadge_link__qRIco,
#stDecoration { display: none !important; }

/* Fullscreen icon only inside the image/chart toolbar, NOT on buttons */
[data-testid="StyledFullScreenButton"] { display: none !important; }

/* ─── HIDE MATERIAL ICON TEXT FALLBACK (exclude sidebar arrow icons) ─────── */
span[data-testid="stIconMaterial"]:not(
  [data-testid="collapsedControl"] span,
  [data-testid="stSidebarCollapseButton"] span
) {
  display: none !important;
  visibility: hidden !important;
  font-size: 0 !important;
  width: 0 !important;
  overflow: hidden !important;
}

/* ─── SIDEBAR COLLAPSE BUTTON (open → close arrow) ─────────────────────── */
[data-testid="stSidebarCollapseButton"] {
  display:    flex !important;
  visibility: visible !important;
  opacity:    1 !important;
}
[data-testid="stSidebarCollapseButton"] button {
  background:    transparent !important;
  border:        1px solid #00d4ff66 !important;
  border-radius: 6px !important;
  box-shadow:    0 0 10px #00d4ff33 !important;
  padding:       6px !important;
  cursor:        pointer !important;
  transition:    box-shadow .25s, background .25s !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
  background: #041e30 !important;
  box-shadow: 0 0 18px #00d4ff66 !important;
}
[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
[data-testid="stSidebarCollapseButton"] span[translate="no"] {
  display:     inline !important;
  visibility:  visible !important;
  font-size:   20px !important;
  width:       auto !important;
  overflow:    visible !important;
  color:       #00d4ff !important;
  font-family: 'Material Icons', 'Material Symbols Outlined', sans-serif !important;
  filter:      drop-shadow(0 0 5px #00d4ffaa) !important;
}

/* ─── SIDEBAR EXPAND BUTTON (closed → open arrow) ───────────────────────── */
[data-testid="collapsedControl"] [data-testid="stIconMaterial"],
[data-testid="collapsedControl"] span[translate="no"] {
  display:     inline !important;
  visibility:  visible !important;
  font-size:   20px !important;
  width:       auto !important;
  overflow:    visible !important;
  color:       #00d4ff !important;
  font-family: 'Material Icons', 'Material Symbols Outlined', sans-serif !important;
  filter:      drop-shadow(0 0 5px #00d4ffaa) !important;
}

/* ─── SIDEBAR COLLAPSE ARROW — fixed to left edge, never over main content ─ */
[data-testid="collapsedControl"] {
  display:       flex !important;
  visibility:    visible !important;
  opacity:       1 !important;
  position:      fixed !important;
  left:          0 !important;
  top:           50vh !important;
  transform:     translateY(-50%) !important;
  z-index:       500 !important;
  width:         auto !important;
  background:    #03111d !important;
  border:        1px solid #00d4ff66 !important;
  border-left:   none !important;
  border-radius: 0 8px 8px 0 !important;
  box-shadow:    3px 0 16px #00d4ff44 !important;
  padding:       10px 6px !important;
  cursor:        pointer !important;
  transition:    box-shadow .25s, background .25s !important;
  pointer-events: auto !important;
}
[data-testid="collapsedControl"]:hover {
  background: #041e30 !important;
  box-shadow: 3px 0 28px #00d4ff77 !important;
}
[data-testid="collapsedControl"] svg {
  display:     block !important;
  fill:        #00d4ff !important;
  color:       #00d4ff !important;
  filter:      drop-shadow(0 0 5px #00d4ffaa) !important;
  width:       18px !important;
  height:      18px !important;
  flex-shrink: 0 !important;
}
[data-testid="collapsedControl"] button {
  background:  transparent !important;
  border:      none !important;
  box-shadow:  none !important;
  padding:     0 !important;
  line-height: 0 !important;
}

/* ─── EXPANDER — sits in normal flow, chevron always visible ─────────────── */
[data-testid="stExpander"] {
  border:        1px solid #00d4ff33 !important;
  border-radius: 8px !important;
  background:    #041825 !important;
  position:      relative !important;
  z-index:       2 !important;
  overflow:      visible !important;
  margin-top:    10px !important;
}
/* Expander summary row */
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] summary {
  display:        flex !important;
  align-items:    center !important;
  gap:            8px !important;
  padding:        10px 14px !important;
  cursor:         pointer !important;
  color:          #00d4ff !important;
  font-size:      .92rem !important;
  letter-spacing: .04em !important;
  list-style:     none !important;
  user-select:    none !important;
}
[data-testid="stExpander"] details > summary::-webkit-details-marker,
[data-testid="stExpander"] summary::-webkit-details-marker {
  display: none !important;
}
[data-testid="stExpander"] summary:hover { color: #66eeff !important; }
/* The chevron SVG */
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] details summary svg,
[data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] {
  display:     inline-block !important;
  visibility:  visible !important;
  fill:        #00d4ff !important;
  color:       #00d4ff !important;
  filter:      drop-shadow(0 0 4px #00d4ff88) !important;
  width:       16px !important;
  height:      16px !important;
  flex-shrink: 0 !important;
}

/* ─── CSS VARIABLES ─────────────────────────────────────────────────────── */
:root {
  --neon:        #00d4ff;
  --neon-dim:    #0099cc;
  --neon-bright: #66eeff;
  --neon-glow:   0 0 8px #00d4ff, 0 0 20px #00d4ff55, 0 0 40px #00d4ff22;
  --neon-glow-md:0 0 4px #00d4ff, 0 0 12px #00d4ff66;
  --bg-deep:     #020b14;
  --bg-panel:    #03111d;
  --bg-card:     #041825;
  --bg-input:    #041522;
  --border:      #00d4ff33;
  --border-md:   #00d4ff66;
  --border-hi:   #00d4ffaa;
  --text-main:   #b8f0ff;
  --text-muted:  #5ba8c4;
  --text-dim:    #2d6a82;
  --ff:          'Times New Roman', Times, serif;
}

/* ─── GLOBAL FONT & BACKGROUND ─────────────────────────────────────────── */
*, *::before, *::after {
  font-family: var(--ff) !important;
  box-sizing: border-box;
}

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
.main, .block-container {
  background: var(--bg-deep) !important;
  color: var(--text-main) !important;
}

/* Animated grid background */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(var(--border) 1px, transparent 1px),
    linear-gradient(90deg, var(--border) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
  z-index: 0;
  opacity: .35;
  animation: gridPulse 6s ease-in-out infinite;
}
@keyframes gridPulse {
  0%,100% { opacity:.25; }
  50%      { opacity:.45; }
}

.block-container {
  position: relative;
  z-index: 1;
  padding-top: 1.5rem !important;
  max-width: 1200px !important;
}

/* ─── SCROLLBAR ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar               { width:6px; height:6px; }
::-webkit-scrollbar-track         { background:var(--bg-deep); }
::-webkit-scrollbar-thumb         { background:var(--neon-dim); border-radius:3px; }
::-webkit-scrollbar-thumb:hover   { background:var(--neon); }

/* ─── SIDEBAR ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-panel) !important;
  border-right: 1px solid var(--border-md) !important;
  box-shadow: 4px 0 30px #00d4ff18 !important;
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] h3 {
  color: var(--neon) !important;
  text-shadow: var(--neon-glow-md);
  letter-spacing: .08em;
  font-size: 1rem !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border-md) !important; }

/* ─── MAIN TITLE ────────────────────────────────────────────────────────── */
h1 {
  color: var(--neon) !important;
  text-shadow: var(--neon-glow) !important;
  font-size: 2rem !important;
  letter-spacing: .12em !important;
  font-style: italic;
  animation: titleFlicker 8s ease-in-out infinite;
}
@keyframes titleFlicker {
  0%,96%,100% { opacity:1; text-shadow: var(--neon-glow); }
  97%         { opacity:.85; text-shadow: 0 0 4px #00d4ff; }
  98%         { opacity:1; }
  99%         { opacity:.9; }
}
h2, h3, h4 {
  color: var(--neon) !important;
  text-shadow: var(--neon-glow-md) !important;
  letter-spacing: .06em;
}

/* ─── CAPTION / SMALL TEXT ──────────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"],
small, caption { color: var(--text-muted) !important; font-style: italic; }

/* ─── INPUTS ────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background: var(--bg-input) !important;
  border: 1px solid var(--border-md) !important;
  border-radius: 6px !important;
  color: var(--neon-bright) !important;
  caret-color: var(--neon) !important;
  transition: border .25s, box-shadow .25s;
  font-size: .95rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--neon) !important;
  box-shadow: var(--neon-glow-md) !important;
  outline: none !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder { color: var(--text-dim) !important; }

/* Label text above inputs */
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label {
  color: var(--text-muted) !important;
  font-size: .85rem !important;
  letter-spacing: .04em;
}

/* ─── SELECTBOX ─────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
  background: var(--bg-input) !important;
  border: 1px solid var(--border-md) !important;
  border-radius: 6px !important;
  color: var(--neon-bright) !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: var(--neon) !important;
  box-shadow: var(--neon-glow-md) !important;
}
/* Dropdown list */
[data-testid="stSelectbox"] ul,
[role="listbox"] {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border-md) !important;
}
[data-testid="stSelectbox"] li,
[role="option"] {
  color: var(--text-main) !important;
}
[role="option"]:hover, [role="option"][aria-selected="true"] {
  background: #00d4ff22 !important;
  color: var(--neon) !important;
}

/* ─── SLIDER ────────────────────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] {
  background: var(--neon) !important;
  box-shadow: var(--neon-glow-md) !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {
  background: var(--border-md) !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div:nth-child(2) {
  background: var(--neon) !important;
}
[data-testid="stSlider"] p { color: var(--neon-bright) !important; }

/* ─── CHECKBOX ──────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] input[type="checkbox"] + div {
  border-color: var(--neon-dim) !important;
  background: var(--bg-input) !important;
}
[data-testid="stCheckbox"] input[type="checkbox"]:checked + div {
  background: var(--neon) !important;
  border-color: var(--neon) !important;
  box-shadow: var(--neon-glow-md) !important;
}

/* ─── BUTTONS ───────────────────────────────────────────────────────────── */
[data-testid="stButton"] button {
  background: transparent !important;
  border: 1px solid var(--neon-dim) !important;
  color: var(--neon) !important;
  border-radius: 6px !important;
  font-size: .85rem !important;
  letter-spacing: .06em !important;
  transition: all .25s !important;
  text-shadow: 0 0 6px #00d4ff88;
}
[data-testid="stButton"] button:hover {
  border-color: var(--neon) !important;
  background: #00d4ff18 !important;
  box-shadow: var(--neon-glow-md) !important;
  color: var(--neon-bright) !important;
  text-shadow: var(--neon-glow) !important;
}
/* Primary buttons */
[data-testid="stButton"] button[kind="primary"] {
  background: linear-gradient(135deg, #003d54, #005f7a) !important;
  border: 1px solid var(--neon) !important;
  box-shadow: var(--neon-glow-md) !important;
  color: var(--neon-bright) !important;
  font-weight: 700 !important;
  letter-spacing: .08em !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
  background: linear-gradient(135deg, #005f7a, #0099cc) !important;
  box-shadow: var(--neon-glow) !important;
  transform: translateY(-1px) !important;
}

/* ─── DOWNLOAD BUTTONS ──────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] {
  position: relative !important;
  z-index: 1 !important;
}
[data-testid="stDownloadButton"] button {
  background: transparent !important;
  border: 1px solid var(--neon-dim) !important;
  color: var(--neon) !important;
  border-radius: 6px !important;
  transition: all .25s !important;
  width: 100% !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
  position: relative !important;
  z-index: 1 !important;
}
[data-testid="stDownloadButton"] button:hover {
  background: #00d4ff18 !important;
  box-shadow: var(--neon-glow-md) !important;
  border-color: var(--neon) !important;
}
/* Fix anchor tag inside download button that causes text doubling */
[data-testid="stDownloadButton"] button a,
[data-testid="stDownloadButton"] a {
  color: var(--neon) !important;
  text-decoration: none !important;
  position: static !important;
  display: contents !important;
}

/* ─── TABS ──────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--border-md) !important;
  gap: 2px;
}
[data-testid="stTabs"] button[role="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border-radius: 6px 6px 0 0 !important;
  border: 1px solid transparent !important;
  border-bottom: none !important;
  font-size: .88rem !important;
  letter-spacing: .05em;
  transition: all .2s;
}
[data-testid="stTabs"] button[role="tab"]:hover {
  color: var(--neon) !important;
  background: #00d4ff0a !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  color: var(--neon) !important;
  background: var(--bg-card) !important;
  border-color: var(--border-md) !important;
  border-bottom-color: var(--bg-card) !important;
  text-shadow: var(--neon-glow-md) !important;
  box-shadow: 0 -2px 12px #00d4ff22 !important;
}

/* ─── EXPANDER CONTAINER ─────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: var(--bg-card) !important;
  position: relative !important;
  z-index: 1 !important;
  overflow: visible !important;
  margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
  color: var(--neon-dim) !important;
}
[data-testid="stExpander"] summary:hover {
  color: var(--neon) !important;
}

/* ─── CHAT MESSAGES ─────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  margin-bottom: .5rem !important;
  animation: msgAppear .3s ease;
}
@keyframes msgAppear {
  from { opacity:0; transform:translateY(6px); }
  to   { opacity:1; transform:translateY(0); }
}
/* User messages */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  border-color: var(--neon-dim) !important;
  background: #04202e !important;
}
/* AI messages */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  border-color: #00d4ff44 !important;
  background: #03141e !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] .stMarkdown { color: var(--text-main) !important; }

/* ─── STCONTAINER (chat area) ───────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--bg-panel) !important;
  border: 1px solid var(--border-md) !important;
  border-radius: 10px !important;
  box-shadow: inset 0 0 30px #00d4ff08, 0 0 20px #00d4ff11 !important;
}

/* ─── SPINNER ───────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p { color: var(--neon-dim) !important; font-style: italic; }
[data-testid="stSpinner"] svg circle { stroke: var(--neon) !important; }

/* ─── ALERT / INFO / SUCCESS / ERROR ────────────────────────────────────── */
[data-testid="stAlert"] {
  background: var(--bg-card) !important;
  border-radius: 8px !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
.stInfo {
  border: 1px solid #00d4ff55 !important;
  background: #011f2e !important;
  color: var(--neon) !important;
}
[data-testid="stAlert"][kind="success"], .stSuccess {
  border: 1px solid #00ff9955 !important;
  background: #011e11 !important;
  color: #66ffbb !important;
}
[data-testid="stAlert"][kind="error"], .stError {
  border: 1px solid #ff335566 !important;
  background: #1e0305 !important;
  color: #ff8899 !important;
}
[data-testid="stAlert"] p { color: inherit !important; }

/* ─── DIVIDER ───────────────────────────────────────────────────────────── */
hr { border-color: var(--border-md) !important; }

/* ─── CODE BLOCKS ───────────────────────────────────────────────────────── */
code, pre {
  background: #041018 !important;
  color: var(--neon-bright) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
}

/* ─── STMARKDOWN general ────────────────────────────────────────────────── */
.stMarkdown p, .stMarkdown li, .stMarkdown span { color: var(--text-main) !important; }
.stMarkdown a { color: var(--neon) !important; text-decoration: underline; }

/* ─── SECTION CONTAINER ─────────────────────────────────────────────────── */
section.main { background: var(--bg-deep) !important; }

/* ─── PASSWORD STRENGTH BAR FIX ─────────────────────────────────────────── */
.strength-bar-bg   { background:#041825 !important; border:1px solid var(--border) !important; }

/* ─── COLUMN / BUTTON LAYOUT — no overlap ───────────────────────────────── */
[data-testid="column"] {
  min-width: 0 !important;
  overflow: visible !important;
}
[data-testid="stButton"],
[data-testid="stDownloadButton"] {
  position: relative !important;
  z-index: 1 !important;
  display: block !important;
  width: 100% !important;
}
[data-testid="stButton"] button,
[data-testid="stDownloadButton"] button {
  position: relative !important;
  width: 100% !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}
/* Sidebar session rows: delete icon won't overlap session name */
[data-testid="stSidebar"] [data-testid="column"] {
  padding: 0 2px !important;
  align-items: center !important;
}
/* Send button column stays compact */
[data-testid="stHorizontalBlock"] [data-testid="column"]:last-child {
  flex: 0 0 auto !important;
  min-width: 82px !important;
}
/* Row containers */
[data-testid="stHorizontalBlock"] {
  gap: 6px !important;
  align-items: flex-end !important;
  flex-wrap: nowrap !important;
}
</style>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE CSS — extended for neon theme
# ══════════════════════════════════════════════════════════════════════════════

_AUTH_CSS = """
<style>
  /* sidebar visible on all pages */

  .auth-logo {
    text-align:center;
    font-size:3.8rem;
    margin-bottom:.4rem;
    animation: logoPulse 3s ease-in-out infinite;
    filter: drop-shadow(0 0 12px #00d4ff);
  }
  @keyframes logoPulse {
    0%,100% { filter: drop-shadow(0 0 12px #00d4ff); }
    50%      { filter: drop-shadow(0 0 24px #00d4ff) drop-shadow(0 0 40px #00d4ff66); }
  }

  .auth-title {
    text-align:center;
    font-size:2rem;
    font-weight:700;
    font-style:italic;
    color:#00d4ff;
    text-shadow: 0 0 8px #00d4ff, 0 0 20px #00d4ff55, 0 0 40px #00d4ff22;
    margin-bottom:.25rem;
    letter-spacing:.12em;
    animation: titleFlicker 8s ease-in-out infinite;
  }
  @keyframes titleFlicker {
    0%,96%,100% { opacity:1; }
    97%         { opacity:.8; }
    99%         { opacity:.93; }
  }

  .auth-sub {
    text-align:center;
    color:#5ba8c4;
    font-size:.9rem;
    font-style:italic;
    margin-bottom:1.4rem;
    letter-spacing:.06em;
  }

  .strength-bar-bg   {
    background:#041825;
    border:1px solid #00d4ff33;
    border-radius:6px;
    height:7px;
    width:100%;
    margin:5px 0 3px;
    overflow:hidden;
  }
  .strength-bar-fill {
    height:7px;
    border-radius:6px;
    box-shadow: 0 0 8px currentColor;
    transition: width .4s ease, background .4s ease;
  }

  .hint-pill {
    display:inline-block;
    background:#03111d;
    border:1px solid #00d4ff33;
    border-radius:20px;
    padding:3px 11px;
    font-size:.78rem;
    color:#5ba8c4;
    margin:3px 3px;
    letter-spacing:.04em;
    transition: all .2s;
  }
  .hint-ok {
    border-color:#00ff99;
    color:#00ff99;
    background:#011e11;
    box-shadow: 0 0 6px #00ff9944;
  }

  .info-box {
    background:#011f2e;
    border:1px solid #00d4ff44;
    border-radius:8px;
    padding:.65rem 1rem;
    font-size:.83rem;
    color:#66ccee;
    margin-top:.9rem;
    text-align:center;
    box-shadow: 0 0 12px #00d4ff11;
  }

  .warn-box {
    background:#1a0505;
    border:1px solid #ff335544;
    border-radius:8px;
    padding:.65rem 1rem;
    font-size:.83rem;
    color:#ff8899;
    margin-top:.9rem;
    text-align:center;
  }

  .otp-box {
    background:#01200e;
    border:1px solid #00ff9966;
    border-radius:8px;
    padding:.75rem 1rem;
    font-size:.86rem;
    color:#66ffbb;
    margin:.7rem 0;
    text-align:center;
    box-shadow: 0 0 12px #00ff9911;
  }
</style>
"""


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
        "users":           ["username", "password_hash", "full_name", "email", "created_at"],
        "sessions":        ["username", "session_id", "name", "created", "updated"],
        "messages":        ["username", "session_id", "role", "content", "ts"],
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
    e = email.lower().strip()
    return any(v["email"] == e for v in load_users().values())


def get_username_by_email(email: str):
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
    sheet = ws("users")
    data  = sheet.get_all_values()
    for i, row in enumerate(data[1:], start=2):
        if row[0] == username:
            sheet.update_cell(i, 2, _hp(new_password))
            break
    _cached_users.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  PASSWORD RESET  (OTP via email)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_otp(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


def send_reset_email(to_email: str, otp: str) -> bool:
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
    e    = email.lower().strip()
    rows = all_rows("password_resets")
    for r in rows:
        if r.get("email", "").lower().strip() == e and str(r.get("otp")) == otp:
            if r.get("used", "no") == "yes":
                return "used"
            if datetime.fromisoformat(str(r["expires_at"])) < datetime.now():
                return "expired"
            return "ok"
    return "invalid"


def consume_otp(email: str, otp: str):
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
    if not password:                           return "Password cannot be empty."
    if len(password) < 6:                      return "Password must be at least 6 characters."
    if not re.search(r"[A-Za-z]", password):  return "Must contain at least one letter."
    if not re.search(r"[0-9]", password):      return "Must contain at least one number."
    return None


def password_strength(password: str):
    score = 0
    if len(password) >= 6:                   score += 1
    if len(password) >= 10:                  score += 1
    if re.search(r"[A-Z]", password):        score += 1
    if re.search(r"[^A-Za-z0-9]", password): score += 1
    labels = {0: "Very Weak", 1: "Weak", 2: "Fair", 3: "Strong", 4: "Very Strong"}
    colors = {0: "#ff3355", 1: "#ff8c00", 2: "#ffd700", 3: "#00d4ff", 4: "#00ff99"}
    return score, labels[score], colors[score]


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
        elif not user_exists(u):
            st.error("❌  User not found. Please sign up first.")
            st.markdown(
                "<div class='warn-box'>"
                "🆕 No account with that username. "
                "Switch to the <b>Sign Up</b> tab to create one."
                "</div>",
                unsafe_allow_html=True,
            )
        elif not auth_user(u, pw):
            st.error("🔑  Incorrect password. Please try again.")
            st.markdown(
                "<div class='info-box'>"
                "Forgot your password? Use the <b>Forgot Password</b> tab →"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.session_state.logged_in = True
            st.session_state.username  = u
            st.success(f"✅  Welcome back, **{u}**!")
            time.sleep(0.6)
            st.rerun()

    st.markdown(
        "<div class='info-box'>No account yet? Switch to <b>Sign Up</b> · "
        "Forgot password? Use <b>Forgot Password</b></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIGN UP PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _signup_panel():
    st.markdown("")
    full_name = st.text_input("🙂  Full Name", key="su_name",
                              placeholder="Enter Your Full Name")

    su_un = st.text_input("👤  Username", key="su_un",
                          placeholder="3–32 chars · letters / numbers / _ . -")
    if su_un:
        err = validate_username(su_un)
        if err:
            st.markdown(f"<span style='color:#ff3355;font-size:.82rem'>⚠ {err}</span>",
                        unsafe_allow_html=True)
        elif user_exists(su_un.strip()):
            st.markdown("<span style='color:#ff3355;font-size:.82rem'>"
                        "⚠ Username already taken.</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#00ff99;font-size:.82rem;text-shadow:0 0 6px #00ff9966'>"
                        "✔ Username available!</span>", unsafe_allow_html=True)

    su_email = st.text_input("📧  Email", key="su_email",
                             placeholder="you@example.com")
    if su_email:
        e_err = validate_email(su_email)
        if e_err:
            st.markdown(f"<span style='color:#ff3355;font-size:.82rem'>⚠ {e_err}</span>",
                        unsafe_allow_html=True)
        elif email_exists(su_email.strip()):
            st.markdown("<span style='color:#ff3355;font-size:.82rem'>"
                        "⚠ Email already registered.</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#00ff99;font-size:.82rem;text-shadow:0 0 6px #00ff9966'>"
                        "✔ Email available!</span>", unsafe_allow_html=True)

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
            f"<div class='strength-bar-fill' style='width:{pct}%;background:{color};box-shadow:0 0 8px {color}'></div>"
            f"</div><span style='color:{color};font-size:.78rem;text-shadow:0 0 6px {color}88'>"
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
            st.markdown("<span style='color:#00ff99;font-size:.82rem;text-shadow:0 0 6px #00ff9966'>✔ Passwords match</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#ff3355;font-size:.82rem'>"
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
#  FORGOT PASSWORD PANEL
# ══════════════════════════════════════════════════════════════════════════════

def _forgot_password_panel():
    st.markdown("")

    if "fp_step" not in st.session_state:
        st.session_state.fp_step     = 1
        st.session_state.fp_email    = ""
        st.session_state.fp_username = ""

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

        st.markdown(
            "<div class='info-box'>Remembered your password? Go back to <b>Sign In →</b></div>",
            unsafe_allow_html=True,
        )

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
                f"<div class='strength-bar-fill' style='width:{pct}%;background:{color};box-shadow:0 0 8px {color}'></div>"
                f"</div><span style='color:{color};font-size:.78rem;text-shadow:0 0 6px {color}88'>"
                f"Strength: <b>{label}</b></span>",
                unsafe_allow_html=True,
            )

        new_pw2 = st.text_input("🔒  Confirm New Password", type="password",
                                key="fp_pw2", placeholder="Re-enter new password")
        if new_pw2 and new_pw1:
            if new_pw1 == new_pw2:
                st.markdown("<span style='color:#00ff99;font-size:.82rem;text-shadow:0 0 6px #00ff9966'>✔ Passwords match</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#ff3355;font-size:.82rem'>"
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
                    else:
                        perr = validate_password(new_pw1)
                        if perr:
                            st.error(f"⚠️  {perr}")
                        elif new_pw1 != new_pw2:
                            st.error("⚠️  Passwords do not match.")
                        else:
                            consume_otp(email, otp)
                            update_password(username, new_pw1)
                            st.success("✅  Password reset successfully! You can now sign in.")
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
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(_AUTH_CSS,  unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1.5, 1])
    with col:
        st.markdown("<div class='auth-logo'>🤖</div>", unsafe_allow_html=True)
        st.markdown("<div class='auth-title'>Groq AI Chatbot</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='auth-sub'>Streamlit · LangChain · Groq · ☁️ Google Sheets</div>",
            unsafe_allow_html=True,
        )
        tab_si, tab_su, tab_fp = st.tabs(["🔑  Sign In", "📝  Sign Up", "🔓  Forgot Password"])
        with tab_si: _signin_panel()
        with tab_su: _signup_panel()
        with tab_fp: _forgot_password_panel()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT APP
# ══════════════════════════════════════════════════════════════════════════════

def chat_app():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    username = st.session_state.username

    sessions = load_sessions(username)
    if not sessions:
        create_session(username, "Chat 1")
        sessions = load_sessions(username)

    if ("active_sid" not in st.session_state or
            not any(s["session_id"] == st.session_state.active_sid for s in sessions)):
        st.session_state.active_sid = sessions[0]["session_id"]

    if "sys_prompt_value" not in st.session_state:
        st.session_state.sys_prompt_value = DEFAULT_PROMPT
    if "_tone_idx" not in st.session_state:
        st.session_state._tone_idx = 0

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<div style='padding:.5rem 0;color:#00d4ff;text-shadow:0 0 8px #00d4ff66;"
            f"font-size:1rem;letter-spacing:.08em;font-style:italic'>"
            f"👤 &nbsp;<b>{username}</b></div>",
            unsafe_allow_html=True,
        )
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.divider()
        st.header("⚙️ Settings")

        groq_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="Enter Your GROQ API Key",
        ).strip() or st.secrets.get("GROQ_API_KEY", "")

        model = st.selectbox("Model", [
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-120b",
            "qwen/qwen3-32b",
            "moonshotai/kimi-k2-instruct",
            "groq/compound",
        ])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.4, 0.1)
        max_tokens  = st.slider("Max Tokens",  64, 2048, 640, 64)

        _tone_options = ["Custom", "Friendly", "Strict", "Teacher"]
        tone_preset   = st.selectbox(
            "Tone Preset",
            _tone_options,
            index=st.session_state._tone_idx,
        )
        new_tone_idx = _tone_options.index(tone_preset)
        if new_tone_idx != st.session_state._tone_idx:
            st.session_state._tone_idx = new_tone_idx
            if tone_preset != "Custom":
                st.session_state.sys_prompt_value = TONE_MAP[tone_preset]

        new_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.sys_prompt_value,
            height=110,
        )
        st.session_state.sys_prompt_value = new_prompt

        if st.button("↺ Reset Prompt"):
            st.session_state.sys_prompt_value = DEFAULT_PROMPT
            st.session_state._tone_idx        = 0
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
                        "<div style='text-align:center;color:#00d4ff88;"
                        "padding-top:90px;font-size:1.1rem;font-style:italic;"
                        "text-shadow:0 0 12px #00d4ff44;letter-spacing:.06em'>"
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
