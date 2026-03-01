"""
Streamlit web UI for the Porsche 993 Repair Assistant.
Uses Pinecone for vector search, Claude for answer generation,
S3 for persistent conversation history, and Google login via st.login().

Run with: streamlit run ui/app.py
"""

import os
import re
import sys
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

# Add parent dir to path so we can import from api/
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)


# --- Page config ---
st.set_page_config(
    page_title="993 Repair Assistant",
    page_icon="wrench",
    layout="centered",
)

# --- Rennlist-Inspired Design System CSS ---
# Tokens:
#   Palette: Rennlist forum DNA (#677887 slate, #36576f header, #a62a2a brick-red hover)
#   Spacing base: 4px  |  Scale: 4, 8, 12, 16, 24, 32, 48
#   Depth: borders-only (1px solid, no shadows)
#   Radius: 3px everywhere (forum-utilitarian)
#   Font: Verdana  |  Sizes: 11px labels, 13px body, 14px subhead, 18px title
st.markdown("""
<style>
    /* ====== DESIGN TOKENS ====== */
    :root {
        --fg-primary: #222222;
        --fg-secondary: #555555;
        --fg-muted: #888888;
        --fg-faint: #aaaaaa;
        --bg-base: #f1f1f1;
        --bg-elevated: #ffffff;
        --bg-inset: #e8e8e8;
        --border-default: #cccccc;
        --border-strong: #222222;
        --accent: #677887;
        --accent-dark: #36576f;
        --accent-hover: #a62a2a;
        --accent-hover-light: rgba(166, 42, 42, 0.08);
        --thead-start: #677887;
        --thead-end: #bdc6cc;
        --radius: 3px;
    }

    /* ====== GLOBAL ====== */
    .stApp {
        font-family: Verdana, Geneva, sans-serif !important;
        background-color: var(--bg-base) !important;
    }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 13px !important;
        line-height: 1.6 !important;
        color: var(--fg-primary) !important;
    }

    /* ====== APP HEADER (dark slate bar) ====== */
    .app-header {
        background: var(--accent-dark);
        margin: -1rem -1rem 24px -1rem;
        padding: 20px 24px;
        border-bottom: 1px solid var(--border-strong);
    }
    .app-header h1.app-header-title,
    .app-header-title {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: 0px !important;
        margin: 0 0 4px 0 !important;
        padding: 0 !important;
        line-height: 1.3 !important;
        border: none !important;
    }
    [data-testid="stMarkdownContainer"] .app-header-sub {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 12px !important;
        color: rgba(255, 255, 255, 0.7) !important;
        margin: 0 !important;
        line-height: 1.5 !important;
    }
    .app-car-tag {
        display: inline-block;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: var(--radius);
        padding: 3px 10px;
        margin-top: 8px;
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 11px;
        font-weight: 400;
        color: rgba(255, 255, 255, 0.85);
        letter-spacing: 0.2px;
    }

    /* ====== SIDEBAR ====== */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-elevated) !important;
        border-right: 1px solid var(--border-default) !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: var(--bg-elevated) !important;
        padding-top: 16px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 4px !important;
    }

    /* Sidebar new-chat button */
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] div,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] span {
        background-color: var(--accent) !important;
        border: 1px solid var(--accent) !important;
        color: #ffffff !important;
        border-radius: var(--radius) !important;
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 12px !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        background-color: var(--accent-dark) !important;
        border-color: var(--accent-dark) !important;
    }
    [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"],
    [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
    }

    /* Sidebar conversation buttons */
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 12px;
        padding: 8px 12px;
        border: none;
        background: transparent;
        color: var(--fg-primary);
        border-radius: var(--radius);
        transition: background 0.15s, color 0.15s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: var(--accent-hover-light);
        color: var(--accent-hover);
    }

    /* Sidebar section labels */
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: var(--fg-muted) !important;
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 700;
    }

    /* Sidebar button text inherits */
    section[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] p {
        color: inherit !important;
    }

    /* ====== SIDEBAR CAR INFO CARD (Rennlist thead-style) ====== */
    .car-info-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius);
        overflow: hidden;
        font-family: Verdana, Geneva, sans-serif !important;
    }
    .car-info-card .card-header {
        background: linear-gradient(to bottom, var(--thead-start), var(--thead-end));
        padding: 8px 12px;
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 11px;
        font-weight: 700;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .car-info-card .card-body {
        padding: 12px;
    }
    .car-info-card .card-label {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        color: var(--fg-muted);
        margin: 0 0 2px 0;
    }
    .car-info-card .card-label:not(:first-child) {
        margin-top: 12px;
    }
    .car-info-card .card-value {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 13px;
        color: var(--fg-primary);
        font-weight: 700;
        margin: 0;
    }
    [data-testid="stMarkdownContainer"] .car-info-card .card-sources {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 11px !important;
        color: var(--fg-muted) !important;
        line-height: 1.6 !important;
        margin-top: 4px;
    }

    /* ====== POPOVER (conversation menu) ====== */
    section[data-testid="stSidebar"] [data-testid="stPopover"] > button,
    section[data-testid="stSidebar"] button:has([data-testid="stIconMaterial"]) {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 4px !important;
        min-height: 0 !important;
        color: var(--fg-muted) !important;
        outline: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stPopover"] > button:hover,
    section[data-testid="stSidebar"] button:has([data-testid="stIconMaterial"]):hover {
        color: var(--accent-hover) !important;
    }
    section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stPopover"],
    section[data-testid="stSidebar"] [data-testid="stPopover"] > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    [data-testid="stPopoverBody"],
    [data-testid="stPopoverBody"] > div,
    [data-testid="stPopoverBody"] div {
        background-color: var(--bg-elevated) !important;
        border-color: var(--border-default) !important;
    }
    [data-testid="stPopoverBody"] {
        border-radius: var(--radius) !important;
        border: 1px solid var(--border-default) !important;
        padding: 4px !important;
        min-width: 0 !important;
    }
    [data-testid="stPopoverBody"] > div {
        padding: 0 !important;
        gap: 0 !important;
    }
    [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
    [data-testid="stPopoverBody"] [data-testid="stElementContainer"] { margin: 0 !important; padding: 0 !important; }
    [data-testid="stPopoverBody"] button {
        color: var(--fg-primary) !important;
        background: transparent !important;
        border: none !important;
        padding: 6px 12px !important;
        min-height: 0 !important;
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 12px !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stPopoverBody"] button:hover {
        background: var(--accent-hover-light) !important;
        color: var(--accent-hover) !important;
    }
    [data-testid="stPopoverBody"] button [data-testid="stMarkdownContainer"] p {
        color: inherit !important;
        font-size: 12px !important;
        margin: 0 !important;
    }

    /* ====== CHAT MESSAGES ====== */
    [data-testid="stChatMessage"] {
        background: var(--bg-elevated) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius) !important;
        margin-bottom: 8px !important;
        padding: 16px !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        color: var(--fg-primary) !important;
    }
    /* User messages — slate left border */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-left: 3px solid var(--accent) !important;
    }
    /* Assistant messages — brick-red left border */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 3px solid var(--accent-hover) !important;
    }
    [data-testid="stChatMessage"] hr {
        border-color: var(--border-default) !important;
    }
    [data-testid="stChatMessage"] strong {
        color: var(--accent-dark) !important;
    }
    /* Part numbers monospace */
    [data-testid="stChatMessage"] code {
        font-family: 'SF Mono', SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace !important;
        font-size: 12px !important;
        background: var(--bg-inset) !important;
        border: 1px solid var(--border-default) !important;
        padding: 2px 6px !important;
        border-radius: var(--radius) !important;
        color: var(--fg-primary) !important;
    }

    /* ====== CHAT INPUT ====== */
    [data-testid="stChatInput"],
    .stChatInput, .stChatInput > div, .stChatInput > div > div,
    .stChatInput > div > div > div, .stChatInput div {
        background-color: var(--bg-elevated) !important;
    }
    .stChatInput textarea {
        background-color: var(--bg-elevated) !important;
        color: var(--fg-primary) !important;
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 13px !important;
    }
    [data-testid="stChatInputTextArea"] {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 13px !important;
        color: var(--fg-primary) !important;
    }
    .stChatInput {
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius) !important;
    }
    .stChatInput:focus-within {
        border-color: var(--accent) !important;
    }
    .stChatInput button {
        background-color: var(--accent) !important;
        color: #ffffff !important;
        border-radius: var(--radius) !important;
    }
    .stChatInput button:hover {
        background-color: var(--accent-dark) !important;
    }
    .stChatInput button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* ====== LINKS ====== */
    a { color: var(--accent-hover) !important; }
    a:hover { color: var(--accent-dark) !important; text-decoration: underline !important; }

    /* ====== BACKGROUNDS — clean up Streamlit defaults ====== */
    header[data-testid="stHeader"] {
        background-color: var(--bg-base) !important;
        height: 0 !important;
        min-height: 0 !important;
        overflow: hidden !important;
    }
    [data-testid="stBottomBlockContainer"] { background-color: var(--bg-base) !important; }
    [data-testid="stBottom"] > div { background-color: var(--bg-base) !important; }
    .main .block-container {
        background-color: var(--bg-base) !important;
        padding-top: 3rem !important;
    }
    [data-testid="stMainBlockContainer"] { background-color: var(--bg-base) !important; }
    .stApp > div, .stApp > div > div { background-color: transparent !important; }
    [data-testid="stPopover"] > div { background-color: var(--bg-elevated) !important; border: 1px solid var(--border-default) !important; }

    /* ====== SPINNER ====== */
    [data-testid="stSpinner"] { color: var(--accent) !important; }

    /* ====== LANDING PAGE ====== */
    .landing-header {
        background: var(--accent-dark);
        margin: -1rem -1rem 0 -1rem;
        padding: 32px 24px 24px 24px;
        text-align: center;
        border-bottom: 1px solid var(--border-strong);
    }
    .landing-header-title {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin: 0 0 8px 0;
    }
    [data-testid="stMarkdownContainer"] .landing-header-desc {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 13px !important;
        color: rgba(255, 255, 255, 0.75) !important;
        line-height: 1.6 !important;
        margin: 0 !important;
        max-width: 420px;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .landing-body {
        max-width: 420px;
        margin: 32px auto 0 auto;
        text-align: center;
        padding: 0 16px;
    }
    [data-testid="stMarkdownContainer"] .landing-footer {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 12px !important;
        color: var(--fg-muted) !important;
        margin-top: 16px;
    }
    .landing-stats {
        display: flex;
        justify-content: center;
        gap: 32px;
        margin-top: 24px;
        padding-top: 20px;
        border-top: 1px solid var(--border-default);
    }
    .landing-stat {
        text-align: center;
    }
    .landing-stat-value {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 18px;
        font-weight: 700;
        color: var(--fg-primary);
    }
    [data-testid="stMarkdownContainer"] .landing-stat-label {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 11px !important;
        color: var(--fg-muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* ====== ONBOARDING ====== */
    .onboard-header {
        text-align: center;
        padding: 32px 0 24px 0;
        border-bottom: 1px solid var(--border-default);
        margin-bottom: 24px;
    }
    .onboard-title {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        color: var(--fg-primary) !important;
        margin: 0 0 4px 0 !important;
        line-height: 1.3 !important;
        border: none !important;
        padding: 0 !important;
    }
    .onboard-sub {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 13px;
        color: var(--fg-secondary);
        margin: 0;
    }

    /* Override Streamlit heading defaults inside our custom containers */
    [data-testid="stMarkdownContainer"] .landing-header-title,
    [data-testid="stMarkdownContainer"] .onboard-title,
    [data-testid="stMarkdownContainer"] .app-header-title {
        font-family: Verdana, Geneva, sans-serif !important;
        border: none !important;
        padding: 0 !important;
    }

    /* ====== FORM STYLING ====== */
    [data-testid="stForm"] {
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius) !important;
        padding: 24px !important;
        background: var(--bg-elevated) !important;
    }

    /* ====== SIDEBAR DIVIDERS ====== */
    section[data-testid="stSidebar"] hr {
        border-color: var(--border-default) !important;
        margin: 4px 0 !important;
    }

    /* ====== SIDEBAR EDIT PROFILE BUTTON ====== */
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        font-size: 11px;
        color: var(--fg-muted);
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        color: var(--accent-hover);
    }

    /* ====== USER INFO (sidebar bottom) ====== */
    .user-info {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 11px;
        color: var(--fg-muted);
        padding: 4px 0;
    }
    .user-info strong {
        color: var(--fg-secondary) !important;
        font-weight: 700;
    }

    /* ====== EMPTY STATE ====== */
    .empty-state {
        text-align: center;
        padding: 64px 24px;
        color: var(--fg-muted);
    }
    [data-testid="stMarkdownContainer"] .empty-state-title {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        color: var(--fg-secondary) !important;
        margin: 0 0 4px 0 !important;
    }
    [data-testid="stMarkdownContainer"] .empty-state-desc {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 13px !important;
        color: var(--fg-muted) !important;
        margin: 0 !important;
    }

    /* ====== STREAMLIT BUTTON OVERRIDES (forms, primary) ====== */
    .stApp [data-testid="stBaseButton-primary"] {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        border-radius: var(--radius) !important;
        font-family: Verdana, Geneva, sans-serif !important;
    }
    .stApp [data-testid="stBaseButton-primary"]:hover {
        background-color: var(--accent-dark) !important;
        border-color: var(--accent-dark) !important;
    }

    /* ====== STREAMLIT INPUT OVERRIDES ====== */
    .stApp [data-testid="stTextInput"] input,
    .stApp [data-testid="stTextArea"] textarea,
    .stApp [data-testid="stSelectbox"] > div > div {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 13px !important;
        border-radius: var(--radius) !important;
    }
    .stApp label {
        font-family: Verdana, Geneva, sans-serif !important;
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# AUTHENTICATION — Google Login via st.login()
# ======================================================================

from api.auth import user_id_from_email, load_user_profile, save_user_profile, decode_vin

# Dev mode: bypass auth when [auth] secrets aren't configured (local development)
_DEV_MODE = False
try:
    _is_logged_in = st.user.is_logged_in
except Exception:
    if os.getenv("DEV_MODE", "").lower() in ("1", "true", "yes"):
        _DEV_MODE = True
        _is_logged_in = True
    else:
        st.error("Authentication is not configured. Check your `[auth]` secrets.")
        st.stop()

if not _is_logged_in:
    # --- Landing page ---
    st.markdown("""
    <div class="landing-header">
        <div class="landing-header-title">993 Repair Assistant</div>
        <p class="landing-header-desc">
            Expert repair advice for your Porsche 993, powered by
            real forum knowledge from Pelican Parts, Rennlist, 911uk, and more.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="landing-body">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 2, 1.2])
    with col2:
        st.button("Sign in with Google", on_click=st.login,
                  use_container_width=True, type="primary")

    st.markdown("""
        <p class="landing-footer">
            Sign in to save your chat history and get advice tailored to your car.
        </p>
        <div class="landing-stats">
            <div class="landing-stat">
                <div class="landing-stat-value">140K+</div>
                <div class="landing-stat-label">Forum Posts</div>
            </div>
            <div class="landing-stat">
                <div class="landing-stat-value">9</div>
                <div class="landing-stat-label">Sources</div>
            </div>
            <div class="landing-stat">
                <div class="landing-stat-value">993</div>
                <div class="landing-stat-label">Focused</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- User is authenticated ---
if _DEV_MODE:
    user_email = "dev@localhost"
    display_name = "Developer"
else:
    try:
        _user_dict = st.user.to_dict()
        user_email = _user_dict.get("email", "")
        display_name = _user_dict.get("name", user_email)
    except Exception:
        try:
            user_email = st.user["email"]
            display_name = st.user.get("name", user_email)
        except Exception:
            st.error("Could not read user info. Please sign out and try again.")
            if st.button("Sign out"):
                st.logout()
            st.stop()

    if not user_email:
        st.warning("No email found. Please sign out and try again.")
        if st.button("Sign out"):
            st.logout()
        st.stop()

user_id = user_id_from_email(user_email)


# ======================================================================
# CAR PROFILE ONBOARDING
# ======================================================================

if "car_profile" not in st.session_state:
    if _DEV_MODE:
        # In dev mode, use a default profile so we can preview the main UI
        st.session_state.car_profile = {
            "vin": "",
            "year": "1997",
            "model": "Targa",
            "transmission": "Tiptronic",
            "mileage": "80,000",
            "known_issues": "",
        }
    else:
        profile = load_user_profile(user_id)
        st.session_state.car_profile = profile


def _show_onboarding():
    """Show the car profile onboarding form. Returns True if completed."""
    st.markdown("""
    <div class="onboard-header">
        <div class="onboard-title">Set up your car profile</div>
        <p class="onboard-sub">We'll personalize all advice to your specific 993.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"Welcome, **{display_name}**.")

    with st.form("car_profile_form"):
        vin = st.text_input(
            "VIN (optional -- auto-fills fields below)",
            max_chars=17,
            placeholder="WP0CB2960VS320001",
        )

        col1, col2 = st.columns(2)
        with col1:
            year = st.selectbox(
                "Year",
                options=[""] + [str(y) for y in range(1998, 1988, -1)],
                index=0,
            )
        with col2:
            model = st.selectbox(
                "Model",
                options=[
                    "", "Carrera", "Carrera S", "Carrera 4",
                    "Carrera 4S", "Targa", "Turbo", "Turbo S", "GT2",
                    "Cabriolet", "Speedster",
                ],
                index=0,
            )

        transmission = st.selectbox(
            "Transmission",
            options=["", "Manual (G50)", "Tiptronic"],
            index=0,
        )
        mileage = st.text_input("Approximate mileage", placeholder="80,000")
        known_issues = st.text_area(
            "Known issues (optional)",
            placeholder="e.g. Oil leak from RMS, soft top motor slow, AC needs recharge...",
            height=100,
        )

        submitted = st.form_submit_button("Save & start chatting", type="primary", use_container_width=True)

        if submitted:
            if vin and len(vin) == 17:
                decoded = decode_vin(vin)
                if decoded:
                    if not year and decoded.get("year"):
                        year = decoded["year"]
                    if not model and decoded.get("model"):
                        model = decoded["model"]

            if not year or not model:
                st.error("Please select at least the year and model.")
                return False

            profile = {
                "vin": vin.strip() if vin else "",
                "year": year,
                "model": model,
                "transmission": transmission,
                "mileage": mileage.strip(),
                "known_issues": known_issues.strip(),
            }
            save_user_profile(user_id, profile)
            st.session_state.car_profile = profile
            return True

    return False


def _show_edit_profile():
    """Show edit profile form inline."""
    profile = st.session_state.car_profile or {}

    st.markdown("""
    <div class="onboard-header">
        <div class="onboard-title">Edit car profile</div>
        <p class="onboard-sub">Update your details to keep advice accurate.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("edit_profile_form"):
        vin = st.text_input("VIN (optional)", value=profile.get("vin", ""), max_chars=17)

        col1, col2 = st.columns(2)
        years_list = [""] + [str(y) for y in range(1998, 1988, -1)]
        models_list = [
            "", "Carrera", "Carrera S", "Carrera 4",
            "Carrera 4S", "Targa", "Turbo", "Turbo S", "GT2",
            "Cabriolet", "Speedster",
        ]
        with col1:
            curr_year = profile.get("year", "")
            year_idx = years_list.index(curr_year) if curr_year in years_list else 0
            year = st.selectbox("Year", options=years_list, index=year_idx)
        with col2:
            curr_model = profile.get("model", "")
            model_idx = models_list.index(curr_model) if curr_model in models_list else 0
            model = st.selectbox("Model", options=models_list, index=model_idx)

        trans_list = ["", "Manual (G50)", "Tiptronic"]
        curr_trans = profile.get("transmission", "")
        trans_idx = trans_list.index(curr_trans) if curr_trans in trans_list else 0
        transmission = st.selectbox("Transmission", options=trans_list, index=trans_idx)

        mileage = st.text_input("Approximate mileage", value=profile.get("mileage", ""))
        known_issues = st.text_area(
            "Known issues",
            value=profile.get("known_issues", ""),
            height=100,
        )

        col_save, col_cancel = st.columns(2)
        with col_save:
            save_clicked = st.form_submit_button("Save changes", type="primary", use_container_width=True)
        with col_cancel:
            cancel_clicked = st.form_submit_button("Cancel", use_container_width=True)

        if save_clicked:
            updated = {
                "vin": vin.strip(),
                "year": year,
                "model": model,
                "transmission": transmission,
                "mileage": mileage.strip(),
                "known_issues": known_issues.strip(),
            }
            save_user_profile(user_id, updated)
            st.session_state.car_profile = updated
            st.session_state.show_edit_profile = False
            st.rerun()

        if cancel_clicked:
            st.session_state.show_edit_profile = False
            st.rerun()


# If no profile yet, show onboarding and stop
if st.session_state.car_profile is None:
    if _show_onboarding():
        st.rerun()
    st.stop()

car_profile = st.session_state.car_profile


# ======================================================================
# CONNECT TO PINECONE (fast — no PyTorch)
# ======================================================================

@st.cache_resource
def connect_pinecone():
    """Connect to Pinecone index (cached). No model loading needed."""
    from api.chat import _get_index
    index = _get_index()
    stats = index.describe_index_stats()
    return index, stats.total_vector_count


try:
    index, chunk_count = connect_pinecone()
except Exception as e:
    st.error(f"Could not connect to Pinecone: {e}")
    st.info("Make sure PINECONE_API_KEY is set in secrets.")
    st.stop()


# ======================================================================
# CHAT STORE (per-user)
# ======================================================================

from api.chat_store import (
    load_index, save_index, load_conversation, save_conversation,
    generate_title, new_conversation_id, delete_conversation,
)

if "conv_index" not in st.session_state:
    st.session_state.conv_index = load_index(user_id=user_id)
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None
if "editing_conv_id" not in st.session_state:
    st.session_state.editing_conv_id = None
if "show_edit_profile" not in st.session_state:
    st.session_state.show_edit_profile = False


# ======================================================================
# SIDEBAR
# ======================================================================

with st.sidebar:
    # --- New chat button ---
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_conv_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # --- Conversation list ---
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)

    conversations = sorted(
        st.session_state.conv_index,
        key=lambda c: c.get("updated_at", ""),
        reverse=True,
    )

    groups = {"Today": [], "Yesterday": [], "This week": [], "Older": []}
    for conv in conversations:
        try:
            conv_date = datetime.fromisoformat(conv["updated_at"]).date()
            if conv_date == today:
                groups["Today"].append(conv)
            elif conv_date == yesterday:
                groups["Yesterday"].append(conv)
            elif conv_date >= week_ago:
                groups["This week"].append(conv)
            else:
                groups["Older"].append(conv)
        except (KeyError, ValueError):
            groups["Older"].append(conv)

    for label, convs in groups.items():
        if not convs:
            continue
        st.caption(label)
        for conv in convs:
            conv_id = conv["id"]
            is_active = conv_id == st.session_state.current_conv_id
            title = conv.get("title", "Untitled")

            # Delete confirmation
            if st.session_state.confirm_delete == conv_id:
                st.warning(f"Delete **{title}**?")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("Delete", key=f"yes_{conv_id}", use_container_width=True):
                        st.session_state.conv_index = delete_conversation(
                            conv_id, st.session_state.conv_index, user_id=user_id
                        )
                        if st.session_state.current_conv_id == conv_id:
                            st.session_state.current_conv_id = None
                            st.session_state.messages = []
                        st.session_state.confirm_delete = None
                        st.rerun()
                with dc2:
                    if st.button("Cancel", key=f"no_{conv_id}", use_container_width=True):
                        st.session_state.confirm_delete = None
                        st.rerun()
                continue

            # Rename inline
            if st.session_state.editing_conv_id == conv_id:
                new_title = st.text_input(
                    "Rename", value=title,
                    key=f"rename_{conv_id}", label_visibility="collapsed",
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("Save", key=f"save_{conv_id}", use_container_width=True):
                        if new_title.strip():
                            for c in st.session_state.conv_index:
                                if c["id"] == conv_id:
                                    c["title"] = new_title.strip()
                                    break
                            save_index(st.session_state.conv_index, user_id=user_id)
                        st.session_state.editing_conv_id = None
                        st.rerun()
                with rc2:
                    if st.button("Cancel", key=f"cancel_{conv_id}", use_container_width=True):
                        st.session_state.editing_conv_id = None
                        st.rerun()
                continue

            # Conversation row
            cols = st.columns([5, 1])
            with cols[0]:
                btn_label = f"{'> ' if is_active else '  '}{title}"
                if st.button(btn_label, key=f"conv_{conv_id}",
                             use_container_width=True, disabled=is_active):
                    loaded = load_conversation(conv_id, user_id=user_id)
                    if loaded is not None:
                        st.session_state.current_conv_id = conv_id
                        st.session_state.messages = loaded
                        st.session_state.confirm_delete = None
                        st.session_state.editing_conv_id = None
                        st.rerun()

            with cols[1]:
                with st.popover("...", use_container_width=True):
                    if st.button("Rename", key=f"ren_{conv_id}", use_container_width=True):
                        st.session_state.editing_conv_id = conv_id
                        st.rerun()
                    if st.button("Delete", key=f"del_{conv_id}", use_container_width=True):
                        st.session_state.confirm_delete = conv_id
                        st.rerun()

    st.divider()

    # --- Car info card (Rennlist thead-style) ---
    car_year = car_profile.get("year", "")
    car_model = car_profile.get("model", "993")
    car_trans = car_profile.get("transmission", "")
    car_miles = car_profile.get("mileage", "")
    car_line = f"{car_year} 993 {car_model}"
    if car_trans:
        car_line += f" &middot; {car_trans}"

    miles_line = ""
    if car_miles:
        miles_line = f'<p class="card-value">~{car_miles} mi</p>'

    st.markdown(f"""
    <div class="car-info-card">
        <div class="card-header">Your 993</div>
        <div class="card-body">
            <p class="card-value">{car_line}</p>
            {miles_line}
            <p class="card-label">Knowledge base</p>
            <p class="card-value">{chunk_count:,} chunks indexed</p>
            <p class="card-sources">
                Pelican Parts &middot; Rennlist &middot; 911uk &middot; 6SpeedOnline &middot; TIPEC &middot; Carpokes &middot; YouTube
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Edit profile
    if st.button("Edit car profile", use_container_width=True):
        st.session_state.show_edit_profile = not st.session_state.show_edit_profile
        st.rerun()

    st.divider()

    # User info + logout
    first_name = display_name.split()[0] if display_name else user_email
    st.markdown(f'<p class="user-info">Signed in as <strong>{first_name}</strong></p>', unsafe_allow_html=True)
    if not _DEV_MODE:
        if st.button("Sign out", use_container_width=True):
            st.logout()


# ======================================================================
# EDIT PROFILE (inline, if toggled)
# ======================================================================

if st.session_state.show_edit_profile:
    _show_edit_profile()
    st.stop()


# ======================================================================
# MAIN CHAT AREA
# ======================================================================

# Header (dark slate bar)
car_badge_text = f"{car_profile.get('year', '')} {car_profile.get('model', '')}"
if car_profile.get("transmission"):
    car_badge_text += f" &middot; {car_profile['transmission']}"
if car_profile.get("mileage"):
    car_badge_text += f" &middot; ~{car_profile['mileage']} mi"

st.markdown(f"""
<div class="app-header">
    <div class="app-header-title">993 Repair Assistant</div>
    <p class="app-header-sub">Ask anything about your Porsche 993 &mdash; powered by real forum knowledge.</p>
    <span class="app-car-tag">{car_badge_text.strip()}</span>
</div>
""", unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <p class="empty-state-title">Start a conversation</p>
        <p class="empty-state-desc">Ask about repairs, maintenance, part numbers, or troubleshooting.</p>
    </div>
    """, unsafe_allow_html=True)

# Chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about your 993..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching forum knowledge..."):
            from api.chat import (
                search, build_context, build_system_prompt,
                extract_part_numbers, generate_parts_links,
                _car_description, rewrite_follow_up,
            )
            import anthropic

            # Rewrite follow-up questions to include conversation context
            # so the RAG search finds relevant chunks
            search_query = rewrite_follow_up(prompt, st.session_state.messages[:-1])
            sources = search(search_query)
            context = build_context(sources)
            system_prompt = build_system_prompt(car_profile)
            car_desc = _car_description(car_profile)

            previous = st.session_state.messages[-11:-1]
            claude_messages = [
                {"role": m["role"], "content": m["content"]} for m in previous
            ]

            user_message = f"""Based on the following knowledge from Porsche forums and technical articles,
answer this question about the owner's {car_desc}:

QUESTION: {prompt}

FORUM KNOWLEDGE:
{context}

Please provide a helpful, practical answer based on this knowledge."""

            claude_messages.append({"role": "user", "content": user_message})

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key or api_key == "your_anthropic_api_key_here":
                st.error("Please set ANTHROPIC_API_KEY in secrets.")
                st.stop()

            client = anthropic.Anthropic(api_key=api_key)

            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=system_prompt,
                messages=claude_messages,
            ) as stream:
                response = st.write_stream(
                    (text for text in stream.text_stream)
                )

            # Source links
            unique_urls = []
            seen = set()
            for s in sources[:5]:
                if s["url"] and s["url"] not in seen:
                    seen.add(s["url"])
                    unique_urls.append((s["title"][:60], s["url"], s["source"]))

            source_md = ""
            if unique_urls:
                source_md = "\n\n---\n**Sources**\n"
                source_md += "\n".join(
                    f"- [{t}]({u}) *({s})*" for t, u, s in unique_urls
                )

            # Parts links
            response_text = response if isinstance(response, str) else str(response)
            part_numbers = extract_part_numbers(response_text)
            parts_md = generate_parts_links(part_numbers)

    full_response = response_text + source_md + parts_md
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
    })

    now = datetime.now().isoformat()
    if st.session_state.current_conv_id is None:
        conv_id = new_conversation_id()
        st.session_state.current_conv_id = conv_id
        title = generate_title(prompt)
        st.session_state.conv_index.append({
            "id": conv_id, "title": title,
            "created_at": now, "updated_at": now,
        })
    else:
        for conv in st.session_state.conv_index:
            if conv["id"] == st.session_state.current_conv_id:
                conv["updated_at"] = now
                break

    save_index(st.session_state.conv_index, user_id=user_id)
    save_conversation(st.session_state.current_conv_id, st.session_state.messages, user_id=user_id)
    st.rerun()
