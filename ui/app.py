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
    page_icon="🔧",
    layout="centered",
)

# --- Forum Theme CSS (Pelican Parts Light Theme) ---
st.markdown("""
<style>
    /* ====== PELICAN PARTS FORUM THEME — LIGHT ====== */

    /* --- Global --- */
    .stApp {
        font-family: Verdana, Arial, Helvetica, sans-serif !important;
        background-color: #FFFFFF !important;
    }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stChatInputTextArea"] {
        font-family: Verdana, Arial, Helvetica, sans-serif !important;
        font-size: 0.9em !important;
    }
    .main [data-testid="stMarkdownContainer"],
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] li {
        color: #333333 !important;
    }
    [data-testid="stChatInputTextArea"] {
        color: #333333 !important;
    }
    [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"],
    [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] p {
        color: inherit !important;
    }

    /* --- Forum Header Banner --- */
    .forum-header {
        background: linear-gradient(180deg, #0B198C 0%, #07104a 100%);
        border: 1px solid #DEDFDF;
        border-radius: 6px;
        margin-bottom: 1.2rem;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.12);
    }
    .forum-topbar {
        background: #050b30;
        color: #FFCF87;
        font-size: 0.65em;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 5px 16px;
        border-bottom: 1px solid rgba(255,207,135,0.2);
        font-family: Tahoma, Verdana, sans-serif;
    }
    .forum-topbar a { color: #FFCF87 !important; text-decoration: none; }
    .forum-banner-content {
        padding: 18px 24px 14px;
        text-align: center;
    }
    .forum-banner-content h1 {
        color: #ffffff;
        font-size: 1.6em;
        font-family: Tahoma, 'Trebuchet MS', Verdana, sans-serif;
        font-weight: bold;
        margin: 0 0 2px 0;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.6);
    }
    .forum-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent 5%, #FFCF87 50%, transparent 95%);
        margin: 8px auto;
        width: 60%;
    }
    .forum-subtitle {
        color: rgba(255,255,255,0.75);
        font-size: 0.78em;
        margin: 4px 0 0 0;
        font-family: Verdana, Arial, sans-serif;
    }
    .forum-car-badge {
        display: inline-block;
        background: rgba(255,207,135,0.15);
        border: 1px solid rgba(255,207,135,0.3);
        border-radius: 4px;
        padding: 3px 10px;
        margin-top: 8px;
        color: #FFCF87;
        font-size: 0.7em;
        letter-spacing: 0.5px;
    }
    .forum-gold-bar {
        height: 3px;
        background: linear-gradient(90deg, #FFCF87, #d4a84b, #FFCF87);
    }

    /* --- Chat Messages --- */
    [data-testid="stChatMessage"] {
        background: #F7F7F7 !important;
        border: 1px solid #DEDFDF !important;
        border-radius: 4px !important;
        margin-bottom: 10px !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        color: #333333 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-left: 3px solid #0B198C !important;
        background: #F0F0F5 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 3px solid #FFBF67 !important;
        background: #F7F7F7 !important;
    }

    /* --- Links --- */
    a { color: #0B198C !important; }
    a:hover { color: #FF6633 !important; text-decoration: underline !important; }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background-color: #F0F1F3 !important;
        border-right: 2px solid #0B198C !important;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #F0F1F3 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] div,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] span {
        background-color: #0B198C !important;
        border: 1px solid #0B198C !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover {
        background-color: #0e1fa8 !important;
    }
    section[data-testid="stSidebar"] .stButton > button {
        text-align: left;
        font-size: 0.8em;
        padding: 6px 10px;
        border: none;
        background: transparent;
        color: #444444;
        font-family: Verdana, Arial, sans-serif;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(11, 25, 140, 0.08);
        color: #0B198C;
    }
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #0B198C !important;
        font-size: 0.7em;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: bold;
    }

    /* --- Popover trigger --- */
    section[data-testid="stSidebar"] [data-testid="stPopover"] > button,
    section[data-testid="stSidebar"] button:has([data-testid="stIconMaterial"]) {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 4px 2px !important;
        min-height: 0 !important;
        color: #888 !important;
        outline: none !important;
    }
    section[data-testid="stSidebar"] [data-testid="stPopover"] > button:hover,
    section[data-testid="stSidebar"] button:has([data-testid="stIconMaterial"]):hover {
        color: #0B198C !important;
        background: transparent !important;
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
        background-color: #FFFFFF !important;
        border-color: #DEDFDF !important;
    }
    [data-testid="stPopoverBody"] {
        border-radius: 4px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
        padding: 2px 0 !important;
        min-width: 0 !important;
    }
    [data-testid="stPopoverBody"] > div {
        padding: 2px 4px !important;
        gap: 0 !important;
    }
    [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
    [data-testid="stPopoverBody"] [data-testid="stElementContainer"] { margin: 0 !important; padding: 0 !important; }
    [data-testid="stPopoverBody"] button {
        color: #333333 !important;
        background: transparent !important;
        border: none !important;
        padding: 4px 12px !important;
        min-height: 0 !important;
        font-size: 0.82em !important;
    }
    [data-testid="stPopoverBody"] button:hover {
        background: #F0F1F3 !important;
        color: #0B198C !important;
    }
    [data-testid="stPopoverBody"] button [data-testid="stMarkdownContainer"] p {
        color: #333333 !important;
        font-size: 0.82em !important;
        margin: 0 !important;
    }
    [data-testid="stPopoverBody"] button:hover [data-testid="stMarkdownContainer"] p {
        color: #0B198C !important;
    }

    /* --- Chat Input --- */
    [data-testid="stChatInput"] {
        background: #FFFFFF !important;
        border: 1px solid #DEDFDF !important;
        border-radius: 6px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #0B198C !important;
        box-shadow: 0 0 0 1px #0B198C !important;
    }

    /* --- Sidebar car info --- */
    .sidebar-car-info {
        background: linear-gradient(180deg, #0B198C, #07104a);
        border: 1px solid #DEDFDF;
        border-radius: 6px;
        padding: 12px;
        font-size: 0.75em;
        color: #d0d4dc;
    }
    .sidebar-car-info strong { color: #FFCF87; }
    .sidebar-sources {
        color: #a0a8b8;
        font-size: 0.9em;
        margin-top: 6px;
    }

    /* --- Spinner --- */
    [data-testid="stSpinner"] { color: #0B198C !important; }

    /* --- Chat hr / bold --- */
    [data-testid="stChatMessage"] hr { border-color: #DEDFDF !important; }
    [data-testid="stChatMessage"] strong { color: #0B198C !important; }

    /* --- Header / bottom / main bg fixes --- */
    header[data-testid="stHeader"] { background-color: #FFFFFF !important; }
    [data-testid="stBottomBlockContainer"] { background-color: #FFFFFF !important; }
    [data-testid="stBottom"] > div { background-color: #FFFFFF !important; }
    .stChatInput, .stChatInput > div, .stChatInput > div > div,
    .stChatInput > div > div > div, .stChatInput div { background-color: #FFFFFF !important; }
    .stChatInput textarea { background-color: #FFFFFF !important; color: #333333 !important; }
    .stChatInput { border: 1px solid #DEDFDF !important; border-radius: 8px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
    .stChatInput:focus-within { border-color: #0B198C !important; box-shadow: 0 0 0 1px #0B198C !important; }
    .stChatInput button { background-color: #0B198C !important; color: #FFFFFF !important; }
    .stChatInput button:hover { background-color: #0e1fa8 !important; }
    .stChatInput button svg { fill: #FFFFFF !important; color: #FFFFFF !important; }
    .main .block-container { background-color: #FFFFFF !important; }
    [data-testid="stMainBlockContainer"] { background-color: #FFFFFF !important; }
    .stApp > div, .stApp > div > div { background-color: transparent !important; }
    [data-testid="stPopover"] > div { background-color: #FFFFFF !important; border: 1px solid #DEDFDF !important; }

    /* --- Google login button styling --- */
    .google-login-btn {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: #FFFFFF;
        border: 2px solid #0B198C;
        border-radius: 6px;
        padding: 12px 28px;
        font-size: 1.05em;
        font-family: Verdana, Arial, sans-serif;
        color: #0B198C;
        cursor: pointer;
        font-weight: bold;
        margin-top: 12px;
        transition: all 0.15s;
    }
    .google-login-btn:hover {
        background: #0B198C;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================
# AUTHENTICATION — Google Login via st.login()
# ======================================================================

from api.auth import user_id_from_email, load_user_profile, save_user_profile, decode_vin

# Debug: show Streamlit version and auth state
_st_version = st.__version__

try:
    _is_logged_in = st.user.is_logged_in
except Exception as _auth_err:
    st.error(f"Auth check failed (Streamlit {_st_version}): {_auth_err}")
    st.info("This may mean your `[auth]` secrets are not configured correctly.")
    st.code(f"st.user type: {type(getattr(st, 'user', 'MISSING'))}")
    st.stop()

if not _is_logged_in:
    # --- Landing page for unauthenticated users ---
    st.markdown("""
    <div class="forum-header">
        <div class="forum-topbar">
            Porsche Forums &rsaquo; 993 Technical &rsaquo; Knowledge Base
        </div>
        <div class="forum-banner-content">
            <h1>&#x1F527; 993 Repair Assistant</h1>
            <div class="forum-divider"></div>
            <p class="forum-subtitle">
                Powered by real Porsche forum knowledge
            </p>
        </div>
        <div class="forum-gold-bar"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### Welcome to the 993 Repair Assistant")
    st.markdown(
        "Get expert repair advice for your Porsche 993, powered by **140,000+ real forum posts** "
        "from Pelican Parts, Rennlist, 911uk, and more."
    )
    st.markdown("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔐  Sign in with Google", use_container_width=True, type="primary"):
            st.login()

    st.markdown("")
    st.caption("Sign in to save your chat history and personalize advice to your car.")
    st.stop()

# --- User is authenticated ---
try:
    _user_dict = st.user.to_dict()
    user_email = _user_dict.get("email", "")
    display_name = _user_dict.get("name", user_email)
except Exception as _ue:
    # Fallback: try direct access
    try:
        user_email = st.user["email"]
        display_name = st.user.get("name", user_email)
    except Exception as _ue2:
        st.error(f"Could not read user info after login: {_ue2}")
        st.code(f"st.user = {st.user}\ntype = {type(st.user)}")
        if st.button("Sign out and try again"):
            st.logout()
        st.stop()

if not user_email:
    st.warning("Logged in but no email found. Please sign out and try again.")
    st.code(f"st.user = {st.user.to_dict() if hasattr(st.user, 'to_dict') else str(st.user)}")
    if st.button("Sign out"):
        st.logout()
    st.stop()

user_id = user_id_from_email(user_email)


# ======================================================================
# CAR PROFILE ONBOARDING
# ======================================================================

if "car_profile" not in st.session_state:
    profile = load_user_profile(user_id)
    st.session_state.car_profile = profile  # None if first visit


def _show_onboarding():
    """Show the car profile onboarding form. Returns True if completed."""
    st.markdown("""
    <div class="forum-header">
        <div class="forum-topbar">
            Porsche Forums &rsaquo; 993 Technical &rsaquo; Setup
        </div>
        <div class="forum-banner-content">
            <h1>&#x1F3CE; Tell us about your 993</h1>
            <div class="forum-divider"></div>
            <p class="forum-subtitle">
                We'll personalize all advice to your specific car
            </p>
        </div>
        <div class="forum-gold-bar"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"Welcome, **{display_name}**! Let's set up your car profile.")

    with st.form("car_profile_form"):
        st.subheader("Your Car Details")

        vin = st.text_input(
            "VIN (optional — auto-fills fields below)",
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
        mileage = st.text_input("Approximate Mileage", placeholder="80,000")
        known_issues = st.text_area(
            "Known Issues (optional)",
            placeholder="e.g. Oil leak from RMS, soft top motor slow, AC needs recharge...",
            height=100,
        )

        submitted = st.form_submit_button("Save & Start Chatting", type="primary", use_container_width=True)

        if submitted:
            if vin and len(vin) == 17:
                decoded = decode_vin(vin)
                if decoded:
                    if not year and decoded.get("year"):
                        year = decoded["year"]
                    if not model and decoded.get("model"):
                        model = decoded["model"]

            if not year or not model:
                st.error("Please at least select the year and model of your 993.")
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

    with st.form("edit_profile_form"):
        st.subheader("Edit Car Profile")

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

        mileage = st.text_input("Approximate Mileage", value=profile.get("mileage", ""))
        known_issues = st.text_area(
            "Known Issues",
            value=profile.get("known_issues", ""),
            height=100,
        )

        if st.form_submit_button("Save Changes", type="primary", use_container_width=True):
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
    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_conv_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    week_ago = today - timedelta(days=7)

    conversations = sorted(
        st.session_state.conv_index,
        key=lambda c: c.get("updated_at", ""),
        reverse=True,
    )

    groups = {"Today": [], "Yesterday": [], "This Week": [], "Older": []}
    for conv in conversations:
        try:
            conv_date = datetime.fromisoformat(conv["updated_at"]).date()
            if conv_date == today:
                groups["Today"].append(conv)
            elif conv_date == yesterday:
                groups["Yesterday"].append(conv)
            elif conv_date >= week_ago:
                groups["This Week"].append(conv)
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

            if st.session_state.confirm_delete == conv_id:
                st.warning(f"Delete **{title}**?", icon="⚠️")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("Yes, delete", key=f"yes_{conv_id}", use_container_width=True):
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

            cols = st.columns([5, 1])
            with cols[0]:
                btn_label = f"{'▸ ' if is_active else '  '}{title}"
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
                with st.popover("···", use_container_width=True):
                    if st.button("Rename", key=f"ren_{conv_id}", use_container_width=True):
                        st.session_state.editing_conv_id = conv_id
                        st.rerun()
                    if st.button("Delete", key=f"del_{conv_id}", use_container_width=True):
                        st.session_state.confirm_delete = conv_id
                        st.rerun()

    if conversations:
        st.divider()

    # Edit profile
    if st.button("✏️ Edit Car Profile", use_container_width=True):
        st.session_state.show_edit_profile = not st.session_state.show_edit_profile
        st.rerun()

    st.divider()

    # Car info footer
    car_year = car_profile.get("year", "")
    car_model = car_profile.get("model", "993")
    car_trans = car_profile.get("transmission", "")
    car_miles = car_profile.get("mileage", "")
    car_line = f"{car_year} 993 {car_model}"
    if car_trans:
        car_line += f" {car_trans}"
    if car_miles:
        car_line += f" · ~{car_miles} mi"

    st.markdown(f"""
    <div class="sidebar-car-info">
        <strong>Your Car</strong><br>
        {car_line}<br><br>
        <strong>Knowledge Base</strong><br>
        {chunk_count:,} forum posts indexed<br>
        <div class="sidebar-sources">
            Pelican Parts · Rennlist · 911uk ·
            6SpeedOnline · TIPEC · Carpokes ·
            p-car.com · YouTube · Blogs
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # User info + logout
    st.caption(f"Signed in as **{display_name}**")
    if st.button("Sign out"):
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

car_badge = f"{car_profile.get('year', '')} {car_profile.get('model', '')} {car_profile.get('transmission', '')}"
if car_profile.get("mileage"):
    car_badge += f" &middot; ~{car_profile['mileage']} mi"

st.markdown(f"""
<div class="forum-header">
    <div class="forum-topbar">
        Porsche Forums &rsaquo; 993 Technical &rsaquo; Knowledge Base
    </div>
    <div class="forum-banner-content">
        <h1>&#x1F527; 993 Repair Assistant</h1>
        <div class="forum-divider"></div>
        <p class="forum-subtitle">
            Powered by real Porsche forum knowledge
        </p>
        <span class="forum-car-badge">
            {car_badge.strip()}
        </span>
    </div>
    <div class="forum-gold-bar"></div>
</div>
""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your 993..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching forum knowledge..."):
            from api.chat import (
                search, build_context, build_system_prompt,
                extract_part_numbers, generate_parts_links,
                _car_description,
            )
            import anthropic

            sources = search(prompt)
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
                source_md = "\n\n---\n**📚 Sources**\n"
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
