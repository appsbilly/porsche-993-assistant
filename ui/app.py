"""
Streamlit web UI for the Porsche 993 Repair Assistant.
Uses Pinecone for vector search, Claude for answer generation,
and S3 for persistent conversation history.

Run with: streamlit run ui/app.py
"""

import os
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
    /* Text color — scoped to avoid overriding button internals */
    .main [data-testid="stMarkdownContainer"],
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] li {
        color: #333333 !important;
    }
    [data-testid="stChatInputTextArea"] {
        color: #333333 !important;
    }
    /* Force white text inside primary buttons (New Chat) */
    [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"],
    [data-testid="stBaseButton-primary"] [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }
    /* Sidebar buttons inherit parent color */
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
    .forum-topbar a {
        color: #FFCF87 !important;
        text-decoration: none;
    }
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
    /* User messages — navy left border */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        border-left: 3px solid #0B198C !important;
        background: #F0F0F5 !important;
    }
    /* Assistant messages — gold left border */
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
    /* Sidebar "New Chat" primary button */
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
    /* Sidebar conversation buttons */
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
    /* Sidebar date group captions */
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #0B198C !important;
        font-size: 0.7em;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: bold;
    }

    /* --- ··· popover trigger (plain text, no box) --- */
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
    /* Hide dropdown arrow icon */
    section[data-testid="stSidebar"] [data-testid="stIconMaterial"] {
        display: none !important;
    }
    /* Remove popover container box/border — target the wrapper div around the button */
    section[data-testid="stSidebar"] [data-testid="stPopover"],
    section[data-testid="stSidebar"] [data-testid="stPopover"] > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    /* Popover dropdown panel — override ALL inner divs */
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
    /* Tighten popover inner container */
    [data-testid="stPopoverBody"] > div {
        padding: 2px 4px !important;
        gap: 0 !important;
    }
    [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }
    [data-testid="stPopoverBody"] [data-testid="stElementContainer"] {
        margin: 0 !important;
        padding: 0 !important;
    }
    /* Rename/Delete buttons inside popover — compact */
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

    /* --- Sidebar car info at bottom --- */
    .sidebar-car-info {
        background: linear-gradient(180deg, #0B198C, #07104a);
        border: 1px solid #DEDFDF;
        border-radius: 6px;
        padding: 12px;
        font-size: 0.75em;
        color: #d0d4dc;
    }
    .sidebar-car-info strong {
        color: #FFCF87;
    }
    .sidebar-sources {
        color: #a0a8b8;
        font-size: 0.9em;
        margin-top: 6px;
    }

    /* --- Spinner/loading --- */
    [data-testid="stSpinner"] {
        color: #0B198C !important;
    }

    /* --- Horizontal rules in chat (source dividers) --- */
    [data-testid="stChatMessage"] hr {
        border-color: #DEDFDF !important;
    }

    /* --- Strong/bold text in chat --- */
    [data-testid="stChatMessage"] strong {
        color: #0B198C !important;
    }

    /* --- Streamlit header bar override --- */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }

    /* --- Bottom block (below chat input) --- */
    [data-testid="stBottomBlockContainer"] {
        background-color: #FFFFFF !important;
    }

    /* --- Fix dark Streamlit bottom wrapper --- */
    [data-testid="stBottom"] > div {
        background-color: #FFFFFF !important;
    }

    /* --- Chat input — override ALL dark internals --- */
    .stChatInput,
    .stChatInput > div,
    .stChatInput > div > div,
    .stChatInput > div > div > div,
    .stChatInput div {
        background-color: #FFFFFF !important;
    }
    .stChatInput textarea {
        background-color: #FFFFFF !important;
        color: #333333 !important;
    }
    /* Chat input outer border */
    .stChatInput {
        border: 1px solid #DEDFDF !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    .stChatInput:focus-within {
        border-color: #0B198C !important;
        box-shadow: 0 0 0 1px #0B198C !important;
    }
    /* Send button */
    .stChatInput button {
        background-color: #0B198C !important;
        color: #FFFFFF !important;
    }
    .stChatInput button:hover {
        background-color: #0e1fa8 !important;
    }
    .stChatInput button svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }

    /* --- Streamlit main content area --- */
    .main .block-container {
        background-color: #FFFFFF !important;
    }
    [data-testid="stMainBlockContainer"] {
        background-color: #FFFFFF !important;
    }

    /* --- Fix any remaining dark backgrounds from Streamlit internals --- */
    .stApp > div,
    .stApp > div > div {
        background-color: transparent !important;
    }

    /* --- Popover menu styling (light) --- */
    [data-testid="stPopover"] > div {
        background-color: #FFFFFF !important;
        border: 1px solid #DEDFDF !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def connect_pinecone():
    """Connect to Pinecone and load embedding model (cached)."""
    from api.chat import _get_index, _get_model
    index = _get_index()
    model = _get_model()
    stats = index.describe_index_stats()
    return index, model, stats.total_vector_count


# --- Connect to Pinecone ---
try:
    index, embed_model, chunk_count = connect_pinecone()
except Exception as e:
    st.error(f"❌ Could not connect to Pinecone: {e}")
    st.info("Make sure you've:\n"
            "1. Set `PINECONE_API_KEY` in your `.env` file\n"
            "2. Run the pipeline: `python embeddings/build_db.py`")
    st.stop()


# --- Chat Store ---
from api.chat_store import (
    load_index, save_index, load_conversation, save_conversation,
    generate_title, new_conversation_id, delete_conversation,
)


# --- Session State Init ---
if "conv_index" not in st.session_state:
    st.session_state.conv_index = load_index()
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None
if "editing_conv_id" not in st.session_state:
    st.session_state.editing_conv_id = None


# --- Sidebar ---
with st.sidebar:
    # New Chat button
    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        st.session_state.current_conv_id = None
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Group conversations by date
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

            # --- Delete confirmation mode ---
            if st.session_state.confirm_delete == conv_id:
                st.warning(f"Delete **{title}**?", icon="⚠️")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("Yes, delete", key=f"yes_{conv_id}", use_container_width=True):
                        st.session_state.conv_index = delete_conversation(
                            conv_id, st.session_state.conv_index
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

            # --- Rename mode ---
            if st.session_state.editing_conv_id == conv_id:
                new_title = st.text_input(
                    "Rename",
                    value=title,
                    key=f"rename_{conv_id}",
                    label_visibility="collapsed",
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("Save", key=f"save_{conv_id}", use_container_width=True):
                        if new_title.strip():
                            for c in st.session_state.conv_index:
                                if c["id"] == conv_id:
                                    c["title"] = new_title.strip()
                                    break
                            save_index(st.session_state.conv_index)
                        st.session_state.editing_conv_id = None
                        st.rerun()
                with rc2:
                    if st.button("Cancel", key=f"cancel_{conv_id}", use_container_width=True):
                        st.session_state.editing_conv_id = None
                        st.rerun()
                continue

            # --- Normal conversation row ---
            cols = st.columns([5, 1])
            with cols[0]:
                btn_label = f"{'▸ ' if is_active else '  '}{title}"
                if st.button(
                    btn_label,
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    disabled=is_active,
                ):
                    loaded = load_conversation(conv_id)
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

    st.divider()
    st.markdown(f"""
    <div class="sidebar-car-info">
        <strong>Your Car</strong><br>
        1997 993 Targa Tiptronic · ~80k mi<br><br>
        <strong>Knowledge Base</strong><br>
        {chunk_count:,} forum posts indexed<br>
        <div class="sidebar-sources">
            Pelican Parts · Rennlist · 911uk ·
            6SpeedOnline · TIPEC · Carpokes ·
            p-car.com · YouTube · Blogs
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Forum Banner Header ---
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
        <span class="forum-car-badge">
            1997 Targa Tiptronic &middot; ~80,000 mi
        </span>
    </div>
    <div class="forum-gold-bar"></div>
</div>
""", unsafe_allow_html=True)


# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# --- Chat input ---
if prompt := st.chat_input("Ask about your 993..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching forum knowledge..."):
            from api.chat import search, build_context, SYSTEM_PROMPT
            import anthropic

            # Search Pinecone
            sources = search(prompt)

            # Build RAG context
            context = build_context(sources)

            # Build messages with conversation history for continuity.
            # Send up to the last 10 messages (5 Q&A pairs) so Claude
            # remembers what was discussed, without blowing up tokens.
            previous = st.session_state.messages[-11:-1]
            claude_messages = []
            for msg in previous:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            # Current message with RAG context attached
            user_message = f"""Based on the following knowledge from Porsche forums and technical articles,
answer this question about the owner's 1997 Porsche 993 Targa Tiptronic (~80k miles):

QUESTION: {prompt}

FORUM KNOWLEDGE:
{context}

Please provide a helpful, practical answer based on this knowledge."""

            claude_messages.append({"role": "user", "content": user_message})

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key or api_key == "your_anthropic_api_key_here":
                st.error("⚠️ Please set ANTHROPIC_API_KEY in your .env file")
                st.stop()

            client = anthropic.Anthropic(api_key=api_key)

            # Stream the response
            with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=claude_messages,
            ) as stream:
                response = st.write_stream(
                    (text for text in stream.text_stream)
                )

            # Build source links to include in the saved message
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

    # Save assistant response + sources to session state
    response_text = response if isinstance(response, str) else str(response)
    full_response = response_text + source_md
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
    })

    # --- Persist conversation to S3 ---
    now = datetime.now().isoformat()

    if st.session_state.current_conv_id is None:
        # First message in a new conversation — generate a title
        conv_id = new_conversation_id()
        st.session_state.current_conv_id = conv_id
        title = generate_title(prompt)
        st.session_state.conv_index.append({
            "id": conv_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        })
    else:
        # Existing conversation — just update the timestamp
        for conv in st.session_state.conv_index:
            if conv["id"] == st.session_state.current_conv_id:
                conv["updated_at"] = now
                break

    save_index(st.session_state.conv_index)
    save_conversation(st.session_state.current_conv_id, st.session_state.messages)

    # Rerun so the sidebar updates with the new/updated conversation
    st.rerun()
