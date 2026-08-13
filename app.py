import streamlit as st
import time
import uuid

from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question


# =============================================================================
# ENVIRONMENT
# =============================================================================

load_dotenv()


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');


/* =========================================================
   ROOT
   ========================================================= */

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;

    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;

    --text: #e8e8f0;
    --text-muted: #7070a0;

    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}


/* =========================================================
   GLOBAL
   ========================================================= */

html,
body,
[data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
}

.stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
}

* {
    font-family: 'JetBrains Mono', monospace;
}


/* =========================================================
   GRID BACKGROUND
   ========================================================= */

.stApp::before {
    content: "";

    position: fixed;

    top: 0;
    left: 0;

    width: 100%;
    height: 100%;

    background-image:
        linear-gradient(
            rgba(124, 58, 237, 0.03) 1px,
            transparent 1px
        ),
        linear-gradient(
            90deg,
            rgba(124, 58, 237, 0.03) 1px,
            transparent 1px
        );

    background-size: 40px 40px;

    pointer-events: none;

    z-index: 0;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}


/* =========================================================
   HEADINGS
   ========================================================= */

h1,
h2,
h3,
h4,
h5,
h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}


/* =========================================================
   HERO
   ========================================================= */

.hero-title {
    font-family: 'Syne', sans-serif;

    font-size: clamp(2rem, 5vw, 3.5rem);

    font-weight: 800;

    line-height: 1.1;

    margin: 0;

    background:
        linear-gradient(
            135deg,
            #ffffff 0%,
            var(--accent-glow) 50%,
            var(--accent-2) 100%
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;

    font-size: 0.8rem;

    color: var(--text-muted);

    letter-spacing: 0.2em;

    text-transform: uppercase;

    margin-top: 0.5rem;
}


/* =========================================================
   CARDS
   ========================================================= */

.card {
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 12px;

    padding: 1.5rem;

    margin-bottom: 1rem;

    position: relative;

    overflow: hidden;

    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: "";

    position: absolute;

    top: 0;
    left: 0;

    width: 3px;
    height: 100%;

    background:
        linear-gradient(
            180deg,
            var(--accent),
            var(--accent-2)
        );
}

.card-title {
    font-family: 'Syne', sans-serif;

    font-size: 0.7rem;

    font-weight: 700;

    letter-spacing: 0.15em;

    text-transform: uppercase;

    color: var(--text-muted);

    margin-bottom: 0.75rem;

    display: flex;

    align-items: center;

    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;

    line-height: 1.7;

    color: var(--text);

    white-space: pre-wrap;

    word-break: break-word;
}


/* =========================================================
   BADGES
   ========================================================= */

.badge {
    display: inline-block;

    padding: 0.2rem 0.6rem;

    border-radius: 4px;

    font-size: 0.65rem;

    font-weight: 600;

    letter-spacing: 0.1em;

    text-transform: uppercase;
}

.badge-purple {
    background: rgba(124, 58, 237, 0.2);

    color: var(--accent-glow);

    border: 1px solid rgba(124, 58, 237, 0.3);
}

.badge-cyan {
    background: rgba(6, 182, 212, 0.15);

    color: var(--accent-2);

    border: 1px solid rgba(6, 182, 212, 0.3);
}

.badge-green {
    background: rgba(16, 185, 129, 0.15);

    color: var(--success);

    border: 1px solid rgba(16, 185, 129, 0.3);
}


/* =========================================================
   INPUTS
   ========================================================= */

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;

    border: 1px solid var(--border) !important;

    border-radius: 8px !important;

    color: var(--text) !important;

    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;

    box-shadow:
        0 0 0 2px rgba(124, 58, 237, 0.2) !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    background:
        linear-gradient(
            135deg,
            var(--accent),
            #5b21b6
        ) !important;

    color: white !important;

    border: none !important;

    border-radius: 8px !important;

    font-family: 'Syne', sans-serif !important;

    font-weight: 700 !important;

    font-size: 0.875rem !important;

    letter-spacing: 0.05em !important;

    padding: 0.6rem 1.5rem !important;

    transition: all 0.2s !important;

    text-transform: uppercase !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;

    box-shadow:
        0 8px 25px rgba(124, 58, 237, 0.4) !important;
}


/* =========================================================
   STATUS
   ========================================================= */

.status-bar {
    display: flex;

    align-items: center;

    gap: 0.75rem;

    padding: 0.75rem 1rem;

    background: var(--surface-2);

    border-radius: 8px;

    margin: 0.4rem 0;

    border: 1px solid var(--border);

    font-size: 0.8rem;
}

.status-dot {
    width: 8px;
    height: 8px;

    border-radius: 50%;

    flex-shrink: 0;
}

.dot-active {
    background: var(--accent-glow);

    box-shadow:
        0 0 8px var(--accent-glow);

    animation: pulse 1.5s infinite;
}

.dot-done {
    background: var(--success);
}

.dot-pending {
    background: var(--border);
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }

    50% {
        opacity: 0.4;
    }
}


/* =========================================================
   CHAT
   ========================================================= */

.chat-container {
    background: var(--surface);

    border: 1px solid var(--border);

    border-radius: 12px;

    padding: 1.25rem;

    max-height: 420px;

    overflow-y: auto;

    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;

    display: flex;

    flex-direction: column;

    gap: 0.2rem;
}

.chat-label {
    font-size: 0.65rem;

    font-weight: 700;

    letter-spacing: 0.15em;

    text-transform: uppercase;
}

.chat-bubble {
    display: inline-block;

    padding: 0.6rem 1rem;

    border-radius: 10px;

    font-size: 0.85rem;

    line-height: 1.6;

    max-width: 90%;

    white-space: pre-wrap;

    word-break: break-word;
}

.user-label {
    color: var(--accent-glow);
}

.bot-label {
    color: var(--accent-2);
}

.user-bubble {
    background: rgba(124, 58, 237, 0.15);

    border: 1px solid rgba(124, 58, 237, 0.25);

    align-self: flex-end;
}

.bot-bubble {
    background: rgba(6, 182, 212, 0.1);

    border: 1px solid rgba(6, 182, 212, 0.2);

    align-self: flex-start;
}


/* =========================================================
   TRANSCRIPT
   ========================================================= */

.transcript-box {
    background: var(--surface-2);

    border: 1px solid var(--border);

    border-radius: 8px;

    padding: 1.25rem;

    font-size: 0.82rem;

    line-height: 1.8;

    max-height: 300px;

    overflow-y: auto;

    color: var(--text-muted);

    white-space: pre-wrap;

    word-break: break-word;
}


/* =========================================================
   STREAMLIT
   ========================================================= */

.stProgress > div > div > div {
    background: var(--accent) !important;
}

.stSpinner > div {
    border-top-color: var(--accent) !important;
}

label {
    color: var(--text-muted) !important;

    font-size: 0.8rem !important;
}


/* =========================================================
   SCROLLBAR
   ========================================================= */

::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}

::-webkit-scrollbar-track {
    background: var(--bg);
}

::-webkit-scrollbar-thumb {
    background: var(--border);

    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}

</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# SESSION STATE
# =============================================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processing" not in st.session_state:
    st.session_state.processing = False

if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = {}

if "video_id" not in st.session_state:
    st.session_state.video_id = None


# =============================================================================
# STATUS HELPER
# =============================================================================

def get_step_class(step):

    status = st.session_state.pipeline_steps.get(
        step,
        "pending"
    )

    if status == "active":
        return "dot-active"

    if status == "done":
        return "dot-done"

    return "dot-pending"


def render_step_bar(
    label,
    step,
    icon
):

    css_class = get_step_class(step)

    # IMPORTANT:
    # Single-line HTML.
    # No indentation-sensitive multiline HTML.

    html = (
        '<div class="status-bar">'
        f'<div class="status-dot {css_class}"></div>'
        f'<span>{icon} {label}</span>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    # Logo
    st.markdown(
        '<div class="hero-title" '
        'style="font-size:1.6rem">🎬 AI<br>Video</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-sub">'
        'Meeting Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<span class="badge badge-purple">INPUT</span>',
        unsafe_allow_html=True
    )

    source = st.text_input(
        "YouTube URL or File Path",
        placeholder=(
            "https://youtube.com/watch?v=... "
            "or /path/to/file.mp4"
        )
    )

    language = st.selectbox(
        "Language",
        [
            "english",
            "hinglish"
        ]
    )

    run_btn = st.button(
        "⚡ ANALYSE",
        use_container_width=True
    )


    # -------------------------------------------------------------------------
    # CURRENT VIDEO
    # -------------------------------------------------------------------------

    if st.session_state.video_id:

        st.markdown("---")

        st.markdown(
            '<span class="badge badge-cyan">'
            'CURRENT VIDEO'
            '</span>',
            unsafe_allow_html=True
        )

        st.caption(
            f"ID: {st.session_state.video_id}"
        )


    # -------------------------------------------------------------------------
    # PIPELINE STATUS
    # -------------------------------------------------------------------------

    if st.session_state.pipeline_done:

        st.markdown("---")

        st.markdown(
            '<span class="badge badge-green">'
            'PIPELINE STATUS'
            '</span>',
            unsafe_allow_html=True
        )

        pipeline_steps = [
            (
                "audio",
                "🔊",
                "Audio Processing"
            ),
            (
                "transcript",
                "📝",
                "Transcription"
            ),
            (
                "title",
                "🏷️",
                "Title Generation"
            ),
            (
                "summary",
                "📋",
                "Summarisation"
            ),
            (
                "extract",
                "🔍",
                "Extraction"
            ),
            (
                "rag",
                "🧠",
                "RAG Engine"
            ),
        ]

        for step, icon, label in pipeline_steps:

            render_step_bar(
                label,
                step,
                icon
            )


# =============================================================================
# MAIN HEADER
# =============================================================================

st.markdown(
    '<div class="hero-title">'
    'AI Video Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-sub">'
    'Transcribe · Summarise · Chat with your meetings'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# =============================================================================
# ANALYSE
# =============================================================================

if run_btn:

    if not source.strip():

        st.error(
            "Please enter a YouTube URL or file path."
        )

    else:

        # ---------------------------------------------------------------------
        # Generate unique ID for this video
        # ---------------------------------------------------------------------

        video_id = str(
            uuid.uuid4()
        )

        st.session_state.video_id = video_id

        # ---------------------------------------------------------------------
        # Reset previous result
        # ---------------------------------------------------------------------

        st.session_state.result = None

        st.session_state.chat_history = []

        st.session_state.pipeline_done = False

        st.session_state.pipeline_steps = {}

        st.session_state.processing = True

        progress_placeholder = st.empty()


        def update_step(
            key,
            state
        ):

            st.session_state.pipeline_steps[key] = state


        try:

            progress_placeholder.info(
                "⚙️ Pipeline running — "
                "see sidebar for live status…"
            )


            # ================================================================
            # AUDIO PROCESSING
            # ================================================================

            update_step(
                "audio",
                "active"
            )

            chunks = process_input(
                source
            )

            update_step(
                "audio",
                "done"
            )


            # ================================================================
            # TRANSCRIPTION
            # ================================================================

            update_step(
                "transcript",
                "active"
            )

            transcript = transcribe_all(
                chunks,
                language
            )

            update_step(
                "transcript",
                "done"
            )


            # ================================================================
            # TITLE
            # ================================================================

            update_step(
                "title",
                "active"
            )

            title = generate_title(
                transcript
            )

            update_step(
                "title",
                "done"
            )


            # ================================================================
            # SUMMARY
            # ================================================================

            update_step(
                "summary",
                "active"
            )

            summary = summarize(
                transcript
            )

            update_step(
                "summary",
                "done"
            )


            # ================================================================
            # EXTRACTION
            # ================================================================

            update_step(
                "extract",
                "active"
            )

            action_items = extract_action_items(
                transcript
            )

            decisions = extract_key_decisions(
                transcript
            )

            questions = extract_questions(
                transcript
            )

            update_step(
                "extract",
                "done"
            )


            # ================================================================
            # RAG
            # ================================================================

            update_step(
                "rag",
                "active"
            )

            # IMPORTANT:
            #
            # The current video's unique ID is passed to the RAG engine.
            #
            # This allows the vector store to distinguish transcripts
            # belonging to different videos.

            rag_chain = build_rag_chain(
                transcript=transcript,
                video_id=video_id
            )

            update_step(
                "rag",
                "done"
            )


            # ================================================================
            # SAVE RESULT
            # ================================================================

            st.session_state.result = {

                "video_id": video_id,

                "title": title,

                "transcript": transcript,

                "summary": summary,

                "action_items": action_items,

                "key_decisions": decisions,

                "open_questions": questions,

                "rag_chain": rag_chain,
            }


            st.session_state.pipeline_done = True

            st.session_state.processing = False


            progress_placeholder.success(
                "✅ Analysis complete!"
            )

            time.sleep(0.5)

            progress_placeholder.empty()

            st.rerun()


        except Exception as e:

            st.session_state.processing = False

            progress_placeholder.error(
                f"❌ Error: {e}"
            )


# =============================================================================
# RESULTS
# =============================================================================

if st.session_state.result:

    r = st.session_state.result


    # =========================================================================
    # TITLE
    # =========================================================================

    st.markdown(
        '<div class="card">'
        '<div class="card-title">'
        '📌 Session Title'
        '</div>'
        '<div style="'
        'font-family:\'Syne\',sans-serif;'
        'font-size:1.4rem;'
        'font-weight:700;'
        'color:var(--text)'
        '">'
        f'{r["title"]}'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


    # =========================================================================
    # VIDEO ID
    # =========================================================================

    st.caption(
        f"Current video ID: {r['video_id']}"
    )


    # =========================================================================
    # SUMMARY + TRANSCRIPT
    # =========================================================================

    col1, col2 = st.columns(
        [3, 2],
        gap="medium"
    )


    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    with col1:

        st.markdown(
            '<div class="card">'
            '<div class="card-title">'
            '📋 Summary'
            '</div>'
            '<div class="card-content">'
            f'{r["summary"]}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # -------------------------------------------------------------------------
    # TRANSCRIPT
    # -------------------------------------------------------------------------

    with col2:

        with st.expander(
            "📝 Full Transcript",
            expanded=False
        ):

            st.markdown(
                '<div class="transcript-box">'
                f'{r["transcript"]}'
                '</div>',
                unsafe_allow_html=True
            )


    # =========================================================================
    # ACTION ITEMS / DECISIONS / QUESTIONS
    # =========================================================================

    c1, c2, c3 = st.columns(
        3,
        gap="medium"
    )


    # -------------------------------------------------------------------------
    # ACTION ITEMS
    # -------------------------------------------------------------------------

    with c1:

        st.markdown(
            '<div class="card">'
            '<div class="card-title">'
            '✅ Action Items'
            '</div>'
            '<div class="card-content">'
            f'{r["action_items"]}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # -------------------------------------------------------------------------
    # KEY DECISIONS
    # -------------------------------------------------------------------------

    with c2:

        st.markdown(
            '<div class="card">'
            '<div class="card-title">'
            '🔑 Key Decisions'
            '</div>'
            '<div class="card-content">'
            f'{r["key_decisions"]}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    # -------------------------------------------------------------------------
    # OPEN QUESTIONS
    # -------------------------------------------------------------------------

    with c3:

        st.markdown(
            '<div class="card">'
            '<div class="card-title">'
            '❓ Open Questions'
            '</div>'
            '<div class="card-content">'
            f'{r["open_questions"]}'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    st.markdown("---")


    # =========================================================================
    # CHAT HEADER
    # =========================================================================

    st.markdown(
        '<div style="'
        'font-family:\'Syne\',sans-serif;'
        'font-size:1.2rem;'
        'font-weight:700;'
        'margin-bottom:1rem'
        '">'
        '💬 Chat with your Meeting'
        '</div>',
        unsafe_allow_html=True
    )


    # =========================================================================
    # CHAT HISTORY
    # =========================================================================

    if st.session_state.chat_history:

        chat_html = (
            '<div class="chat-container">'
        )

        for msg in st.session_state.chat_history:

            if msg["role"] == "user":

                chat_html += (
                    '<div class="chat-msg" '
                    'style="align-items:flex-end">'
                    '<span class="chat-label user-label">'
                    'You'
                    '</span>'
                    '<div class="chat-bubble user-bubble">'
                    f'{msg["content"]}'
                    '</div>'
                    '</div>'
                )

            else:

                chat_html += (
                    '<div class="chat-msg" '
                    'style="align-items:flex-start">'
                    '<span class="chat-label bot-label">'
                    '🤖 Assistant'
                    '</span>'
                    '<div class="chat-bubble bot-bubble">'
                    f'{msg["content"]}'
                    '</div>'
                    '</div>'
                )

        chat_html += '</div>'

        st.markdown(
            chat_html,
            unsafe_allow_html=True
        )


    # =========================================================================
    # CHAT INPUT
    # =========================================================================

    chat_col1, chat_col2 = st.columns(
        [5, 1],
        gap="small"
    )


    with chat_col1:

        user_input = st.text_input(
            "Your question",
            placeholder=(
                "What were the main decisions made?"
            ),
            label_visibility="collapsed"
        )


    with chat_col2:

        send_btn = st.button(
            "Send →",
            use_container_width=True
        )


    # =========================================================================
    # ASK QUESTION
    # =========================================================================

    if send_btn and user_input.strip():

        question = user_input.strip()

        with st.spinner(
            "Thinking…"
        ):

            answer = ask_question(
                r["rag_chain"],
                question
            )


        # Store question
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        # Store answer
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        st.rerun()


    # =========================================================================
    # CLEAR CHAT
    # =========================================================================

    if st.session_state.chat_history:

        if st.button(
            "🗑️ Clear Chat",
            type="secondary"
        ):

            st.session_state.chat_history = []

            st.rerun()


# =============================================================================
# EMPTY STATE
# =============================================================================

else:

    # No custom HTML here.
    # Native Streamlit components are used deliberately.

    st.write("")

    empty_col1, empty_col2, empty_col3 = st.columns(
        [1, 2, 1]
    )


    with empty_col2:

        st.markdown(
            "### 🎬"
        )

        st.markdown(
            "## Ready to Analyse"
        )

        st.caption(
            "Paste a YouTube URL or local file path "
            "in the sidebar, choose your language, "
            "and hit Analyse to get started."
        )

        st.write("")

        badge_col1, badge_col2, badge_col3 = st.columns(
            3
        )


        with badge_col1:

            st.info(
                "🎙️ Transcription"
            )


        with badge_col2:

            st.info(
                "📋 Summarisation"
            )


        with badge_col3:

            st.success(
                "💬 RAG Chat"
            )