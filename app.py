import os

# ============================================================
# TRANSFORMERS CONFIGURATION
# ============================================================

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"


# ============================================================
# IMPORTS
# ============================================================

import io
from html import escape
from typing import List, Dict

import pandas as pd
import streamlit as st

from src.utils.parser import extract_text_from_file
from src.utils.resume_extractor import extract_all_resume_features

from src.processing.nlp import (
    extract_keywords,
    normalize_text,
)

from src.processing.embedding import (
    get_embedder,
    embed_texts,
    cosine_similarity_matrix,
)

from src.scoring import compute_scores


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Screener",
    layout="wide",
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedder():
    """
    Load the Sentence Transformer model once and cache it.
    """
    return get_embedder()


# ============================================================
# DISPLAY HELPERS
# ============================================================

def format_score(value) -> str:
    """
    Safely format a score as a percentage.

    If the score is not applicable (None/NaN),
    display N/A instead of nan%.
    """
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass

    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def format_experience(value) -> str:
    """
    Safely format experience years.
    """
    if value is None:
        return "Not detected"

    try:
        if pd.isna(value):
            return "Not detected"
    except (TypeError, ValueError):
        pass

    try:
        return f"{float(value):.1f} years"
    except (TypeError, ValueError):
        return str(value)


def format_list_value(value) -> str:
    """
    Safely format list/dict/string values for display.
    """
    if value is None:
        return "Not specified"

    try:
        if pd.isna(value):
            return "Not specified"
    except (TypeError, ValueError):
        pass

    if isinstance(value, list):
        if not value:
            return "None detected"
        return ", ".join(str(item) for item in value)

    if isinstance(value, dict):
        if not value:
            return "Not detected"

        detected = []

        for key, present in value.items():
            if present:
                detected.append(str(key).replace("_", " ").title())

        if detected:
            return ", ".join(detected)

        return "Not detected"

    text = str(value).strip()

    return text if text else "Not specified"

def _safe_skill_list(value):
    if value is None: return []
    if isinstance(value, (list, tuple, set)): raw_items = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "[]"}: return []
        raw_items = text.split(",")
    return [str(item).strip().replace("_", " ") for item in raw_items if str(item).strip()]

def render_skill_pills(skills, pill_class="match", prefix=""):
    items = _safe_skill_list(skills)
    if not items: return False
    pills = "".join(f'<span class="skill-pill {pill_class}">{escape(prefix + item)}</span>' for item in items)
    st.markdown(f'<div class="skill-pill-wrap">{pills}</div>', unsafe_allow_html=True)
    return True

def render_score_card(title, value, subtitle="", icon="📊"):
    st.markdown(f'<div class="score-card"><div class="score-card-icon">{icon}</div><div class="score-card-title">{escape(title)}</div><div class="score-card-value">{escape(format_score(value))}</div><div class="score-card-subtitle">{escape(subtitle)}</div></div>', unsafe_allow_html=True)

def render_match_ring(score):
    try: numeric_score = max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError): numeric_score = 0.0
    pct = numeric_score * 100
    st.markdown(f'<div class="match-ring" style="--score:{pct:.2f}%"><div class="match-ring-inner"><div class="match-ring-value">{pct:.0f}%</div><div class="match-ring-label">OVERALL MATCH</div></div></div>', unsafe_allow_html=True)

# ============================================================
# MAIN APPLICATION
# ============================================================

def main() -> None:

    # ============================================================
    # PREMIUM AI UI THEME
    # ============================================================

    st.markdown(
        """
        <style>

        /* --------------------------------------------------------
           GLOBAL PAGE
        -------------------------------------------------------- */

        .stApp {
            background:
                radial-gradient(
                    circle at 10% 10%,
                    rgba(124, 58, 237, 0.10),
                    transparent 28%
                ),
                radial-gradient(
                    circle at 90% 5%,
                    rgba(37, 99, 235, 0.10),
                    transparent 30%
                ),
                linear-gradient(
                    135deg,
                    #f8f9ff 0%,
                    #f5f3ff 45%,
                    #eef4ff 100%
                );
        }

        /* --------------------------------------------------------
           MAIN CONTENT
        -------------------------------------------------------- */

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1450px;
        }

        /* --------------------------------------------------------
           HEADINGS
        -------------------------------------------------------- */

        h1 {
            font-weight: 800 !important;
            letter-spacing: -1.5px !important;
            background: linear-gradient(
                90deg,
                #4f46e5,
                #7c3aed,
                #2563eb
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        h2, h3 {
            font-weight: 750 !important;
            letter-spacing: -0.5px;
        }

        /* --------------------------------------------------------
           SUBTITLE / CAPTION
        -------------------------------------------------------- */

        .stCaption {
            color: #64748b !important;
            font-size: 1rem !important;
        }

        /* --------------------------------------------------------
           SIDEBAR
        -------------------------------------------------------- */

        section[data-testid="stSidebar"] {
            background:
                linear-gradient(
                    180deg,
                    #f7f7ff 0%,
                    #eef2ff 50%,
                    #f8faff 100%
                );
            border-right: 1px solid rgba(99, 102, 241, 0.15);
        }

        section[data-testid="stSidebar"] > div {
            padding-top: 1.5rem;
        }

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #312e81 !important;
        }

        /* --------------------------------------------------------
           BUTTONS
        -------------------------------------------------------- */

        .stButton > button {
            border-radius: 12px !important;
            border: 1px solid rgba(99, 102, 241, 0.18) !important;
            background: rgba(255, 255, 255, 0.75) !important;
            color: #312e81 !important;
            font-weight: 650 !important;
            transition:
                transform 0.2s ease,
                box-shadow 0.2s ease,
                border-color 0.2s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            border-color: rgba(124, 58, 237, 0.45) !important;
            box-shadow:
                0 8px 24px rgba(79, 70, 229, 0.16);
        }

        /* --------------------------------------------------------
           PRIMARY ANALYZE BUTTON
        -------------------------------------------------------- */

        button[kind="primary"] {
            background:
                linear-gradient(
                    90deg,
                    #4f46e5,
                    #7c3aed,
                    #2563eb
                ) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            min-height: 3rem;
            font-size: 1.05rem !important;
            font-weight: 750 !important;
            box-shadow:
                0 8px 25px rgba(79, 70, 229, 0.30);
            transition:
                transform 0.2s ease,
                box-shadow 0.2s ease;
        }

        button[kind="primary"]:hover {
            transform: translateY(-2px);
            box-shadow:
                0 12px 32px rgba(124, 58, 237, 0.40);
        }

        /* --------------------------------------------------------
           TEXT AREA
        -------------------------------------------------------- */

        .stTextArea textarea {
            border-radius: 14px !important;
            border: 1px solid
                rgba(99, 102, 241, 0.20) !important;
            background: rgba(255, 255, 255, 0.82) !important;
            box-shadow:
                0 4px 18px rgba(30, 41, 59, 0.04);
            transition:
                border-color 0.2s ease,
                box-shadow 0.2s ease;
        }

        .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow:
                0 0 0 3px rgba(99, 102, 241, 0.12) !important;
        }

        /* --------------------------------------------------------
           FILE UPLOADER
        -------------------------------------------------------- */

        [data-testid="stFileUploader"] {
            background: rgba(255, 255, 255, 0.70);
            border: 1px dashed
                rgba(99, 102, 241, 0.35);
            border-radius: 16px;
            padding: 0.5rem;
            transition:
                border-color 0.2s ease,
                box-shadow 0.2s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #7c3aed;
            box-shadow:
                0 8px 28px rgba(124, 58, 237, 0.10);
        }

        /* --------------------------------------------------------
           EXPANDERS
        -------------------------------------------------------- */

        [data-testid="stExpander"] {
            border-radius: 14px !important;
            border: 1px solid
                rgba(99, 102, 241, 0.14) !important;
            background: rgba(255, 255, 255, 0.70) !important;
            box-shadow:
                0 5px 20px rgba(30, 41, 59, 0.04);
        }

        /* --------------------------------------------------------
           METRICS
        -------------------------------------------------------- */

        [data-testid="stMetric"] {
            background:
                linear-gradient(
                    145deg,
                    rgba(255,255,255,0.92),
                    rgba(245,243,255,0.85)
                );
            border: 1px solid
                rgba(99, 102, 241, 0.12);
            border-radius: 16px;
            padding: 1rem;
            box-shadow:
                0 8px 25px rgba(30, 41, 59, 0.05);
            transition:
                transform 0.2s ease,
                box-shadow 0.2s ease;
        }

        [data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow:
                0 12px 30px rgba(79, 70, 229, 0.12);
        }

        /* --------------------------------------------------------
           DATAFRAME
        -------------------------------------------------------- */

        [data-testid="stDataFrame"] {
            border-radius: 14px !important;
            overflow: hidden;
            box-shadow:
                0 6px 24px rgba(30, 41, 59, 0.05);
        }

        /* --------------------------------------------------------
           ALERTS
        -------------------------------------------------------- */

        [data-testid="stAlert"] {
            border-radius: 12px !important;
        }

        /* --------------------------------------------------------
           SLIDERS
        -------------------------------------------------------- */

        [data-testid="stSlider"] {
            padding-top: 0.2rem;
            padding-bottom: 0.4rem;
        }

        /* --------------------------------------------------------
           DIVIDERS
        -------------------------------------------------------- */

        hr {
            border: none !important;
            height: 1px !important;
            background:
                linear-gradient(
                    90deg,
                    transparent,
                    rgba(99,102,241,0.20),
                    transparent
                ) !important;
        }

        /* --------------------------------------------------------
           SMOOTH PAGE FEEL
        -------------------------------------------------------- */

        * {
            scroll-behavior: smooth;
        }



        /* ========================================================
           PHASE 2-9 PREMIUM UI SYSTEM
        ======================================================== */
        .hero-shell{position:relative;overflow:hidden;margin:.5rem 0 1.8rem;padding:2.35rem;border:1px solid rgba(124,58,237,.18);border-radius:28px;background:radial-gradient(circle at 92% 8%,rgba(124,58,237,.22),transparent 25%),radial-gradient(circle at 5% 95%,rgba(37,99,235,.12),transparent 28%),linear-gradient(135deg,rgba(255,255,255,.96),rgba(246,243,255,.94) 55%,rgba(239,246,255,.96));box-shadow:0 22px 65px rgba(79,70,229,.12),inset 0 1px 0 rgba(255,255,255,.9)}
        .hero-shell:before{content:"";position:absolute;width:220px;height:220px;right:-80px;top:-100px;border-radius:50%;background:rgba(139,92,246,.18);filter:blur(35px);pointer-events:none}
        .hero-badge{display:inline-flex;align-items:center;gap:.45rem;padding:.48rem .82rem;border-radius:999px;border:1px solid rgba(124,58,237,.2);background:rgba(255,255,255,.82);color:#5b21b6;font-size:.72rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase;box-shadow:0 8px 24px rgba(124,58,237,.1)}
        .hero-title{position:relative;margin:1rem 0 .45rem;font-size:clamp(2.7rem,5vw,4.65rem);line-height:.98;font-weight:900;letter-spacing:-.055em;background:linear-gradient(100deg,#4338ca 0%,#7c3aed 48%,#2563eb 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
        .hero-subtitle{max-width:850px;margin:0;color:#52607a;font-size:1.03rem;line-height:1.7}
        .feature-grid{position:relative;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.85rem;margin-top:1.45rem}
        .feature-card{padding:1rem;border:1px solid rgba(99,102,241,.13);border-radius:18px;background:rgba(255,255,255,.72);box-shadow:0 10px 30px rgba(30,41,59,.045);transition:transform .22s ease,box-shadow .22s ease,border-color .22s ease}
        .feature-card:hover{transform:translateY(-4px);border-color:rgba(124,58,237,.28);box-shadow:0 16px 34px rgba(79,70,229,.12)}
        .feature-icon{font-size:1.35rem;margin-bottom:.45rem}.feature-name{color:#27245f;font-weight:800;font-size:.93rem}.feature-desc{margin-top:.2rem;color:#64748b;font-size:.75rem;line-height:1.45}
        .section-kicker{display:inline-flex;align-items:center;gap:.45rem;margin:.35rem 0 .65rem;color:#312e81;font-size:1.18rem;font-weight:850}
        [data-testid="stTextArea"]{padding:.35rem;border-radius:20px;background:rgba(255,255,255,.42);box-shadow:0 14px 38px rgba(79,70,229,.06)}
        [data-testid="stTextArea"] label,[data-testid="stFileUploader"] label{font-weight:700!important;color:#37306f!important}
        [data-testid="stFileUploader"]{margin-top:.25rem;box-shadow:0 14px 38px rgba(79,70,229,.06)}
        .analyze-note{margin:.7rem 0 .35rem;text-align:center;color:#64748b;font-size:.78rem}
        .results-banner{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin:1.8rem 0 1rem;padding:1rem 1.2rem;border-radius:18px;border:1px solid rgba(99,102,241,.16);background:linear-gradient(100deg,rgba(255,255,255,.9),rgba(245,243,255,.9));box-shadow:0 12px 32px rgba(79,70,229,.07)}
        .results-banner-title{color:#27245f;font-size:1.25rem;font-weight:850}.results-banner-subtitle{margin-top:.15rem;color:#64748b;font-size:.78rem}.status-chip{padding:.45rem .72rem;border-radius:999px;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.2);color:#047857;font-size:.72rem;font-weight:800}
        .ranking-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.9rem;margin:.85rem 0 1.1rem}.rank-card{position:relative;overflow:hidden;padding:1.05rem;border-radius:20px;border:1px solid rgba(99,102,241,.14);background:rgba(255,255,255,.84);box-shadow:0 12px 34px rgba(30,41,59,.06);transition:transform .22s ease,box-shadow .22s ease}.rank-card:hover{transform:translateY(-4px);box-shadow:0 18px 40px rgba(79,70,229,.12)}.rank-card:after{content:"";position:absolute;width:100px;height:100px;right:-40px;top:-40px;border-radius:50%;background:rgba(124,58,237,.1);filter:blur(20px)}.rank-number{color:#7c3aed;font-size:.72rem;font-weight:850;text-transform:uppercase;letter-spacing:.08em}.rank-name{margin-top:.28rem;color:#1e1b4b;font-size:.95rem;font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.rank-score{margin-top:.6rem;font-size:1.7rem;font-weight:900;background:linear-gradient(90deg,#4f46e5,#7c3aed,#2563eb);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.rank-meta{margin-top:.15rem;color:#64748b;font-size:.72rem}
        .skill-pill-wrap{display:flex;flex-wrap:wrap;gap:.48rem;margin-top:.6rem}.skill-pill{display:inline-flex;align-items:center;padding:.42rem .7rem;border-radius:999px;font-size:.76rem;font-weight:700;line-height:1}.skill-pill.match{color:#047857;background:#ecfdf5;border:1px solid #bbf7d0}.skill-pill.missing{color:#b91c1c;background:#fef2f2;border:1px solid #fecaca}.skill-pill.extra{color:#4338ca;background:#eef2ff;border:1px solid #c7d2fe}
        .score-card{padding:1.1rem;border-radius:20px;border:1px solid rgba(99,102,241,.14);background:linear-gradient(145deg,rgba(255,255,255,.96),rgba(245,243,255,.9));box-shadow:0 10px 30px rgba(30,41,59,.055);min-height:150px;transition:.22s}.score-card:hover{transform:translateY(-4px);box-shadow:0 16px 36px rgba(79,70,229,.12)}.score-card-icon{font-size:1.2rem}.score-card-title{margin-top:.4rem;color:#475569;font-size:.75rem;font-weight:800;text-transform:uppercase;letter-spacing:.06em}.score-card-value{margin-top:.25rem;font-size:1.7rem;font-weight:900;color:#312e81}.score-card-subtitle{color:#94a3b8;font-size:.7rem}.match-hero{padding:1.45rem 1.6rem;margin:.6rem 0 1.25rem;border-radius:24px;border:1px solid rgba(99,102,241,.16);background:linear-gradient(135deg,rgba(255,255,255,.96),rgba(245,243,255,.9));box-shadow:0 14px 42px rgba(79,70,229,.09)}.match-ring{--ring-size:190px;width:var(--ring-size);height:var(--ring-size);border-radius:50%;display:grid;place-items:center;position:relative;margin:auto;background:conic-gradient(#4f46e5 var(--score),#e2e8f0 0);box-shadow:0 12px 35px rgba(79,70,229,.18);animation:ringPulse 3s ease-in-out infinite}.match-ring:before{content:"";position:absolute;inset:9px;border-radius:50%;background:linear-gradient(145deg,#fff,#f6f3ff)}.match-ring-inner{position:relative;z-index:1;text-align:center}.match-ring-value{font-size:2.55rem;font-weight:950;color:#312e81}.match-ring-label{font-size:.62rem;font-weight:850;letter-spacing:.1em;color:#64748b}.premium-candidate{padding:1.4rem;border-radius:24px;border:1px solid rgba(245,158,11,.3);background:radial-gradient(circle at 100% 0%,rgba(251,191,36,.17),transparent 30%),linear-gradient(135deg,#fffdf7,#fff 55%,#f5f3ff);box-shadow:0 18px 50px rgba(124,58,237,.11);margin:.7rem 0}.premium-candidate:before{content:"";display:block;height:4px;margin:-1.4rem -1.4rem 1rem;background:linear-gradient(90deg,#f59e0b,#7c3aed,#2563eb)}.premium-topline{display:flex;justify-content:space-between;align-items:center}.premium-badge{padding:.42rem .7rem;border-radius:999px;background:#fffbeb;border:1px solid #fde68a;color:#92400e;font-size:.7rem;font-weight:900}.premium-rank{color:#7c3aed;font-size:.75rem;font-weight:850}.premium-name{margin:.5rem 0 .2rem;color:#1e1b4b;font-size:1.45rem;font-weight:900}.premium-meta{color:#64748b;font-size:.78rem}.premium-body{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-top:1rem}.premium-stat{padding:.8rem;border-radius:14px;background:rgba(255,255,255,.72);border:1px solid rgba(99,102,241,.1)}.premium-stat-label{font-size:.67rem;font-weight:800;text-transform:uppercase;color:#64748b}.premium-stat-value{margin-top:.18rem;font-weight:850;color:#312e81}.premium-score{font-size:2rem;font-weight:950;color:#312e81}.premium-skill-label{margin-top:.8rem;color:#475569;font-size:.72rem;font-weight:800}@keyframes ringPulse{0%,100%{transform:scale(1)}50%{transform:scale(1.025)}}
        [data-testid="stExpander"]:hover{border-color:rgba(124,58,237,.28)!important;box-shadow:0 10px 30px rgba(79,70,229,.08)}
        section[data-testid="stSidebar"] .stButton>button{min-height:2.55rem;font-size:.8rem!important} section[data-testid="stSidebar"] [data-testid="stSlider"]{margin:.1rem 0 .55rem} section[data-testid="stSidebar"] [data-testid="stProgress"]>div{border-radius:999px}
        @keyframes heroFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-2px)}} @keyframes softGlow{0%,100%{box-shadow:0 8px 25px rgba(79,70,229,.22)}50%{box-shadow:0 12px 34px rgba(124,58,237,.34)}} .hero-badge{animation:heroFloat 4s ease-in-out infinite} button[kind="primary"]{animation:softGlow 3.2s ease-in-out infinite}
        @media(max-width:900px){.feature-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.ranking-grid{grid-template-columns:1fr}.hero-shell{padding:1.6rem}} @media(max-width:600px){.feature-grid{grid-template-columns:1fr}.hero-title{font-size:2.6rem}}

        </style>
        """,
        unsafe_allow_html=True,
    )


    # --------------------------------------------------------
    # PREMIUM HERO / HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-badge">✨ AI-POWERED RECRUITMENT</div>
            <div class="hero-title">AI Resume Screener</div>
            <p class="hero-subtitle">Find the right candidate faster with intelligent NLP analysis, semantic understanding, skill matching, and transparent scoring.</p>
            <div class="feature-grid">
                <div class="feature-card"><div class="feature-icon">🧠</div><div class="feature-name">NLP Analysis</div><div class="feature-desc">Extract skills and candidate signals from resumes.</div></div>
                <div class="feature-card"><div class="feature-icon">⚡</div><div class="feature-name">Semantic AI</div><div class="feature-desc">Understand meaning beyond simple keyword overlap.</div></div>
                <div class="feature-card"><div class="feature-icon">🎯</div><div class="feature-name">Skill Matching</div><div class="feature-desc">Identify matched, missing, and additional skills.</div></div>
                <div class="feature-card"><div class="feature-icon">📊</div><div class="feature-name">Smart Ranking</div><div class="feature-desc">Rank candidates using transparent weighted scores.</div></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.header("Scoring Weights")


        # ----------------------------------------------------
        # QUICK PRESETS
        # ----------------------------------------------------

        st.subheader("📋 Quick Presets")

        preset_col1, preset_col2 = st.columns(2)


        # ----------------------------------------------------
        # INITIAL DEFAULT WEIGHTS
        # ----------------------------------------------------

        if "weights" not in st.session_state:

            st.session_state.weights = {
                "similarity": 0.40,
                "keywords": 0.30,
                "experience": 0.15,
                "education": 0.10,
                "certs": 0.05,
            }


        # ----------------------------------------------------
        # PRESET BUTTONS
        # ----------------------------------------------------

        with preset_col1:

            if st.button(
                "🎯 Balanced",
                use_container_width=True,
            ):

                st.session_state.weights = {
                    "similarity": 0.40,
                    "keywords": 0.30,
                    "experience": 0.15,
                    "education": 0.10,
                    "certs": 0.05,
                }

                st.rerun()


            if st.button(
                "💼 Senior Role",
                use_container_width=True,
            ):

                st.session_state.weights = {
                    "similarity": 0.30,
                    "keywords": 0.25,
                    "experience": 0.30,
                    "education": 0.10,
                    "certs": 0.05,
                }

                st.rerun()


        with preset_col2:

            if st.button(
                "🚀 Entry Level",
                use_container_width=True,
            ):

                st.session_state.weights = {
                    "similarity": 0.35,
                    "keywords": 0.40,
                    "experience": 0.05,
                    "education": 0.15,
                    "certs": 0.05,
                }

                st.rerun()


            if st.button(
                "☁️ Cloud Role",
                use_container_width=True,
            ):

                st.session_state.weights = {
                    "similarity": 0.30,
                    "keywords": 0.30,
                    "experience": 0.15,
                    "education": 0.10,
                    "certs": 0.15,
                }

                st.rerun()


        # ----------------------------------------------------
        # CUSTOM WEIGHTS
        # ----------------------------------------------------

        default_weights = st.session_state.weights

        st.divider()

        st.subheader("⚙️ Custom Weights")


        # ----------------------------------------------------
        # WEIGHT EXPLANATION
        # ----------------------------------------------------

        with st.expander("ℹ️ Weights Kya Hain?"):

            st.markdown(
                """
                **Weights = Har factor ko kitna importance do**

                - **0.0** = Is factor ko ignore karo
                - **1.0** = Maximum importance
                - **0.5** = Medium importance

                **Example:**

                - Senior role ke liye Experience ko zyada
                  weight do (0.3–0.4)
                - Entry level ke liye Skills ko zyada
                  weight do (0.4–0.5)
                - Cloud jobs ke liye Certifications ko
                  zyada weight do (0.2–0.3)
                """
            )


        # ----------------------------------------------------
        # SEMANTIC SIMILARITY
        # ----------------------------------------------------

        weight_similarity = st.slider(
            "Semantic Similarity",
            0.0,
            1.0,
            default_weights["similarity"],
            0.05,
            key="weight_similarity",
            help=(
                "Meaning-based similarity between the "
                "job description and resume."
            ),
        )


        # ----------------------------------------------------
        # KEYWORD / SKILL COVERAGE
        # ----------------------------------------------------

        weight_keywords = st.slider(
            "Skill Coverage",
            0.0,
            1.0,
            default_weights["keywords"],
            0.05,
            key="weight_keywords",
            help=(
                "Technical skills required by the JD "
                "and detected in the resume."
            ),
        )


        # ----------------------------------------------------
        # EXPERIENCE
        # ----------------------------------------------------

        weight_experience = st.slider(
            "Experience Match",
            0.0,
            1.0,
            default_weights["experience"],
            0.05,
            key="weight_experience",
            help=(
                "Match between required experience in the "
                "JD and candidate experience."
            ),
        )


        # ----------------------------------------------------
        # EDUCATION
        # ----------------------------------------------------

        weight_education = st.slider(
            "Education Match",
            0.0,
            1.0,
            default_weights["education"],
            0.05,
            key="weight_education",
            help=(
                "Match between required degree level and "
                "candidate education."
            ),
        )


        # ----------------------------------------------------
        # CERTIFICATIONS
        # ----------------------------------------------------

        weight_certifications = st.slider(
            "Certifications",
            0.0,
            1.0,
            default_weights["certs"],
            0.05,
            key="weight_certifications",
            help=(
                "Relevance of candidate certifications "
                "to the job description."
            ),
        )


        # ----------------------------------------------------
        # SAVE CURRENT WEIGHTS
        # ----------------------------------------------------

        st.session_state.weights = {
            "similarity": weight_similarity,
            "keywords": weight_keywords,
            "experience": weight_experience,
            "education": weight_education,
            "certs": weight_certifications,
        }


        # ----------------------------------------------------
        # TOTAL WEIGHT
        # ----------------------------------------------------

        total_weight = (
            weight_similarity
            + weight_keywords
            + weight_experience
            + weight_education
            + weight_certifications
        )


        if total_weight == 0:

            st.warning(
                "⚠️ Increase at least one weight to proceed."
            )

        else:

            st.info(
                f"📊 Total Weight: {total_weight:.2f}"
            )


            # ------------------------------------------------
            # WEIGHT DISTRIBUTION
            # ------------------------------------------------

            st.markdown("**Weight Distribution:**")

            weight_data = {

                "Similarity":
                    weight_similarity
                    / total_weight
                    * 100,

                "Skills":
                    weight_keywords
                    / total_weight
                    * 100,

                "Experience":
                    weight_experience
                    / total_weight
                    * 100,

                "Education":
                    weight_education
                    / total_weight
                    * 100,

                "Certifications":
                    weight_certifications
                    / total_weight
                    * 100,
            }


            for name, pct in weight_data.items():

                st.progress(
                    pct / 100,
                    text=f"{name}: {pct:.1f}%",
                )


        # ----------------------------------------------------
        # NLP OPTIONS
        # ----------------------------------------------------

        st.divider()

        st.header("🔎 NLP Options")


        num_keywords = st.slider(
            "Skills to Extract",
            5,
            50,
            50,
            help=(
                "Maximum number of technical skills "
                "used during matching. A higher limit helps "
                "avoid dropping relevant skills from resumes "
                "with many technical skills."
            ),
        )


        use_synonyms = st.checkbox(
            "Use Skill Synonyms",
            value=True,
            help=(
                "Recognize alternative names and abbreviations "
                "for the same skill, such as ML → Machine Learning "
                "and K8s → Kubernetes."
            ),
        )


    # ========================================================
    # JOB DESCRIPTION
    # ========================================================

    st.markdown('<div class="section-kicker">📄 Job Description</div>', unsafe_allow_html=True)

    jd_text = st.text_area(
        "Paste the job description here",
        height=200,
        placeholder=(
            "Responsibilities...\n"
            "Requirements...\n"
            "Technical skills...\n"
            "Nice to have..."
        ),
    )


    # ========================================================
    # RESUME UPLOAD
    # ========================================================

    st.markdown('<div class="section-kicker">📂 Upload Resumes</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload resume files (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    st.markdown(
        '<div class="analyze-note">Powered by NLP extraction + semantic embeddings + weighted candidate scoring</div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "🚀 Analyze Resumes",
        type="primary",
        use_container_width=True,
    ):

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not jd_text.strip():

            st.error(
                "Please provide a job description."
            )

            return


        if not uploaded_files:

            st.error(
                "Please upload at least one resume."
            )

            return


        if total_weight == 0:

            st.error(
                "Please increase at least one scoring weight."
            )

            return


        # ====================================================
        # ANALYSIS PIPELINE
        # ====================================================

        with st.spinner(
            "Loading NLP models and analyzing resumes..."
        ):

            # ------------------------------------------------
            # LOAD EMBEDDING MODEL
            # ------------------------------------------------

            embedder = load_embedder()


            # ------------------------------------------------
            # PROCESS JOB DESCRIPTION
            # ------------------------------------------------

            jd_clean = normalize_text(
                jd_text
            )


            jd_keywords = extract_keywords(
                jd_clean,
                top_k=num_keywords,
                use_synonyms=use_synonyms,
            )


            # ------------------------------------------------
            # PROCESS RESUMES
            # ------------------------------------------------

            resume_records: List[Dict] = []


            for uploaded_file in uploaded_files:

                try:

                    # ----------------------------------------
                    # READ UPLOADED FILE
                    # ----------------------------------------

                    content_bytes = (
                        uploaded_file.getvalue()
                    )


                    # ----------------------------------------
                    # EXTRACT TEXT
                    # ----------------------------------------

                    text = extract_text_from_file(
                        uploaded_file.name,
                        io.BytesIO(content_bytes),
                    )


                    # ----------------------------------------
                    # NORMALIZE TEXT
                    # ----------------------------------------

                    text_clean = normalize_text(
                        text
                    )


                    # ----------------------------------------
                    # EXTRACT TECHNICAL SKILLS
                    # ----------------------------------------

                    resume_keywords = extract_keywords(
                        text_clean,
                        top_k=num_keywords,
                        use_synonyms=use_synonyms,
                    )


                    # ----------------------------------------
                    # EXTRACT STRUCTURED FEATURES
                    # ----------------------------------------

                    features = extract_all_resume_features(
                        text_clean
                    )


                    # ----------------------------------------
                    # STORE RESUME DATA
                    # ----------------------------------------

                    resume_records.append({

                        "filename":
                            uploaded_file.name,

                        "text":
                            text_clean,

                        "keywords":
                            resume_keywords,

                        "experience_years":
                            features.get(
                                "experience_years"
                            ),

                        "education":
                            features.get(
                                "education",
                                {},
                            ),

                        "certifications":
                            features.get(
                                "certifications",
                                [],
                            ),

                        "recent_roles":
                            features.get(
                                "recent_roles",
                                [],
                            ),
                    })


                except Exception as error:

                    st.warning(
                        f"Failed to process "
                        f"{uploaded_file.name}: {error}"
                    )


            # ------------------------------------------------
            # CHECK PROCESSED RESUMES
            # ------------------------------------------------

            if not resume_records:

                st.error(
                    "No resumes could be processed."
                )

                return


            # =================================================
            # SEMANTIC EMBEDDINGS
            # =================================================

            all_texts = (
                [jd_clean]
                + [
                    resume["text"]
                    for resume in resume_records
                ]
            )


            embeddings = embed_texts(
                embedder,
                all_texts,
            )


            sim_matrix = cosine_similarity_matrix(
                embeddings
            )


            # =================================================
            # CALCULATE SCORES
            # =================================================

            scores_df = compute_scores(

                jd_text=jd_clean,

                jd_keywords=jd_keywords,

                resumes=resume_records,

                sim_matrix=sim_matrix,

                weight_similarity=
                    weight_similarity,

                weight_keywords=
                    weight_keywords,

                weight_experience=
                    weight_experience,

                weight_education=
                    weight_education,

                weight_certifications=
                    weight_certifications,
            )


        # ====================================================
        # ANALYSIS COMPLETE
        # ====================================================

        st.success(
            "✅ Analysis complete"
        )

        # ====================================================
        # PREMIUM RESULTS DASHBOARD
        # ====================================================

        candidate_count = len(scores_df)
        best_score = float(scores_df["total_score"].max()) if candidate_count else 0.0
        avg_score = float(scores_df["total_score"].mean()) if candidate_count else 0.0
        best_candidate = str(scores_df.loc[scores_df["total_score"].idxmax(), "filename"]) if candidate_count else "—"

        st.markdown(
            f"""
            <div class="results-banner">
                <div><div class="results-banner-title">📊 Screening Intelligence Dashboard</div><div class="results-banner-subtitle">Candidates analyzed and ranked using your selected scoring profile.</div></div>
                <div class="status-chip">● ANALYSIS COMPLETE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        best_row = scores_df.loc[scores_df["total_score"].idxmax()]
        st.markdown('<div class="match-hero">', unsafe_allow_html=True)
        ring_col, summary_col = st.columns([1, 3])
        with ring_col:
            render_match_ring(best_score)
        with summary_col:
            st.markdown('<div class="section-kicker">🎯 Overall Screening Match</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="premium-name">{escape(best_candidate)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="premium-meta">Top candidate based on the selected weighted scoring profile.</div>', unsafe_allow_html=True)
            summary_cols = st.columns(4)
            vals = [("Candidates", str(candidate_count), "👥"),("Average Match", f"{avg_score:.1%}", "📈"),("Skill Coverage", format_score(best_row.get("keyword_coverage")), "🧩"),("Semantic Fit", format_score(best_row.get("similarity")), "🧠")]
            for col,(label,val,icon) in zip(summary_cols,vals):
                with col: st.markdown(f'<div class="premium-stat"><div class="premium-stat-label">{icon} {escape(label)}</div><div class="premium-stat-value">{escape(val)}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="premium-candidate"><div class="premium-topline"><span class="premium-badge">🏆 TOP CANDIDATE</span><span class="premium-rank">RANK {int(best_row["rank"])}</span></div><div class="premium-name">{escape(str(best_row["filename"]))}</div><div class="premium-meta">Best overall match across all enabled scoring factors.</div><div class="premium-body"><div class="premium-stat"><div class="premium-stat-label">Overall Match</div><div class="premium-score">{float(best_row["total_score"]):.1%}</div></div><div class="premium-stat"><div class="premium-stat-label">Experience</div><div class="premium-stat-value">{escape(format_experience(best_row.get("experience_years")))}</div></div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="premium-skill-label">Matched Skills</div>', unsafe_allow_html=True)
        render_skill_pills(best_row.get("matched_skills"), "match", "✓ ")
        if _safe_skill_list(best_row.get("missing_skills")):
            st.markdown('<div class="premium-skill-label">Missing Skills</div>', unsafe_allow_html=True)
            render_skill_pills(best_row.get("missing_skills"), "missing", "! ")
        if _safe_skill_list(best_row.get("additional_skills")):
            st.markdown('<div class="premium-skill-label">Additional Skills</div>', unsafe_allow_html=True)
            render_skill_pills(best_row.get("additional_skills"), "extra", "+ ")

        # ====================================================
        # JD SKILLS
        # ====================================================

        with st.expander(
            "🔎 Detected Job Description Skills",
            expanded=True,
        ):

            if jd_keywords:

                st.write(
                    ", ".join(jd_keywords)
                )

            else:

                st.info(
                    "No technical skills were detected "
                    "in the job description."
                )


        # ====================================================
        # RANKED RESUMES
        # ====================================================

        st.markdown('<div class="section-kicker">📋 Candidate Ranking — Top 3</div>', unsafe_allow_html=True)

        top_candidates = scores_df.sort_values("rank").head(3)
        ranking_cards = []
        for _, top_candidate in top_candidates.iterrows():
            safe_name = escape(str(top_candidate["filename"]))
            ranking_cards.append(
                f"""<div class="rank-card"><div class="rank-number">Rank {int(top_candidate["rank"])}</div><div class="rank-name" title="{safe_name}">{safe_name}</div><div class="rank-score">{float(top_candidate["total_score"]):.1%}</div><div class="rank-meta">Overall candidate match score</div></div>"""
            )
        if ranking_cards:
            st.markdown('<div class="ranking-grid">' + ''.join(ranking_cards) + '</div>', unsafe_allow_html=True)


        display_cols = [

            "rank",

            "filename",

            "total_score",

            "similarity",

            "keyword_coverage",

            "experience_score",

            "education_score",

            "certification_score",

            "experience_years",

            "education",

            "certifications",

            "matched_skills",

            "missing_skills",

            "additional_skills",
        ]


        # ----------------------------------------------------
        # CREATE DISPLAY COPY
        # ----------------------------------------------------

        display_df = (
            scores_df[display_cols]
            .sort_values("rank")
            .copy()
        )


        # ----------------------------------------------------
        # FORMAT N/A-AWARE SCORE COLUMNS
        # ----------------------------------------------------

        for column in [
            "total_score",
            "similarity",
            "keyword_coverage",
            "experience_score",
            "education_score",
            "certification_score",
        ]:

            display_df[column] = display_df[column].apply(
                format_score
            )


        # ----------------------------------------------------
        # FORMAT EXPERIENCE
        # ----------------------------------------------------

        display_df["experience_years"] = (
            display_df["experience_years"]
            .apply(format_experience)
        )


        # ----------------------------------------------------
        # FORMAT EDUCATION
        # ----------------------------------------------------

        display_df["education"] = (
            display_df["education"]
            .apply(format_list_value)
        )


        # ----------------------------------------------------
        # FORMAT CERTIFICATIONS
        # ----------------------------------------------------

        display_df["certifications"] = (
            display_df["certifications"]
            .apply(format_list_value)
        )


        st.dataframe(

            display_df,

            use_container_width=True,

            hide_index=True,
        )


        # ====================================================
        # DETAILED CANDIDATE ANALYSIS
        # ====================================================

        st.subheader(
            "🔍 Detailed Candidate Analysis"
        )


        for _, candidate in scores_df.iterrows():

            score_percent = (
                float(candidate["total_score"])
                * 100
            )


            candidate_title = (
                f"🏆 Rank "
                f"{int(candidate['rank'])} — "
                f"{candidate['filename']} — "
                f"{score_percent:.1f}% Match"
            )


            with st.expander(
                candidate_title
            ):

                # --------------------------------------------
                # SCORE METRICS
                # --------------------------------------------

                st.markdown("### 📊 Score Breakdown")
                metric_cols = st.columns(3)
                card_data = [("Overall Match", candidate["total_score"], "Final weighted score", "🎯"),("Semantic Similarity", candidate["similarity"], "Meaning-based fit", "🧠"),("Skill Coverage", candidate["keyword_coverage"], "Required skills found", "🧩"),("Experience Match", candidate["experience_score"], "Experience fit", "💼"),("Education Match", candidate["education_score"], "Education fit", "🎓"),("Certification Match", candidate["certification_score"], "Relevant certifications", "🏅")]
                for index,(title,value,subtitle,icon) in enumerate(card_data):
                    with metric_cols[index % 3]:
                        render_score_card(title,value,subtitle,icon)
                st.divider()


                # --------------------------------------------
                # MATCHED AND MISSING SKILLS
                # --------------------------------------------

                skills_col1, skills_col2 = (
                    st.columns(2)
                )


                # --------------------------------------------
                # MATCHED SKILLS
                # --------------------------------------------

                with skills_col1:

                    st.markdown(
                        "### ✅ Matched Skills"
                    )


                    matched = str(
                        candidate["matched_skills"]
                    ).strip()


                    if matched:
                        render_skill_pills(matched, "match", "✓ ")
                    else:

                        st.info(
                            "No matching technical skills "
                            "were detected."
                        )


                # --------------------------------------------
                # MISSING SKILLS
                # --------------------------------------------

                with skills_col2:

                    st.markdown(
                        "### ❌ Missing Skills"
                    )


                    missing = str(
                        candidate["missing_skills"]
                    ).strip()


                    if missing:
                        render_skill_pills(missing, "missing", "! ")
                    else:

                        st.success(
                            "No required skills are missing."
                        )


                # --------------------------------------------
                # ADDITIONAL SKILLS
                # --------------------------------------------

                st.markdown(
                    "### ➕ Additional Skills"
                )


                additional = str(
                    candidate["additional_skills"]
                ).strip()


                if additional:
                    render_skill_pills(additional, "extra", "+ ")
                else:

                    st.info(
                        "No additional skills detected."
                    )


                st.divider()


                # --------------------------------------------
                # CANDIDATE INFORMATION
                # --------------------------------------------

                st.markdown(
                    "### 📋 Candidate Information"
                )


                info_col1, info_col2, info_col3 = (
                    st.columns(3)
                )


                with info_col1:

                    st.write(
                        f"**Experience:** "
                        f"{format_experience(candidate['experience_years'])}"
                    )


                with info_col2:

                    st.write(
                        f"**Education:** "
                        f"{format_list_value(candidate['education'])}"
                    )


                with info_col3:

                    st.write(
                        f"**Certifications:** "
                        f"{format_list_value(candidate['certifications'])}"
                    )


                # --------------------------------------------
                # SCORE EXPLANATION
                # --------------------------------------------

                st.markdown(
                    "### 💡 Why This Score?"
                )


                st.info(
                    candidate["score_explanation"]
                )


        # ====================================================
        # SCORE VISUALIZATION
        # ====================================================

        st.markdown('<div class="section-kicker">📈 Score Visualization</div>', unsafe_allow_html=True)


        chart_col1, chart_col2 = (
            st.columns(2)
        )


        # ----------------------------------------------------
        # OVERALL SCORES
        # ----------------------------------------------------

        with chart_col1:

            st.write(
                "**Overall Scores**"
            )


            chart_df = scores_df[
                [
                    "filename",
                    "similarity",
                    "keyword_coverage",
                    "total_score",
                ]
            ].copy()


            chart_df = chart_df.set_index(
                "filename"
            )


            st.bar_chart(
                chart_df
            )


        # ----------------------------------------------------
        # FEATURE SCORES
        # ----------------------------------------------------

        with chart_col2:

            st.write(
                "**Feature Scores**"
            )


            feature_chart_df = scores_df[
                [
                    "filename",
                    "experience_score",
                    "education_score",
                    "certification_score",
                ]
            ].copy()


            # -----------------------------------------------
            # Convert scores to numeric
            # -----------------------------------------------

            score_columns = [
                "experience_score",
                "education_score",
                "certification_score",
            ]


            for column in score_columns:

                feature_chart_df[column] = pd.to_numeric(
                    feature_chart_df[column],
                    errors="coerce",
                )


            # -----------------------------------------------
            # Remove columns that are completely unavailable
            # -----------------------------------------------

            available_columns = [

                column

                for column in score_columns

                if feature_chart_df[column].notna().any()
            ]


            if available_columns:

                feature_chart_df = feature_chart_df[
                    [
                        "filename"
                    ]
                    + available_columns
                ]


                feature_chart_df = feature_chart_df.set_index(
                    "filename"
                )


                st.bar_chart(
                    feature_chart_df
                )

            else:

                st.info(
                    "No optional feature scores are applicable "
                    "for this job description."
                )


        # ====================================================
        # DETAILED METRICS
        # ====================================================

        with st.expander(
            "📊 Additional Candidate Metrics"
        ):

            metrics_df = scores_df[
                [
                    "filename",
                    "experience_years",
                    "education",
                    "certifications",
                ]
            ].copy()


            metrics_df["experience_years"] = (
                metrics_df["experience_years"]
                .apply(format_experience)
            )


            metrics_df["education"] = (
                metrics_df["education"]
                .apply(format_list_value)
            )


            metrics_df["certifications"] = (
                metrics_df["certifications"]
                .apply(format_list_value)
            )


            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True,
            )


        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        st.markdown('<div class="section-kicker">📥 Export Results</div>', unsafe_allow_html=True)


        csv_bytes = (
            scores_df
            .to_csv(index=False)
            .encode("utf-8")
        )


        st.download_button(

            label="Download Complete Results (CSV)",

            data=csv_bytes,

            file_name="resume_screening_results.csv",

            mime="text/csv",

            use_container_width=True,
        )

        st.markdown(
            "<div style='text-align:center;color:#94a3b8;font-size:.72rem;margin-top:1.2rem;'>AI Resume Screener • NLP-powered candidate analysis • Transparent scoring</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()