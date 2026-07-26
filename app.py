from datetime import datetime
from pathlib import Path
import html
import os
import re

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from preprocessing import clean_text
from report_generator import generate_report

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


st.set_page_config(
    page_title="Truth Lens - Fact-Checking Platform",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_assets():
    model = joblib.load(MODELS_DIR / "fake_news_model.pkl")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer


def article_stats(text: str):
    words = len(text.split())
    characters = len(text)
    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    average_words = words / sentences
    reading_time = max(1, round(words / 200, 1))
    return words, characters, sentences, average_words, reading_time


def confidence_donut(value, label, color, bg):
    fig = go.Figure(
        go.Pie(
            values=[value, 100 - value],
            hole=0.75,
            sort=False,
            direction="clockwise",
            rotation=90,
            marker=dict(colors=[color, bg], line=dict(color="rgba(0,0,0,0)", width=0)),
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        text=f"<b>{value:.1f}%</b><br><span style='font-size:11px;color:#64748b'>{label}</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(color="#0f172a", size=18),
        align="center",
    )
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


model, vectorizer = load_assets()

if "history" not in st.session_state:
    st.session_state.history = []

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None


# --- Custom Styling (Light Theme) ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
        --card-bg: rgba(255, 255, 255, 0.9);
        --card-border: rgba(226, 232, 240, 0.9);
        --text-primary: #0f172a;
        --text-muted: #475569;
        --accent-purple: #7c3aed;
        --accent-blue: #2563eb;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-glow-green: rgba(16, 185, 129, 0.12);
        --accent-glow-red: rgba(239, 68, 68, 0.12);
        --shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.05), 0 8px 10px -6px rgba(15, 23, 42, 0.03);
    }

    /* Overall App Background & Typography */
    .stApp {
        background: var(--bg-gradient);
        color: var(--text-primary);
        font-family: 'Outfit', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    /* Hide standard Streamlit header decoration */
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.95);
        border-right: 1px solid var(--card-border);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        font-family: 'Outfit', sans-serif;
    }

    /* Header & Hero Section Styling (Attractive & Unique UI) */
    .title-container {
        text-align: center;
        padding: 2.25rem 1.5rem 2.25rem 1.5rem;
        margin-bottom: 2rem;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.85) 100%);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 24px;
        box-shadow: 0 20px 40px -15px rgba(37, 99, 235, 0.07), 0 10px 20px -10px rgba(15, 23, 42, 0.04);
        position: relative;
        overflow: hidden;
    }

    .title-container::before {
        content: '';
        position: absolute;
        top: -60px;
        left: 50%;
        transform: translateX(-50%);
        width: 340px;
        height: 120px;
        background: radial-gradient(ellipse at center, rgba(37, 99, 235, 0.12) 0%, rgba(255, 255, 255, 0) 75%);
        pointer-events: none;
    }
    
    .brand-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1d4ed8;
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
        transition: transform 0.2s ease;
    }

    .brand-tag-icon {
        font-size: 0.95rem;
    }

    .brand-title {
        color: #0f172a;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0 0 0.6rem 0;
        line-height: 1.1;
    }

    .brand-accent {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        color: #475569;
        font-size: 1.125rem;
        font-weight: 450;
        max-width: 680px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Custom Light Glassmorphic Card Panel */
    .glass-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 20px;
        padding: 1.75rem;
        box-shadow: var(--shadow);
        backdrop-filter: blur(16px);
        margin-bottom: 1.5rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        transform: translateY(-2px);
        border-color: #cbd5e1;
        box-shadow: 0 20px 35px -10px rgba(15, 23, 42, 0.08);
    }

    /* Card Titles */
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metric Panels */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.2rem 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }

    .metric-card:hover {
        background: #ffffff;
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--text-primary);
    }

    /* Prediction Badges */
    .verdict-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.85rem 1.75rem;
        border-radius: 999px;
        font-weight: 800;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
    }

    .verdict-real {
        color: #166534;
        background: #dcfce7;
        border: 1px solid #86efac;
        box-shadow: 0 0 20px var(--accent-glow-green);
    }

    .verdict-fake {
        color: #991b1b;
        background: #fee2e2;
        border: 1px solid #fca5a5;
        box-shadow: 0 0 20px var(--accent-glow-red);
    }

    /* Widget Labels (Dark, Sharp & Visible) */
    label[data-testid="stWidgetLabel"], label[data-testid="stWidgetLabel"] p, label, .stSelectbox label, .stTextArea label {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Primary Button (Analyze Article) */
    div.stButton > button[kind="primary"], div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    div.stButton > button[kind="primary"]:hover, div.stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
        color: #ffffff !important;
    }

    /* Secondary Button (Clear Text Area) */
    div.stButton > button[kind="secondary"], button[data-testid="stBaseButton-secondary"] {
        background: #f1f5f9 !important;
        border: 1px solid #cbd5e1 !important;
        color: #334155 !important;
        border-radius: 12px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: none !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    div.stButton > button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover {
        background: #e2e8f0 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
    }

    /* PDF Download Button Override (Highly Visible & High Contrast) */
    div.stDownloadButton > button, button[data-testid="stDownloadButton"] {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: #ffffff !important;
        border: 1px solid #059669 !important;
        border-radius: 14px !important;
        padding: 0.85rem 1.75rem !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }
    div.stDownloadButton > button:hover, button[data-testid="stDownloadButton"]:hover {
        background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    div.stDownloadButton > button *, button[data-testid="stDownloadButton"] * {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Text Area Styling & Visible Black Cursor */
    textarea {
        background-color: #ffffff !important;
        color: #0f172a !important;
        caret-color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-family: 'Outfit', sans-serif !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        padding: 1rem !important;
        transition: all 0.2s ease !important;
    }

    textarea::placeholder, input::placeholder {
        color: #64748b !important;
        opacity: 1 !important;
    }

    textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.18) !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        caret-color: #000000 !important;
    }

    /* Selectbox styling & Popover Options (Cursor Removed from Selectbox) */
    div[data-baseweb="select"] {
        border-radius: 12px !important;
    }
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        user-select: none !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="select"] span, div[data-baseweb="select"] p {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] svg {
        fill: #0f172a !important;
    }

    /* Dropdown Popover List */
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1) !important;
    }
    li[role="option"] {
        color: #0f172a !important;
        background-color: #ffffff !important;
        font-weight: 500 !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #eff6ff !important;
        color: #2563eb !important;
    }

    /* Streamlit Expander styling */
    .stExpander, div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        background-color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03) !important;
    }
    .stExpander summary, div[data-testid="stExpander"] summary {
        color: #0f172a !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #f1f5f9;
        padding: 0.5rem;
        border-radius: 14px;
        border: 1px solid var(--card-border);
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 0.65rem 1.3rem;
        background-color: transparent;
        border-radius: 10px;
        color: var(--text-muted);
        font-weight: 600;
        font-size: 0.95rem;
        border: none;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background-color: rgba(255, 255, 255, 0.6);
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: var(--text-primary) !important;
        border: 1px solid #cbd5e1 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05) !important;
    }
    
    /* Info/Warning/Success override */
    div.stAlert {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 14px !important;
        color: var(--text-primary) !important;
        box-shadow: var(--shadow) !important;
    }

    /* Table styling */
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--card-border);
    }
    
    /* Divider spacing */
    hr {
        margin: 2.5rem 0 !important;
        border-color: var(--card-border) !important;
    }

    /* Footer styles */
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.9rem;
        margin-top: 3.5rem;
        padding: 1.5rem 0;
        border-top: 1px solid var(--card-border);
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar Content ---
st.sidebar.markdown(
    """
    <div style="padding: 1.25rem 0 1rem 0; text-align: center; border-bottom: 1px solid #e2e8f0; margin-bottom: 1.25rem;">
        <div style="font-size: 1.45rem; font-weight: 800; color: #0f172a; letter-spacing: -0.03em; display: inline-flex; align-items: center; gap: 0.4rem;">
            <span style="color: #2563eb;">🔎</span> Truth<span style="color: #2563eb;">Lens</span>
        </div>
        <div style="font-size: 0.72rem; color: #64748b; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.25rem;">
            News Verification Engine
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 1.2rem; border-radius: 14px; margin-bottom: 1.2rem;">
        <span style="font-size: 0.85rem; font-weight: 700; color: var(--accent-blue); display: block; margin-bottom: 0.5rem; text-transform: uppercase;">Fact-Checking Guide</span>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin: 0 0 0.5rem 0;">
            1. Select or paste article content in the workspace.
        </p>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin: 0 0 0.5rem 0;">
            2. Run analysis to trigger the ML classifier.
        </p>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin: 0;">
            3. Download a detailed PDF report for distribution.
        </p>
    </div>
    
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 1.2rem; border-radius: 14px; margin-bottom: 1.2rem;">
        <span style="font-size: 0.85rem; font-weight: 700; color: var(--accent-purple); display: block; margin-bottom: 0.5rem; text-transform: uppercase;">Pipeline Features</span>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin: 0 0 0.25rem 0;">• Logistic Regression model</p>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin: 0 0 0.25rem 0;">• TF-IDF feature extraction</p>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin: 0 0 0.25rem 0;">• Porter Stemmer & NLTK</p>
        <p style="font-size: 0.85rem; color: var(--text-muted); line-height: 1.5; margin: 0;">• Automatic PDF report writer</p>
    </div>
    
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 1.2rem; border-radius: 14px;">
        <span style="font-size: 0.85rem; font-weight: 700; color: var(--accent-green); display: block; margin-bottom: 0.5rem; text-transform: uppercase;">Model Accuracy</span>
        <p style="font-size: 1.2rem; font-weight: 800; color: var(--text-primary); margin: 0;">98.58%</p>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin: 0.2rem 0 0 0;">Validation F1-score: 0.99</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Header Section ---
st.markdown(
    """
    <div class="title-container">
        <div class="brand-tag">
            <span class="brand-tag-icon">🔍</span>
            <span>Fake News Detection</span>
        </div>
        <h1 class="brand-title">Truth<span class="brand-accent">Lens</span></h1>
        <p class="subtitle">A machine learning application that analyzes news articles and predicts whether they are real or fake.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

sample_articles = {
    "Balanced news story": """City health officials launched a community wellness drive focused on exercise, nutrition, and regular screenings. The program will expand clinic access and bring free workshops to several neighborhoods. Organizers said the effort is meant to improve long-term health outcomes through prevention and education.""",
    "Questionable viral claim": """Breaking: a hidden invention has allegedly been confirmed to cure every disease in one day with zero side effects. Anonymous sources claim the breakthrough has been buried by powerful companies for years. Readers were urged to repost the message before it is removed.""",
    "Straightforward report": """Fire crews responded to a warehouse blaze late Tuesday after residents reported smoke from the industrial zone. The fire was contained within two hours, and no injuries were reported. Investigators are reviewing footage and determining the cause while nearby businesses remain closed.""",
}

# --- Tabs Setup ---
tab_verify, tab_insights, tab_history, tab_model = st.tabs([
    "🔎 News Verifier",
    "📊 Dataset Insights",
    "🕒 Prediction History",
    "⚙️ Model & Tech Stack"
])


# ==========================================
# TAB 1: NEWS VERIFIER
# ==========================================
with tab_verify:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">✍️ Verification Workspace</div>
            <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 0.5rem; line-height: 1.6;">
                Paste a news article or select a preset from the options below. The model will analyze the syntax, sentiment, and patterns to predict whether it is likely genuine or contains questionable claims.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    left, right = st.columns([1.2, 0.8], gap="medium")
    
    with left:
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        preset = st.selectbox(
            "Select a sample article to load",
            ["Write custom article"] + list(sample_articles.keys())
        )
        
        # Handle clear text state
        if st.session_state.get("clear_text"):
            st.session_state.clear_text = False
            news_default = ""
        else:
            news_default = sample_articles.get(preset, "") if preset != "Write custom article" else ""
            
        news = st.text_area(
            "Article Content",
            value=news_default,
            height=300,
            placeholder="Paste news text here (typically 2-3 paragraphs or more yields more accurate results)..."
        )
        
        b1, b2 = st.columns(2)
        with b1:
            predict_clicked = st.button("Analyze Article", use_container_width=True, type="primary")
        with b2:
            clear_clicked = st.button("Clear Text Area", use_container_width=True, type="secondary", key="clear_btn")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        if clear_clicked:
            st.session_state.clear_text = True
            st.session_state.latest_prediction = None
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div class="card-title">ℹ️ Fact-Checking Guide</div>
                <div style="margin-bottom: 1.25rem;">
                    <span style="font-weight: 600; color: var(--accent-blue);">1. Provide Context</span>
                    <p style="font-size: 0.9rem; color: var(--text-muted); margin: 0.25rem 0 0 0; line-height: 1.5;">
                        Short phrases or single sentences may have low prediction confidence. Full-length articles work best.
                    </p>
                </div>
                <div style="margin-bottom: 1.25rem;">
                    <span style="font-weight: 600; color: var(--accent-purple);">2. Check Cleaned Text</span>
                    <p style="font-size: 0.9rem; color: var(--text-muted); margin: 0.25rem 0 0 0; line-height: 1.5;">
                        Examine the preprocessed text to see the stemmed words and removed stopwords used by the TF-IDF vectorizer.
                    </p>
                </div>
                <div style="margin-bottom: 1.25rem;">
                    <span style="font-weight: 600; color: var(--accent-green);">3. Verify Source Metrics</span>
                    <p style="font-size: 0.9rem; color: var(--text-muted); margin: 0.25rem 0 0 0; line-height: 1.5;">
                        A typical news story has standard readability patterns, word lengths, and sentence frequencies.
                    </p>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 12px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary); text-transform: uppercase; display: block; margin-bottom: 0.25rem;">AI Classifier Snapshot</span>
                    <p style="font-size: 0.9rem; color: var(--text-muted); margin: 0; line-height: 1.5;">
                        The classifier is trained on <b>44,000+</b> balanced articles, achieving a validation accuracy of <b>98.58%</b>.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Prediction Processing Trigger
    if predict_clicked:
        if not news.strip():
            st.warning("Please enter or select some news text before analyzing.")
        else:
            cleaned = clean_text(news)
            word_count, character_count, sentence_count, average_words, reading_time = article_stats(news)
            
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector)[0]
            fake_prob = float(probability[0] * 100)
            real_prob = float(probability[1] * 100)
            
            result = "Real News" if prediction == 1 else "Fake News"
            confidence = real_prob if prediction == 1 else fake_prob
            status_class = "verdict-real" if prediction == 1 else "verdict-fake"
            status_text = "Likely genuine" if prediction == 1 else "Needs review"
            
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Prediction": result,
                "Confidence": f"{confidence:.2f}%",
                "Words": word_count,
                "Sentences": sentence_count,
            })
            
            report_name = "Fake_News_Report.pdf"
            generate_report(
                filename=report_name,
                prediction=result,
                confidence=confidence,
                original_text=news,
                cleaned_text=cleaned,
                word_count=word_count,
                character_count=character_count,
                sentence_count=sentence_count,
                average_words=average_words,
                reading_time=reading_time,
            )
            
            st.session_state.latest_prediction = {
                "result": result,
                "confidence": confidence,
                "real_prob": real_prob,
                "fake_prob": fake_prob,
                "status_class": status_class,
                "status_text": status_text,
                "word_count": word_count,
                "character_count": character_count,
                "sentence_count": sentence_count,
                "average_words": average_words,
                "reading_time": reading_time,
                "cleaned": cleaned,
                "original": news,
                "report_name": report_name
            }
            
    # Display Latest Results
    if st.session_state.latest_prediction:
        res = st.session_state.latest_prediction
        
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<h3>Verification Analysis Results</h3>", unsafe_allow_html=True)
        
        # Verdict Banner
        st.markdown(
            f"""
            <div class="verdict-badge {res['status_class']}">
                {res['status_text'].upper()} &nbsp;|&nbsp; {res['result'].upper()} ({res['confidence']:.2f}% Confidence)
            </div>
            """,
            unsafe_allow_html=True
        )
        
        res_left, res_right = st.columns([1, 1], gap="medium")
        with res_left:
            st.markdown(
                f"""
                <div class="glass-card" style="height: 100%;">
                    <div class="card-title">🔮 Machine Learning Verdict</div>
                    <h2 style="font-size: 2.2rem; margin: 0.5rem 0; font-weight: 800; background: linear-gradient(135deg, #0f172a 0%, #334155 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {res['result']}
                    </h2>
                    <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.6; margin-top: 1rem;">
                        The TF-IDF vectorizer extracted semantic features from the article, and the Logistic Regression classifier evaluated the likelihood of the news being genuine vs fabricated.
                    </p>
                    <div style="margin-top: 1.5rem; background: #f8fafc; border: 1px solid #e2e8f0; padding: 1rem; border-radius: 12px;">
                        <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-muted); display: block; margin-bottom: 0.5rem; text-transform: uppercase;">Confidence Breakdown</span>
                        <div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-top: 0.5rem;">
                            <span>Real News probability:</span>
                            <span style="font-weight: 700; color: var(--accent-green);">{res['real_prob']:.2f}%</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-top: 0.25rem;">
                            <span>Fake News probability:</span>
                            <span style="font-weight: 700; color: var(--accent-red);">{res['fake_prob']:.2f}%</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with res_right:
            st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Probability Donut Charts</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    confidence_donut(res['real_prob'], "Real Confidence", "#10b981", "#e2e8f0"),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            with c2:
                st.plotly_chart(
                    confidence_donut(res['fake_prob'], "Fake Confidence", "#ef4444", "#e2e8f0"),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.write("")
        
        # Article Metrics Card
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="card-title">📐 Article Statistics & Readability</div>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-label">Word Count</div>
                        <div class="metric-value">{res['word_count']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Characters</div>
                        <div class="metric-value">{res['character_count']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sentences</div>
                        <div class="metric-value">{res['sentence_count']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Words / Sentence</div>
                        <div class="metric-value">{res['average_words']:.1f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Reading Time</div>
                        <div class="metric-value">{res['reading_time']:.1f}m</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Cleaned Text Expander
        with st.expander("📝 View Cleaned Text (NLP Preprocessed)", expanded=False):
            st.info(res['cleaned'])
            
        st.write("")
        
        # Download button for Report
        if os.path.exists(res['report_name']):
            with open(res['report_name'], "rb") as pdf_file:
                st.download_button(
                    label="📄 Download Full Prediction Report (PDF)",
                    data=pdf_file,
                    file_name=res['report_name'],
                    mime="application/pdf",
                    use_container_width=True,
                )


# ==========================================
# TAB 2: DATASET INSIGHTS
# ==========================================
with tab_insights:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">📊 Dataset Visualizations & Exploratory Data Analysis</div>
            <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.6;">
                Explore the underlying data used to train the machine learning models. The datasets comprise approximately <b>44,000+</b> news articles, split into 21,417 real articles (from Reuters) and 23,481 fake articles (flagged by fact-checkers).
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_dist, col_desc = st.columns([1.1, 0.9], gap="medium")
    class_dist_path = OUTPUTS_DIR / "class_distribution.png"
    
    with col_dist:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📈 Category Distribution</div>', unsafe_allow_html=True)
        if class_dist_path.exists():
            st.image(str(class_dist_path), use_container_width=True)
        else:
            st.warning("Category distribution plot is missing.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_desc:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div class="card-title">💡 Key Dataset Takeaways</div>
                <ul style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.8; padding-left: 1.25rem;">
                    <li><b>Highly Balanced</b>: The class distribution contains 48% real news and 52% fake news, preventing the classifier from developing class biases.</li>
                    <li><b>Lexical Discrepancies</b>: Fake news and real news have noticeably different vocabularies. Words like "video", "trump", and "reuters" serve as strong textual markers.</li>
                    <li><b>Source Reliability</b>: Real articles were scraped from Reuters.com, which adheres to high journalistic standards, whereas fake articles are sourced from flagged online blogs and unreliable media sites.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.write("")
    
    st.markdown("<h3>Textual Patterns & Word Clouds</h3>", unsafe_allow_html=True)
    subtab_fake, subtab_real = st.tabs(["⚠️ Fake News Vocabulary", "✅ Real News Vocabulary"])
    
    fake_wc_path = OUTPUTS_DIR / "fake_wordcloud.png"
    fake_top_path = OUTPUTS_DIR / "fake_top_words.png"
    real_wc_path = OUTPUTS_DIR / "real_wordcloud.png"
    real_top_path = OUTPUTS_DIR / "real_top_words.png"
    
    with subtab_fake:
        st.markdown(
            """
            <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">
                These visualizations highlight the most common words and themes found in stories flagged as fake news. Notice a higher frequency of sensationalist keywords and political figures.
            </p>
            """,
            unsafe_allow_html=True
        )
        cf1, cf2 = st.columns(2, gap="medium")
        with cf1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">☁️ Word Cloud (Fake News)</div>', unsafe_allow_html=True)
            if fake_wc_path.exists():
                st.image(str(fake_wc_path), use_container_width=True)
            else:
                st.warning("Fake news wordcloud is missing.")
            st.markdown('</div>', unsafe_allow_html=True)
        with cf2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Top 20 Words Frequency (Fake News)</div>', unsafe_allow_html=True)
            if fake_top_path.exists():
                st.image(str(fake_top_path), use_container_width=True)
            else:
                st.warning("Fake news word counts plot is missing.")
            st.markdown('</div>', unsafe_allow_html=True)
            
    with subtab_real:
        st.markdown(
            """
            <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1.5rem;">
                These visualizations show the lexical patterns of real, fact-checked news. Real news tends to use formal journalistic language, quoting reliable sources, and focusing heavily on official titles and regional reporting (e.g. "reuters").
            </p>
            """,
            unsafe_allow_html=True
        )
        cr1, cr2 = st.columns(2, gap="medium")
        with cr1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">☁️ Word Cloud (Real News)</div>', unsafe_allow_html=True)
            if real_wc_path.exists():
                st.image(str(real_wc_path), use_container_width=True)
            else:
                st.warning("Real news wordcloud is missing.")
            st.markdown('</div>', unsafe_allow_html=True)
        with cr2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Top 20 Words Frequency (Real News)</div>', unsafe_allow_html=True)
            if real_top_path.exists():
                st.image(str(real_top_path), use_container_width=True)
            else:
                st.warning("Real news word counts plot is missing.")
            st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# TAB 3: PREDICTION HISTORY
# ==========================================
with tab_history:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">🕒 Local Prediction Logs & History</div>
            <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.6;">
                Below is the history of news verifications run during this session. This data is stored locally in the session memory and can be exported as a CSV report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        
        st.write("")
        h1, h2 = st.columns(2)
        with h1:
            csv = history_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Prediction History (CSV)",
                data=csv,
                file_name="news_prediction_history.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with h2:
            if st.button("🗑️ Clear History Logs", use_container_width=True):
                st.session_state.history = []
                st.session_state.latest_prediction = None
                st.success("Session history cleared successfully.")
                st.rerun()
    else:
        st.info("No predictions checked in this session yet. Use the verifier workspace to start analyzing.")


# ==========================================
# TAB 4: MODEL INFO & TECH STACK
# ==========================================
with tab_model:
    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">⚙️ Machine Learning Pipeline Architecture</div>
            <p style="color: var(--text-muted); font-size: 0.95rem; line-height: 1.6;">
                Truth Lens uses a supervised Natural Language Processing (NLP) pipeline to detect linguistic signals, patterns, and anomalies in text.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    col_pipe1, col_pipe2 = st.columns(2, gap="medium")
    with col_pipe1:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div class="card-title">🛠️ Preprocessing & NLP Pipeline</div>
                <ul style="color: var(--text-muted); font-size: 0.9rem; line-height: 1.7; padding-left: 1.25rem;">
                    <li><b>Lowercase Conversion</b>: Standardizes all text to lowercase to maintain word-count uniformity.</li>
                    <li><b>Noise Removal</b>: Filters out web links, HTML tags, punctuation, special symbols, and numbers.</li>
                    <li><b>Stopwords Filtering</b>: Eliminates common English filler words (e.g. "the", "is", "at") via NLTK corpus.</li>
                    <li><b>Stemming</b>: Reduces words to their core grammatical root using the <i>Porter Stemmer</i> algorithm (e.g. "running" -> "run").</li>
                    <li><b>Feature Extraction</b>: Vectorizes textual tokens using <b>TF-IDF (Term Frequency-Inverse Document Frequency)</b> with a vocabulary cap of 5,000 top words.</li>
                    <li><b>Train-Test Split</b>: Splits the dataset into 80% training and 20% testing data.</li>
                    <li><b>Model Training</b>: Trains a Logistic Regression classifier on the extracted features.</li>
                    <li><b>Model Evaluation</b>: Evaluates performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.</li>
                    <li><b>Model Deployment</b>: Saves the trained model and vectorizer for use in the Streamlit web application.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
    with col_pipe2:
        st.markdown(
            """
            <div class="glass-card" style="height: 100%;">
                <div class="card-title">🔬 Model Specifications & Validation</div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 1.1rem; border-radius: 12px; margin-bottom: 1rem;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: var(--accent-purple); display: block; margin-bottom: 0.25rem;">CLASSIFICATION MODEL</span>
                    <p style="font-size: 0.95rem; color: var(--text-primary); font-weight: 600; margin: 0;">Logistic Regression (L2 Regularization)</p>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 1.1rem; border-radius: 12px; margin-bottom: 1rem;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: var(--accent-blue); display: block; margin-bottom: 0.25rem;">MODEL PERFORMANCE ACCURACY</span>
                    <p style="font-size: 0.95rem; color: var(--text-primary); font-weight: 600; margin: 0;">98.58% Validation Accuracy</p>
                </div>
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; padding: 1.1rem; border-radius: 12px;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: var(--accent-green); display: block; margin-bottom: 0.25rem;">TRAINING DURATION</span>
                    <p style="font-size: 0.95rem; color: var(--text-primary); font-weight: 600; margin: 0;">Trained on 44,898 total samples</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --- Footer ---
st.markdown(
    """
    <div class="footer">
        Truth Lens Fact-Checking AI • Created by Charmi Khunt
    </div>
    """,
    unsafe_allow_html=True,
)
