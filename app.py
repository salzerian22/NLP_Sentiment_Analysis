"""
Movie Review Sentiment Analysis — Streamlit App
================================================
Setup:
  pip install streamlit scikit-learn nltk pandas matplotlib seaborn

Run:
  streamlit run app.py

Make sure best_model.pkl and tfidf_vectorizer.pkl are in the same folder.
"""

import streamlit as st
import pickle
import re
import nltk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Download NLTK data (only on first run) ──────────────────────
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="CineScope — Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0d0d0d;
        color: #f0f0f0;
    }

    /* Hero title */
    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4rem;
        letter-spacing: 4px;
        background: linear-gradient(135deg, #f5c842 0%, #ff6b35 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        line-height: 1;
        margin-bottom: 0;
    }

    .hero-sub {
        text-align: center;
        color: #888;
        font-size: 0.9rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #f5c842, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* Result cards */
    .result-positive {
        background: linear-gradient(135deg, #0a2e1a, #0d3b21);
        border: 1px solid #2ecc71;
        border-left: 5px solid #2ecc71;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        text-align: center;
    }

    .result-negative {
        background: linear-gradient(135deg, #2e0a0a, #3b0d0d);
        border: 1px solid #e74c3c;
        border-left: 5px solid #e74c3c;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        text-align: center;
    }

    .result-label {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.5rem;
        letter-spacing: 3px;
        margin: 0;
    }

    .positive-text { color: #2ecc71; }
    .negative-text { color: #e74c3c; }

    .result-emoji {
        font-size: 3rem;
        display: block;
        margin-bottom: 0.3rem;
    }

    /* Info tiles */
    .info-tile {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }

    .info-tile-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f5c842;
        margin: 0;
    }

    .info-tile-label {
        font-size: 0.75rem;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Text area */
    .stTextArea textarea {
        background: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #f5c842 !important;
        box-shadow: 0 0 0 2px rgba(245, 200, 66, 0.15) !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #f5c842, #ff6b35);
        color: #0d0d0d;
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 50px;
        padding: 0.7rem 2.5rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        width: 100%;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111 !important;
        border-right: 1px solid #222;
    }

    [data-testid="stSidebar"] * {
        color: #ccc !important;
    }

    h3, h4 {
        color: #f5c842 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Preprocessing (must match Colab) ───────────────────────────
@st.cache_resource
def load_assets():
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
        return model, tfidf, stop_words, stemmer, True
    except FileNotFoundError:
        return None, None, stop_words, stemmer, False

def preprocess_text(text, stop_words, stemmer):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

model, tfidf, stop_words, stemmer, model_loaded = load_assets()


# ─── Header ─────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🎬 CineScope</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Movie Review Sentiment Analyzer · NLP Practical 8</p>',
            unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

if not model_loaded:
    st.error("""
    **Model files not found!**

    Make sure `best_model.pkl` and `tfidf_vectorizer.pkl` are in the same
    directory as this `app.py` file.

    Run the Colab notebook first, then download both `.pkl` files.
    """)
    st.stop()


# ─── Sidebar: Model Info ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Info")
    try:
        with open('model_metadata.pkl', 'rb') as f:
            meta = pickle.load(f)
        st.markdown(f"**Best Model:** {meta['best_model_name']}")
        st.markdown(f"**Accuracy:** `{meta['best_accuracy']}%`")

        st.markdown("---")
        st.markdown("### 📊 All Model Scores")
        for name, scores in meta['all_results'].items():
            st.markdown(f"**{name}**")
            st.markdown(f"Acc: `{scores['Accuracy']}%` · F1: `{scores['F1-Score']}%`")
    except FileNotFoundError:
        st.info("model_metadata.pkl not found.\nPlace it alongside the app for model stats.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    Trained on the **IMDB 50k** movie review dataset.

    **Pipeline:**
    - HTML & noise removal
    - Stopword filtering
    - Porter Stemming
    - TF-IDF (10k features, bigrams)
    - Best of 5 classifiers
    """)


# ─── Main Input ─────────────────────────────────────────────────
st.markdown("#### ✍️ Enter a Movie Review")
review_input = st.text_area(
    label="",
    placeholder="e.g. The cinematography was breathtaking and the performances were extraordinary...",
    height=160,
    max_chars=5000
)

word_count = len(review_input.split()) if review_input.strip() else 0
st.markdown(f"<p style='color:#555; font-size:0.8rem; text-align:right;'>{word_count} words</p>",
            unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🎬 Predict Sentiment")


# ─── Prediction ─────────────────────────────────────────────────
if predict_btn:
    if not review_input.strip():
        st.warning("⚠️ Please enter a movie review before predicting.")
    elif word_count < 3:
        st.warning("⚠️ Review is too short. Please enter at least a few words.")
    else:
        with st.spinner("Analyzing sentiment..."):
            cleaned = preprocess_text(review_input, stop_words, stemmer)
            vectorized = tfidf.transform([cleaned])
            prediction = model.predict(vectorized)[0]

            # Decision score for confidence bar
            try:
                score = model.decision_function(vectorized)[0]
                confidence = min(100, int(50 + abs(score) * 10))
            except AttributeError:
                try:
                    proba = model.predict_proba(vectorized)[0]
                    confidence = int(max(proba) * 100)
                except AttributeError:
                    confidence = None

        # Result Card
        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <span class="result-emoji">😊</span>
                <p class="result-label positive-text">Positive Sentiment</p>
                <p style="color:#aaa; margin:0.5rem 0 0; font-size:0.9rem;">
                    This review expresses a <strong style="color:#2ecc71">favourable</strong> opinion.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <span class="result-emoji">😞</span>
                <p class="result-label negative-text">Negative Sentiment</p>
                <p style="color:#aaa; margin:0.5rem 0 0; font-size:0.9rem;">
                    This review expresses an <strong style="color:#e74c3c">unfavourable</strong> opinion.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar
        if confidence is not None:
            st.markdown("**Confidence Level:**")
            bar_color = "#2ecc71" if prediction == 1 else "#e74c3c"
            st.markdown(f"""
            <div style="background:#1a1a1a; border-radius:50px; height:14px; width:100%; overflow:hidden; border:1px solid #333;">
                <div style="background:{bar_color}; width:{confidence}%; height:100%;
                            border-radius:50px; transition:width 0.5s ease;"></div>
            </div>
            <p style="color:#888; font-size:0.8rem; margin:4px 0 0;">{confidence}% confidence</p>
            """, unsafe_allow_html=True)

        # Preprocessing detail (expandable)
        with st.expander("🔍 See preprocessing output"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Original (first 300 chars):**")
                st.code(review_input[:300], language=None)
            with col_b:
                st.markdown("**After preprocessing:**")
                st.code(cleaned[:300], language=None)
            st.markdown(f"Original tokens: `{word_count}` → Cleaned tokens: `{len(cleaned.split())}`")


# ─── Try Example Reviews ─────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("#### 💡 Try an Example")

examples = {
    "⭐ Glowing Review":
        "One of the greatest films ever made. The performances are stunning, "
        "the direction is masterful, and the story is deeply moving. "
        "An absolute triumph of cinema.",
    "💀 Scathing Review":
        "Absolute rubbish. The plot makes no sense, the acting is wooden, "
        "and the CGI looks like it was done in 2003. "
        "I want my two hours back.",
    "😐 Mixed Review":
        "The first half was genuinely gripping with some great performances, "
        "but the second half fell apart completely. "
        "Not terrible, but a real missed opportunity."
}

cols = st.columns(3)
for col, (label, text) in zip(cols, examples.items()):
    if col.button(label, use_container_width=True):
        st.session_state['example_text'] = text
        st.rerun()

# If an example was selected, show it in a code block with note
if 'example_text' in st.session_state:
    st.info(f"📋 Copied to clipboard concept — paste this into the text area above:\n\n{st.session_state['example_text']}")
    del st.session_state['example_text']


# ─── Footer ─────────────────────────────────────────────────────
st.markdown("""
<hr class="divider">
<p style="text-align:center; color:#444; font-size:0.8rem; letter-spacing:1px;">
    PRACTICAL 8 · BTech CSE · Sentiment Analysis using NLP
</p>
""", unsafe_allow_html=True)
