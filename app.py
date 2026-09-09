# ============================================================
# app.py — CHT SAFETY MONITOR (Fast, multi-image background)
# ============================================================

import base64
import io
from pathlib import Path

import streamlit as st
import folium

# -------------------- DOWNLOAD MODELS FROM GOOGLE DRIVE --------------------
import os
import gdown

# REPLACE WITH YOUR ACTUAL FILE IDs
MODEL_IDS = {
    'best_binary.pt': '1GEn_pvq4cc4PPnj3JgwRYWJrIHwg1HR5',
    'best_multi.pt': '1BZEUuDuv3hHEA9GHZtdXknKNyd9RQ_I2'
}

def download_models():
    """Download models from Google Drive if they don't exist locally"""
    os.makedirs('models', exist_ok=True)
    for filename, file_id in MODEL_IDS.items():
        path = f'models/{filename}'
        if not os.path.exists(path):
            with st.spinner(f"📥 Downloading {filename} (this may take a few minutes)..."):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, path, quiet=False)
                st.success(f"✅ Downloaded {filename}")

# Download models before loading
download_models()
# -------------------- END DOWNLOAD CODE --------------------

from folium.plugins import Fullscreen
from utils.models import (
    load_binary_model, load_multi_model, tokenizer,
    load_gazetteer, load_coordinates, load_severity_weights,
    predict
)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="CHT Safety Monitor",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- BACKGROUND IMAGE LOADING (BASE64) --------------------
# WHY THE APP WAS SLOW:
# The original code read each background image's raw bytes and base64-encoded
# them directly into the page's <style> block. Base64 inflates size by ~33%,
# and a handful of full-resolution phone/camera photos (often 3-10MB each)
# turns into 10-40MB+ of inline HTML that the browser must parse before it
# can paint anything. Adding more images (picture3-6.jpg) makes this worse
# linearly. The fix below downsizes + recompresses each image ONCE (cached),
# so the actual payload sent to the browser is tiny (usually <150KB/image)
# no matter how large the source files are.

ASSETS_DIR = Path(__file__).parent / "assets"

BG_FILENAMES = [
    "picture1.jpg",
    "picture2.jpg",
    "picture3.jpg",
    "picture4.jpg",
    "picture5.jpg",
    "picture6.jpg",
]

# Tune these if you want sharper backgrounds at the cost of a bit more load time.
MAX_WIDTH = 1600       # px — full-viewport backgrounds rarely need to be wider
JPEG_QUALITY = 65       # 1-95, lower = smaller/faster, 60-70 looks fine as a blurred/darkened bg


def _compress_image_bytes(raw_bytes: bytes) -> bytes:
    """Resize + recompress an image to keep the base64 payload small."""
    if not PIL_AVAILABLE:
        return raw_bytes  # fallback: use original bytes as-is

    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if img.width > MAX_WIDTH:
        new_height = int(img.height * (MAX_WIDTH / img.width))
        img = img.resize((MAX_WIDTH, new_height), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def load_bg_images_as_base64(filenames):
    encoded = []
    for name in filenames:
        path = ASSETS_DIR / name
        if not path.exists():
            continue
        with open(path, "rb") as f:
            raw = f.read()
        compressed = _compress_image_bytes(raw)
        encoded.append("data:image/jpeg;base64," + base64.b64encode(compressed).decode())
    return encoded


bg_images = load_bg_images_as_base64(BG_FILENAMES)
n_images = len(bg_images)

# -------------------- DYNAMIC CROSSFADE CSS --------------------
# Generates N background layers (one per image found) that crossfade through
# each other in sequence, using a single shared @keyframes rule reused via
# animation-delay — this works for any number of images (2, 6, or more)
# instead of the old hardcoded 2-layer version.

SECONDS_PER_IMAGE = 6  # how long each image is fully visible before fading

if n_images == 0:
    bg_layers_html = ""
    bg_layers_css = """
    .bg-fallback {
        position: fixed; inset: 0; z-index: -2;
        background: radial-gradient(circle at 20% 20%, #1e3a5f, #0f172a 60%);
    }
    """
    bg_layers_html = '<div class="bg-fallback"></div>'
elif n_images == 1:
    bg_layers_css = f"""
    .bg-layer-single {{
        position: fixed; inset: 0; z-index: -2;
        background-image: url('{bg_images[0]}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
    bg_layers_html = '<div class="bg-layer-single"></div><div class="bg-layer-single"></div>'
    # (reuse the ::after darkening overlay defined globally below via .bg-layer-single::after)
else:
    total_duration = SECONDS_PER_IMAGE * n_images
    window_pct = 100 / n_images
    fade_pct = window_pct * 0.3  # portion of the window spent fading in/out

    layer_divs = []
    layer_rules = []
    for i in range(n_images):
        start = i * window_pct
        fade_in_end = start + fade_pct
        fade_out_start = start + window_pct - fade_pct
        fade_out_end = start + window_pct

        # Wrap percentages > 100 back around (for keyframe correctness)
        def wrap(p):
            return p % 100

        keyframe_name = f"bgFade{i}"
        layer_rules.append(f"""
@keyframes {keyframe_name} {{
    0% {{ opacity: 0; }}
    {wrap(start):.2f}% {{ opacity: 0; }}
    {wrap(fade_in_end):.2f}% {{ opacity: 1; }}
    {wrap(fade_out_start):.2f}% {{ opacity: 1; }}
    {wrap(fade_out_end):.2f}% {{ opacity: 0; }}
    100% {{ opacity: 0; }}
}}
.bg-layer-{i} {{
    position: fixed;
    inset: 0;
    z-index: -2;
    background-image: url('{bg_images[i]}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    opacity: 0;
    animation: {keyframe_name} {total_duration}s ease-in-out infinite;
}}
""")
        layer_divs.append(f'<div class="bg-layer bg-layer-{i}"></div>')

    bg_layers_css = "\n".join(layer_rules)
    bg_layers_html = "\n".join(layer_divs)

# -------------------- GLOBAL STYLE / ANIMATIONS --------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif;
}}

/* Darkening overlay shared by every background layer */
.bg-layer::after, .bg-layer-single::after, .bg-fallback::after {{
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background: linear-gradient(160deg, rgba(10,15,30,0.75), rgba(30,10,45,0.55));
}}

{bg_layers_css}

/* Floating gradient orbs for extra depth/atmosphere */
.orb {{
    position: fixed;
    border-radius: 50%;
    filter: blur(70px);
    z-index: -1;
    opacity: 0.55;
    pointer-events: none;
}}
.orb-a {{
    width: 420px; height: 420px;
    top: -120px; left: -100px;
    background: radial-gradient(circle, #ff6a00, transparent 70%);
    animation: floatA 14s ease-in-out infinite;
}}
.orb-b {{
    width: 380px; height: 380px;
    bottom: -140px; right: -80px;
    background: radial-gradient(circle, #00c9ff, transparent 70%);
    animation: floatB 18s ease-in-out infinite;
}}
.orb-c {{
    width: 300px; height: 300px;
    top: 40%; right: 10%;
    background: radial-gradient(circle, #a855f7, transparent 70%);
    animation: floatC 20s ease-in-out infinite;
}}
@keyframes floatA {{
    0%, 100% {{ transform: translate(0,0) scale(1); }}
    50%      {{ transform: translate(60px, 40px) scale(1.15); }}
}}
@keyframes floatB {{
    0%, 100% {{ transform: translate(0,0) scale(1); }}
    50%      {{ transform: translate(-50px, -30px) scale(1.1); }}
}}
@keyframes floatC {{
    0%, 100% {{ transform: translate(0,0) scale(1); }}
    50%      {{ transform: translate(-30px, 50px) scale(1.2); }}
}}

.stApp {{
    background: transparent;
}}

/* Fade-in for the whole main block */
[data-testid="stAppViewContainer"] .main .block-container {{
    animation: fadeInUp 0.8s ease-out;
}}
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* Hero title */
.hero-title {{
    text-align: center;
    font-size: 3.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #f7b733, #fc4a1a, #ff2fd0, #f7b733);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
    margin-bottom: 0;
    text-shadow: 0 0 40px rgba(252, 74, 26, 0.25);
}}
@keyframes shine {{
    to {{ background-position: 300% center; }}
}}
.hero-subtitle {{
    text-align: center;
    color: #e0f7fa;
    font-size: 1.15rem;
    font-weight: 400;
    margin-top: -6px;
    opacity: 0.9;
    letter-spacing: 0.3px;
}}

/* Glassmorphism card base */
.glass-card {{
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 20px;
    padding: 22px 20px;
    color: #fff;
    box-shadow: 0 8px 32px rgba(0,0,0,0.28);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    animation: popIn 0.5s ease-out;
    position: relative;
    overflow: hidden;
}}
.glass-card::before {{
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 30%, rgba(255,255,255,0.10) 50%, transparent 70%);
    background-size: 250% 250%;
    animation: sheen 5s ease-in-out infinite;
    pointer-events: none;
}}
@keyframes sheen {{
    0%   {{ background-position: 200% 0; }}
    100% {{ background-position: -50% 0; }}
}}
.glass-card:hover {{
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 18px 45px rgba(0,0,0,0.4);
    border-color: rgba(255,255,255,0.4);
}}
@keyframes popIn {{
    from {{ opacity: 0; transform: scale(0.92); }}
    to   {{ opacity: 1; transform: scale(1); }}
}}

/* Force all cards to be the same height */
.result-card-fixed {{
    height: 100%;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}

.card-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    opacity: 0.75;
    font-weight: 600;
    margin-bottom: 4px;
    position: relative;
    z-index: 1;
}}
.card-value {{
    font-size: 1.5rem;
    font-weight: 800;
    margin: 2px 0;
    line-height: 1.3;
    position: relative;
    z-index: 1;
}}
.card-sub {{
    font-size: 0.8rem;
    opacity: 0.8;
    margin-top: 2px;
    position: relative;
    z-index: 1;
}}

/* Pulsing dot for "accident" state */
.pulse-dot {{
    height: 12px;
    width: 12px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: pulse 1.4s infinite;
}}
@keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(255,255,255,0.6); }}
    70%  {{ box-shadow: 0 0 0 12px rgba(255,255,255,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(255,255,255,0); }}
}}

/* Risk meter bar with shimmer */
.risk-track {{
    width: 100%;
    height: 12px;
    border-radius: 8px;
    background: rgba(255,255,255,0.15);
    overflow: hidden;
    margin-top: 10px;
    position: relative;
    z-index: 1;
}}
.risk-fill {{
    height: 100%;
    border-radius: 8px;
    animation: growBar 1.2s ease-out forwards;
    position: relative;
    overflow: hidden;
}}
.risk-fill::after {{
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.55), transparent);
    animation: shimmerBar 1.8s linear infinite;
}}
@keyframes growBar {{
    from {{ width: 0%; }}
}}
@keyframes shimmerBar {{
    0%   {{ transform: translateX(-100%); }}
    100% {{ transform: translateX(200%); }}
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, #ff6a00, #ee0979, #a855f7);
    background-size: 200% auto;
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: 0.6em 1em;
    transition: all 0.3s ease;
    box-shadow: 0 4px 18px rgba(238, 9, 121, 0.35);
}}
.stButton > button:hover {{
    background-position: right center;
    transform: translateY(-3px);
    box-shadow: 0 10px 26px rgba(238, 9, 121, 0.55);
    color: white;
}}
.stButton > button:active {{
    transform: translateY(0px) scale(0.98);
}}

/* Text area */
.stTextArea textarea {{
    background: rgba(255,255,255,0.08);
    color: #fff;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.25);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}}
.stTextArea textarea:focus {{
    border-color: #fc4a1a;
    box-shadow: 0 0 0 3px rgba(252, 74, 26, 0.25);
}}
.stTextArea textarea::placeholder {{
    color: rgba(255,255,255,0.55);
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #16222a, #3a1f5f 60%, #3a6073);
}}
[data-testid="stSidebar"] * {{
    color: #f0f0f0 !important;
}}

/* Section headers */
h2, h3 {{
    color: #ffffff !important;
    text-shadow: 0 2px 12px rgba(0,0,0,0.4);
}}

/* Footer */
.footer-note {{
    text-align: center;
    color: rgba(255,255,255,0.65);
    font-size: 0.85rem;
    padding: 25px 0 10px 0;
}}

hr {{
    border-color: rgba(255,255,255,0.15) !important;
}}

/* Make columns equal height */
[data-testid="column"] {{
    display: flex;
    flex-direction: column;
}}
[data-testid="column"] > div {{
    flex: 1;
    display: flex;
}}
[data-testid="column"] > div > div {{
    width: 100%;
}}
</style>

{bg_layers_html}
<div class="orb orb-a"></div>
<div class="orb orb-b"></div>
<div class="orb orb-c"></div>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS (CACHED) --------------------
@st.cache_resource(show_spinner=False)
def load_models():
    binary_model = load_binary_model('models/best_binary.pt')
    multi_model = load_multi_model('models/best_multi.pt')
    return binary_model, multi_model

@st.cache_data(show_spinner=False)
def load_data():
    known_list = load_gazetteer('data/location_gazetteer.json')
    coord_dict = load_coordinates('data/cht_coordinates.csv')
    severity_weights = load_severity_weights('data/severity_weights.json')
    return known_list, coord_dict, severity_weights

with st.spinner("🔄 Waking up the AI models..."):
    binary_model, multi_model = load_models()
    known_list, coord_dict, severity_weights = load_data()

# -------------------- SESSION STATE for example text --------------------
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def set_example(text):
    st.session_state.user_input = text

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🏔️ CHT Safety Monitor")
    st.caption("AI-powered accident detection & risk mapping for the Chittagong Hill Tracts")
    st.markdown("---")
    st.markdown("### 🌐 How it works")
    st.markdown("""
    1. 📝 Type or paste a post (Bangla / Romanized / mixed)
    2. 🤖 The model checks if it's an accident
    3. 🏷️ It classifies the accident type
    4. 📍 It finds the location mentioned
    5. 🗺️ It plots it on the risk map
    """)
    st.markdown("---")
    st.markdown("### 📍 Extracted Locations")
    if "last_locations" in st.session_state and st.session_state.last_locations:
        for loc in st.session_state.last_locations:
            st.markdown(f"✅ {loc}")
    else:
        st.caption("Locations will appear here after analysis.")
    st.markdown("---")
    st.caption("🌿 Built for tourism safety in CHT")

# -------------------- HERO --------------------
st.markdown('<div class="hero-title">🏔️ CHT Safety Monitor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Real-time accident detection & risk mapping for Bandarban · Rangamati · Khagrachari</div>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# -------------------- INPUT SECTION --------------------
st.subheader("📝 What's happening?")

user_input = st.text_area(
    "",
    height=120,
    placeholder="Paste or type Bangla, Romanized, or code-mixed text here...\n\nExample: বান্দরবানে সড়ক দুর্ঘটনায় ৩ পর্যটক আহত",
    key="user_input",
    label_visibility="collapsed"
)

st.markdown("**💡 Try a quick example:**")
ex_col1, ex_col2, ex_col3, ex_col4, ex_col5, ex_col6 = st.columns(6)

examples = [
    ("🚗 Road", "বান্দরবানে সড়ক দুর্ঘটনায় ৩ পর্যটক আহত"),
    ("🌊 Waterway", "রাঙ্গামাটিতে বোটডুবির ঘটনায় দুই নিখোঁজ"),
    ("🌪️ Disaster", "খাগড়াছড়িতে বন্যায় ভেসে গেছে তিনটি পর্যটন কটেজ"),
    ("🔪 Crime", "রাঙ্গামাটির রিজার্ভ বাজার এলাকায় পর্যটকের ব্যাগ ছিনতাই"),
    ("🐘 Wildlife", "বান্দরবানের জাদিপাই ঝিরিতে পর্যটকের ওপর হাতির আক্রমণ"),
    ("☀️ Safe", "সাজেক ভ্যালি ঘুরতে খুব মজা লাগলো, আবহাওয়া চমৎকার"),
]
for col, (label, text) in zip([ex_col1, ex_col2, ex_col3, ex_col4, ex_col5, ex_col6], examples):
    with col:
        st.button(label, use_container_width=True, on_click=set_example, args=(text,))

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🔍 Analyze Text", use_container_width=True)

# -------------------- ANALYSIS --------------------
if analyze_btn and user_input.strip():
    with st.spinner("🧠 Reading between the lines..."):
        result = predict(
            user_input, binary_model, multi_model, tokenizer,
            known_list, coord_dict, severity_weights
        )

    st.session_state.last_locations = result['locations']

    st.markdown("---")
    st.subheader("📊 Analysis Results")

    is_accident = result['is_accident']
    label = result['binary_label']
    conf = result['binary_confidence']
    accident_type = result['accident_type'] if is_accident else None
    risk_level = result['risk_level']
    risk_score = result['risk_score']

    # ---- Result Cards ----
    col1, col2, col3, col4 = st.columns(4, gap="medium")

    # 1. Binary Detection
    with col1:
        icon = "⚠️" if is_accident else "✅"
        accent = "#ff3b3b" if is_accident else "#22c55e"
        dot_color = "#ff3b3b" if is_accident else "#22c55e"

        st.markdown(f"""
        <div class="glass-card result-card-fixed" style="border-top: 4px solid {accent};">
            <div class="card-label">{icon} Detection Result</div>
            <div class="card-value">
                <span class="pulse-dot" style="background:{dot_color};"></span>{label}
            </div>
            <div class="card-sub">Confidence: {conf:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Accident Type
    with col2:
        if is_accident:
            type_emoji = {
                "Road accident": "🚗",
                "Crime": "🔪",
                "Natural disaster": "🌪️",
                "Waterway": "🚤",
                "Wildlife attack": "🐘"
            }.get(accident_type, "📌")
            type_conf = result['type_confidence']

            st.markdown(f"""
            <div class="glass-card result-card-fixed" style="border-top: 4px solid #38bdf8;">
                <div class="card-label">{type_emoji} Accident Type</div>
                <div class="card-value" style="font-size:1.2rem;">{accident_type}</div>
                <div class="card-sub">Confidence: {type_conf:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card result-card-fixed" style="border-top: 4px solid #94a3b8;">
                <div class="card-label">📌 Accident Type</div>
                <div class="card-value" style="opacity:0.6; font-size:1.5rem;">—</div>
                <div class="card-sub">No accident detected</div>
            </div>
            """, unsafe_allow_html=True)

    # 3. Risk Score
    with col3:
        risk_color = {
            "Very High": "#b91c1c",
            "High": "#f97316",
            "Medium": "#eab308",
            "Low": "#22c55e",
            "None": "#94a3b8"
        }.get(risk_level, "#94a3b8")

        risk_emoji = {
            "Very High": "🔴",
            "High": "🟠",
            "Medium": "🟡",
            "Low": "🟢",
            "None": "⚪"
        }.get(risk_level, "⚪")

        st.markdown(f"""
        <div class="glass-card result-card-fixed" style="border-top: 4px solid {risk_color};">
            <div class="card-label">{risk_emoji} Risk Assessment</div>
            <div class="card-value" style="color:{risk_color};">{risk_score}</div>
            <div class="card-sub">Level: <b style="color:{risk_color};">{risk_level}</b></div>
            <div class="risk-track">
                <div class="risk-fill" style="width:{risk_score}%; background:{risk_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Locations
    with col4:
        locations = result['locations']
        loc_str = ", ".join(locations) if locations else "No locations found"
        loc_count = len(locations)

        st.markdown(f"""
        <div class="glass-card result-card-fixed" style="border-top: 4px solid #f59e0b;">
            <div class="card-label">📍 Location Extraction</div>
            <div class="card-value" style="font-size:1.1rem;">{loc_str}</div>
            <div class="card-sub">{loc_count} location{'s' if loc_count != 1 else ''} found</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- MAP ----
    st.markdown("<br>", unsafe_allow_html=True)
    if result['coordinates']:
        st.subheader("🗺️ Location Map")

        lat = result['coordinates']['lat']
        lon = result['coordinates']['lon']

        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            st.error("⚠️ Invalid coordinates format.")
            lat = None
            lon = None

        if lat is not None and lon is not None:
            loc_name = result['locations'][0] if result['locations'] else "Detected location"

            m = folium.Map(
                location=[lat, lon],
                zoom_start=13,
                tiles='OpenStreetMap',
                control_scale=True,
                zoom_control=True
            )

            marker_color = 'red' if is_accident else 'green'

            folium.Marker(
                [lat, lon],
                popup=f"""
                <div style="font-family: 'Poppins', sans-serif; padding: 8px; min-width: 150px;">
                    <b style="font-size: 16px;">📍 {loc_name}</b><br>
                    <span style="color:{'#ff3b3b' if is_accident else '#22c55e'}; font-weight:600;">
                        {label}
                    </span><br>
                    {f"Type: {accident_type}" if is_accident else ""}
                    <br>Risk: <b>{risk_level}</b>
                </div>
                """,
                icon=folium.Icon(color=marker_color, icon='info-sign', prefix='glyphicon')
            ).add_to(m)

            folium.Circle(
                [lat, lon],
                radius=5000,
                color=marker_color,
                fill=True,
                fill_opacity=0.2,
                weight=3,
                popup=f"📍 {loc_name}<br>Risk Zone: {risk_level}"
            ).add_to(m)

            folium.Circle(
                [lat, lon],
                radius=1000,
                color=marker_color,
                fill=True,
                fill_opacity=0.4,
                weight=2
            ).add_to(m)

            Fullscreen().add_to(m)

            st.components.v1.html(m._repr_html_(), height=500)
            st.caption(f"📍 **Location:** {loc_name} | **Coordinates:** {lat}, {lon}")
        else:
            st.warning("⚠️ Could not display map due to invalid coordinates.")
    else:
        st.info("ℹ️ No specific location coordinates found. Try mentioning a place like Bandarban, Rangamati, Sajek, or Kaptai.")

    # ---- DETAILED OUTPUT ----
    with st.expander("📄 View Detailed JSON Output"):
        st.json(result)

elif analyze_btn and not user_input.strip():
    st.warning("⚠️ Please enter some text to analyze.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div class="footer-note">
    <div style="margin-bottom: 10px;">
        🌿 Built with ❤️ for tourism safety in the Chittagong Hill Tracts
    </div>
    <div style="font-size: 0.85rem; opacity: 0.8; line-height: 1.8;">
        <strong>Developed by:</strong><br>
        Purna Paul · <a href="mailto:paulpurna673@gmail.com" style="color: #81c784; text-decoration: none;">paulpurna673@gmail.com</a><br>
        Sukanta Datta · <a href="mailto:dattasukanta412@gmail.com" style="color: #81c784; text-decoration: none;">dattasukanta412@gmail.com</a><br>
        Dr. Tanjim Mahmud · <a href="mailto:tanjim_cse@yahoo.com" style="color: #81c784; text-decoration: none;">tanjim_cse@yahoo.com</a>
    </div>
    <div style="margin-top: 12px; font-size: 0.8rem; opacity: 0.6;">
        &copy; 2026 CHT Safety Monitor · All rights reserved
    </div>
</div>
""", unsafe_allow_html=True)
