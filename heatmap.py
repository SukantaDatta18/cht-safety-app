import streamlit as st
import folium
import pandas as pd
import re
import os
from folium.plugins import HeatMap
from collections import Counter

st.set_page_config(
    page_title="CHT Risk Heatmap",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 CHT Accident Risk Heatmap")
st.markdown("""
This map shows **real accident hotspots** from the dataset.
**Red areas** = High accident frequency | **Yellow** = Medium | **Blue** = Low
""")

# -------------------- SIMPLE FILE LOADER --------------------
@st.cache_data
def load_data():
    """
    Simple file loader - looks in the data folder
    """
    # Try to load from data folder
    file_path = "data/DatasetPurna.xlsx"
    
    # Check if file exists
    if not os.path.exists(file_path):
        st.error(f"""
        ❌ **File Not Found!**
        
        Please make sure `DatasetPurna.xlsx` is in the `data/` folder.
        
        **Current file path tried:** `{file_path}`
        
        **How to fix:**
        1. Check if the file name is exactly `DatasetPurna.xlsx`
        2. Make sure it's in the `data/` folder
        3. Check if the file is uploaded to GitHub
        """)
        return None, None
    
    try:
        df = pd.read_excel(file_path)
        st.success(f"✅ Loaded dataset: {file_path}")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        return None, None
    
    # Show dataset info
    st.sidebar.markdown("### 📊 Dataset Info")
    st.sidebar.write(f"Rows: {len(df)}")
    st.sidebar.write(f"Columns: {list(df.columns)}")
    
    # Check required columns
    if 'label 1' not in df.columns:
        st.error(f"❌ Column 'label 1' not found. Available: {', '.join(df.columns)}")
        return None, None
    
    if 'Location Annotation' not in df.columns:
        st.error(f"❌ Column 'Location Annotation' not found. Available: {', '.join(df.columns)}")
        return None, None
    
    # Filter only accidents
    accident_df = df[df['label 1'] == 1].copy()
    
    if len(accident_df) == 0:
        st.warning("⚠️ No accident records found (label 1 == 1)")
        return None, None
    
    # Extract locations
    all_locations = []
    for ann in accident_df['Location Annotation'].dropna():
        if isinstance(ann, str):
            for part in re.split(r'[,،\n]+', ann):
                part = part.strip()
                if part and len(part) > 1:
                    all_locations.append(part)
    
    if not all_locations:
        st.warning("⚠️ No location annotations found.")
        return None, None
    
    loc_counts = Counter(all_locations)
    
    st.sidebar.success(f"✅ {len(accident_df)} accidents, {len(loc_counts)} unique locations")
    
    return accident_df, loc_counts

# Load data
with st.spinner("📊 Loading accident data..."):
    accident_df, loc_counts = load_data()

if accident_df is None or loc_counts is None:
    st.stop()

# -------------------- LOAD COORDINATES --------------------
@st.cache_data
def load_coords():
    coord_path = "data/cht_coordinates.csv"
    if not os.path.exists(coord_path):
        st.error(f"❌ Coordinates file not found: {coord_path}")
        return None
    try:
        df = pd.read_csv(coord_path)
        st.sidebar.success(f"✅ Loaded coordinates: {coord_path}")
        return df
    except Exception as e:
        st.error(f"❌ Error loading coordinates: {e}")
        return None

coords = load_coords()
if coords is None:
    st.stop()

# -------------------- BUILD HEATMAP DATA --------------------
heat_data = []
marker_data = []
max_count = max(loc_counts.values()) if loc_counts else 1

for _, row in coords.iterrows():
    loc_name = row['name']
    count = loc_counts.get(loc_name, 0)
    if count > 0:
        heat_data.append([row['lat'], row['lon'], count / max_count])
        marker_data.append({
            'name': loc_name,
            'lat': row['lat'],
            'lon': row['lon'],
            'count': count
        })

# -------------------- SHOW STATS --------------------
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Accidents", len(accident_df))

with col2:
    st.metric("📍 Locations Found", len(marker_data))

with col3:
    if marker_data:
        top = max(marker_data, key=lambda x: x['count'])
        st.metric("🔥 Hottest Spot", f"{top['name']} ({top['count']})")

with col4:
    active_places = len([x for x in marker_data if x['count'] > 0])
    st.metric("🌐 Active Places", active_places)

# -------------------- CREATE MAP --------------------
m = folium.Map(
    location=[22.5, 92.2],
    zoom_start=9,
    tiles='OpenStreetMap',
    control_scale=True
)

if heat_data:
    HeatMap(heat_data, radius=25, blur=15, min_opacity=0.3).add_to(m)

# Add markers
for loc in marker_data:
    if loc['count'] > 20:
        color = 'red'
    elif loc['count'] > 10:
        color = 'orange'
    else:
        color = 'green'

    folium.Marker(
        [loc['lat'], loc['lon']],
        popup=f"<b>{loc['name']}</b><br>🚨 Accidents: {loc['count']}",
        icon=folium.Icon(color=color, icon='info-sign')
    ).add_to(m)

# Legend
from branca.element import Template, MacroElement

legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000; background: rgba(255,255,255,0.9); padding: 15px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-family: Arial, sans-serif;">
    <h4 style="margin: 0 0 8px 0;">📊 Accident Risk Level</h4>
    <div style="display: flex; align-items: center; margin: 4px 0;">
        <div style="width: 20px; height: 20px; background: green; margin-right: 8px; border-radius: 4px;"></div>
        <span>Low (1-10 accidents)</span>
    </div>
    <div style="display: flex; align-items: center; margin: 4px 0;">
        <div style="width: 20px; height: 20px; background: orange; margin-right: 8px; border-radius: 4px;"></div>
        <span>Medium (11-20 accidents)</span>
    </div>
    <div style="display: flex; align-items: center; margin: 4px 0;">
        <div style="width: 20px; height: 20px; background: red; margin-right: 8px; border-radius: 4px;"></div>
        <span>High (20+ accidents)</span>
    </div>
    <p style="margin-top: 8px; font-size: 11px; color: #888;">Based on {total} accident reports</p>
</div>
"""
legend = MacroElement()
legend._template = Template(legend_html.format(total=len(accident_df)))
m.get_root().add_child(legend)

# Display map
st.components.v1.html(m._repr_html_(), height=600)

# Show top 10 hotspots
st.markdown("---")
st.subheader("📍 Top 10 Accident Hotspots")

top_locations = sorted(marker_data, key=lambda x: x['count'], reverse=True)[:10]

if top_locations:
    for i, loc in enumerate(top_locations, 1):
        st.markdown(f"{i}. **{loc['name']}** — {loc['count']} accidents")
else:
    st.info("No hotspot data available.")