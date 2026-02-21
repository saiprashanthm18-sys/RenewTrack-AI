import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
from streamlit_autorefresh import st_autorefresh
import pytz
import requests
import json

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RenewTrack AI – Smart Renewable Energy Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR THEME ---
st.markdown("""
    <style>
    /* Dark & Blue Tech Theme */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .main {
        background: transparent;
    }
    
    /* Neon Glassmorphism Headers */
    .main-header {
        background: linear-gradient(135deg, rgba(10, 25, 47, 0.95), rgba(0, 242, 254, 0.2));
        padding: 2.5rem;
        border-radius: 20px;
        color: #00f2fe;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.3);
        backdrop-filter: blur(15px);
    }
    
    h1, h2, h3 {
        color: #00f2fe !important; /* Neon Blue */
        font-family: 'Inter', sans-serif;
        text-shadow: 0 0 10px rgba(0, 242, 254, 0.3);
    }
    
    /* Sidebar Styling - Cyber Dark */
    section[data-testid="stSidebar"] {
        background-image: linear-gradient(180deg, #0a192f 0%, #020c1b 100%);
        border-right: 1px solid rgba(0, 242, 254, 0.1);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #ccd6f6;
    }
    
    /* Card Styles - Dark Glass */
    .stMetric {
        background: rgba(17, 34, 64, 0.7);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(0, 242, 254, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
    }
    
    /* Footer - Sleek Dark */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(10, 25, 47, 0.98);
        color: #64ffda;
        text-align: center;
        padding: 12px;
        font-weight: 500;
        z-index: 999;
        backdrop-filter: blur(10px);
        border-top: 1px solid rgba(0, 242, 254, 0.2);
        letter-spacing: 1px;
    }
    
    /* Tech Highlights */
    .st-emotion-cache-1kyxreq {
        border-bottom: 3px solid #00f2fe;
    }

    /* Cyber scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #020c1b;
    }
    ::-webkit-scrollbar-thumb {
        background: #112240;
        border-radius: 10px;
        border: 1px solid #00f2fe;
    }
    </style>
    """, unsafe_allow_html=True)

# --- IMAGE URLS ---
IMAGES = {
    "hero": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80&w=2072", # Global blue tech network
    "solar": "https://images.unsplash.com/photo-1558449028-b53a39d100fc?auto=format&fit=crop&q=80&w=2070", # Solar panels at night/dark
    "prediction": "https://images.unsplash.com/photo-1510511459019-5dee2c127bb0?auto=format&fit=crop&q=80&w=2070", # Deep data/AI
    "calculator": "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=2070", # Digital tech green
    "recommendation": "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?auto=format&fit=crop&q=80&w=2072", # Satellite/Earth at night
    "utilization": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=2070", # Tech circuits
    "map": "https://images.unsplash.com/photo-1526778548025-fa2f459cd5c1?auto=format&fit=crop&q=80&w=2070", # Nocturnal city lights/Map
    "cap_bg": "https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&q=80&w=600", # Tech collab
    "gen_bg": "https://images.unsplash.com/photo-1531297484001-80022131f5a1?auto=format&fit=crop&q=80&w=600", # Energy tech
    "util_bg": "https://images.unsplash.com/photo-1550745165-9bc0b252726f?auto=format&fit=crop&q=80&w=600", # High tech hardware
    "co2_bg": "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=80&w=600", # Neon green energy
}

# --- CUSTOM COMPONENTS ---
def styled_metric(label, value, delta, img_url, is_positive=True):
    delta_color = "#00ff00" if is_positive else "#ff4b4b"
    st.markdown(f"""
    <div style="
        background-image: url('{img_url}');
        background-size: cover;
        background-position: center;
        border-radius: 20px;
        height: 180px;
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(0, 242, 254, 0.2);
        transition: transform 0.3s ease;">
        <div style="
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: linear-gradient(to bottom, rgba(10, 25, 47, 0.4), rgba(2, 12, 27, 0.8));
            backdrop-filter: blur(3px);
            padding: 25px;
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <p style="margin: 0; font-size: 13px; font-weight: 600; color: #00f2fe; text-transform: uppercase; letter-spacing: 1.5px; display: flex; align-items: center;">
                <span style="margin-right: 5px;">⚡</span> {label}
            </p>
            <h2 style="margin: 8px 0; font-size: 28px; font-weight: 800; color: #ffffff; text-shadow: 0 0 15px rgba(0, 242, 254, 0.4);">{value}</h2>
            <div style="display: flex; align-items: center; gap: 8px;">
                <p style="margin: 0; font-size: 15px; font-weight: bold; color: {delta_color}; background: rgba(0, 242, 254, 0.1); padding: 2px 8px; border-radius: 5px;">{delta}</p>
                <span style="font-size: 10px; opacity: 0.8; color: #ccd6f6;">Grid Tracking</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- AUTOMATION HELPERS ---
def trigger_n8n_alert(type="Manual Grid Alert", state="National", utilization=0):
    webhook_url = "https://analytiqsolutions.app.n8n.cloud/webhook/grid-overload-alert"
    payload = {
        "type": type,
        "state": state,
        "utilization": f"{utilization:.2f}%",
        "timestamp": datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'),
        "message": f"EMERGENCY: {type} detected for {state}. Immediate action required."
    }
    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        print(f"Webhook error: {e}")
        return False

# --- DATA GENERATION ---
@st.cache_data
def load_data():
    states_data = {
        'State': [
            'Rajasthan', 'Gujarat', 'Tamil Nadu', 'Karnataka', 'Maharashtra', 
            'Andhra Pradesh', 'Madhya Pradesh', 'Telangana', 'Uttar Pradesh', 'Punjab',
            'Haryana', 'Kerala', 'Odisha', 'West Bengal', 'Bihar', 
            'Chhattisgarh', 'Uttarakhand', 'Himachal Pradesh', 'Assam', 'Jharkhand',
            'Jammu and Kashmir', 'Goa', 'Tripura', 'Manipur', 'Meghalaya', 
            'Nagaland', 'Arunachal Pradesh', 'Mizoram', 'Sikkim', 'Delhi',
            'Puducherry', 'Chandigarh', 'Andaman and Nicobar', 'Ladakh', 'Dadra and Nagar Haveli', 'Lakshadweep'
        ],
        'Lat': [
            27.02, 22.25, 11.05, 15.31, 19.60, 
            15.91, 23.47, 17.12, 26.84, 31.14,
            29.05, 10.85, 20.95, 22.98, 25.09,
            21.27, 30.06, 31.10, 26.20, 23.61,
            33.77, 15.29, 23.74, 24.66, 25.57,
            26.15, 28.21, 23.16, 27.53, 28.70,
            11.94, 30.73, 10.21, 34.15, 20.39, 10.56
        ],
        'Lon': [
            74.21, 71.19, 78.38, 75.71, 75.55,
            79.74, 77.94, 79.20, 80.94, 75.34,
            76.08, 76.27, 85.09, 87.85, 85.31,
            81.86, 79.01, 77.17, 92.93, 85.27,
            76.57, 74.12, 91.74, 93.90, 91.88,
            94.56, 94.72, 92.93, 88.51, 77.10,
            79.80, 76.77, 92.57, 77.57, 72.98, 72.64
        ],
        'Installed_Capacity_MW': [
            34140, 33390, 25240, 22500, 19800,
            17500, 14200, 12800, 10500, 8500,
            7800, 6500, 5800, 5200, 4800,
            4200, 3800, 3500, 2800, 2500,
            2100, 1200, 850, 720, 680,
            540, 480, 320, 280, 250,
            180, 150, 120, 95, 80, 45
        ],
        'Solar_Percentage': [
            85, 60, 45, 40, 35, 55, 65, 70, 90, 45,
            50, 30, 40, 35, 95, 60, 25, 20, 40, 80,
            30, 90, 85, 95, 40, 35, 20, 15, 10, 98,
            95, 90, 30, 95, 99, 100
        ],
        'Wind_Percentage': [
            15, 40, 55, 60, 65, 45, 35, 30, 10, 55,
            50, 70, 60, 65, 5, 40, 75, 80, 60, 20,
            70, 10, 15, 5, 60, 65, 80, 85, 90, 2,
            5, 10, 70, 5, 1, 0
        ]
    }
    df = pd.DataFrame(states_data)
    # Realistic generation simulation based on MNRE typical load factors
    df['Daily_Generation_MW'] = df['Installed_Capacity_MW'] * 0.72 # Industry Avg
    df['Temp'] = np.random.randint(25, 42, size=len(df))
    df['Wind_Speed'] = np.random.uniform(4, 12, size=len(df))
    df['Irradiance'] = np.random.randint(600, 1000, size=len(df))
    df['Utilization'] = (df['Daily_Generation_MW'] / df['Installed_Capacity_MW']) * 100
    df['CO2_Saved_Tons'] = df['Daily_Generation_MW'] * 24 * 0.9
    return df

@st.cache_data
def generate_timeseries_data():
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]
    dates.reverse()
    data = {
        'Date': dates,
        'Generation_MW': [12000 + np.random.randint(-1000, 2000) for _ in range(30)]
    }
    return pd.DataFrame(data)

df = load_data()
ts_df = generate_timeseries_data()

# --- GLOBAL LIVE SIMULATION ---
st_autorefresh(interval=10000, key="global_refresh")

# Inject random noise for live tracking across ALL states
def get_live_data(base_df):
    live_df = base_df.copy()
    # Vary generation by +/- 2-5% for a realistic live feel
    noise = np.random.uniform(0.95, 1.05, size=len(live_df))
    live_df['Daily_Generation_MW'] = live_df['Daily_Generation_MW'] * noise
    # Re-calculate dependent metrics
    live_df['Utilization'] = (live_df['Daily_Generation_MW'] / live_df['Installed_Capacity_MW']) * 100
    live_df['CO2_Saved_Tons'] = live_df['Daily_Generation_MW'] * 24 * 0.9
    return live_df

live_df = get_live_data(df)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌱 RenewTrack AI")
st.sidebar.markdown("---")

# --- EMERGENCY BROADCAST ---
with st.sidebar.expander("🚨 Emergency Automation", expanded=True):
    st.write("Broadcast alerts via n8n workflow")
    if st.button("Trigger National Alert", type="primary", use_container_width=True):
        with st.spinner("Broadcasting..."):
            if trigger_n8n_alert():
                st.success("Broadcast Sent!")
            else:
                st.error("Automation error")

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate to", [
    "📊 Dashboard Overview",
    "📈 Utilization Analysis",
    "🤖 AI Prediction Module",
    "🍃 CO2 Reduction Calculator",
    "💡 AI Recommendation Engine",
    "🗺️ India Heatmap",
    "🎲 Scenario Simulator"
])

# --- PAGE 1: DASHBOARD OVERVIEW ---
if page == "📊 Dashboard Overview":
    st.markdown('<div class="main-header"><h1>RenewTrack AI – Smart Optimization</h1><p>Harnessing the power of AI for a Sustainable India</p></div>', unsafe_allow_html=True)
    
    st.image(IMAGES["hero"], width="stretch", caption="Optimizing National Grid Resilience")

    st.markdown("""
    ### 🔋 Our Optimization Ecosystem
    RenewTrack AI operates in three core phases to ensure maximum energy efficiency:
    1. **Data Ingestion**: Real-time monitoring of state-wise generation and weather patterns.
    2. **AI Analysis**: Machine Learning models identify underperformance and predict future yields.
    3. **Actionable Insights**: Smart recommendations for grid optimization and infrastructure maintenance.
    """)
    st.markdown("---")

    # --- LIVE AGGREGATED METRICS ---
    total_capacity = live_df['Installed_Capacity_MW'].sum()
    total_generation = live_df['Daily_Generation_MW'].sum()
    avg_utilization = live_df['Utilization'].mean()
    total_co2 = live_df['CO2_Saved_Tons'].sum()
    
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime('%H:%M:%S')
    st.markdown(f"**Last Updated:** {current_time} | 📡 Live simulated renewable energy feed (All States Active)")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        styled_metric("Total Capacity", f"{total_capacity:,.0f} MW", "National", IMAGES["cap_bg"])
    with col2:
        styled_metric("Current Gen", f"{total_generation:,.0f} MW", "Live Tracking", IMAGES["gen_bg"])
    with col3:
        styled_metric("Avg Utilization", f"{avg_utilization:.1f}%", "Real-time", IMAGES["util_bg"])
    with col4:
        styled_metric("CO2 Reduction", f"{total_co2/1e6:.2f}M Tons", "Yield", IMAGES["co2_bg"])

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("30-Day Generation Trend")
        fig_line = px.line(ts_df, x='Date', y='Generation_MW', title="National Generation Trend (MW)")
        fig_line.update_layout(template="plotly_dark")
        st.plotly_chart(fig_line, width="stretch")

    with c2:
        st.subheader("Solar vs Wind Capacity")
        fig_bar = px.bar(df, x='State', y=['Solar_Percentage', 'Wind_Percentage'], 
                         title="Energy Source Split (%)", barmode='group',
                         color_discrete_sequence=['#00f2fe', '#64ffda'],
                         template="plotly_dark")
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("---")
    st.subheader("⭐ Highlight: Tamil Nadu Performance")
    tn_data = df[df['State'] == 'Tamil Nadu'].iloc[0]
    sc1, sc2, sc3 = st.columns(3)
    sc1.write(f"**Capacity:** {tn_data['Installed_Capacity_MW']} MW")
    sc2.write(f"**Utilization:** {tn_data['Utilization']:.1f}%")
    sc3.write(f"**Source:** {tn_data['Wind_Percentage']}% Wind / {tn_data['Solar_Percentage']}% Solar")

# --- PAGE 2: UTILIZATION ANALYSIS ---
elif page == "📈 Utilization Analysis":
    st.title("📈 Utilization & Performance Analysis")
    st.image(IMAGES["utilization"], width="stretch")
    
    # Tracking message
    ist = pytz.timezone('Asia/Kolkata')
    st.info(f"📡 **Live Feed Active**: Data refreshed at {datetime.now(ist).strftime('%H:%M:%S')}. Monitoring all 36 States & UTs.")

    df_sorted = live_df.sort_values(by='Utilization', ascending=False)
    
    st.subheader("State Efficiency Rankings (Live)")
    fig_efficiency = px.bar(df_sorted, x='State', y='Utilization', color='Utilization',
                            color_continuous_scale='Blues', title="Live Utilization Percentage by State",
                            template="plotly_dark")
    st.plotly_chart(fig_efficiency, width="stretch")

    underperforming = live_df[live_df['Utilization'] < 70]
    overloaded = live_df[live_df['Utilization'] > 95]

    if not overloaded.empty:
        for _, row in overloaded.iterrows():
            st.error(f"🚨 **CRITICAL OVERLOAD**: {row['State']} is at {row['Utilization']:.2f}% utilization!")
            if st.button(f"Alert Authorities in {row['State']}", key=f"alert_{row['State']}"):
                if trigger_n8n_alert("Grid Overload", row['State'], row['Utilization']):
                    st.toast(f"Email notifications sent for {row['State']}!")

    if not underperforming.empty:
        st.warning(f"⚠ Detected {len(underperforming)} underperforming states with < 70% utilization.")
        st.table(underperforming[['State', 'Installed_Capacity_MW', 'Daily_Generation_MW', 'Utilization']])
    else:
        st.success("✅ All states are performing efficiently (>= 70% utilization).")

    st.subheader("Detailed Breakdown (Live Analytics)")
    st.dataframe(live_df[['State', 'Installed_Capacity_MW', 'Daily_Generation_MW', 'Utilization', 'CO2_Saved_Tons']].style.format({
        'Installed_Capacity_MW': '{:,.0f}',
        'Daily_Generation_MW': '{:,.0f}',
        'Utilization': '{:.2f}%',
        'CO2_Saved_Tons': '{:,.0f}'
    }))

# --- PAGE 3: AI PREDICTION MODULE ---
elif page == "🤖 AI Prediction Module":
    st.title("🤖 AI Energy Forecast")
    st.image(IMAGES["prediction"], width="stretch")
    st.write("Using historical weather data (Temperature, Wind Speed, Irradiance) to predict future energy generation.")

    # Synthetic Training Data
    X = df[['Temp', 'Wind_Speed', 'Irradiance']]
    y = df['Daily_Generation_MW']

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    st.success(f"Model trained successfully! Accuracy (R² Score): **{r2:.4f}**")

    st.subheader("7-Day Generation Forecast")
    future_dates = [datetime.now() + timedelta(days=x) for x in range(1, 8)]
    # Simple forecast logic: drift existing generation with random noise
    base_gen = df['Daily_Generation_MW'].sum()
    predictions = [base_gen * (1 + (np.random.normal(0, 0.05))) for _ in range(7)]
    
    forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted_Generation_MW': predictions})
    
    fig_forecast = px.area(forecast_df, x='Date', y='Predicted_Generation_MW', 
                           title="Projected National Generation (Next 7 Days)",
                           line_shape='spline', color_discrete_sequence=['#00f2fe'],
                           template="plotly_dark")
    st.plotly_chart(fig_forecast, width="stretch")

# --- PAGE 4: CO2 REDUCTION CALCULATOR ---
elif page == "🍃 CO2 Reduction Calculator":
    st.title("🍃 CO2 Reduction & Sustainability Impact")
    st.image(IMAGES["calculator"], width="stretch")
    st.write("Calculate how increasing efficiency impacts the environment.")

    eff_boost = st.slider("Select Efficiency Improvement (%)", 0, 50, 10)
    
    # Logic
    current_gen = df['Daily_Generation_MW'].sum()
    new_gen = current_gen * (1 + eff_boost/100)
    additional_gen_yearly = (new_gen - current_gen) * 24 * 365
    co2_saved = additional_gen_yearly * 0.9 # 0.9 tons per MWh from coal
    trees_equivalent = co2_saved * 45 # approx 45 trees per ton of CO2 absorbed in lifetime
    cars_removed = co2_saved / 4.6 # approx 4.6 tons CO2 per car per year

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CO2 Saved / Year", f"{co2_saved/1e3:.1f}k Tons", f"+{eff_boost}%")
    with col2:
        st.metric("Tree Planting Equivalent", f"{trees_equivalent/1e6:.2f}M Trees", "🌳")
    with col3:
        st.metric("Cars Removed Equivalent", f"{cars_removed:,.0f} Cars", "🚗")

    st.info("💡 Formula: Coal emission factor of 0.9 tons CO2 per MWh is used as the baseline for offsets.")

# --- PAGE 5: AI RECOMMENDATION ENGINE ---
elif page == "💡 AI Recommendation Engine":
    st.title("💡 AI Smart Recommendations")
    st.image(IMAGES["recommendation"], width="stretch")
    
    target_state = st.selectbox("Select State to Analyze", live_df['State'].unique())
    state_row = live_df[live_df['State'] == target_state].iloc[0]
    
    st.write(f"Analyzing efficiency for **{target_state}**...")
    
    if state_row['Utilization'] < 75:
        st.error(f"⚠ **Underperformance Detected**: {target_state}'s current utilization is only {state_row['Utilization']:.1f}%")
        
        with st.container():
            st.markdown(f"### 🛠 Suggested Interventions for {target_state}:")
            if state_row['Utilization'] < 50:
                st.markdown("""
                1. **Grid Stabilization**: Immediate voltage regulation required.
                2. **Full System Audit**: High risk of component failure detected.
                3. **Emergency Storage**: Deploy mobile battery units to handle fluctuations.
                """)
            elif state_row['Utilization'] < 75:
                st.markdown("""
                1. **Solar Panel Cleaning**: Dust accumulation in this region reduces efficiency by ~12%.
                2. **Predictive Maintenance**: Check Wind Turbines for abnormal vibration patterns.
                3. **Battery Storage**: Optimize grid curtailment during peak solar hours.
                """)
            else:
                st.markdown("""
                1. **Inverter Tuning**: Fine-tune phase matching for synchronous grid integration.
                2. **Load Balancing**: Distribute evening peaks across regional microgrids.
                """)
            
            est_gain = 15.0
            rev_gain = state_row['Installed_Capacity_MW'] * (est_gain/100) * 24 * 365 * 0.05 # $0.05/kWh
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Est. Efficiency Gain", f"+{est_gain}%")
            c2.metric("Revenue Potential", f"${rev_gain/1e6:.2f}M")
            c3.metric("CO2 Saved", f"{(state_row['Installed_Capacity_MW']*0.15*24*365*0.9)/1000:.1f}k Tons")
    else:
        st.success(f"✅ **Peak Performance**: {target_state} is operating at high efficiency ({state_row['Utilization']:.1f}%).")
        st.write("Maintain current protocols and focus on long-term infrastructure scaling.")

# --- PAGE 6: INDIA HEATMAP ---
elif page == "🗺️ India Heatmap":
    st.title("🗺️ National Energy Efficiency Heatmap")
    st.image(IMAGES["map"], width="stretch")
    st.write("Visualizing state-wise performance across India.")

    # Create map centered on India with Satellite View
    m = folium.Map(location=[22, 78], zoom_start=5)
    
    # Add Google Hybrid tiles (Satellite + Labels)
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Hybrid',
        overlay=False,
        control=True
    ).add_to(m)

    # Add markers
    for _, row in live_df.iterrows():
        color = 'green' if row['Utilization'] > 85 else 'orange' if row['Utilization'] > 70 else 'red'
        
        popup_text = f"""
        <b>State:</b> {row['State']}<br>
        <b>Utilization:</b> {row['Utilization']:.1f}%<br>
        <b>CO2 Saved:</b> {row['CO2_Saved_Tons']:,.0f} Tons
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=8,
            popup=folium.Popup(popup_text, max_width=250),
            tooltip=row['State'],
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)

    folium_static(m)
    
    st.markdown("""
    **Legend:** 
    - 🟢 High Efficiency (>85%) 
    - 🟠 Medium Efficiency (70-85%) 
    - 🔴 Low Efficiency (<70%)
    - *Circle size corresponds to Installed Capacity.*
    """)

# --- PAGE 7: SCENARIO SIMULATOR ---
elif page == "🎲 Scenario Simulator":
    st.title("🎲 Strategic Scenario Simulator")
    st.image(IMAGES["solar"], width="stretch")
    st.write("Simulate the national impact of strategic efficiency improvements.")

    with st.expander("Configure Simulation Parameters", expanded=True):
        eff_inc = st.slider("National Efficiency Improvement (%)", 0, 30, 10)
        cost_savings_per_mwh = st.number_input("Cost Savings per MWh ($)", value=45)

    base_gen = live_df['Daily_Generation_MW'].sum()
    sim_gen = base_gen * (1 + eff_inc/100)
    
    st.subheader("Simulation Results: Impact Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_comp = go.Figure(data=[
            go.Bar(name='Current', x=['National Generation'], y=[base_gen], marker_color='#1e3a8a'),
            go.Bar(name='Projected', x=['National Generation'], y=[sim_gen], marker_color='#2e7d32')
        ])
        fig_comp.update_layout(title="Generation Comparison (MW)", barmode='group')
        st.plotly_chart(fig_comp, width="stretch")
        
    with col2:
        yearly_gain = (sim_gen - base_gen) * 24 * 365
        financial_impact = yearly_gain * cost_savings_per_mwh / 1e6
        st.metric("Additional Annual Generation", f"{yearly_gain/1e6:.2f}M MWh")
        st.metric("Projected Financial Gain", f"${financial_impact:.2f}M")
        st.metric("CO2 Footprint Reduction", f"{(yearly_gain * 0.9)/1e6:.2f}M Tons")

# --- FOOTER ---
st.markdown("---")
st.markdown('<div class="footer">RenewTrack AI – Optimizing Energy Today, Protecting the Planet Tomorrow</div>', unsafe_allow_html=True)
