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
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RenewTrack AI – Integrated Renewable Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR PREMIUM THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    /* Global Overrides */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #050b18;
        color: #ffffff;
    }
    
    .main {
        background: transparent;
    }

    /* Glassmorphism Containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 254, 0.4);
    }
    
    /* Neon Headers */
    .main-header {
        background: linear-gradient(135deg, rgba(16, 20, 24, 0.9), rgba(0, 242, 254, 0.1));
        padding: 3rem 2rem;
        border-radius: 30px;
        text-align: center;
        margin-bottom: 3rem;
        border: 1px solid rgba(0, 242, 254, 0.2);
    }
    
    h1, h2, h3 {
        color: #00f2fe !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }

    .gradient-text {
        background: linear-gradient(90deg, #00f2fe, #bcff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background: #020c1b;
        border-right: 1px solid rgba(0, 242, 254, 0.1);
    }

    /* Hide standard streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- IMAGE URLS ---
IMAGES = {
    "hero": "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&q=80&w=2070",
    "solar": "https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?auto=format&fit=crop&q=80&w=2072", 
    "home": "https://images.unsplash.com/photo-1558002038-1055907df827?auto=format&fit=crop&q=80&w=2070", 
    "invest": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&q=80&w=2015", 
    "prediction": "https://images.unsplash.com/photo-1518186239717-2e909873173d?auto=format&fit=crop&q=80&w=2070",
}

# --- EMAIL CONFIGURATION ---
SENDER_EMAIL = "solutionsanalytiq@gmail.com"
RECEIVER_EMAIL = "saiprashanthm18@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

def get_app_password():
    try:
        return st.secrets["GMAIL_APP_PASSWORD"]
    except Exception:
        return os.environ.get("GMAIL_APP_PASSWORD", "")

def send_email_alert(alert_type="Manual Grid Alert", state="National", utilization=0):
    ist = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S IST')
    
    subject = f"⚡ RenewTrack AI: {alert_type} - {state}"
    
    html_body = f"""
    <div style="background:#0e1117; color:white; padding:40px; font-family:sans-serif; border-radius:15px; border:1px solid #00f2fe;">
        <h1 style="color:#00f2fe;">RenewTrack AI</h1>
        <h2>{alert_type}</h2>
        <p><b>State:</b> {state}</p>
        <p><b>Utilization:</b> {utilization:.2f}%</p>
        <p><b>Timestamp:</b> {timestamp}</p>
        <hr style="border:0; border-top:1px solid #1e293b;">
        <p style="font-size:0.8rem; color:#8892b0;">Automated System Intelligence • Do not reply</p>
    </div>
    """
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    app_password = get_app_password()
    if not app_password: return False, "No Password Configured"

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, app_password)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        return True, "Email Sent Successfully"
    except Exception as e:
        return False, str(e)

# --- DATA GENERATION ---
@st.cache_data
def load_data():
    states_data = {
        'State': ['Rajasthan', 'Gujarat', 'Tamil Nadu', 'Karnataka', 'Maharashtra', 'Andhra Pradesh', 'Madhya Pradesh', 'Telangana', 'Uttar Pradesh', 'Punjab', 'Haryana', 'Kerala', 'Odisha', 'West Bengal', 'Bihar', 'Chhattisgarh', 'Uttarakhand', 'Himachal Pradesh', 'Assam', 'Jharkhand'],
        'Lat': [27.02, 22.25, 11.05, 15.31, 19.60, 15.91, 23.47, 17.12, 26.84, 31.14, 29.05, 10.85, 20.95, 22.98, 25.09, 21.27, 30.06, 31.10, 26.20, 23.61],
        'Lon': [74.21, 71.19, 78.38, 75.71, 75.55, 79.74, 77.94, 79.20, 80.94, 75.34, 76.08, 76.27, 85.09, 87.85, 85.31, 81.86, 79.01, 77.17, 92.93, 85.27],
        'Installed_Capacity_MW': [34140, 33390, 25240, 22500, 19800, 17500, 14200, 12800, 10500, 8500, 7800, 6500, 5800, 5200, 4800, 4200, 3800, 3500, 2800, 2500],
        'Solar_Percentage': [85, 60, 45, 40, 35, 55, 65, 70, 90, 45, 50, 30, 40, 35, 95, 60, 25, 20, 40, 80],
        'Wind_Percentage': [15, 40, 55, 60, 65, 45, 35, 30, 10, 55, 50, 70, 60, 65, 5, 40, 75, 80, 60, 20],
        'Solar_Potential': [142, 120, 95, 100, 88, 92, 110, 105, 80, 75, 78, 65, 70, 68, 72, 85, 60, 55, 62, 70],
        'Wind_Potential': [45, 95, 100, 85, 78, 70, 40, 35, 15, 20, 25, 55, 60, 50, 10, 30, 40, 45, 15, 25],
        'Policy_Score': [9.2, 9.5, 9.0, 8.8, 8.5, 8.2, 8.0, 8.1, 7.5, 7.0, 7.2, 8.4, 7.8, 7.6, 6.5, 7.0, 7.5, 7.8, 6.8, 7.1]
    }
    df = pd.DataFrame(states_data)
    df['Utilization'] = np.random.uniform(72, 98, size=len(df))
    df['Daily_Generation_MW'] = df['Installed_Capacity_MW'] * (df['Utilization']/100)
    df['CO2_Saved_Tons'] = df['Daily_Generation_MW'] * 24 * 0.9
    return df

base_df = load_data()
st_autorefresh(interval=10000, key="global_refresh")

def get_live_data(df):
    live_df = df.copy()
    noise = np.random.uniform(0.96, 1.04, size=len(live_df))
    live_df['Daily_Generation_MW'] = live_df['Daily_Generation_MW'] * noise
    live_df['Utilization'] = (live_df['Daily_Generation_MW'] / live_df['Installed_Capacity_MW']) * 100
    live_df['CO2_Saved_Tons'] = live_df['Daily_Generation_MW'] * 24 * 0.9
    return live_df

live_df = get_live_data(base_df)

# --- SIDEBAR & SENTINEL ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00f2fe;'>⚡ RenewTrack AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sentinel
    underperforming = live_df[live_df['Utilization'] < 75]
    if not underperforming.empty:
        st.error(f"⚠ {len(underperforming)} States Underperforming")
        if st.button("📩 Alert Authorities", help="Sends emergency report to administration"):
            success, status = send_email_alert("Sentinel Reporting", "National Summary", live_df['Utilization'].mean())
            if success: st.success("Alert Dispatched!")
            else: st.error(f"Failed: {status}")
    else:
        st.success("✅ Grid Performance Optimal")
    
    st.markdown("---")
    page = st.radio("MAIN NAVIGATION", [
        "📊 National Dashboard",
        "📈 Utilization & Alerts",
        "🗺️ Intelligence Map",
        "💰 Investment Strategy",
        "🔮 Future Predictor",
        "🏠 Home Optimizer",
        "🎲 Scenario Simulator"
    ])
    st.markdown("---")
    st.info("💡 **AI INSIGHT**: Residential solar adoption is up by 18% this month.")
    st.caption("v4.5.0 • Enterprise Edition")

# --- NAVIGATION LOGIC ---
if page == "📊 National Dashboard":
    st.markdown('<div class="main-header"><h1>National <span class="gradient-text">Energy Dashboard</span></h1><p>Integrated real-time analytics for India\'s renewable energy grid</p></div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="glass-card"><h6>TOTAL CAPACITY</h6><h2>{live_df["Installed_Capacity_MW"].sum()/1000:,.1f} GW</h2></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="glass-card"><h6>LIVE GENERATION</h6><h2>{live_df["Daily_Generation_MW"].sum()/1000:,.1f} GW</h2></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="glass-card"><h6>GRID UTILIZATION</h6><h2>{live_df["Utilization"].mean():.1f}%</h2></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="glass-card"><h6>CO2 OFFSET / DAY</h6><h2>{live_df["CO2_Saved_Tons"].sum()/1e6:.2f}M T</h2></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("State-wise Installed Capacity (MW)")
        fig = px.bar(live_df, x='State', y='Installed_Capacity_MW', color='Utilization', 
                     color_continuous_scale='Teal', template="plotly_dark")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Energy Source Mix")
        fig_pie = px.pie(values=[base_df['Solar_Percentage'].mean(), base_df['Wind_Percentage'].mean()], 
                         names=["Solar", "Wind"], hole=0.5, color_discrete_sequence=['#00f2fe', '#bcff00'])
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

elif page == "📈 Utilization & Alerts":
    st.title("📈 Grid Performance & Automated Alerts")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("State Efficiency Rankings")
    df_sorted = live_df.sort_values('Utilization', ascending=False)
    st.dataframe(df_sorted[['State', 'Installed_Capacity_MW', 'Utilization', 'CO2_Saved_Tons']].style.background_gradient(subset=['Utilization'], cmap='RdYlGn'), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    overloaded = live_df[live_df['Utilization'] > 95]
    if not overloaded.empty:
        for _, row in overloaded.iterrows():
            st.error(f"🚨 **CRITICAL OVERLOAD**: {row['State']} is at {row['Utilization']:.1f}% capacity!")
            if st.button(f"Email Alert for {row['State']}", key=f"alert_{row['State']}"):
                success, status = send_email_alert("Overload Emergency", row['State'], row['Utilization'])
                if success: st.success(f"Emergency Alert Sent for {row['State']}")
                else: st.error(f"Failed to send alert: {status}")

elif page == "🗺️ Intelligence Map":
    st.markdown('<div class="main-header"><h1>National <span class="gradient-text">Intelligence Map</span></h1><p>Spatial distribution of energy assets and efficiency levels</p></div>', unsafe_allow_html=True)
    m = folium.Map(location=[22, 78], zoom_start=5, tiles='CartoDB dark_matter')
    for _, row in live_df.iterrows():
        color = '#00f2fe' if row['Utilization'] > 85 else '#bcff00' if row['Utilization'] > 75 else '#ff4b4b'
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=row['Installed_Capacity_MW']/3500 + 4,
            popup=f"<b>{row['State']}</b><br>Capacity: {row['Installed_Capacity_MW']} MW<br>Util: {row['Utilization']:.1f}%",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    folium_static(m, width=1200, height=600)

elif page == "💰 Investment Strategy":
    st.markdown('<div class="main-header"><h1>Investment <span class="gradient-text">Recommendation Engine</span></h1><p>AI-driven insights for high-yield renewable projects</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Your Profile")
        budget = st.selectbox("Budget Segment", ["Micro (Residential)", "SME (Commercial)", "Utility Scale (Project)"])
        e_type = st.radio("Primary Energy Type", ["Solar PV", "Wind Sector", "Hybrid Grid"])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        # Recommendation logic
        metric = 'Solar_Potential' if "Solar" in e_type else 'Wind_Potential'
        top_state = live_df.sort_values([metric, 'Policy_Score'], ascending=False).iloc[0]
        st.markdown(f"""
        <div class="glass-card" style="border-left: 5px solid #00f2fe;">
            <h2>Top State: <span style="color:#bcff00;">{top_state['State']}</span></h2>
            <p style="font-size:1.1rem;">Superior {e_type} feasibility combined with an industry-leading policy score of <b>{top_state['Policy_Score']}/10</b>.</p>
            <div style="display:flex; gap:20px; margin-top:20px;">
                <div style="background:rgba(0,242,254,0.1); padding:15px; border-radius:12px; flex:1; text-align:center;">
                    <p style="margin:0; font-size:0.8rem;">EST. ROI</p>
                    <p style="margin:0; font-size:1.4rem; font-weight:800;">22.4%</p>
                </div>
                <div style="background:rgba(188,255,0,0.1); padding:15px; border-radius:12px; flex:1; text-align:center;">
                    <p style="margin:0; font-size:0.8rem;">PAYBACK</p>
                    <p style="margin:0; font-size:1.4rem; font-weight:800;">4.2 Years</p>
                </div>
            </div>
            <p style="margin-top:20px; font-style:italic; color:#8892b0;">Reasoning: {top_state['State']} has seen a 12% growth in grid connectivity infrastructure in the last 12 months.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "🔮 Future Predictor":
    st.markdown('<div class="main-header"><h1>Future <span class="gradient-text">Opportunity Predictor</span></h1><p>Predictive analytics for the next decade of energy</p></div>', unsafe_allow_html=True)
    st.markdown('''
    <div class="glass-card">
        <h3>🚀 Forecast Summary</h3>
        <ul>
            <li><b>Wind Energy</b> will grow fastest in <b>Gujarat</b> by 2028 (Est. 18% CAGR).</li>
            <li><b>Rajasthan</b> will achieve 100GW installed capacity by 2030.</li>
            <li><b>Green Hydrogen</b> hubs are projected for Tamil Nadu coastal regions.</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    years = [2024, 2025, 2026, 2027, 2028, 2029, 2030]
    solar_grow = [75 + i*15 for i in range(7)]
    wind_grow = [45 + i*8 for i in range(7)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=solar_grow, name="Solar (GW)", line=dict(color='#00f2fe', width=4)))
    fig.add_trace(go.Scatter(x=years, y=wind_grow, name="Wind (GW)", line=dict(color='#bcff00', width=4)))
    fig.update_layout(title="National Capacity Forecast (2024-2030)", template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

elif page == "🏠 Home Optimizer":
    st.markdown('<div class="main-header"><h1>Smart <span class="gradient-text">Home Energy</span> Optimizer</h1><p>Appliance-level consumption mapping and AI savings</p></div>', unsafe_allow_html=True)
    st.subheader("💡 Daily Usage Input")
    c1, c2, c3 = st.columns(3)
    ac_h = c1.slider("How many hours do you use AC?", 0, 24, 6)
    fan_n = c2.number_input("How many fans do you use?", 0, 10, 4)
    light_h = c3.slider("How long are lights ON?", 0, 24, 8)
    fridge_on = st.toggle("Is Refrigerator always ON?", value=True)
    
    # Calculations
    ac_units = ac_h * 1.5
    fan_units = fan_n * 0.075 * 12
    light_units = light_h * 0.05
    fridge_units = 0.2 * 24 if fridge_on else 0
    total_daily = ac_units + fan_units + light_units + fridge_units
    monthly_cost = total_daily * 30 * 8 # ₹8 unit
    
    col_x, col_y = st.columns([1, 1])
    with col_x:
        st.subheader("Device Contribution")
        fig_pie = px.pie(values=[ac_units, fan_units, light_units, fridge_units], 
                         names=["AC", "Fans", "Lights", "Fridge"], hole=0.5,
                         color_discrete_sequence=px.colors.sequential.Teal)
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_y:
        st.subheader("AI Recommendations")
        st.markdown(f"""
        <div class="glass-card" style="text-align:center;">
            <p style="margin:0;">Est. Monthly Bill</p>
            <h1 style="margin:0; font-size:3rem;">₹{monthly_cost:,.0f}</h1>
        </div>
        """, unsafe_allow_html=True)
        if ac_h > 5:
            save = (1.5 * 8 * 30)
            st.warning(f"🤖 **AI Suggestion**: Reduce AC by <b>1 hour</b> to save **₹{save:,.0f}** per month.")
        if light_h > 10:
            st.info("🤖 **AI Suggestion**: Switch to Motion-Sensor LEDs to reduce light consumption by 40%.")

elif page == "🎲 Scenario Simulator":
    st.markdown('<div class="main-header"><h1>Strategic <span class="gradient-text">Scenario Simulator</span></h1><p>Model the national impact of efficiency improvements</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    eff_boost = st.select_slider("Select National Efficiency Improvement (%)", options=[0, 5, 10, 15, 20, 25, 30, 40, 50], value=15)
    
    current_gen = live_df['Daily_Generation_MW'].sum()
    projected_gen = current_gen * (1 + eff_boost/100)
    additional_co2 = (projected_gen - current_gen) * 24 * 0.9
    st.markdown(f"**Impact at +{eff_boost}% Improvement:**")
    sc1, sc2 = st.columns(2)
    sc1.metric("Projected Generation", f"{projected_gen/1000:,.1f} GW", f"+{projected_gen - current_gen:,.0f} MW")
    sc2.metric("Extra CO2 Offset / Day", f"{additional_co2:,.0f} Tons", "🌳")
    st.markdown('</div>', unsafe_allow_html=True)
    
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8892b0;'>RenewTrack AI © 2026 • Powering a Cleaner Tomorrow with Advanced Analytics</p>", unsafe_allow_html=True)
