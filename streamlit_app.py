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
    page_title="RenewTrack AI – Renewable Energy Intelligence",
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
        transition: transform 0.3s ease, border 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 254, 0.4);
    }
    
    /* Neon Headers */
    .main-header {
        background: linear-gradient(135deg, rgba(16, 20, 24, 0.9), rgba(0, 242, 254, 0.1));
        padding: 4rem 2rem;
        border-radius: 30px;
        text-align: center;
        margin-bottom: 3rem;
        border: 1px solid rgba(0, 242, 254, 0.2);
        box-shadow: 0 10px 50px rgba(0, 0, 0, 0.5);
    }
    
    h1, h2, h3 {
        color: #00f2fe !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px !important;
    }

    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(90deg, #00f2fe, #bcff00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe, #00d2ff);
        color: #050b18 !important;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.6);
        transform: scale(1.02);
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background: #020c1b;
        border-right: 1px solid rgba(0, 242, 254, 0.1);
    }

    /* Metric Styling */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #00f2fe;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Hide standard streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- IMAGE URLS ---
IMAGES = {
    "hero": "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&q=80&w=2070", # Wind turbine sunset
    "solar": "https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?auto=format&fit=crop&q=80&w=2072", 
    "home": "https://images.unsplash.com/photo-1558002038-1055907df827?auto=format&fit=crop&q=80&w=2070", # Smart home
    "invest": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&q=80&w=2015", # Finance charts
    "prediction": "https://images.unsplash.com/photo-1518186239717-2e909873173d?auto=format&fit=crop&q=80&w=2070",
}

# --- DATA GENERATION ---
@st.cache_data
def load_data():
    states_data = {
        'State': [
            'Rajasthan', 'Gujarat', 'Tamil Nadu', 'Karnataka', 'Maharashtra', 
            'Andhra Pradesh', 'Madhya Pradesh', 'Telangana', 'Uttar Pradesh', 'Punjab',
            'Haryana', 'Kerala', 'Odisha', 'West Bengal', 'Bihar', 
            'Chhattisgarh', 'Uttarakhand', 'Himachal Pradesh', 'Assam', 'Jharkhand'
        ],
        'Lat': [27.02, 22.25, 11.05, 15.31, 19.60, 15.91, 23.47, 17.12, 26.84, 31.14, 29.05, 10.85, 20.95, 22.98, 25.09, 21.27, 30.06, 31.10, 26.20, 23.61],
        'Lon': [74.21, 71.19, 78.38, 75.71, 75.55, 79.74, 77.94, 79.20, 80.94, 75.34, 76.08, 76.27, 85.09, 87.85, 85.31, 81.86, 79.01, 77.17, 92.93, 85.27],
        'Installed_Capacity_MW': [34140, 33390, 25240, 22500, 19800, 17500, 14200, 12800, 10500, 8500, 7800, 6500, 5800, 5200, 4800, 4200, 3800, 3500, 2800, 2500],
        'Solar_Potential': [142, 120, 95, 100, 88, 92, 110, 105, 80, 75, 78, 65, 70, 68, 72, 85, 60, 55, 62, 70],  # Relative scale
        'Wind_Potential': [45, 95, 100, 85, 78, 70, 40, 35, 15, 20, 25, 55, 60, 50, 10, 30, 40, 45, 15, 25],   # Relative scale
        'Policy_Score': [9.2, 9.5, 9.0, 8.8, 8.5, 8.2, 8.0, 8.1, 7.5, 7.0, 7.2, 8.4, 7.8, 7.6, 6.5, 7.0, 7.5, 7.8, 6.8, 7.1]
    }
    df = pd.DataFrame(states_data)
    df['Utilization'] = np.random.uniform(70, 95, size=len(df))
    df['Daily_Generation_MW'] = df['Installed_Capacity_MW'] * (df['Utilization']/100)
    return df

df = load_data()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #00f2fe;'>⚡ RenewTrack AI</h1>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("MAIN MENU", [
        "📊 National Grid Dashboard",
        "💰 Investment Strategy AI",
        "🔮 Future Opportunity Predictor",
        "🏠 Smart Home Energy Optimizer",
        "🌐 National Intelligence Map"
    ])
    st.markdown("---")
    st.info("💡 **AI INSIGHT**: Solar adoption in residential areas has increased by 14% this quarter.")

# --- PAGE 1: DASHBOARD OVERVIEW ---
if page == "📊 National Grid Dashboard":
    st.markdown('<div class="main-header"><h1><span class="gradient-text">RenewTrack AI</span> Dashboard</h1><p>Real-time analytics for India\'s renewable energy transition</p></div>', unsafe_allow_html=True)
    
    # Global Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="glass-card"><p class="metric-label">Total Capacity</p><p class="metric-value">{df["Installed_Capacity_MW"].sum() / 1000:,.1f} GW</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="glass-card"><p class="metric-label">Live Generation</p><p class="metric-value">{df["Daily_Generation_MW"].sum() / 1000:,.1f} GW</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="glass-card"><p class="metric-label">Avg Utilization</p><p class="metric-value">{df["Utilization"].mean():.1f}%</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="glass-card"><p class="metric-label">CO2 Saved / Day</p><p class="metric-value">{df["Daily_Generation_MW"].sum() * 1.2 / 1000:,.1f}k T</p></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Performing States")
        fig_cap = px.bar(df.sort_values('Installed_Capacity_MW', ascending=False).head(10), 
                         x='State', y='Installed_Capacity_MW', 
                         color='Installed_Capacity_MW', color_continuous_scale='Blues',
                         template="plotly_dark")
        fig_cap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_cap, use_container_width=True)

    with c2:
        st.subheader("Efficiency Distribution")
        fig_pie = px.pie(df, values='Installed_Capacity_MW', names='State', 
                         hole=0.4, color_discrete_sequence=px.colors.sequential.Deep_r)
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- PAGE 2: INVESTMENT STRATEGY AI ---
elif page == "💰 Investment Strategy AI":
    st.markdown('<div class="main-header"><h1>Investment <span class="gradient-text">Recommendation Engine</span></h1><p>Data-driven insights for high-yield renewable investments</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Investment Profile")
        budget = st.selectbox("Your Budget (₹ INR)", [
            "Below 10 Lakhs (Residential)", 
            "10 Lakhs - 50 Lakhs (SME)", 
            "50 Lakhs - 5 Crores (Commercial)", 
            "Above 5 Crores (Utility Scale)"
        ])
        energy_type = st.radio("Primary Focus", ["Solar PV", "Wind Energy", "Hybrid Solar-Wind"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Simple Logic for Recommendation
        if energy_type == "Solar PV":
            top_state = df.sort_values(by=['Solar_Potential', 'Policy_Score'], ascending=False).iloc[0]
            reasoning = f"{top_state['State']} offers the highest solar irradiance in India with an average of 6.2 kWh/m²/day. " \
                        f"The state's {top_state['Policy_Score']}/10 policy favorability ensures faster permit approvals and better net-metering benefits."
        elif energy_type == "Wind Energy":
            top_state = df.sort_values(by=['Wind_Potential', 'Policy_Score'], ascending=False).iloc[0]
            reasoning = f"{top_state['State']} is a pioneer in wind energy with vast coastal stretches. " \
                        f"Policy scores here are among the best for repowering old turbines and new offshore projects."
        else:
            top_state = df[df['State'] == 'Gujarat'].iloc[0] # Gujarat is excellent for hybrid
            reasoning = "Gujarat ranks highest for Hybrid projects due to its massive Renewable Energy parks (Khavda) " \
                        "where solar and wind complement each other's generation cycles."
        
        st.markdown(f'''
        <div class="glass-card">
            <h3>Best State for Investment: <span style="color:#bcff00;">{top_state['State']}</span></h3>
            <p style="font-size: 1.1rem; line-height: 1.6; color: #ccd6f6;">{reasoning}</p>
            <div style="display: flex; gap: 20px; margin-top: 20px;">
                <div style="flex: 1; background: rgba(0,242,254,0.1); padding: 15px; border-radius: 12px; text-align: center;">
                    <p style="margin:0; font-size: 0.8rem; color: #00f2fe;">EST. ROI</p>
                    <p style="margin:0; font-size: 1.5rem; font-weight: 800;">18-22%</p>
                </div>
                <div style="flex: 1; background: rgba(188,255,0,0.1); padding: 15px; border-radius: 12px; text-align: center;">
                    <p style="margin:0; font-size: 0.8rem; color: #bcff00;">BREAK EVEN</p>
                    <p style="margin:0; font-size: 1.5rem; font-weight: 800;">4.2 Years</p>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.subheader("Comparison of Growth Potential")
    fig_potential = px.scatter(df, x='Solar_Potential', y='Wind_Potential', color='Policy_Score',
                              size='Installed_Capacity_MW', hover_name='State',
                              title="Opportunity Matrix: Potential vs Policy Framework",
                              labels={'Solar_Potential': 'Solar Growth Index', 'Wind_Potential': 'Wind Growth Index'},
                              color_continuous_scale='Viridis', template="plotly_dark")
    st.plotly_chart(fig_potential, use_container_width=True)

# --- PAGE 3: FUTURE OPPORTUNITY PREDICTOR ---
elif page == "🔮 Future Opportunity Predictor":
    st.markdown('<div class="main-header"><h1>Future <span class="gradient-text">Opportunity Predictor</span></h1><p>Predictive analysis for the next decade of energy</p></div>', unsafe_allow_html=True)
    
    st.markdown('''
    <div class="glass-card" style="border-left: 5px solid #00f2fe;">
        <h3>🚀 AI Forecasting Summary</h3>
        <ul>
            <li><b>Wind energy</b> will grow fastest in <b>Gujarat</b> by 2028 due to massive offshore wind auctions.</li>
            <li><b>Rajasthan</b> will maintain dominance in <b>Solar</b> reaching 100GW capacity by 2030.</li>
            <li><b>Tamil Nadu</b> is expected to emerge as the <b>Green Hydrogen</b> hub of Asia.</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

    # Simulated Forecast Data
    years = list(range(2024, 2031))
    forecast = pd.DataFrame({
        'Year': years,
        'Solar (GW)': [75 + (i-2024)*15 for i in years],
        'Wind (GW)': [45 + (i-2024)*8 for i in years],
        'Hybrid/Storage (GW)': [5 + (i-2024)*12 for i in years]
    })
    
    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=forecast['Year'], y=forecast['Solar (GW)'], name='Solar Projection', fill='tozeroy', line_color='#00f2fe'))
    fig_future.add_trace(go.Scatter(x=forecast['Year'], y=forecast['Wind (GW)'], name='Wind Projection', fill='tozeroy', line_color='#bcff00'))
    fig_future.add_trace(go.Scatter(x=forecast['Year'], y=forecast['Hybrid/Storage (GW)'], name='Storage Integration', fill='tozeroy', line_color='#60A5FA'))
    
    fig_future.update_layout(title="National Capacity Projection (2024-2030)", template="plotly_dark", 
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             xaxis_title="Year", yaxis_title="Capacity (GW)")
    st.plotly_chart(fig_future, use_container_width=True)

# --- PAGE 4: HOME ENERGY OPTIMIZER ---
elif page == "🏠 Smart Home Energy Optimizer":
    st.markdown('<div class="main-header"><h1>Smart <span class="gradient-text">Home Energy</span> Optimizer</h1><p>Personalized appliance-based analytics & AI recommendations</p></div>', unsafe_allow_html=True)
    
    with st.container():
        st.subheader("💡 Energy Consumption Questionnaire")
        st.write("Tell us about your daily usage to get a custom breakdown.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ac_hours = st.select_slider("How many hours do you use AC?", options=list(range(0, 25)), value=6)
            fan_count = st.number_input("How many fans do you use?", min_value=0, max_value=20, value=4)
        with col2:
            fan_hours = st.select_slider("How many hours are fans running?", options=list(range(0, 25)), value=12)
            light_hours = st.select_slider("How long are lights ON (Total)?", options=list(range(0, 25)), value=6)
        with col3:
            fridge_status = st.toggle("Is Refrigerator always ON?", value=True)
            other_load = st.number_input("Other Loads (W) - TV, PC, etc.", value=200)

    # Standard Watt Values
    APPLIANCES = {
        "AC (1.5 Ton)": {"watts": 1500, "hours": ac_hours},
        "Fans": {"watts": 75 * fan_count, "hours": fan_hours},
        "Lights": {"watts": 50, "hours": light_hours},
        "Fridge": {"watts": 200, "hours": 24 if fridge_status else 0},
        "Others": {"watts": other_load, "hours": 4}
    }
    
    # Calculations
    results = []
    total_wh = 0
    for name, data in APPLIANCES.items():
        wh = data['watts'] * data['hours']
        units = wh / 1000
        cost = units * 8.0 # ₹8 per unit
        total_wh += wh
        results.append({"Device": name, "Units Consumed": units, "Monthly Cost (₹)": cost * 30})
    
    res_df = pd.DataFrame(results)
    
    st.markdown("---")
    
    # Insights Dashboard
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Device-wise Energy Contribution")
        fig_donut = px.pie(res_df, values='Units Consumed', names='Device', hole=0.5,
                           color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig_donut.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_donut, use_container_width=True)
    
    with c2:
        st.subheader("Cost Breakdown")
        total_monthly_cost = res_df["Monthly Cost (₹)"].sum()
        st.markdown(f'''
        <div class="glass-card" style="text-align: center;">
            <p class="metric-label">Estimated Monthly Bill</p>
            <p class="metric-value">₹{total_monthly_cost:,.0f}</p>
            <p style="color: #bcff00;">@ ₹8.0 per Unit</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # AI Recommendation
        st.subheader("🤖 AI Smart Suggestion")
        if ac_hours > 5:
            save_amt = (1500 * 1 / 1000) * 8 * 30
            st.warning(f"💡 Reduce AC usage by **1 hour** to save **₹{save_amt:,.0f}** monthly!")
        if total_wh/1000 > 15:
            st.error("⚠ **High Consumption Pattern**: Your usage is 40% above average. Consider switching to 5-star BEE rated appliances.")
        else:
            st.success("✅ **Efficient Usage**: Your consumption is within the green zone.")

# --- PAGE 5: NATIONAL INTELLIGENCE MAP ---
elif page == "🌐 National Intelligence Map":
    st.markdown('<div class="main-header"><h1>National <span class="gradient-text">Intelligence Map</span></h1><p>Interactive spatial distribution of India\'s energy assets</p></div>', unsafe_allow_html=True)
    
    # Map logic
    m = folium.Map(location=[22, 78], zoom_start=5, tiles='CartoDB dark_matter')
    
    for _, row in df.iterrows():
        color = '#00f2fe' if row['Utilization'] > 85 else '#bcff00' if row['Utilization'] > 75 else '#ff4b4b'
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=row['Installed_Capacity_MW']/3000 + 5,
            popup=f"<b>{row['State']}</b><br>Capacity: {row['Installed_Capacity_MW']} MW",
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
    
    folium_static(m, width=1100, height=600)
    
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px;">
        <span>🔵 High Efficiency (>85%)</span>
        <span>🟢 Optimal (75-85%)</span>
        <span>🔴 Maintenance Required (<75%)</span>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #8892b0;'>RenewTrack AI © 2026 • Powering a Cleaner Tomorrow with Advanced Analytics</p>", unsafe_allow_html=True)
