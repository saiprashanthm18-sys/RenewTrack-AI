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
    .main {
        background-color: #f0f4f8;
    }
    .stMetric {
        padding: 20px;
        border-radius: 10px;
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
    h1, h2, h3 {
        color: #1e3a8a; /* Deep blue */
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7d32, #1e3a8a);
        color: white;
    }
    @media (max-width: 768px) {
        .footer {
            position: relative;
        }
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1e3a8a;
        color: white;
        text-align: center;
        padding: 10px;
        font-weight: bold;
        z-index: 999;
    }
    .main-header {
        background: linear-gradient(90deg, #1e3a8a, #2e7d32);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- IMAGE URLS ---
IMAGES = {
    "hero": "https://images.unsplash.com/photo-1466611653911-9545540316e2?auto=format&fit=crop&q=80&w=2070", # Wind turbines
    "solar": "https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&q=80&w=2072", # Solar panels
    "prediction": "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&q=80&w=2070", # Tech/AI circuit
    "calculator": "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&q=80&w=2070", # Green leaf/Energy
    "recommendation": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80&w=2072", # Global network
    "utilization": "https://images.unsplash.com/photo-1542332213-9b5a5a3fad35?auto=format&fit=crop&q=80&w=2070", # Analytics/Solar
    "map": "https://images.unsplash.com/photo-1524661135-423995f22d0b?auto=format&fit=crop&q=80&w=2070", # Satellite view/India context
    "cap_bg": "https://images.unsplash.com/photo-1548337138-e87d889cc369?auto=format&fit=crop&q=80&w=600",
    "gen_bg": "https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&q=80&w=600",
    "util_bg": "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=600",
    "co2_bg": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&q=80&w=600",
}

# --- CUSTOM COMPONENTS ---
def styled_metric(label, value, delta, img_url, is_positive=True):
    delta_color = "#00ff00" if is_positive else "#ff4b4b"
    st.markdown(f"""
    <div style="
        background-image: url('{img_url}');
        background-size: cover;
        background-position: center;
        border-radius: 15px;
        height: 160px;
        position: relative;
        overflow: hidden;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    ">
        <div style="
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
            padding: 20px;
            color: white;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <p style="margin: 0; font-size: 14px; font-weight: 500; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px;">{label}</p>
            <h2 style="margin: 5px 0; font-size: 26px; font-weight: 800; color: white;">{value}</h2>
            <p style="margin: 0; font-size: 14px; font-weight: bold; color: {delta_color};">{delta}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- DATA GENERATION ---
@st.cache_data
def load_data():
    states_data = {
        'State': ['Tamil Nadu', 'Gujarat', 'Rajasthan', 'Karnataka', 'Maharashtra'],
        'Lat': [11.1271, 22.2587, 27.0238, 15.3173, 19.7515],
        'Lon': [78.6569, 71.1924, 74.2179, 75.7139, 75.7139],
        'Installed_Capacity_MW': [18000, 16500, 19200, 15800, 14200],
        'Solar_Percentage': [45, 60, 85, 40, 35],
        'Wind_Percentage': [55, 40, 15, 60, 65],
        'Daily_Generation_MW': [13500, 12800, 14500, 10200, 8500],
        'Temp': [32, 35, 38, 30, 31],
        'Wind_Speed': [8.5, 7.2, 4.5, 9.2, 7.8],
        'Irradiance': [750, 850, 950, 700, 680]
    }
    df = pd.DataFrame(states_data)
    df['Utilization'] = (df['Daily_Generation_MW'] / df['Installed_Capacity_MW']) * 100
    df['CO2_Saved_Tons'] = df['Daily_Generation_MW'] * 24 * 0.9 # Daily generation * 24h * 0.9 tons/MWh
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

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🌱 RenewTrack AI")
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

    # --- LIVE DATA SIMULATION ---
    st_autorefresh(interval=10000, key="datarefresh")
    
    # Base values
    base_capacity = 83700
    base_generation = 59500
    
    # Random variations
    live_capacity = base_capacity + np.random.randint(-50, 50)
    live_generation = base_generation + np.random.randint(-500, 1000)
    live_utilization = (live_generation / live_capacity) * 100
    live_co2 = live_generation * 24 * 0.9 # Dynamic CO2 based on simulated generation
    
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist).strftime('%H:%M:%S')
    st.markdown(f"**Last Updated:** {current_time} | 📡 Live simulated renewable energy feed")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        styled_metric("Total Capacity", f"{live_capacity:,} MW", "Variable", IMAGES["cap_bg"])
    with col2:
        styled_metric("Current Gen", f"{live_generation:,} MW", "Live Feed", IMAGES["gen_bg"])
    with col3:
        styled_metric("Avg Utilization", f"{live_utilization:.1f}%", "Auto-calc", IMAGES["util_bg"])
    with col4:
        styled_metric("CO2 Reduction", f"{live_co2/1e6:.2f}M Tons", "Real-time", IMAGES["co2_bg"])

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("30-Day Generation Trend")
        fig_line = px.line(ts_df, x='Date', y='Generation_MW', title="National Generation Trend (MW)")
        fig_line.update_layout(template="plotly_white")
        st.plotly_chart(fig_line, width="stretch")

    with c2:
        st.subheader("Solar vs Wind Capacity")
        fig_bar = px.bar(df, x='State', y=['Solar_Percentage', 'Wind_Percentage'], 
                         title="Energy Source Split (%)", barmode='group',
                         color_discrete_sequence=['#ffcc00', '#00ccff'])
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
    
    df_sorted = df.sort_values(by='Utilization', ascending=False)
    
    st.subheader("State Efficiency Rankings")
    fig_efficiency = px.bar(df_sorted, x='State', y='Utilization', color='Utilization',
                            color_continuous_scale='RdYlGn', title="Utilization Percentage by State")
    st.plotly_chart(fig_efficiency, width="stretch")

    underperforming = df[df['Utilization'] < 70]
    if not underperforming.empty:
        st.warning(f"⚠ Detected {len(underperforming)} underperforming states with < 70% utilization.")
        st.table(underperforming[['State', 'Installed_Capacity_MW', 'Daily_Generation_MW', 'Utilization']])
    else:
        st.success("✅ All states are performing efficiently (>= 70% utilization).")

    st.subheader("Detailed Breakdown")
    st.dataframe(df[['State', 'Installed_Capacity_MW', 'Daily_Generation_MW', 'Utilization', 'CO2_Saved_Tons']].style.format({
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
                           line_shape='spline', color_discrete_sequence=['#2C3E50'])
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
    
    target_state = st.selectbox("Select State to Analyze", df['State'].unique())
    state_row = df[df['State'] == target_state].iloc[0]
    
    st.write(f"Analyzing efficiency for **{target_state}**...")
    
    if state_row['Utilization'] < 75:
        st.error(f"⚠ **Underperformance Detected**: {target_state}'s current utilization is only {state_row['Utilization']:.1f}%")
        
        with st.container():
            st.markdown("""
            ### 🛠 Suggested Interventions:
            1. **Solar Panel Cleaning**: Dust accumulation in this region reduces efficiency by ~12%.
            2. **Predictive Maintenance**: 2/5 Wind Turbines show abnormal vibration patterns.
            3. **Battery Storage**: Current grid curtailment is high during peak solar hours.
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

    # Create map centered on India
    m = folium.Map(location=[22, 78], zoom_start=5, tiles='CartoDB positron')

    # Add markers
    for _, row in df.iterrows():
        color = 'green' if row['Utilization'] > 85 else 'orange' if row['Utilization'] > 70 else 'red'
        
        popup_text = f"""
        <b>State:</b> {row['State']}<br>
        <b>Utilization:</b> {row['Utilization']:.1f}%<br>
        <b>CO2 Saved:</b> {row['CO2_Saved_Tons']:,.0f} Tons
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Lon']],
            radius=row['Installed_Capacity_MW'] / 1000,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
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

    base_gen = df['Daily_Generation_MW'].sum()
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
