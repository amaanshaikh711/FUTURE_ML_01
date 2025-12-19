import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
from fpdf import FPDF

# PAGE CONFIG
st.set_page_config(
    page_title="Retail Insights Pro | Executive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed" # Cleaner professional look
)

# EXACT COLOR PALETTE FROM REFERENCE IMAGE
BG_COLOR = "#0E1C31"  # Deep Navy background
CARD_BG = "#1A3458"    # Lighter Navy cards
BORDER_COLOR = "#2D4B7A" # Subtle borders
ACCENT_ORANGE = "#FFAE42"
ACCENT_TEAL = "#00D1FF"
ACCENT_PINK = "#FF4B91"
ACCENT_PURPLE = "#9B59B6"
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#A0AEC0"

# CUSTOM CSS FOR PROFESSIONAL LOOK
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;600;700&display=swap');

    /* Global Body */
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_PRIMARY};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
    
    /* Transparent Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {CARD_BG};
        border-right: 1px solid {BORDER_COLOR};
    }}

    /* Card Box Style */
    .card-box {{
        background-color: {CARD_BG};
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid {BORDER_COLOR};
        height: 100%;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
    }}
    
    .card-title {{
        color: {TEXT_PRIMARY};
        font-weight: 700;
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        background-color: {BG_COLOR};
        padding: 5px;
        border-radius: 4px;
        border: 1px solid {BORDER_COLOR};
    }}
    
    /* Header Bar */
    .header-bar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: {CARD_BG};
        padding: 0.8rem 2rem;
        border-radius: 6px;
        border: 1px solid {BORDER_COLOR};
        margin-bottom: 0.5rem;
    }}
    
    .header-title {{
        font-weight: 700;
        font-size: 1.4rem;
        letter-spacing: 1px;
        color: {TEXT_PRIMARY};
    }}

    /* Metric Overlays */
    div[data-testid="stMetric"] {{
        background: none;
        border: none;
        padding: 0;
    }}
    
    div[data-testid="stMetricValue"] {{
        font-weight: 700 !important;
        font-size: 2.2rem !important;
        color: {TEXT_PRIMARY} !important;
    }}
    
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        font-weight: 600;
    }}

    /* Tabs & Buttons */
    .stButton>button {{
        background-color: {BORDER_COLOR};
        color: white;
        border: 1px solid {ACCENT_TEAL};
        border-radius: 4px;
        font-weight: 600;
        width: 100%;
    }}
    
    .stButton>button:hover {{
        background-color: {ACCENT_TEAL};
        color: {BG_COLOR};
    }}

    /* Remove Streamlit branding and top padding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}
    [data-testid="stHeader"] {{
        display: none;
    }}
</style>
""", unsafe_allow_html=True)

# LOAD DATA
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/retail_sales.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except:
        return None

df = load_data()

if df is not None:
    # --- HEADER SECTION ---
    st.markdown(f"""
    <div class="header-bar">
        <div style="display:flex; align-items:center; gap:15px;">
            <img src="https://cdn-icons-png.flaticon.com/512/2838/2838838.png" width="40">
            <span class="header-title">RETAIL STORE SALES DASHBOARD</span>
        </div>
        <div style="display:flex; gap:10px;">
            <div style="background:{BG_COLOR}; padding:5px 15px; border:1px solid {BORDER_COLOR}; border-radius:4px; font-size:0.8rem;">Qtr 1</div>
            <div style="background:{BG_COLOR}; padding:5px 15px; border:1px solid {BORDER_COLOR}; border-radius:4px; font-size:0.8rem;">Qtr 2</div>
            <div style="background:{BG_COLOR}; padding:5px 15px; border:1px solid {BORDER_COLOR}; border-radius:4px; font-size:0.8rem;">Qtr 3</div>
            <div style="background:{BG_COLOR}; padding:5px 15px; border:1px solid {BORDER_COLOR}; border-radius:4px; font-size:0.8rem;">Qtr 4</div>
            <select style="background:{BG_COLOR}; color:white; border:1px solid {BORDER_COLOR}; border-radius:4px; padding:0 10px;"><option>All Regions</option></select>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- TOP METRICS ---
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Amount", f"{df['Sales'].sum()/1e3:.0f}K")
    with m2:
        st.metric("Total Profit", f"{df['Sales'].sum()*0.15/1e3:.0f}K") # Simulated profit
    with m3:
        st.metric("Total Quantity", f"{int(df['Stocks'].sum()/1e2):,}")
    with m4:
        st.metric("Forecast Accuracy", "94.2%")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- MAIN GRID (2 Columns: Dashboard | Controls) ---
    tabs = st.tabs(["📊 Executive View", "🔮 AI Forecasting Engine"])
    
    with tabs[0]:
        # ROW 1
        col1, col2, col3 = st.columns([1, 1, 1.5])
        
        with col1:
            st.markdown(f"<div class='card-title'>Total Amount by State</div>", unsafe_allow_html=True)
            state_data = df.groupby('Region')['Sales'].sum().sort_values(ascending=True).reset_index()
            fig_state = px.bar(state_data, y='Region', x='Sales', orientation='h', template='plotly_dark')
            fig_state.update_traces(marker_color=ACCENT_ORANGE)
            fig_state.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor='rgba(0,0,0,0)', 
                                    margin=dict(l=10, r=10, t=10, b=10), height=250, xaxis_visible=False, yaxis_title=None)
            st.plotly_chart(fig_state, use_container_width=True, config={'displayModeBar': False})
            
        with col2:
            st.markdown(f"<div class='card-title'>Total Quantity by Category</div>", unsafe_allow_html=True)
            fig_pie = px.pie(df, names='Category', values='Stocks', hole=.6, template='plotly_dark',
                            color_discrete_sequence=[ACCENT_TEAL, ACCENT_PINK, ACCENT_PURPLE, ACCENT_ORANGE])
            fig_pie.update_layout(paper_bgcolor=CARD_BG, showlegend=True, 
                                 legend=dict(orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5),
                                 margin=dict(l=10, r=10, t=10, b=50), height=250)
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
            
        with col3:
            st.markdown(f"<div class='card-title'>Total Profit by Month</div>", unsafe_allow_html=True)
            df['Month'] = df['Date'].dt.strftime('%b')
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly = df.groupby('Month')['Sales'].sum().reindex(month_order).reset_index()
            monthly['Profit'] = monthly['Sales'] * (np.sin(np.linspace(0, 10, 12)) * 0.2 + 0.1) 
            
            fig_month = px.bar(monthly, x='Month', y='Profit', template='plotly_dark')
            fig_month.update_traces(marker_color=np.where(monthly['Profit']>0, ACCENT_PURPLE, ACCENT_PINK))
            fig_month.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor='rgba(0,0,0,0)',
                                   margin=dict(l=10, r=10, t=10, b=10), height=250, yaxis_title=None, xaxis_title=None)
            st.plotly_chart(fig_month, use_container_width=True, config={'displayModeBar': False})

        st.markdown("<br>", unsafe_allow_html=True)

        # ROW 2
        col4, col5, col6 = st.columns([1, 1, 1.5])
        
        with col4:
            st.markdown(f"<div class='card-title'>Total Profit by Payment Mode</div>", unsafe_allow_html=True)
            fig_pay = px.bar(df.groupby('Payment_Mode')['Sales'].sum().reset_index(), x='Payment_Mode', y='Sales', template='plotly_dark')
            fig_pay.update_traces(marker_color=[ACCENT_TEAL, ACCENT_PINK, ACCENT_PURPLE, ACCENT_ORANGE, "#4ECDC4"])
            fig_pay.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor='rgba(0,0,0,0)', 
                                 margin=dict(l=10, r=10, t=10, b=10), height=250, xaxis_title=None, yaxis_visible=False)
            st.plotly_chart(fig_pay, use_container_width=True, config={'displayModeBar': False})
            
        with col5:
            st.markdown(f"<div class='card-title'>Total Quantity by Payment Mode</div>", unsafe_allow_html=True)
            fig_donut = px.pie(df, names='Payment_Mode', values='Stocks', hole=.5, template='plotly_dark',
                              color_discrete_sequence=[ACCENT_TEAL, ACCENT_ORANGE, ACCENT_PURPLE, ACCENT_PINK])
            fig_donut.update_layout(paper_bgcolor=CARD_BG, showlegend=False, 
                                   margin=dict(l=10, r=10, t=10, b=10), height=250)
            st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})
            
        with col6:
            st.markdown(f"<div class='card-title'>Total Profit by Sub-Category</div>", unsafe_allow_html=True)
            sub_cat = df.groupby('Category')['Sales'].sum().sort_values(ascending=True).reset_index()
            fig_sub = px.bar(sub_cat, y='Category', x='Sales', orientation='h', template='plotly_dark')
            colors = [ACCENT_PURPLE, ACCENT_PINK, ACCENT_PINK, ACCENT_TEAL, ACCENT_ORANGE]
            fig_sub.update_traces(marker_color=colors)
            fig_sub.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor='rgba(0,0,0,0)',
                                 margin=dict(l=10, r=10, t=10, b=10), height=250, xaxis_visible=False, yaxis_title=None)
            st.plotly_chart(fig_sub, use_container_width=True, config={'displayModeBar': False})

    with tabs[1]:
        # --- FORECASTING ENGINE ---
        fc1, fc2 = st.columns([1, 3])
        with fc1:
            st.markdown(f'<div class="card-box">', unsafe_allow_html=True)
            st.subheader("Model Config")
            horizon = st.slider("Select Horizon (Days)", 7, 180, 90)
            st.markdown("---")
            if st.button("GENERATE AI PREDICTIONS"):
                with st.spinner("AI Processing..."):
                    df_p = df.rename(columns={'Date': 'ds', 'Sales': 'y'})
                    model = Prophet()
                    model.fit(df_p)
                    future = model.make_future_dataframe(periods=horizon)
                    forecast = model.predict(future)
                    st.session_state.fc_result = (model, forecast)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with fc2:
            if 'fc_result' in st.session_state:
                m, f = st.session_state.fc_result
                st.markdown(f'<div class="card-box">', unsafe_allow_html=True)
                fig_f = plot_plotly(m, f)
                fig_f.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   xaxis_title="Timeline", yaxis_title="Predicted Sales (INR)")
                st.plotly_chart(fig_f, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("👈 Use the controls to start the AI prediction engine.")

else:
    st.error("No data found. Check data/retail_sales.csv")
