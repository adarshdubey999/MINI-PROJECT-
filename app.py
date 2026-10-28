"""
AI-BASED DYSCALCULIA SCREENING TOOL
Main Streamlit Web Application - Fully Interactive & Responsive Edition
Developed according to the Project Report by Aadarsh Dubey (Seat No: 510510555)
Supervised by Prof. Javed Pathan, Department of Computer Science,
Rizvi College of Arts, Science and Commerce (Affiliated to University of Mumbai).

Tech Stack (Chapter 4 Survey of Technology):
- Python 3.13
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SQLite
"""

import os
import sys
import time
import json
import tempfile
from datetime import datetime
from pathlib import Path

# Ensure Matplotlib temp directory is set cleanly to avoid permissions warnings on Windows
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import streamlit as st
import pandas as pd
import numpy as np

# Import internal modules
from database import (
    register_user, authenticate_user, get_user_by_id,
    save_assessment_record, get_user_assessments, get_assessment_details,
    get_system_stats, init_db
)
from ml_engine import (
    load_screening_model, predict_dyscalculia_risk, FEATURE_COLUMNS
)
from assessment_data import (
    get_all_questions, get_category_names, generate_dot_svg
)
from visualizations import (
    plot_category_performance, plot_cognitive_radar_chart,
    plot_response_times, plot_historical_progress, plot_ai_model_metrics
)
from recommendations import get_recommendations_for_result


# -----------------------------------------------------------------------------
# 1. Page Configuration & Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Based Dyscalculia Screening Tool",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Accessibility CSS (Dyscalculia-friendly typography, contrast & cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 50%, #3B82F6 100%);
        color: white;
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
    }
    
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: #E0E7FF;
        font-size: 1.02rem;
        margin-bottom: 0px;
    }
    
    .institution-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 10px;
    }

    .card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 18px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1E3A8A;
        line-height: 1.2;
    }
    
    .metric-label {
        font-size: 0.88rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .risk-banner {
        padding: 22px;
        border-radius: 14px;
        color: white;
        font-weight: 600;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #15803D 0%, #16A34A 100%);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #B45309 0%, #D97706 100%);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #B91C1C 0%, #DC2626 100%);
    }

    .question-box {
        background: #F8FAFC;
        border: 2px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
    }

    .comparison-card {
        background: #EFF6FF;
        border: 2px dashed #93C5FD;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E40AF;
        margin: 10px 0;
    }

    .equation-card {
        background: #F1F5F9;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: 2px;
        margin: 10px 0;
    }

    .sequence-card {
        background: #FAF5FF;
        border: 2px solid #E9D5FF;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 800;
        color: #6B21A8;
        letter-spacing: 3px;
        margin: 10px 0;
    }

    .disclaimer-box {
        background: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 16px 20px;
        border-radius: 8px;
        font-size: 0.92rem;
        color: #92400E;
        line-height: 1.5;
        margin-top: 15px;
    }

    /* Quick Question Navigation Bar */
    .q-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.82rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. State & Navigation Controller
# -----------------------------------------------------------------------------
NAV_PAGES = [
    "🏠 Home & Overview",
    "📊 Dashboard",
    "🧠 Screening Assessment",
    "📑 Results & Risk Report",
    "🤖 AI Screening Assistant",
    "📈 User Assessment History",
    "🎮 Math Practice Arena",
    "🔐 Account & Login"
]

if 'nav_page' not in st.session_state:
    st.session_state['nav_page'] = "🏠 Home & Overview"

def navigate_to(page: str):
    """Centralized navigation dispatcher that changes page and triggers instant rerun."""
    st.session_state['nav_page'] = page
    st.rerun()

if 'logged_in_user' not in st.session_state:
    # Default to sample student Rohan
    st.session_state['logged_in_user'] = {
        'user_id': 3,
        'name': 'Demo Student (Rohan)',
        'email': 'rohan@example.com',
        'role': 'Student',
        'age': 10
    }

if 'current_q_index' not in st.session_state:
    st.session_state['current_q_index'] = 0

if 'test_responses' not in st.session_state:
    st.session_state['test_responses'] = {}

if 'question_start_time' not in st.session_state:
    st.session_state['question_start_time'] = time.time()

if 'latest_assessment_id' not in st.session_state:
    st.session_state['latest_assessment_id'] = None

if 'latest_result' not in st.session_state:
    st.session_state['latest_result'] = None


# -----------------------------------------------------------------------------
# 3. Sidebar Navigation & Academic Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 12px;">
        <span style="font-size: 2.8rem;">🧮</span>
        <h3 style="margin: 4px 0 0 0; color: #1E3A8A; font-weight: 800;">Dyscalculia AI</h3>
        <p style="font-size: 0.78rem; color: #64748B; margin: 0;">Screening & Diagnostic Tool</p>
    </div>
    """, unsafe_allow_html=True)

    # Academic Citation Box
    st.markdown("""
    <div style="background: #F1F5F9; border-radius: 10px; padding: 12px; margin-bottom: 16px; border-left: 4px solid #2563EB;">
        <p style="font-size: 0.78rem; font-weight: 700; color: #1E293B; margin: 0 0 4px 0;">PROJECT REPORT</p>
        <p style="font-size: 0.75rem; color: #475569; margin: 0;"><b>Author:</b> Aadarsh Dubey (Seat No: 510510555)</p>
        <p style="font-size: 0.75rem; color: #475569; margin: 0;"><b>Guide:</b> Prof. Javed Pathan</p>
        <p style="font-size: 0.75rem; color: #475569; margin: 0;"><b>Institution:</b> Rizvi College of ASC, Mumbai</p>
        <p style="font-size: 0.75rem; color: #475569; margin: 0;"><b>Degree:</b> B.Sc. Computer Science (Sem V)</p>
    </div>
    """, unsafe_allow_html=True)

    # User Profile Pill
    user = st.session_state['logged_in_user']
    if user:
        st.markdown(f"""
        <div style="background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 10px; padding: 10px 14px; margin-bottom: 16px;">
            <p style="font-size: 0.75rem; color: #1D4ED8; font-weight: 700; text-transform: uppercase; margin: 0;">Active Profile</p>
            <p style="font-size: 0.95rem; font-weight: 800; color: #1E3A8A; margin: 0;">{user['name']}</p>
            <p style="font-size: 0.78rem; color: #3B82F6; margin: 0;">{user['role']} • Age {user.get('age', 12)}</p>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar Navigation Radio with synchronized index
    curr_idx = NAV_PAGES.index(st.session_state['nav_page']) if st.session_state['nav_page'] in NAV_PAGES else 0
    selected_menu = st.radio(
        "Navigation Menu",
        NAV_PAGES,
        index=curr_idx,
        key="main_sidebar_radio"
    )

    # Sync selection if user clicked a sidebar radio button directly
    if selected_menu != st.session_state['nav_page']:
        st.session_state['nav_page'] = selected_menu
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.74rem; color: #94A3B8; text-align: center; line-height: 1.4;">
        <b>Tech Stack (Chapter 4):</b><br/>
        Python • Streamlit • Pandas • NumPy • Scikit-learn • Matplotlib • SQLite
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Top Navigation Bar (Always visible even if sidebar is collapsed)
# -----------------------------------------------------------------------------
top_c1, top_c2, top_c3, top_c4 = st.columns(4)
with top_c1:
    if st.button("🏠 Home", use_container_width=True, type="primary" if st.session_state['nav_page'] == "🏠 Home & Overview" else "secondary"):
        navigate_to("🏠 Home & Overview")
with top_c2:
    if st.button("📊 Dashboard", use_container_width=True, type="primary" if st.session_state['nav_page'] == "📊 Dashboard" else "secondary"):
        navigate_to("📊 Dashboard")
with top_c3:
    if st.button("🧠 Take Test", use_container_width=True, type="primary" if st.session_state['nav_page'] == "🧠 Screening Assessment" else "secondary"):
        st.session_state['current_q_index'] = 0
        st.session_state['question_start_time'] = time.time()
        navigate_to("🧠 Screening Assessment")
with top_c4:
    if st.button("📑 Results", use_container_width=True, type="primary" if st.session_state['nav_page'] == "📑 Results & Risk Report" else "secondary"):
        navigate_to("📑 Results & Risk Report")

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 1: 🏠 Home & Overview (Chapter 1, 2, 3)
# -----------------------------------------------------------------------------
if st.session_state['nav_page'] == "🏠 Home & Overview":
    st.markdown("""
    <div class="main-header">
        <span class="institution-badge">Rizvi College of Arts, Science and Commerce • University of Mumbai</span>
        <h1>AI-Based Dyscalculia Screening Tool</h1>
        <p>An intelligent educational platform for preliminary identification and cognitive analysis of mathematical learning difficulties.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.8, 1.2])

    with col1:
        st.markdown("""
        ### 📌 What is Developmental Dyscalculia?
        **Dyscalculia** is a specific learning difficulty (SLD) that affects an individual's innate ability to understand numbers, acquire arithmetic facts, and perform mathematical reasoning. While affecting ~5-7% of school children, it remains significantly underdiagnosed due to a lack of accessible preliminary screening facilities.

        #### 🧩 Key Manifestations Identified in Chapter 1:
        - **Subitizing Deficit:** Inability to recognize small quantities (1 to 4 dots) without counting one-by-one.
        - **Magnitude Comparison Difficulties:** Trouble determining which number is larger or understanding the distance effect (e.g. 78 vs 87).
        - **Arithmetic Fact Retrieval Impairment:** Reliance on slow finger-counting rather than memory retrieval for facts like `7 + 8` or `6 × 4`.
        - **Spatial Number Line Weakness:** Struggling to estimate where a number sits on an unmarked number line (e.g. 50 on a 0-100 scale).
        - **Sequence & Logic Confusion:** Challenges recognizing skip counting sequences and understanding mathematical word problem relationships.
        """)

        st.markdown("---")
        st.markdown("### 🚀 How the System Works (Spiral Model - Chapter 2.6)")

        step_c1, step_c2, step_c3, step_c4 = st.columns(4)
        with step_c1:
            st.markdown("""
            <div class="card" style="text-align: center; min-height: 175px;">
                <span style="font-size: 2rem;">📝</span>
                <p style="font-weight: 700; margin: 6px 0 2px 0;">1. Assessment</p>
                <p style="font-size: 0.8rem; color: #64748B;">Attempt 15 multi-domain questions covering core number sense.</p>
            </div>
            """, unsafe_allow_html=True)
        with step_c2:
            st.markdown("""
            <div class="card" style="text-align: center; min-height: 175px;">
                <span style="font-size: 2rem;">⏱️</span>
                <p style="font-weight: 700; margin: 6px 0 2px 0;">2. Latency</p>
                <p style="font-size: 0.8rem; color: #64748B;">Measures cognitive processing speed in milliseconds.</p>
            </div>
            """, unsafe_allow_html=True)
        with step_c3:
            st.markdown("""
            <div class="card" style="text-align: center; min-height: 175px;">
                <span style="font-size: 2rem;">🤖</span>
                <p style="font-weight: 700; margin: 6px 0 2px 0;">3. AI Analysis</p>
                <p style="font-size: 0.8rem; color: #64748B;">Scikit-learn Random Forest model calculates risk probability.</p>
            </div>
            """, unsafe_allow_html=True)
        with step_c4:
            st.markdown("""
            <div class="card" style="text-align: center; min-height: 175px;">
                <span style="font-size: 2rem;">📊</span>
                <p style="font-weight: 700; margin: 6px 0 2px 0;">4. Interventions</p>
                <p style="font-size: 0.8rem; color: #64748B;">Generates Matplotlib charts and tailored educational guidance.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("🚀 Start Dyscalculia Screening Assessment", type="primary", use_container_width=True, key="home_start_btn"):
            st.session_state['current_q_index'] = 0
            st.session_state['test_responses'] = {}
            st.session_state['question_start_time'] = time.time()
            navigate_to("🧠 Screening Assessment")

    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="margin-top: 0; color: #1E3A8A;">📋 Project Objectives (Chapter 1.4)</h4>
            <ul style="font-size: 0.88rem; color: #334155; padding-left: 20px;">
                <li><b>Accessible Interface:</b> Clean, interactive web platform built with Streamlit.</li>
                <li><b>Multi-Domain Screening:</b> Assesses subitizing, magnitude, arithmetic, estimation & reasoning.</li>
                <li><b>Machine Learning:</b> Scikit-learn Random Forest pipeline trained on clinical cognitive psychometrics.</li>
                <li><b>Visual Performance:</b> Matplotlib category-wise and radar charts.</li>
                <li><b>Secure Storage:</b> SQLite relational database tracking assessment history.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer-box">
            <b>⚠️ Ethical & Responsible AI Disclaimer (Chapter 1.4 & 1.6):</b><br/>
            This tool is designed as an <b>educational preliminary screening aid</b>. It provides valuable early awareness indicators but <b>does not replace a clinical diagnosis</b> by an educational psychologist, neuropsychologist, or medical doctor.
        </div>
        """, unsafe_allow_html=True)

        # Quick stats box
        sys_stats = get_system_stats()
        st.markdown(f"""
        <div class="card" style="margin-top: 15px;">
            <h5 style="margin-top: 0; color: #1E3A8A;">📈 Platform Statistics</h5>
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <span style="font-size: 1.6rem; font-weight: 800; color: #2563EB;">{sys_stats['total_assessments']}</span>
                    <p style="font-size: 0.75rem; color: #64748B; margin: 0;">Assessments</p>
                </div>
                <div>
                    <span style="font-size: 1.6rem; font-weight: 800; color: #16A34A;">{sys_stats['avg_score']}%</span>
                    <p style="font-size: 0.75rem; color: #64748B; margin: 0;">Avg Score</p>
                </div>
                <div>
                    <span style="font-size: 1.6rem; font-weight: 800; color: #7C3AED;">{sys_stats['total_users']}</span>
                    <p style="font-size: 0.75rem; color: #64748B; margin: 0;">Users</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 2: 📊 Dashboard (Section 7.2.2)
# -----------------------------------------------------------------------------
elif st.session_state['nav_page'] == "📊 Dashboard":
    user = st.session_state['logged_in_user']
    user_name = user['name'] if user else "Guest Student"
    user_id = user['user_id'] if user else 1

    st.markdown(f"""
    <div class="main-header">
        <h1>📊 Student Screening Dashboard</h1>
        <p>Welcome back, <b>{user_name}</b>! Track your assessment performance, screening milestones, and cognitive progress.</p>
    </div>
    """, unsafe_allow_html=True)

    # Retrieve user's past records from SQLite
    user_history = get_user_assessments(user_id)
    total_completed = len(user_history)

    # Top KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_completed}</div>
            <div class="metric-label">Assessments Taken</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        avg_score = round(sum(a['percentage'] for a in user_history) / total_completed, 1) if total_completed > 0 else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #16A34A;">{avg_score}%</div>
            <div class="metric-label">Average Score</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        best_score = round(max(a['percentage'] for a in user_history), 1) if total_completed > 0 else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #2563EB;">{best_score}%</div>
            <div class="metric-label">Personal Best</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        latest_risk = user_history[0]['risk_level'] if total_completed > 0 else "Not Assessed"
        risk_color = "#16A34A" if "Low" in latest_risk else ("#D97706" if "Moderate" in latest_risk else "#DC2626")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {risk_color}; font-size: 1.5rem;">{latest_risk}</div>
            <div class="metric-label">Latest Screening Status</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    d_col1, d_col2 = st.columns([1.6, 1.4])

    with d_col1:
        st.markdown("### 🚀 Quick Actions")
        qa1, qa2 = st.columns(2)
        with qa1:
            st.markdown("""
            <div class="card" style="border-top: 4px solid #2563EB;">
                <h4 style="margin-top: 0; color: #1E3A8A;">🧠 Take Screening</h4>
                <p style="font-size: 0.85rem; color: #64748B;">Complete the 15-question psychometric mathematical test.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start Assessment", key="dash_start", type="primary", use_container_width=True):
                st.session_state['current_q_index'] = 0
                st.session_state['test_responses'] = {}
                st.session_state['question_start_time'] = time.time()
                navigate_to("🧠 Screening Assessment")

        with qa2:
            st.markdown("""
            <div class="card" style="border-top: 4px solid #7C3AED;">
                <h4 style="margin-top: 0; color: #6B21A8;">🤖 AI Assistant</h4>
                <p style="font-size: 0.85rem; color: #64748B;">Test what-if diagnostic parameters with our AI model.</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open AI Assistant", key="dash_ai", use_container_width=True):
                navigate_to("🤖 AI Screening Assistant")

        # Historical Trend Chart if assessments exist
        if total_completed >= 2:
            st.markdown("### 📈 Performance Progression Over Time")
            fig_hist = plot_historical_progress(user_history)
            st.pyplot(fig_hist)
        elif total_completed == 1:
            st.info("💡 You have completed 1 assessment. Take another assessment in the future to view your performance progression trend chart over time!")

    with d_col2:
        st.markdown("### 📋 Recent Assessment Summary")
        if total_completed > 0:
            latest = user_history[0]
            st.markdown(f"""
            <div class="card">
                <p style="font-size: 0.82rem; color: #64748B; margin: 0;">Date: {latest['assessment_date']}</p>
                <h3 style="margin: 4px 0 8px 0; color: #1E3A8A;">Assessment #{latest['assessment_id']}</h3>
                <p style="font-size: 0.92rem; margin: 0 0 6px 0;"><b>Score:</b> {latest['total_score']} / {latest['max_score']} ({latest['percentage']}%)</p>
                <p style="font-size: 0.92rem; margin: 0 0 6px 0;"><b>Total Time:</b> {latest['total_time_seconds']} seconds</p>
                <p style="font-size: 0.92rem; margin: 0 0 6px 0;"><b>Outcome:</b> <span style="font-weight: 700; color: {risk_color};">{latest['risk_level']}</span></p>
                <p style="font-size: 0.92rem; margin: 0 0 6px 0;"><b>Identified Primary Deficit:</b> {latest.get('primary_deficit', 'None')}</p>
                <div style="background: #F8FAFC; border-radius: 8px; padding: 10px; margin-top: 10px; font-size: 0.84rem; color: #475569;">
                    <b>Intervention Note:</b> {latest.get('recommendations', 'Continue standard mathematical development.')[:150]}...
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("📄 View Full Screening Report", key="dash_view_rep", use_container_width=True):
                navigate_to("📑 Results & Risk Report")
        else:
            st.markdown("""
            <div class="card" style="text-align: center; padding: 35px 20px;">
                <span style="font-size: 3rem;">📝</span>
                <h4 style="color: #1E3A8A; margin-top: 10px 0 6px 0;">No Assessments Yet</h4>
                <p style="font-size: 0.88rem; color: #64748B;">You haven't completed a screening assessment yet. Click 'Start Assessment' to begin your first evaluation.</p>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 3: 🧠 Dyscalculia Screening Assessment (Section 7.2.3 & Chapter 6.2 Flowchart)
# -----------------------------------------------------------------------------
elif st.session_state['nav_page'] == "🧠 Screening Assessment":
    questions = get_all_questions()
    total_q = len(questions)

    st.markdown("""
    <div class="main-header">
        <h1>🧠 Dyscalculia Screening Assessment</h1>
        <p>Interactive 15-question evaluation covering all 5 core mathematical cognition domains.</p>
    </div>
    """, unsafe_allow_html=True)

    # Question Navigator Grid (Allows jumping directly to any question)
    st.markdown("##### 📍 Question Navigator (Click any number to jump):")
    nav_cols = st.columns(15)
    for idx, col in enumerate(nav_cols):
        with col:
            q_id = questions[idx]['id']
            is_answered = q_id in st.session_state['test_responses']
            is_curr = (idx == st.session_state['current_q_index'])
            btn_label = f"✓{idx+1}" if is_answered else f"{idx+1}"
            btn_type = "primary" if is_curr else ("secondary")
            if st.button(btn_label, key=f"nav_q_btn_{idx}", type=btn_type, use_container_width=True):
                st.session_state['current_q_index'] = idx
                st.session_state['question_start_time'] = time.time()
                st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Current Question State
    q_idx = st.session_state['current_q_index']
    curr_q = questions[q_idx]

    # Progress Bar & Category Header
    progress_val = (q_idx + 1) / total_q
    st.progress(progress_val)
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span style="font-weight: 700; color: #1E3A8A; font-size: 1.05rem;">
            Question {q_idx + 1} of {total_q}
        </span>
        <span style="background: #DBEAFE; color: #1E40AF; padding: 4px 12px; border-radius: 9999px; font-size: 0.82rem; font-weight: 700;">
            {curr_q['category']}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Question Card Container
    st.markdown(f"""
    <div class="question-box">
        <h3 style="color: #0F172A; margin-top: 0;">{curr_q['title']}</h3>
        <p style="font-size: 1.05rem; color: #334155; margin-bottom: 15px;">{curr_q['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    # Render Specific Visual Type
    v_type = curr_q.get('visual_type')
    if v_type == "svg_dots":
        dot_html = generate_dot_svg(
            count=curr_q['dot_count'],
            color=curr_q.get('dot_color', '#2563EB'),
            size=200
        )
        st.markdown(dot_html, unsafe_allow_html=True)

    elif v_type == "comparison_boxes":
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(f"""
            <div class="comparison-card">
                <span style="font-size: 0.85rem; display: block; color: #64748B;">Box A</span>
                {curr_q['box_a']}
            </div>
            """, unsafe_allow_html=True)
        with c_b:
            st.markdown(f"""
            <div class="comparison-card" style="background: #FDF2F8; border-color: #F472B6; color: #9D174D;">
                <span style="font-size: 0.85rem; display: block; color: #64748B;">Box B</span>
                {curr_q['box_b']}
            </div>
            """, unsafe_allow_html=True)

    elif v_type == "fraction_comparison":
        f_a, f_b = st.columns(2)
        with f_a:
            st.markdown(f"""
            <div class="comparison-card" style="font-size: 1.8rem;">
                🍕 Option A<br/><b>{curr_q['box_a']}</b>
            </div>
            """, unsafe_allow_html=True)
        with f_b:
            st.markdown(f"""
            <div class="comparison-card" style="background: #FEF3C7; border-color: #FCD34D; color: #92400E; font-size: 1.8rem;">
                🍕 Option B<br/><b>{curr_q['box_b']}</b>
            </div>
            """, unsafe_allow_html=True)

    elif v_type == "equation":
        st.markdown(f"""
        <div class="equation-card">
            {curr_q['equation']}
        </div>
        """, unsafe_allow_html=True)

    elif v_type == "sequence":
        st.markdown(f"""
        <div class="sequence-card">
            {curr_q['sequence']}
        </div>
        """, unsafe_allow_html=True)

    elif v_type == "text_card":
        st.markdown(f"""
        <div class="comparison-card" style="font-size: 2rem;">
            "{curr_q['card_text']}"
        </div>
        """, unsafe_allow_html=True)

    elif v_type == "number_line":
        st.markdown("""
        <div style="text-align: center; margin: 15px 0;">
            <div style="position: relative; width: 85%; margin: 0 auto; height: 10px; background: #94A3B8; border-radius: 5px;">
                <span style="position: absolute; left: 0; top: 15px; font-weight: 700; color: #1E293B;">0</span>
                <span style="position: absolute; left: 50%; top: -12px; transform: translateX(-50%); font-size: 1.5rem; color: #DC2626;">▼ <b>?</b></span>
                <span style="position: absolute; right: 0; top: 15px; font-weight: 700; color: #1E293B;">100</span>
            </div>
            <div style="height: 35px;"></div>
        </div>
        """, unsafe_allow_html=True)

    elif v_type == "scenario":
        st.markdown(f"""
        <div style="background: #F0FDF4; border: 2px solid #BBF7D0; border-radius: 12px; padding: 18px; font-size: 1.15rem; color: #166534; font-weight: 600; text-align: center; margin: 10px 0;">
            ✏️ {curr_q['text']}
        </div>
        """, unsafe_allow_html=True)

    # Options Selection
    st.markdown(f"#### ❓ {curr_q['question_text']}")
    saved_ans = st.session_state['test_responses'].get(curr_q['id'], {}).get('submitted_answer', None)

    # Option Selection via Radio
    selected_option = st.radio(
        "Choose your answer from options:",
        curr_q['options'],
        index=curr_q['options'].index(saved_ans) if saved_ans in curr_q['options'] else 0,
        key=f"q_radio_item_{curr_q['id']}"
    )

    # Immediately record this choice into session_state
    elapsed_ms = (time.time() - st.session_state['question_start_time']) * 1000.0
    st.session_state['test_responses'][curr_q['id']] = {
        'question_id': curr_q['id'],
        'category': curr_q['category'],
        'submitted_answer': selected_option,
        'correct_answer': curr_q['correct_answer'],
        'is_correct': (selected_option == curr_q['correct_answer']),
        'response_time_ms': round(max(300.0, elapsed_ms), 1)
    }

    st.markdown(f"""
    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 8px; padding: 8px 14px; margin: 10px 0; font-size: 0.9rem; color: #166534;">
        <b>Current Choice Recorded:</b> <code>{selected_option}</code> (Answer saved automatically)
    </div>
    """, unsafe_allow_html=True)

    with st.expander("💡 View Helpful Hint (If stuck)"):
        st.write(curr_q['hint'])

    st.markdown("<br/>", unsafe_allow_html=True)

    # Navigation Buttons (Previous, Next, Submit)
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1.4])

    with btn_col1:
        if q_idx > 0:
            if st.button("⬅️ Previous Question", use_container_width=True, key="prev_q_btn"):
                st.session_state['current_q_index'] -= 1
                st.session_state['question_start_time'] = time.time()
                st.rerun()

    with btn_col2:
        if q_idx < total_q - 1:
            if st.button("Save & Next Question ➡️", type="primary", use_container_width=True, key="next_q_btn"):
                st.session_state['current_q_index'] += 1
                st.session_state['question_start_time'] = time.time()
                st.rerun()

    with btn_col3:
        if q_idx == total_q - 1:
            if st.button("✅ Submit Assessment & Run AI Model", type="primary", use_container_width=True, key="submit_q_btn"):
                # Evaluate Assessment
                responses = list(st.session_state['test_responses'].values())
                total_score = sum(1 for r in responses if r['is_correct'])
                max_score = len(questions)
                percentage = round((total_score / max_score) * 100.0, 1)
                total_time_sec = round(sum(r['response_time_ms'] for r in responses) / 1000.0, 1)

                # Compute Category-wise metrics
                category_acc = {}
                category_rt = {}
                for cat in get_category_names():
                    cat_responses = [r for r in responses if r['category'] == cat]
                    if cat_responses:
                        c_correct = sum(1 for r in cat_responses if r['is_correct'])
                        c_acc = (c_correct / len(cat_responses)) * 100.0
                        c_rt = sum(r['response_time_ms'] for r in cat_responses) / len(cat_responses)
                    else:
                        c_acc = 100.0
                        c_rt = 1000.0
                    category_acc[cat] = round(c_acc, 1)
                    category_rt[cat] = round(c_rt, 1)

                avg_overall_rt = float(np.mean(list(category_rt.values())))

                # Prepare input features for AI Random Forest Model (Chapter 4.6 & 6.3.2)
                user = st.session_state['logged_in_user']
                age = user.get('age', 12) if user else 12

                ai_features = {
                    'age': age,
                    'subitizing_acc': category_acc.get('Number Recognition & Subitizing', 80.0),
                    'subitizing_rt_ms': category_rt.get('Number Recognition & Subitizing', 1000.0),
                    'magnitude_acc': category_acc.get('Numerical Magnitude Comparison', 80.0),
                    'magnitude_rt_ms': category_rt.get('Numerical Magnitude Comparison', 1000.0),
                    'arithmetic_acc': category_acc.get('Basic Arithmetic Operations', 80.0),
                    'arithmetic_rt_ms': category_rt.get('Basic Arithmetic Operations', 2200.0),
                    'estimation_acc': category_acc.get('Quantity Understanding & Estimation', 80.0),
                    'reasoning_acc': category_acc.get('Mathematical Reasoning & Sequences', 80.0),
                    'overall_accuracy': percentage,
                    'avg_response_time_ms': avg_overall_rt
                }

                # Live AI Inference
                ai_result = predict_dyscalculia_risk(ai_features)

                # Generate recommendations
                rec_info = get_recommendations_for_result(
                    ai_result['risk_outcome'],
                    ai_result['primary_deficit']
                )

                # Save record to SQLite
                user_id = user['user_id'] if user else 1
                screening_data = {
                    'risk_outcome': ai_result['risk_outcome'],
                    'risk_probability': ai_result['risk_probability'],
                    'subitizing_acc': ai_features['subitizing_acc'],
                    'magnitude_acc': ai_features['magnitude_acc'],
                    'arithmetic_acc': ai_features['arithmetic_acc'],
                    'estimation_acc': ai_features['estimation_acc'],
                    'reasoning_acc': ai_features['reasoning_acc'],
                    'avg_response_time_ms': avg_overall_rt,
                    'primary_deficit': ai_result['primary_deficit'],
                    'recommendations': "; ".join(rec_info['general']['action_plan']),
                    'model_version': ai_result['model_version']
                }

                new_ass_id = save_assessment_record(
                    user_id=user_id,
                    total_score=float(total_score),
                    max_score=float(max_score),
                    percentage=percentage,
                    total_time_seconds=total_time_sec,
                    risk_level=ai_result['risk_outcome'],
                    responses_list=responses,
                    screening_result=screening_data
                )

                st.session_state['latest_assessment_id'] = new_ass_id
                st.session_state['latest_result'] = {
                    'assessment_id': new_ass_id,
                    'total_score': total_score,
                    'max_score': max_score,
                    'percentage': percentage,
                    'total_time_seconds': total_time_sec,
                    'ai_result': ai_result,
                    'category_acc': category_acc,
                    'category_rt': category_rt,
                    'responses': responses,
                    'rec_info': rec_info
                }

                # Transition automatically to Results page
                navigate_to("📑 Results & Risk Report")


# -----------------------------------------------------------------------------
# PAGE 4: 📑 Results & Risk Report (Section 7.2.4 & 7.2.5)
# -----------------------------------------------------------------------------
elif st.session_state['nav_page'] == "📑 Results & Risk Report":
    latest = st.session_state.get('latest_result')

    # If no test completed in this session, fetch the latest from SQLite
    if not latest:
        user = st.session_state['logged_in_user']
        user_id = user['user_id'] if user else 1
        history = get_user_assessments(user_id)
        if history:
            ass_details = get_assessment_details(history[0]['assessment_id'])
            if ass_details:
                cat_acc = {
                    'Number Recognition & Subitizing': ass_details['subitizing_acc'],
                    'Numerical Magnitude Comparison': ass_details['magnitude_acc'],
                    'Basic Arithmetic Operations': ass_details['arithmetic_acc'],
                    'Quantity Understanding & Estimation': ass_details['estimation_acc'],
                    'Mathematical Reasoning & Sequences': ass_details['reasoning_acc']
                }
                ai_res = {
                    'risk_outcome': ass_details['risk_outcome'],
                    'risk_probability': ass_details['risk_probability'],
                    'risk_color': "#16A34A" if "Low" in ass_details['risk_outcome'] else ("#D97706" if "Moderate" in ass_details['risk_outcome'] else "#DC2626"),
                    'risk_description': "Screening evaluated via AI Random Forest model.",
                    'primary_deficit': ass_details['primary_deficit'],
                    'category_scores': cat_acc,
                    'model_version': 'RandomForest-1.0'
                }
                rec_info = get_recommendations_for_result(ass_details['risk_outcome'], ass_details['primary_deficit'])
                latest = {
                    'assessment_id': ass_details['assessment_id'],
                    'total_score': ass_details['total_score'],
                    'max_score': ass_details['max_score'],
                    'percentage': ass_details['percentage'],
                    'total_time_seconds': ass_details['total_time_seconds'],
                    'ai_result': ai_res,
                    'category_acc': cat_acc,
                    'category_rt': {k: ass_details['avg_response_time_ms'] for k in cat_acc.keys()},
                    'responses': ass_details.get('responses', []),
                    'rec_info': rec_info
                }

    if not latest:
        st.warning("⚠️ No assessment report found. Please take the screening assessment first.")
        if st.button("🚀 Go to Screening Assessment Now", type="primary"):
            st.session_state['current_q_index'] = 0
            navigate_to("🧠 Screening Assessment")
    else:
        ai_res = latest['ai_result']
        risk_class = "risk-low" if "Low" in ai_res['risk_outcome'] else ("risk-moderate" if "Moderate" in ai_res['risk_outcome'] else "risk-high")

        st.markdown(f"""
        <div class="main-header">
            <span class="institution-badge">Screening Report • Assessment #{latest['assessment_id']}</span>
            <h1>Diagnostic Screening & Risk Analysis</h1>
            <p>Comprehensive cognitive profile, AI classification, and evidence-based learning recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

        # Prominent AI Risk Banner
        st.markdown(f"""
        <div class="risk-banner {risk_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9;">AI Screening Outcome</span>
                    <h2 style="color: white; margin: 4px 0; font-size: 2.2rem;">{ai_res['risk_outcome']}</h2>
                    <p style="margin: 0; font-size: 1.05rem; opacity: 0.95;">{ai_res['risk_description']}</p>
                </div>
                <div style="background: rgba(255,255,255,0.22); border-radius: 12px; padding: 12px 20px; text-align: center;">
                    <span style="font-size: 2rem; font-weight: 800;">{ai_res['risk_probability']}%</span><br/>
                    <span style="font-size: 0.8rem; text-transform: uppercase;">Dyscalculia Index</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # KPI Summary Row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{int(latest['total_score'])} / {int(latest['max_score'])}</div>
                <div class="metric-label">Total Score</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #2563EB;">{latest['percentage']}%</div>
                <div class="metric-label">Accuracy Rate</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #7C3AED;">{latest['total_time_seconds']}s</div>
                <div class="metric-label">Completion Time</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            primary_def = ai_res['primary_deficit'].split(',')[0]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #0F172A; font-size: 1.25rem;">{primary_def}</div>
                <div class="metric-label">Primary Deficit Area</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Performance Visualizations (Chapter 4.7 & 7.2.5 Matplotlib charts)
        st.markdown("### 📊 Performance Analysis & Visualization (Chapter 7.2.5)")
        v_col1, v_col2 = st.columns(2)

        with v_col1:
            st.markdown("#### 1. Category-Wise Accuracy vs Peer Norm")
            fig_bar = plot_category_performance(latest['category_acc'])
            st.pyplot(fig_bar)

        with v_col2:
            st.markdown("#### 2. 5-Domain Cognitive Radar Profile")
            fig_radar = plot_cognitive_radar_chart(latest['category_acc'])
            st.pyplot(fig_radar)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### 3. Cognitive Processing Latency (Response Time Analysis)")
        fig_rt = plot_response_times(latest['category_rt'])
        st.pyplot(fig_rt)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Personalized Recommendations (Chapter 7.2.4 & 8.2)
        rec_data = latest['rec_info']
        st.markdown("### 💡 Tailored Educational Guidance & Interventions")

        st.markdown(f"""
        <div class="card" style="border-left: 5px solid #2563EB;">
            <h4 style="color: #1E3A8A; margin-top: 0;">🎯 Clinical Guidance: {rec_data['general']['badge']}</h4>
            <p style="font-size: 0.95rem; color: #334155;">{rec_data['general']['summary']}</p>
            <h5 style="color: #1E40AF; margin-bottom: 6px;">Recommended Immediate Action Plan:</h5>
            <ul style="font-size: 0.9rem; color: #334155;">
                {"".join([f"<li>{act}</li>" for act in rec_data['general']['action_plan']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Specific domain interventions
        for domain in rec_data.get('specific_interventions', []):
            with st.expander(f"📘 Strategies for: {domain['title']}", expanded=True):
                st.markdown("**Core Learning Strategies:**")
                for s in domain['strategies']:
                    st.markdown(f"- {s}")
                st.markdown("**Classroom & Exam Accommodations:**")
                for acc in domain['classroom_accommodations']:
                    st.markdown(f"- ✏️ {acc}")

        # Review All Questions Attempted
        if latest.get('responses'):
            with st.expander("📝 Detailed Question-by-Question Response Audit", expanded=False):
                resp_df = pd.DataFrame(latest['responses'])
                if not resp_df.empty:
                    resp_df['Result'] = resp_df['is_correct'].apply(lambda x: "✅ Correct" if x == 1 else "❌ Incorrect")
                    resp_df['Latency (s)'] = resp_df['response_time_ms'].apply(lambda x: f"{x/1000.0:.2f}s")
                    display_cols = ['category', 'submitted_answer', 'correct_answer', 'Result', 'Latency (s)']
                    st.dataframe(resp_df[display_cols].rename(columns={
                        'category': 'Domain Category',
                        'submitted_answer': 'Your Answer',
                        'correct_answer': 'Correct Key'
                    }), use_container_width=True)

        # Action Buttons
        st.markdown("---")
        act_c1, act_c2, act_c3 = st.columns(3)
        with act_c1:
            if st.button("🔄 Retake Assessment", use_container_width=True):
                st.session_state['current_q_index'] = 0
                st.session_state['test_responses'] = {}
                st.session_state['question_start_time'] = time.time()
                navigate_to("🧠 Screening Assessment")
        with act_c2:
            if st.button("📈 View Full History", use_container_width=True):
                navigate_to("📈 User Assessment History")
        with act_c3:
            report_text = f"""
============================================================
AI-BASED DYSCALCULIA SCREENING TOOL - OFFICIAL REPORT
Rizvi College of Arts, Science and Commerce • University of Mumbai
Project by Aadarsh Dubey (Seat: 510510555) • Guide: Prof. Javed Pathan
============================================================

Assessment ID: #{latest['assessment_id']}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Student: {st.session_state['logged_in_user']['name']} (Age: {st.session_state['logged_in_user'].get('age', 12)})

AI SCREENING OUTCOME: {ai_res['risk_outcome']}
Dyscalculia Risk Probability: {ai_res['risk_probability']}%
Overall Accuracy: {latest['percentage']}% ({int(latest['total_score'])} / {int(latest['max_score'])})
Total Completion Time: {latest['total_time_seconds']} seconds
Identified Primary Deficit: {ai_res['primary_deficit']}

CATEGORY ACCURACY BREAKDOWN:
- Number Recognition & Subitizing: {latest['category_acc'].get('Number Recognition & Subitizing')}%
- Numerical Magnitude Comparison: {latest['category_acc'].get('Numerical Magnitude Comparison')}%
- Basic Arithmetic Operations: {latest['category_acc'].get('Basic Arithmetic Operations')}%
- Quantity Understanding & Estimation: {latest['category_acc'].get('Quantity Understanding & Estimation')}%
- Mathematical Reasoning & Sequences: {latest['category_acc'].get('Mathematical Reasoning & Sequences')}%

RECOMMENDATIONS & ACTION PLAN:
{chr(10).join(['* ' + a for a in rec_data['general']['action_plan']])}

CLINICAL DISCLAIMER:
This report is generated by an Artificial Intelligence model for preliminary educational screening.
It does NOT constitute a formal clinical diagnosis. Consult an educational psychologist for formal assessment.
============================================================
            """
            st.download_button(
                label="📥 Download Screening Report (TXT)",
                data=report_text,
                file_name=f"Dyscalculia_Screening_Report_{latest['assessment_id']}.txt",
                mime="text/plain",
                type="primary",
                use_container_width=True
            )


# -----------------------------------------------------------------------------
# PAGE 5: 🤖 AI-Based Screening Assistant & Simulator (Section 7.2.6)
# -----------------------------------------------------------------------------
elif st.session_state['nav_page'] == "🤖 AI Screening Assistant":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Based Screening Assistant & Diagnostic Simulator</h1>
        <p>Explore the Scikit-learn machine learning pipeline, test custom parameters, and simulate risk predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "🔬 Interactive What-If Simulator",
        "⚙️ AI Model Architecture & Metrics",
        "💬 Dyscalculia Screening Advisor (Q&A)"
    ])

    with tab1:
        st.markdown("### 🎛️ Clinical What-If Parameter Simulator")
        st.markdown("Adjust the student's cognitive metrics below to observe how the AI Random Forest model calculates dyscalculia risk probabilities in real time:")

        sim_c1, sim_c2 = st.columns(2)

        with sim_c1:
            st.markdown("##### 👤 Demographic & Rapid Subitizing")
            s_age = st.slider("Student Age (Years)", 6, 18, 10)
            s_sub_acc = st.slider("Subitizing Accuracy (%)", 0, 100, 85, help="Recognition of 1 to 4 dot patterns")
            s_sub_rt = st.slider("Subitizing Latency (ms)", 400, 3500, 950, help="Response speed for dot enumeration")
            s_mag_acc = st.slider("Magnitude Comparison Accuracy (%)", 0, 100, 90)
            s_mag_rt = st.slider("Magnitude Latency (ms)", 400, 3500, 1050)

        with sim_c2:
            st.markdown("##### 🧮 Calculation & Estimation Faculties")
            s_arith_acc = st.slider("Mental Arithmetic Accuracy (%)", 0, 100, 80)
            s_arith_rt = st.slider("Arithmetic Retrieval Latency (ms)", 800, 6000, 2100)
            s_est_acc = st.slider("Quantity Estimation Accuracy (%)", 0, 100, 85)
            s_reas_acc = st.slider("Reasoning & Sequences Accuracy (%)", 0, 100, 85)

        s_overall_acc = float(np.mean([s_sub_acc, s_mag_acc, s_arith_acc, s_est_acc, s_reas_acc]))
        s_overall_rt = float(np.mean([s_sub_rt, s_mag_rt, s_arith_rt]))

        if st.button("⚡ Run AI Risk Prediction Simulation", type="primary", use_container_width=True, key="sim_run_btn"):
            sim_input = {
                'age': s_age,
                'subitizing_acc': s_sub_acc,
                'subitizing_rt_ms': s_sub_rt,
                'magnitude_acc': s_mag_acc,
                'magnitude_rt_ms': s_mag_rt,
                'arithmetic_acc': s_arith_acc,
                'arithmetic_rt_ms': s_arith_rt,
                'estimation_acc': s_est_acc,
                'reasoning_acc': s_reas_acc,
                'overall_accuracy': s_overall_acc,
                'avg_response_time_ms': s_overall_rt
            }

            pred = predict_dyscalculia_risk(sim_input)

            st.markdown("---")
            p_c1, p_c2 = st.columns([1.5, 1])
            with p_c1:
                r_color = pred['risk_color']
                st.markdown(f"""
                <div class="card" style="border-top: 5px solid {r_color};">
                    <h3 style="color: {r_color}; margin-top: 0;">Simulated Outcome: {pred['risk_outcome']}</h3>
                    <p style="font-size: 1.05rem; color: #334155;">{pred['risk_description']}</p>
                    <p><b>Primary Deficit:</b> {pred['primary_deficit']}</p>
                    <p><b>Model Version:</b> {pred['model_version']}</p>
                </div>
                """, unsafe_allow_html=True)
            with p_c2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {pred['risk_color']};">{pred['risk_probability']}%</div>
                    <div class="metric-label">Predicted Dyscalculia Probability</div>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### ⚙️ Machine Learning Pipeline Architecture (Chapter 4.6 & 7.2.6)")
        model, metrics = load_screening_model()

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        with m_c1:
            st.metric("Model Accuracy", f"{metrics['accuracy']}%")
        with m_c2:
            st.metric("Precision", f"{metrics['precision']}%")
        with m_c3:
            st.metric("Recall (Sensitivity)", f"{metrics['recall']}%")
        with m_c4:
            st.metric("ROC-AUC Score", f"{metrics['roc_auc']}")

        st.markdown("<br/>", unsafe_allow_html=True)
        # Confusion Matrix and Performance Visualization
        fig_cm = plot_ai_model_metrics(metrics['confusion_matrix'], metrics)
        st.pyplot(fig_cm)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Feature Importance Analysis (Random Forest)")
        st.markdown("Relative weight assigned by the machine learning model to each cognitive psychometric feature:")

        feat_df = pd.DataFrame(
            list(metrics['feature_importances'].items()),
            columns=['Cognitive Feature', 'Relative Importance']
        ).sort_values(by='Relative Importance', ascending=False)

        st.dataframe(feat_df, use_container_width=True)

    with tab3:
        st.markdown("### 💬 Dyscalculia Screening Advisor & Knowledge Assistant")
        st.markdown("Select a common question to receive evidence-based educational insights:")

        faq = {
            "How does Dyscalculia differ from general Math Anxiety?": (
                "While dyscalculia and math anxiety frequently co-occur, they have distinct neurocognitive roots. "
                "Dyscalculia is a specific learning disorder characterized by a deficit in the brain's intraparietal sulcus "
                "(the core 'number module' responsible for subitizing and magnitude comparison). "
                "Math anxiety is an emotional reaction involving intrusive thoughts and sympathetic arousal that consumes working memory. "
                "Importantly, a dyscalculic student struggles even in zero-pressure environments when manipulating basic quantities."
            ),
            "What are the earliest signs of dyscalculia in primary school?": (
                "Key red flags in early childhood include: difficulty learning to count while pointing to objects (one-to-one correspondence); "
                "inability to instantly see that a cluster of 3 dots is 3 without counting fingers; "
                "reversing numbers (writing 14 as 41); trouble learning basic addition facts like 2+3; "
                "and severe confusion with concepts of time, money, or left-right orientation."
            ),
            "What classroom accommodations are most effective?": (
                "Evidence-based accommodations include: 1) Extended testing time (25% to 50%); "
                "2) Allowed use of four-function calculators for higher-order reasoning problems; "
                "3) Concrete manipulatives (Numicon tiles, base-10 blocks, Cuisenaire rods); "
                "4) Visual desk number strips; 5) Graph paper with large grids to assist vertical column alignment; "
                "and 6) Eliminating high-stress timed oral math drills."
            ),
            "What clinical diagnostic systems recognize Dyscalculia?": (
                "Dyscalculia is formally recognized by the World Health Organization (WHO) in ICD-11 under code "
                "**6A03.0: Developmental Learning Disorder with Impairment in Mathematics**, and by the American "
                "Psychiatric Association (APA) in DSM-5 as **Specific Learning Disorder (with impairment in mathematics)**."
            )
        }

        selected_q = st.selectbox("Select a question to consult:", list(faq.keys()))
        if selected_q:
            st.markdown(f"""
            <div class="card" style="border-left: 5px solid #10B981; margin-top: 15px;">
                <h4 style="color: #065F46; margin-top: 0;">💡 Educational Insight:</h4>
                <p style="font-size: 0.98rem; color: #1E293B; line-height: 1.6;">{faq[selected_q]}</p>
            </div>
            """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 6: 📈 User Assessment History (Section 7.2.7)
# -----------------------------------------------------------------------------
elif st.session_state['nav_page'] == "📈 User Assessment History":
    user = st.session_state['logged_in_user']
    user_id = user['user_id'] if user else 1

    st.markdown("""
    <div class="main-header">
        <h1>📈 User Profile & Assessment History</h1>
        <p>Comprehensive SQLite records tracking all previous screening sessions, accuracy scores, and risk classifications.</p>
    </div>
    """, unsafe_allow_html=True)

    # User Profile Details
    if user:
        p_c1, p_c2, p_c3, p_c4 = st.columns(4)
        with p_c1:
            st.markdown(f"**Name:** {user['name']}")
        with p_c2:
            st.markdown(f"**Email:** {user['email']}")
        with p_c3:
            st.markdown(f"**Role:** {user['role']}")
        with p_c4:
            st.markdown(f"**Age:** {user.get('age', 12)} years")

    st.markdown("---")

    # Fetch History from SQLite
    history = get_user_assessments(user_id)

    if not history:
        st.info("No past assessment records found for this user account. Take an assessment to record your initial results.")
        if st.button("Start Screening Assessment Now"):
            st.session_state['current_q_index'] = 0
            navigate_to("🧠 Screening Assessment")
    else:
        st.markdown(f"### 📋 Stored Assessments ({len(history)} sessions found in SQLite database)")

        # Display Summary Table
        table_records = []
        for a in history:
            table_records.append({
                'ID': f"#{a['assessment_id']}",
                'Date': a['assessment_date'],
                'Score': f"{int(a['total_score'])} / {int(a['max_score'])}",
                'Accuracy': f"{a['percentage']}%",
                'Total Time': f"{a['total_time_seconds']}s",
                'AI Screening Risk': a['risk_level'],
                'Primary Deficit': a.get('primary_deficit', 'None')
            })
        st.dataframe(pd.DataFrame(table_records), use_container_width=True)

        if len(history) >= 2:
            st.markdown("### 📊 Performance Progression Trend (Chapter 4.7)")
            fig_hist = plot_historical_progress(history)
            st.pyplot(fig_hist)

        # Drill Down Inspector
        st.markdown("### 🔍 Inspect Past Assessment Details")
        selected_ass_id = st.selectbox(
            "Choose an assessment to review:",
            [a['assessment_id'] for a in history],
            format_func=lambda x: f"Assessment #{x} ({next((a['assessment_date'] for a in history if a['assessment_id'] == x), '')})"
        )

        if selected_ass_id:
            details = get_assessment_details(selected_ass_id)
            if details:
                st.markdown(f"""
                <div class="card">
                    <h4 style="color: #1E3A8A; margin-top: 0;">Assessment #{details['assessment_id']} Overview</h4>
                    <p><b>Date:</b> {details['assessment_date']} | <b>Score:</b> {details['total_score']} / {details['max_score']} ({details['percentage']}%)</p>
                    <p><b>Risk Outcome:</b> <span style="font-weight: 700;">{details['risk_outcome']}</span> | <b>Dyscalculia Probability:</b> {details['risk_probability']}%</p>
                    <p><b>Identified Deficit:</b> {details['primary_deficit']}</p>
                    <p><b>Guidance:</b> {details['recommendations']}</p>
                </div>
                """, unsafe_allow_html=True)

                if details.get('responses'):
                    with st.expander("View Question Responses for this Session"):
                        r_df = pd.DataFrame(details['responses'])
                        r_df['Status'] = r_df['is_correct'].apply(lambda x: "✅ Correct" if x == 1 else "❌ Incorrect")
                        st.dataframe(r_df[['category', 'submitted_answer', 'correct_answer', 'Status', 'response_time_ms']], use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 7: 🎮 Math Practice Arena
# -----------------------------------------------------------------------------
elif st.session_state['nav_page'] == "🎮 Math Practice Arena":
    st.markdown("""
    <div class="main-header">
        <h1>🎮 Interactive Math Practice Arena</h1>
        <p>Targeted multi-sensory activities to strengthen rapid subitizing, mental number lines, and arithmetic fluency.</p>
    </div>
    """, unsafe_allow_html=True)

    game_tab1, game_tab2, game_tab3 = st.tabs([
        "⚡ Rapid Dot Subitizing Flash",
        "📏 Mental Number Line Placement",
        "🧩 Arithmetic Fact Fluency"
    ])

    with game_tab1:
        st.markdown("### ⚡ Rapid Dot Subitizing")
        st.markdown("Look at the dots and enter the count as quickly as possible without finger-counting:")

        if 'subit_target' not in st.session_state:
            st.session_state['subit_target'] = np.random.randint(2, 7)

        sub_dots = generate_dot_svg(st.session_state['subit_target'], color="#2563EB", size=180)
        st.markdown(sub_dots, unsafe_allow_html=True)

        user_sub_ans = st.number_input("How many dots did you see?", min_value=1, max_value=12, step=1, key="subit_input")
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            if st.button("Check Subitizing Answer", key="subit_btn", type="primary", use_container_width=True):
                if user_sub_ans == st.session_state['subit_target']:
                    st.success(f"🌟 Correct! There are exactly {st.session_state['subit_target']} dots.")
                else:
                    st.error(f"Try again! There were {st.session_state['subit_target']} dots.")
        with c_sub2:
            if st.button("Next Random Dot Card 🎲", key="subit_next_btn", use_container_width=True):
                st.session_state['subit_target'] = np.random.randint(2, 7)
                st.rerun()

    with game_tab2:
        st.markdown("### 📏 Mental Number Line Slider")
        st.markdown("Estimate where the target number belongs on a 0 to 100 number line:")

        if 'nl_target' not in st.session_state:
            st.session_state['nl_target'] = np.random.randint(15, 85)

        st.markdown(f"<h3 style='text-align: center; color: #2563EB;'>Target Number: {st.session_state['nl_target']}</h3>", unsafe_allow_html=True)
        user_guess = st.slider("Position the slider on the number line (0 to 100):", 0, 100, 50, key="nl_slider")

        c_nl1, c_nl2 = st.columns(2)
        with c_nl1:
            if st.button("Submit Placement", key="nl_btn", type="primary", use_container_width=True):
                diff = abs(user_guess - st.session_state['nl_target'])
                if diff <= 5:
                    st.success(f"🎯 Outstanding estimation! You guessed {user_guess}, target was {st.session_state['nl_target']} (Difference: {diff})")
                elif diff <= 12:
                    st.info(f"👍 Good estimate! You guessed {user_guess}, target was {st.session_state['nl_target']} (Difference: {diff})")
                else:
                    st.warning(f"Off by {diff}. The target {st.session_state['nl_target']} was placed at {user_guess}.")
        with c_nl2:
            if st.button("New Target Number 🎯", key="nl_next_btn", use_container_width=True):
                st.session_state['nl_target'] = np.random.randint(15, 85)
                st.rerun()

    with game_tab3:
        st.markdown("### 🧩 Arithmetic Fact Fluency")
        st.markdown("Bridge-to-10 practice challenge:")

        if 'fact_a' not in st.session_state:
            st.session_state['fact_a'] = np.random.randint(6, 10)
            st.session_state['fact_b'] = np.random.randint(5, 9)

        fa = st.session_state['fact_a']
        fb = st.session_state['fact_b']
        expected_sum = fa + fb

        st.markdown(f"""
        <div class="equation-card">
            {fa} + {fb} = ?
        </div>
        """, unsafe_allow_html=True)

        user_fact = st.number_input("Enter the sum:", min_value=1, max_value=30, step=1, key="fact_input")
        c_fact1, c_fact2 = st.columns(2)
        with c_fact1:
            if st.button("Check Fact", key="fact_btn", type="primary", use_container_width=True):
                if user_fact == expected_sum:
                    st.success(f"🎉 Correct! {fa} + {fb} = {expected_sum}")
                else:
                    st.error(f"Not quite. {fa} + {fb} = {expected_sum} (Hint: {fa} + {10 - fa} = 10, then add {fb - (10 - fa)})")
        with c_fact2:
            if st.button("Next Problem ➕", key="fact_next_btn", use_container_width=True):
                st.session_state['fact_a'] = np.random.randint(6, 10)
                st.session_state['fact_b'] = np.random.randint(5, 9)
                st.rerun()


# -----------------------------------------------------------------------------
# PAGE 8: 🔐 Account & Login (Section 7.2.1)
# -----------------------------------------------------------------------------
elif st.session_state['nav_page'] == "🔐 Account & Login":
    st.markdown("""
    <div class="main-header">
        <h1>🔐 User Authentication & Account Management</h1>
        <p>Register or log in to securely store your assessment history in SQLite (Chapter 4.8 & 7.2.1).</p>
    </div>
    """, unsafe_allow_html=True)

    auth_tab1, auth_tab2 = st.tabs(["🔑 Sign In", "📝 Create New Account"])

    with auth_tab1:
        st.markdown("### Quick Profile Switch (One-Click Testing)")
        q_sw1, q_sw2, q_sw3 = st.columns(3)
        with q_sw1:
            if st.button("👤 Switch to Rohan (Student)", use_container_width=True):
                user = authenticate_user("rohan@example.com", "rohan123")
                if user:
                    st.session_state['logged_in_user'] = user
                    st.success(f"Switched to {user['name']}!")
                    navigate_to("📊 Dashboard")
        with q_sw2:
            if st.button("👨‍🏫 Switch to Prof. Javed (Educator)", use_container_width=True):
                user = authenticate_user("javed.pathan@rizvi.edu", "faculty123")
                if user:
                    st.session_state['logged_in_user'] = user
                    st.success(f"Switched to {user['name']}!")
                    navigate_to("📊 Dashboard")
        with q_sw3:
            if st.button("🎓 Switch to Aadarsh Dubey (Author)", use_container_width=True):
                user = authenticate_user("aadarsh@rizvi.edu", "student123")
                if user:
                    st.session_state['logged_in_user'] = user
                    st.success(f"Switched to {user['name']}!")
                    navigate_to("📊 Dashboard")

        st.markdown("---")
        st.markdown("### Manual Sign In")
        login_email = st.text_input("Email Address", value="rohan@example.com", key="login_email_manual")
        login_pwd = st.text_input("Password", type="password", value="rohan123", key="login_pwd_manual")

        if st.button("Sign In", type="primary", key="signin_submit_btn"):
            user = authenticate_user(login_email, login_pwd)
            if user:
                st.session_state['logged_in_user'] = user
                st.success(f"Welcome back, {user['name']}! Logged in as {user['role']}.")
                navigate_to("📊 Dashboard")
            else:
                st.error("Invalid email or password. Please verify your credentials.")

    with auth_tab2:
        st.markdown("### Register New Student or Educator Account")
        r_name = st.text_input("Full Name", placeholder="e.g. Rahul Sharma", key="reg_name")
        r_email = st.text_input("Email Address", placeholder="e.g. rahul@example.com", key="reg_email")
        r_pwd = st.text_input("Choose Password", type="password", key="reg_pwd")
        r_role = st.selectbox("Role", ["Student", "Parent", "Educator", "Clinician"], key="reg_role")
        r_age = st.number_input("Age (Years)", min_value=5, max_value=99, value=11, key="reg_age")

        if st.button("Create Account", type="primary", key="reg_submit_btn"):
            if not r_name or not r_email or not r_pwd:
                st.error("Please fill in all required fields.")
            else:
                success, msg = register_user(r_name, r_email, r_pwd, r_role, r_age)
                if success:
                    st.success(msg)
                    new_user = authenticate_user(r_email, r_pwd)
                    if new_user:
                        st.session_state['logged_in_user'] = new_user
                        navigate_to("📊 Dashboard")
                else:
                    st.error(msg)
