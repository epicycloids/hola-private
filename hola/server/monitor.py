"""Monitoring dashboard for distributed optimization."""

import time
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Configure the page
st.set_page_config(
    page_title="HOLA Optimization Monitor",
    page_icon="📊",
    layout="wide",
)

# Initialize session state
if "start_time" not in st.session_state:
    st.session_state.start_time = datetime.now()
if "data" not in st.session_state:
    st.session_state.data = {
        "timestamp": [],
        "active_workers": [],
        "total_evaluations": [],
        "best_objectives": [],
    }

# Title and description
st.title("HOLA Optimization Monitor")
st.markdown("""
This dashboard provides real-time monitoring of the distributed optimization process.
""")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="monitor_refresh")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", "http://localhost:8000")
    refresh_interval = st.slider("Refresh Interval (seconds)", 1, 30, 5)

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Active Workers",
        st.session_state.data["active_workers"][-1] if st.session_state.data["active_workers"] else 0,
    )

with col2:
    st.metric(
        "Total Evaluations",
        st.session_state.data["total_evaluations"][-1] if st.session_state.data["total_evaluations"] else 0,
    )

with col3:
    elapsed = datetime.now() - st.session_state.start_time
    st.metric("Elapsed Time", f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s")

# Fetch current status
try:
    response = requests.get(f"{api_url}/status")
    if response.status_code == 200:
        status = response.json()
        current_time = datetime.now()

        # Update data
        st.session_state.data["timestamp"].append(current_time)
        st.session_state.data["active_workers"].append(status["active_workers"])
        st.session_state.data["total_evaluations"].append(status["total_evaluations"])
        st.session_state.data["best_objectives"].append(status["best_result"])

        # Keep only last 100 data points
        max_points = 100
        for key in st.session_state.data:
            st.session_state.data[key] = st.session_state.data[key][-max_points:]

        # Create DataFrame for plotting
        df = pd.DataFrame(st.session_state.data)

        # Plot active workers
        workers_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="timestamp:T",
                y="active_workers:Q",
                tooltip=["timestamp:T", "active_workers:Q"],
            )
            .properties(title="Active Workers Over Time")
        )
        st.altair_chart(workers_chart, use_container_width=True)

        # Plot total evaluations
        evals_chart = (
            alt.Chart(df)
            .mark_line()
            .encode(
                x="timestamp:T",
                y="total_evaluations:Q",
                tooltip=["timestamp:T", "total_evaluations:Q"],
            )
            .properties(title="Total Evaluations Over Time")
        )
        st.altair_chart(evals_chart, use_container_width=True)

        # Display best objectives
        if status["best_result"]:
            st.subheader("Best Objectives")
            for obj_name, value in status["best_result"]["objectives"].items():
                st.metric(obj_name, f"{value:.4f}")

except Exception as e:
    st.error(f"Error fetching status: {str(e)}")

# Footer
st.markdown("---")
st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")