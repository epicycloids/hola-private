"""
Streamlit dashboard for monitoring HOLA optimization.

This dashboard connects to a running optimization scheduler and displays
real-time information about the optimization process.
"""

import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from hola.core.monitor import OptimizationMonitor

# Initialize session state for the dashboard
if 'monitor' not in st.session_state:
    st.session_state.monitor = None

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = 0

if 'status' not in st.session_state:
    st.session_state.status = {}

def initialize_monitor(scheduler_address: str = "tcp://localhost:5555"):
    """Initialize the optimization monitor."""
    try:
        st.session_state.monitor = OptimizationMonitor(scheduler_address=scheduler_address)
        st.session_state.status = st.session_state.monitor.get_status()
        st.session_state.last_update_time = time.time()
        return True
    except Exception as e:
        st.error(f"Failed to connect to scheduler: {e}")
        return False

def update_status():
    """Update the optimization status from the monitor."""
    if st.session_state.monitor:
        try:
            st.session_state.status = st.session_state.monitor.get_status()
            st.session_state.last_update_time = time.time()
            return True
        except Exception as e:
            st.error(f"Error updating status: {e}")
    return False

def show_connection_settings():
    """Display connection settings UI."""
    st.sidebar.header("Connection Settings")

    with st.sidebar.form("connection_settings"):
        scheduler_address = st.text_input("Scheduler Address", value="tcp://localhost:5555")
        submit = st.form_submit_button("Connect")

        if submit:
            if initialize_monitor(scheduler_address):
                st.success("Connected to scheduler")
                st.rerun()

def show_statistics():
    """Display optimization statistics."""
    if 'monitor' not in st.session_state or not st.session_state.monitor:
        st.info("Not connected to any optimization scheduler.")
        return

    status = st.session_state.status

    # Create metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Trials", status.get("total_evaluations", 0))

    with col2:
        st.metric("Active Workers", status.get("active_workers", 0))

    with col3:
        running = status.get("is_running", False)
        status_text = "Running" if running else "Stopped"
        st.metric("Status", status_text)

    # Show best trial
    best_objectives = status.get("best_objectives")
    if best_objectives:
        st.subheader("Best Objectives")
        st.json(best_objectives)

    # Get trial data for additional statistics
    try:
        df = st.session_state.monitor.get_all_trials_dataframe()
        if not df.empty:
            st.subheader("Trial Statistics")

            feasible_trials = len(df[df.get('Is Feasible', False)])
            ranked_trials = len(df[df.get('Is Ranked', False)])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Feasible Trials", feasible_trials)

            with col2:
                st.metric("Ranked Trials", ranked_trials)
    except Exception as e:
        st.warning(f"Could not fetch trial statistics: {e}")

def plot_pareto_fronts():
    """Plot Pareto fronts of the optimization."""
    if 'monitor' not in st.session_state or not st.session_state.monitor:
        return

    try:
        # Get data
        df = st.session_state.monitor.get_all_trials_dataframe()

        if df.empty:
            st.info("No trial data available yet.")
            return

        # Plot objectives
        st.subheader("Objective Values")

        objective_cols = [col for col in df.columns if col.startswith('f') and col not in
                         ['Front', 'Is Feasible', 'Is Ranked']]

        if len(objective_cols) >= 2:
            tab1, tab2, tab3 = st.tabs(["2D Plot", "3D Plot", "Parallel Coordinates"])

            with tab1:
                # 2D scatter plot of first two objectives
                fig = px.scatter(
                    df,
                    x=objective_cols[0],
                    y=objective_cols[1],
                    color='Is Ranked',
                    hover_data=['Trial'] + objective_cols,
                    title=f"{objective_cols[0]} vs {objective_cols[1]}"
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # 3D scatter plot if we have at least 3 objectives
                if len(objective_cols) >= 3:
                    fig = px.scatter_3d(
                        df,
                        x=objective_cols[0],
                        y=objective_cols[1],
                        z=objective_cols[2],
                        color='Is Ranked',
                        hover_data=['Trial'] + objective_cols,
                        title=f"{objective_cols[0]} vs {objective_cols[1]} vs {objective_cols[2]}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 3 objectives for 3D plot")

            with tab3:
                # Parallel coordinates plot
                fig = px.parallel_coordinates(
                    df,
                    dimensions=objective_cols,
                    color='Is Ranked',
                    title="Parallel Coordinates Plot of Objectives"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Plot parameter space
        st.subheader("Parameter Space")

        # Find parameter columns (those not related to objectives or metadata)
        param_cols = [col for col in df.columns if col not in objective_cols and col not in
                     ['Trial', 'Is Ranked', 'Is Feasible', 'Front', 'Crowding Distance', 'Group 0 Score', 'Group 1 Score']]

        if len(param_cols) >= 2:
            fig = px.scatter(
                df,
                x=param_cols[0],
                y=param_cols[1],
                color='Is Ranked',
                hover_data=['Trial'] + objective_cols,
                title=f"Parameter Space: {param_cols[0]} vs {param_cols[1]}"
            )
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error plotting data: {e}")

def show_data_tables():
    """Display data tables with trial information."""
    if 'monitor' not in st.session_state or not st.session_state.monitor:
        return

    try:
        # Get data
        df_ranked = st.session_state.monitor.get_trials_dataframe(ranked_only=True)
        df_all = st.session_state.monitor.get_all_trials_dataframe()
        metadata_df = st.session_state.monitor.get_trials_metadata()

        tab1, tab2, tab3 = st.tabs(["Ranked Trials", "All Trials", "Metadata"])

        with tab1:
            st.write(f"Ranked Trials: {len(df_ranked)} rows")
            st.dataframe(df_ranked, use_container_width=True)

        with tab2:
            st.write(f"All Trials: {len(df_all)} rows")
            st.dataframe(df_all, use_container_width=True)

        with tab3:
            st.write(f"Metadata: {len(metadata_df)} rows")
            st.dataframe(metadata_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying data tables: {e}")

def main():
    """Main function for the Streamlit dashboard."""
    st.set_page_config(
        page_title="HOLA Optimization Monitor",
        page_icon="📊",
        layout="wide",
    )

    st.title("HOLA Optimization Monitor")

    # Show connection settings in sidebar
    show_connection_settings()

    # Update status if connected
    if st.session_state.monitor:
        update_status()

        # Show timestamp of last update
        st.sidebar.info(f"Last updated: {time.strftime('%H:%M:%S', time.localtime(st.session_state.last_update_time))}")

        # Display tabs for different views
        tab1, tab2, tab3 = st.tabs(["Statistics", "Visualizations", "Data Tables"])

        with tab1:
            show_statistics()

        with tab2:
            plot_pareto_fronts()

        with tab3:
            show_data_tables()
    else:
        st.info("Please connect to a scheduler using the sidebar.")

    # Auto-refresh while optimization is running
    if st.session_state.monitor and st.session_state.status.get("is_running", False):
        time.sleep(1)  # Small delay to prevent too rapid refreshes
        st.rerun()

if __name__ == "__main__":
    main()