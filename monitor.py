import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
from typing import Dict, Any
import json


class OptimizationMonitor:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url

        # Initialize session state
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()

    def get_optimization_status(self) -> Dict[str, Any]:
        """Fetch current optimization status from the server"""
        try:
            response = requests.get(f"{self.api_base_url}/status")
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse server response: {e}")
                    return {}
            else:
                st.error(f"Server returned status code: {response.status_code}")
                return {}
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to server: {e}")
            return {}

    def update_dashboard(self):
        """Update all dashboard components"""
        # Fetch current status
        status = self.get_optimization_status()

        if not status:
            st.warning("Unable to fetch optimization status")
            return

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Workers", status.get('active_workers', 0))
        with col2:
            st.metric("Total Evaluations", status.get('total_evaluations', 0))
        with col3:
            best_result = status.get('best_result', {})
            if isinstance(best_result, dict):
                objectives = best_result.get('objectives', {})
                if isinstance(objectives, dict):
                    best_value = objectives.get('objective1')
                    if best_value is not None:
                        st.metric("Best Objective Value", f"{best_value:.4f}")

        # Update history if we have new data
        best_result = status.get('best_result')
        if best_result and (not st.session_state.history or
                          best_result != st.session_state.history[-1]):
            st.session_state.history.append(best_result)

        # Plot optimization progress
        if st.session_state.history:
            try:
                data = []
                for i, result in enumerate(st.session_state.history, 1):
                    if isinstance(result, dict):
                        objectives = result.get('objectives', {})
                        if isinstance(objectives, dict):
                            value = objectives.get('objective1')
                            if value is not None:
                                data.append({
                                    'trial': i,
                                    'objective_value': value
                                })

                if data:
                    df = pd.DataFrame(data)
                    fig = px.line(df, x='trial', y='objective_value',
                                title='Optimization Progress',
                                labels={'trial': 'Trial Number',
                                       'objective_value': 'Objective Value'})
                    st.plotly_chart(fig)

                    # Display history table
                    st.subheader("Optimization History")
                    st.dataframe(df)
                else:
                    st.info("No optimization data available yet")
            except Exception as e:
                st.error(f"Error creating visualization: {e}")

    def run(self):
        st.title("Hyperparameter Optimization Monitor")

        # Sidebar controls
        st.sidebar.title("Controls")
        update_interval = st.sidebar.slider("Update Interval (seconds)", 1, 10, 2)

        if st.sidebar.button("Clear History"):
            st.session_state.history = []

        # Main content area
        placeholder = st.empty()

        # Auto-refresh using st.empty() and a placeholder
        while True:
            with placeholder.container():
                self.update_dashboard()
            time.sleep(update_interval)
            st.rerun()  # Using st.rerun() instead of experimental_rerun()

def main():
    st.set_page_config(
        page_title="Optimization Monitor",
        page_icon="📈",
        layout="wide",
    )

    monitor = OptimizationMonitor()
    monitor.run()

if __name__ == "__main__":
    main()