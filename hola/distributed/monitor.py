"""
Streamlit-based monitoring dashboard for HOLA distributed optimization.

This module provides a real-time web dashboard to monitor the progress of
distributed optimization jobs, including active workers, current evaluations,
and optimization metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import threading
import json
import zmq
import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from hola.core.coordinator import OptimizationCoordinator
from hola.distributed.server import OptimizationServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hola.distributed.monitor")


class OptimizationMonitor:
    """
    A monitoring dashboard for the HOLA distributed optimization system.

    This class connects to a running OptimizationServer via ZMQ and displays
    real-time information about the optimization progress, including:
    - Overall statistics
    - Active workers and their current jobs
    - Visualization of objective values and parameters
    - Pareto fronts
    """

    def __init__(
        self,
        server_zmq_endpoint: str,
        update_interval: float = 2.0,
        display_limit: int = 100
    ):
        """
        Initialize the optimization monitor.

        :param server_zmq_endpoint: ZMQ endpoint of the optimization server
        :param update_interval: How often to update the dashboard (seconds)
        :param display_limit: Maximum number of trials to display in tables
        """
        self.server_zmq_endpoint = server_zmq_endpoint
        self.update_interval = update_interval
        self.display_limit = display_limit

        # Initialize ZMQ context and socket
        self.zmq_context = zmq.Context()
        self.zmq_socket = None

        # Dashboard state
        self.server_info = {}
        self.workers_info = {}
        self.jobs_info = {}
        self.coordinator_stats = {}
        self.trials_data = pd.DataFrame()
        self.last_update_time = 0
        self.connected = False
        self.connection_error = None

        # Store the last successful data for when disconnected
        self.last_successful_server_info = {}
        self.last_successful_coordinator_stats = {}
        self.last_successful_trials_data = pd.DataFrame()
        self.was_previously_connected = False

    def connect(self) -> bool:
        """
        Connect to the optimization server.

        :return: True if connection successful, False otherwise
        """
        try:
            # Create REQ socket for communication with server
            if self.zmq_socket:
                self.zmq_socket.close()

            self.zmq_socket = self.zmq_context.socket(zmq.REQ)
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout (increased from 3s)

            # Log connection attempt
            logger.info(f"Attempting to connect to server at {self.server_zmq_endpoint}")
            self.zmq_socket.connect(self.server_zmq_endpoint)

            # Ping server to check connection
            logger.info("Sending ping to server")
            self.zmq_socket.send_json({"action": "ping"})

            try:
                response = self.zmq_socket.recv_json()

                if response.get("status") == "ok":
                    self.connected = True
                    self.connection_error = None
                    logger.info(f"Successfully connected to server at {self.server_zmq_endpoint}")
                    return True

                self.connection_error = f"Server returned an error status: {response}"
                logger.warning(f"Server returned error: {response}")
                return False

            except zmq.error.Again:
                # Timeout waiting for response
                self.connection_error = "Timeout waiting for server response. Server may not be running."
                logger.warning("Timeout waiting for server response")
                return False

        except Exception as e:
            self.connected = False
            self.connection_error = str(e)
            logger.warning(f"Failed to connect to server: {str(e)}")
            return False

    def disconnect(self):
        """Close ZMQ connection to the server."""
        if self.zmq_socket:
            self.zmq_socket.close()
        self.connected = False

    def request_data(self, action: str, **kwargs) -> Dict:
        """
        Send a request to the server and return the response.

        :param action: Action to request from the server
        :param kwargs: Additional parameters for the request
        :return: Server response as a dictionary
        """
        if not self.connected or not self.zmq_socket:
            return {"status": "error", "message": "Not connected to server"}

        try:
            # Prepare request
            request = {"action": action, **kwargs}

            # Send request
            self.zmq_socket.send_json(request)

            # Receive response
            response = self.zmq_socket.recv_json()
            return response

        except Exception as e:
            logger.warning(f"Error in request: {str(e)}")
            self.connected = False
            self.connection_error = str(e)
            return {"status": "error", "message": str(e)}

    def update_data(self):
        """Fetch updated data from the server."""
        # If not connected, try to reconnect first
        if not self.connected:
            if not self.connect():
                return False

        try:
            # Update server info
            server_info = self.request_data("get_server_info")
            if server_info.get("status") == "ok":
                self.server_info = server_info.get("data", {})

            # Update workers info
            workers_info = self.request_data("get_workers_info")
            if workers_info.get("status") == "ok":
                self.workers_info = workers_info.get("data", {})

            # Update active jobs
            jobs_info = self.request_data("get_jobs_info")
            if jobs_info.get("status") == "ok":
                self.jobs_info = jobs_info.get("data", {})

            # Update optimization stats
            coordinator_stats = self.request_data("get_coordinator_stats")
            if coordinator_stats.get("status") == "ok":
                self.coordinator_stats = coordinator_stats.get("data", {})

            # Get trials data
            trials_data = self.request_data("get_trials_data", limit=self.display_limit)
            if trials_data.get("status") == "ok" and "data" in trials_data:
                df = pd.DataFrame(trials_data.get("data", []))
                if not df.empty:
                    self.trials_data = df

            self.last_update_time = time.time()
            return True

        except Exception as e:
            logger.exception(f"Error updating data: {str(e)}")
            self.connected = False
            self.connection_error = str(e)
            return False

    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="HOLA Distributed Optimization Monitor",
            page_icon="📊",
            layout="wide",
        )

        st.title("HOLA Distributed Optimization Monitor")

        # Connection settings
        with st.sidebar:
            st.subheader("Connection Settings")

            # Display connection status prominently
            if self.connected:
                st.success("✅ Connected to server")
            elif self.connection_error:
                st.error(f"⚠️ Connection error: {self.connection_error}")
            else:
                st.warning("⚠️ Not connected to server")

            # Server endpoint input with better default display
            server_endpoint = st.text_input(
                "Server ZMQ Endpoint",
                value=self.server_zmq_endpoint,
                help="The ZMQ endpoint of the optimization server (e.g., tcp://127.0.0.1:5555)"
            )

            # Check if endpoint changed
            endpoint_changed = server_endpoint != self.server_zmq_endpoint

            # Connect button with better visual feedback
            if st.button("Connect" if not self.connected else "Reconnect",
                         type="primary" if not self.connected else "secondary"):
                if endpoint_changed:
                    self.server_zmq_endpoint = server_endpoint
                    self.disconnect()  # Make sure we disconnect first

                with st.spinner("Connecting to server..."):
                    # Try multiple connection attempts
                    for attempt in range(3):
                        if self.connect():
                            st.success(f"Connected to server at {self.server_zmq_endpoint}")
                            # Store the successful endpoint for future default
                            break
                        time.sleep(0.5)

                    if not self.connected:
                        st.error(f"Could not connect after 3 attempts: {self.connection_error}")
                        st.info("Make sure the server is running and the endpoint is correct. Try running:\n\n```python -m hola.distributed.zmq_monitor_example --evals 50```")

            st.divider()

            # Auto-refresh settings
            st.subheader("Auto-Refresh Settings")
            auto_refresh = st.checkbox("Auto-refresh", value=True)
            if auto_refresh:
                refresh_interval = st.slider("Refresh interval (seconds)",
                                             min_value=1, max_value=30,
                                             value=int(self.update_interval))
                self.update_interval = refresh_interval

            st.divider()

            # Display options
            st.subheader("Display Options")
            self.display_limit = st.slider("Display limit",
                                          min_value=10, max_value=1000,
                                          value=self.display_limit)

        # Main content
        if not self.connected:
            # If we were previously connected but now disconnected,
            # likely the optimization has completed
            if self.was_previously_connected and self.last_successful_coordinator_stats:
                # Display a completion message
                st.success("✅ Optimization completed! The server has been stopped.")
                st.info(f"Showing last known data from completed optimization run ({self.last_successful_coordinator_stats.get('total_trials', 0)} evaluations)")

                # We'll show the last data we had and display a completion message
                self.server_info = self.last_successful_server_info
                self.coordinator_stats = self.last_successful_coordinator_stats
                self.trials_data = self.last_successful_trials_data

                # Display content with our cached data
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Overview",
                    "Workers & Jobs",
                    "Optimization Results",
                    "Visualizations"
                ])

                with tab1:
                    self._display_overview()

                with tab3:
                    self._display_optimization_results()

                with tab4:
                    self._display_visualizations()
            else:
                # No previous connection or data - provide helpful guidance
                st.warning("Not connected to any optimization server")

                # Provide helpful instructions
                st.markdown("""
                ### Connection Instructions

                To connect to an optimization server:

                1. Make sure an optimization server is running with a ZMQ endpoint
                2. Enter the server's ZMQ endpoint in the sidebar (e.g., `tcp://127.0.0.1:5555`)
                3. Click the "Connect" button

                You can start a server with the example script:
                ```bash
                python -m hola.distributed.zmq_monitor_example --workers 4 --evals 50
                ```

                Or connect to an existing server by specifying its ZMQ endpoint.
                """)

                # Try to connect automatically if endpoint is set and we haven't tried yet
                if not self.connection_error:
                    with st.spinner("Attempting initial connection..."):
                        self.connect()
                        if self.connected:
                            st.success("Connected successfully!")
                            st.rerun()  # Refresh to show connected state
        else:
            # Update data from server
            update_success = self.update_data()
            if update_success:
                # Store the successful data for when we disconnect
                self.was_previously_connected = True
                self.last_successful_server_info = self.server_info.copy()
                self.last_successful_coordinator_stats = self.coordinator_stats.copy()
                if not self.trials_data.empty:
                    self.last_successful_trials_data = self.trials_data.copy()
            else:
                st.error("Failed to update data from server")
                if self.connection_error:
                    st.error(f"Error: {self.connection_error}")

                    # Check for server completion
                    if "Resource temporarily unavailable" in self.connection_error and self.was_previously_connected:
                        st.warning("Server appears to have stopped. Showing last known data.")

                        # Use last known data
                        self.server_info = self.last_successful_server_info
                        self.coordinator_stats = self.last_successful_coordinator_stats
                        self.trials_data = self.last_successful_trials_data
                    else:
                        return

            # Display content
            tab1, tab2, tab3, tab4 = st.tabs([
                "Overview",
                "Workers & Jobs",
                "Optimization Results",
                "Visualizations"
            ])

            with tab1:
                self._display_overview()

            with tab2:
                self._display_workers_and_jobs()

            with tab3:
                self._display_optimization_results()

            with tab4:
                self._display_visualizations()

            # Auto-refresh
            if auto_refresh and self.connected:
                time.sleep(0.1)  # Small delay to prevent UI freezing
                st.rerun()

    def _display_overview(self):
        """Display optimization overview."""
        st.header("Optimization Overview")

        # Check if server has been stopped (optimization completed)
        if self.connection_error and "Resource temporarily unavailable" in self.connection_error:
            st.success("✅ Optimization completed! The server has been stopped.")

            if self.coordinator_stats:
                # Display completion info
                total_trials = self.coordinator_stats.get("total_trials", 0)
                st.info(f"Total evaluations completed: {total_trials}")

                # Display best trial if available
                if "best_trial" in self.coordinator_stats:
                    best_trial = self.coordinator_stats["best_trial"]
                    st.subheader("Best Trial Found")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Parameters:")
                        st.json(best_trial.get("parameters", {}))

                    with col2:
                        st.write("Objectives:")
                        st.json(best_trial.get("objectives", {}))

            return

        # Server info
        st.subheader("Server Information")
        if self.server_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Server Status", self.server_info.get("status", "Unknown"))
            with col2:
                st.metric("Uptime (s)", self.server_info.get("uptime", 0))
            with col3:
                st.metric("Workers Connected", len(self.workers_info))

            # Progress indicators
            if "total_trials" in self.coordinator_stats:
                total_trials = self.coordinator_stats.get("total_trials", 0)
                max_trials = self.server_info.get("max_evaluations", 0)

                if max_trials > 0:
                    progress = min(total_trials / max_trials, 1.0)
                    st.progress(progress, text=f"Progress: {total_trials}/{max_trials} evaluations")
                else:
                    st.info(f"Total evaluations: {total_trials}")
        else:
            st.warning("No server information available")

    def _display_workers_and_jobs(self):
        """Display workers and their jobs."""
        st.header("Workers and Jobs")

        # Workers table
        st.subheader("Active Workers")
        if self.workers_info:
            workers_data = []
            for worker_id, worker_info in self.workers_info.items():
                workers_data.append({
                    "Worker ID": worker_id,
                    "Status": worker_info.get("status", "Unknown"),
                    "Last Heartbeat": worker_info.get("last_heartbeat", "Unknown"),
                    "Total Jobs": worker_info.get("total_jobs", 0),
                    "Successful Jobs": worker_info.get("successful_jobs", 0),
                    "Failed Jobs": worker_info.get("failed_jobs", 0)
                })

            workers_df = pd.DataFrame(workers_data)
            st.dataframe(workers_df, use_container_width=True)
        else:
            st.warning("No active workers")

        # Active jobs
        st.subheader("Active Jobs")
        if self.jobs_info:
            jobs_data = []
            for job_id, job_info in self.jobs_info.items():
                jobs_data.append({
                    "Job ID": job_id,
                    "Worker ID": job_info.get("worker_id", "Unknown"),
                    "Status": job_info.get("status", "Unknown"),
                    "Parameters": json.dumps(job_info.get("parameters", {})),
                    "Started": job_info.get("start_time", "Unknown"),
                    "Attempts": job_info.get("attempts", 0)
                })

            jobs_df = pd.DataFrame(jobs_data)
            st.dataframe(jobs_df, use_container_width=True)
        else:
            st.info("No active jobs")

    def _display_optimization_results(self):
        """Display optimization results and statistics."""
        st.header("Optimization Results")

        if self.coordinator_stats:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Trials", self.coordinator_stats.get("total_trials", 0))

            with col2:
                st.metric("Feasible Trials", self.coordinator_stats.get("feasible_trials", 0))

            with col3:
                st.metric("Ranked Trials", self.coordinator_stats.get("ranked_trials", 0))

            with col4:
                st.metric("Infeasible Trials", self.coordinator_stats.get("infeasible_trials", 0))

            # Display best trial if available
            if "best_trial" in self.coordinator_stats:
                best_trial = self.coordinator_stats["best_trial"]

                st.subheader("Best Trial")
                col1, col2 = st.columns(2)

                with col1:
                    st.write("Parameters:")
                    st.json(best_trial.get("parameters", {}))

                with col2:
                    st.write("Objectives:")
                    st.json(best_trial.get("objectives", {}))
        else:
            st.warning("No optimization results available")

        # Display trials data
        if not self.trials_data.empty:
            st.subheader(f"Latest Trials (showing {len(self.trials_data)} of {self.coordinator_stats.get('total_trials', 0)})")
            st.dataframe(self.trials_data, use_container_width=True)
        else:
            st.info("No trials data available")

    def _display_visualizations(self):
        """Display visualizations of optimization data."""
        st.header("Visualizations")

        if self.trials_data.empty:
            st.warning("No data available for visualization")
            return

        # Identify objective and parameter columns
        objective_cols = [col for col in self.trials_data.columns if col.startswith('f')]
        param_cols = [col for col in self.trials_data.columns
                     if col not in objective_cols
                     and col not in ['Trial', 'Is Ranked', 'Is Feasible', 'Front']]

        # Plot objectives
        if len(objective_cols) >= 2:
            st.subheader("Objective Values")
            tab1, tab2, tab3 = st.tabs(["2D Plot", "3D Plot", "Parallel Coordinates"])

            with tab1:
                # 2D scatter plot of first two objectives
                fig = px.scatter(
                    self.trials_data,
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
                        self.trials_data,
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
                    self.trials_data,
                    dimensions=objective_cols,
                    color='Is Ranked',
                    title="Parallel Coordinates Plot of Objectives"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Plot parameter space
        if len(param_cols) >= 2:
            st.subheader("Parameter Space")
            fig = px.scatter(
                self.trials_data,
                x=param_cols[0],
                y=param_cols[1],
                color='Is Ranked',
                hover_data=['Trial'] + objective_cols,
                title=f"Parameter Space: {param_cols[0]} vs {param_cols[1]}"
            )
            st.plotly_chart(fig, use_container_width=True)


def run_monitor():
    """
    Run the monitoring dashboard as a standalone application.

    Parses command-line arguments to configure the monitor.
    """
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="HOLA Optimization Monitor")
        parser.add_argument("--zmq_endpoint", type=str, default="tcp://localhost:5555",
                        help="ZMQ endpoint for connecting to the optimization server")
        parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

        # When run through streamlit, sys.argv contains streamlit's own arguments first
        # Find where our arguments start (after --)
        args_start = 0
        for i, arg in enumerate(sys.argv):
            if arg == "--":
                args_start = i + 1
                break

        # Parse only our arguments
        if args_start > 0 and args_start < len(sys.argv):
            args = parser.parse_args(sys.argv[args_start:])
        else:
            args = parser.parse_args([])

        # Configure logging based on verbosity
        if args.verbose:
            logging.getLogger("hola").setLevel(logging.DEBUG)
        else:
            logging.getLogger("hola").setLevel(logging.INFO)
            # Reduce third-party logging
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("streamlit").setLevel(logging.WARNING)
            logging.getLogger("zmq").setLevel(logging.WARNING)

        # Create and run the monitor
        monitor = OptimizationMonitor(
            server_zmq_endpoint=args.zmq_endpoint
        )

        monitor.run_dashboard()

    except Exception as e:
        st.error(f"Error initializing monitor: {str(e)}")
        logger.exception("Monitor initialization error")


if __name__ == "__main__":
    run_monitor()