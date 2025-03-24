import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import threading
import random
import queue
import os
import pickle
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler
from hola.core.leaderboard import Trial

# Constants for file paths
TEMP_DIR = Path("temp_optimization")
STATE_FILE = TEMP_DIR / "optimization_state.pkl"
PROGRESS_FILE = TEMP_DIR / "progress.json"
STOP_FILE = TEMP_DIR / "stop.txt"
COMPLETE_FILE = TEMP_DIR / "complete.txt"

# Create temp directory if it doesn't exist
TEMP_DIR.mkdir(exist_ok=True)

# Initialize session state
if 'coordinator' not in st.session_state:
    st.session_state.coordinator = None

if 'optimization_running' not in st.session_state:
    st.session_state.optimization_running = False

if 'total_iterations' not in st.session_state:
    st.session_state.total_iterations = 100

if 'progress' not in st.session_state:
    st.session_state.progress = 0.0

if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = 0

def save_state(coordinator, progress):
    """Save the current optimization state to disk"""
    try:
        # Create backup of existing file if it exists
        if STATE_FILE.exists():
            backup_file = STATE_FILE.with_suffix('.bak')
            STATE_FILE.rename(backup_file)

        with open(STATE_FILE, "wb") as f:
            state = {
                "progress": progress,
                "coordinator": coordinator
            }
            pickle.dump(state, f)
    except Exception as e:
        print(f"Error saving state: {e}")
        # Try to restore backup if save failed
        backup_file = STATE_FILE.with_suffix('.bak')
        if backup_file.exists():
            try:
                backup_file.rename(STATE_FILE)
                print("Restored state file from backup")
            except:
                pass

def load_state():
    """Load the optimization state from disk"""
    if STATE_FILE.exists() and STATE_FILE.stat().st_size > 0:
        try:
            with open(STATE_FILE, "rb") as f:
                state = pickle.load(f)
                st.session_state.progress = state.get("progress", 0.0)
                st.session_state.coordinator = state.get("coordinator")
                print(f"Loaded state from {STATE_FILE}")
                return True
        except Exception as e:
            print(f"Error loading state: {e}")
            # Try to load from backup
            backup_file = STATE_FILE.with_suffix('.bak')
            if backup_file.exists():
                try:
                    with open(backup_file, "rb") as f:
                        state = pickle.load(f)
                        st.session_state.progress = state.get("progress", 0.0)
                        st.session_state.coordinator = state.get("coordinator")
                        print(f"Loaded state from backup {backup_file}")
                        return True
                except Exception as e2:
                    print(f"Error loading backup state: {e2}")
    return False

def save_progress(progress: float, total_evaluations: int):
    """Save progress to a JSON file"""
    try:
        data = {
            "progress": progress,
            "total_evaluations": total_evaluations,
            "timestamp": time.time()
        }
        # Write to temporary file first, then rename to avoid corruption
        temp_file = PROGRESS_FILE.with_suffix('.tmp')
        with open(temp_file, "w") as f:
            json.dump(data, f)
        temp_file.rename(PROGRESS_FILE)
    except Exception as e:
        print(f"Error saving progress: {e}")

def load_progress():
    """Load progress from JSON file"""
    if PROGRESS_FILE.exists() and PROGRESS_FILE.stat().st_size > 0:
        try:
            with open(PROGRESS_FILE, "r") as f:
                data = f.read()
                if data.strip():  # Check if file is not empty
                    return json.loads(data)
        except Exception as e:
            print(f"Error loading progress: {e}")
    return None

# Define default parameters and objectives for the test problem
def get_default_parameters():
    return {
        "x": {"type": "continuous", "min": 0.0, "max": 10.0},
        "y": {"type": "continuous", "min": 0.0, "max": 10.0},
    }

def get_default_objectives():
    return {
        "f1": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 1.0,
            "comparison_group": 0
        },
        "f2": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 0.8,
            "comparison_group": 0
        },
        "f3": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 0.5,
            "comparison_group": 1
        },
    }

# Define the evaluation function
def evaluate(x: float, y: float) -> dict[str, float]:
    f1 = x**2 + y**2
    f2 = (x-2)**2 + (y-2)**2
    f3 = (x-4)**2 + (y-4)**2
    return {"f1": f1, "f2": f2, "f3": f3}

# Function to run optimization in a separate thread
def run_optimization(iterations: int):
    print("Starting optimization thread")

    # Track thread state in a file
    TEMP_DIR.mkdir(exist_ok=True)
    thread_status_file = TEMP_DIR / "thread_status.json"

    try:
        # Create samplers for exploration and exploitation
        explore_sampler = SobolSampler(dimension=2)
        exploit_sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)

        # Create an explore-exploit sampler (combines exploration and exploitation)
        sampler = ExploreExploitSampler(
            explore_sampler=explore_sampler,
            exploit_sampler=exploit_sampler,
        )

        # Create coordinator
        coordinator = OptimizationCoordinator.from_dict(
            hypercube_sampler=sampler,
            objectives_dict=get_default_objectives(),
            parameters_dict=get_default_parameters()
        )

        # Remove any existing completion file
        if COMPLETE_FILE.exists():
            COMPLETE_FILE.unlink()

        # Write thread status
        with open(thread_status_file, 'w') as f:
            json.dump({"status": "running", "time": time.time()}, f)

        print(f"Starting {iterations} iterations")

        # Run optimization
        for i in range(iterations):
            # Check if stop file exists
            if STOP_FILE.exists():
                print("Stop signal received, terminating optimization")
                try:
                    STOP_FILE.unlink()
                except:
                    pass
                break

            params_list, metadata = coordinator.suggest_parameters()

            for params in params_list:
                objectives = evaluate(**params)
                coordinator.record_evaluation(params, objectives, metadata)

            # Update progress
            progress = (i + 1) / iterations
            save_progress(progress, coordinator.get_total_evaluations())

            if i % 5 == 0 or i == iterations - 1:
                print(f"Iteration {i+1}/{iterations}, progress: {progress:.1%}")
                # Save state periodically
                save_state(coordinator, progress)

                # Update thread status
                with open(thread_status_file, 'w') as f:
                    json.dump({"status": "running", "iteration": i+1, "time": time.time()}, f)

            time.sleep(0.1)  # Sleep a bit to simulate real work

        print("Optimization thread completed")
        # Final state save
        save_state(coordinator, progress)

        # Signal completion with a dedicated file
        COMPLETE_FILE.touch()

        # Update thread status
        with open(thread_status_file, 'w') as f:
            json.dump({"status": "completed", "time": time.time()}, f)

    except Exception as e:
        print(f"Error in optimization thread: {e}")
        # Update thread status on error
        with open(thread_status_file, 'w') as f:
            json.dump({"status": "error", "error": str(e), "time": time.time()}, f)
        raise
    finally:
        # Make sure we always signal completion if thread exits for any reason
        if not COMPLETE_FILE.exists():
            COMPLETE_FILE.touch()

# UI to start/stop optimization
def optimization_controls():
    # Direct check for completion file
    if COMPLETE_FILE.exists() and st.session_state.optimization_running:
        print("optimization_controls: Detected completion file")
        st.session_state.optimization_running = False

    col1, col2 = st.columns([3, 1])

    with col1:
        iterations = st.slider("Total iterations", min_value=10, max_value=500, value=st.session_state.total_iterations, step=10)
        st.session_state.total_iterations = iterations

    with col2:
        if st.session_state.optimization_running:
            if st.button("Stop Optimization"):
                # Create stop file
                STOP_FILE.touch()
                st.session_state.optimization_running = False
                print("Stop button clicked")
        else:
            if st.button("Start Optimization"):
                # Clear any existing stop file
                if STOP_FILE.exists():
                    STOP_FILE.unlink()

                # Clear any existing completion file
                if COMPLETE_FILE.exists():
                    COMPLETE_FILE.unlink()

                st.session_state.progress = 0.0
                st.session_state.optimization_running = True
                st.session_state.last_update_time = time.time()

                print("Start button clicked")

                # Start optimization in a new thread
                optimization_thread = threading.Thread(
                    target=run_optimization,
                    args=(iterations,)
                )
                optimization_thread.daemon = True
                optimization_thread.start()

# Display progress bar
def show_progress():
    # Force check with filesystem operations to ensure state is current
    completion_flag = COMPLETE_FILE.exists()

    # Check for completion
    if completion_flag and st.session_state.optimization_running:
        print("Detected optimization completion")
        st.session_state.optimization_running = False
        # Force refresh on completion
        st.rerun()

    # Update progress from file if optimization is running
    if st.session_state.optimization_running:
        progress_data = load_progress()
        if progress_data:
            st.session_state.progress = progress_data["progress"]
            st.session_state.last_update_time = progress_data["timestamp"]

    if st.session_state.optimization_running:
        progress_text = f"Optimization in progress... {int(st.session_state.progress * 100)}%"
    else:
        if completion_flag and st.session_state.progress > 0:
            progress_text = f"Optimization complete! {int(st.session_state.progress * 100)}%"
        elif st.session_state.progress > 0:
            progress_text = f"Optimization stopped at {int(st.session_state.progress * 100)}%"
        else:
            progress_text = "Optimization not started"

    st.progress(st.session_state.progress, text=progress_text)

# Function to display optimization statistics
def show_statistics():
    if st.session_state.coordinator is None:
        st.info("No optimization data available. Start optimization to see statistics.")
        return

    coordinator = st.session_state.coordinator

    # Create metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trials", coordinator.get_total_evaluations())

    with col2:
        st.metric("Feasible Trials", coordinator.get_feasible_count())

    with col3:
        st.metric("Ranked Trials", coordinator.get_ranked_count())

    with col4:
        st.metric("Infeasible Trials", coordinator.get_infeasible_count())

    # Show best trial
    best_trial = coordinator.get_best_trial()
    if best_trial:
        st.subheader("Best Trial")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Parameters:")
            st.json(best_trial.parameters)

        with col2:
            st.write("Objectives:")
            st.json(best_trial.objectives)

# Function to plot pareto fronts
def plot_pareto_fronts():
    if st.session_state.coordinator is None or st.session_state.coordinator.get_total_evaluations() == 0:
        return

    coordinator = st.session_state.coordinator

    # Get data
    df = coordinator.get_all_trials_dataframe()

    # Plot objectives
    st.subheader("Objective Values")

    objective_cols = [col for col in df.columns if col.startswith('f')]

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

# Function to display data tables
def show_data_tables():
    if st.session_state.coordinator is None or st.session_state.coordinator.get_total_evaluations() == 0:
        return

    coordinator = st.session_state.coordinator

    # Get data
    df_ranked = coordinator.get_trials_dataframe(ranked_only=True)
    df_all = coordinator.get_all_trials_dataframe()
    metadata_df = coordinator.get_trials_metadata()

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

# Main app
def main():
    st.set_page_config(
        page_title="HOLA Optimization Monitor",
        page_icon="📊",
        layout="wide",
    )

    # Initialize the app state
    initialize_app_state()

    st.title("HOLA Optimization Monitor")

    # Controls and progress
    optimization_controls()
    show_progress()

    # Display tabs for different views
    tab1, tab2, tab3 = st.tabs(["Statistics", "Visualizations", "Data Tables"])

    with tab1:
        show_statistics()

    with tab2:
        plot_pareto_fronts()

    with tab3:
        show_data_tables()

    # Add a debug section to show thread status
    with st.expander("Debug Info"):
        st.write(f"Optimization running: {st.session_state.optimization_running}")
        st.write(f"Current progress: {st.session_state.progress:.2%}")
        st.write(f"Last update time: {st.session_state.last_update_time}")
        st.write(f"Completion file exists: {COMPLETE_FILE.exists()}")

        # Show thread status if available
        thread_status_file = TEMP_DIR / "thread_status.json"
        if thread_status_file.exists():
            try:
                with open(thread_status_file, 'r') as f:
                    thread_status = json.load(f)
                    st.write("Thread status:")
                    st.json(thread_status)
            except:
                st.write("Error reading thread status")

        progress_data = load_progress()
        if progress_data:
            st.write("Latest progress data:")
            st.json(progress_data)

        if st.session_state.coordinator:
            st.write(f"Total evaluations: {st.session_state.coordinator.get_total_evaluations()}")

    # Auto-refresh while optimization is running
    if st.session_state.optimization_running:
        time.sleep(0.5)  # Small delay to prevent too rapid refreshes
        st.rerun()

def initialize_app_state():
    """Initialize the app state by checking files and loading state"""
    # Always try to load the latest state
    load_state()

    # Check for completion file at startup
    if COMPLETE_FILE.exists():
        print("Initial check: Detected completion file")
        st.session_state.optimization_running = False

    # Check thread status file
    thread_status_file = TEMP_DIR / "thread_status.json"
    if thread_status_file.exists():
        try:
            with open(thread_status_file, 'r') as f:
                status = json.load(f)
                time_diff = time.time() - status.get('time', 0)
                # If thread status is recent (last 10 seconds) and running, update our state
                if time_diff < 10 and status.get('status') == 'running':
                    print("Thread appears to be active")
                    st.session_state.optimization_running = True
                # If status is completed or error, make sure we're not running
                elif status.get('status') in ('completed', 'error'):
                    print(f"Thread status: {status.get('status')}")
                    st.session_state.optimization_running = False
        except Exception as e:
            print(f"Error reading thread status: {e}")

    # Double-check: if running but completion file exists, not running
    if st.session_state.optimization_running and COMPLETE_FILE.exists():
        print("State conflict: running flag but completion file exists")
        st.session_state.optimization_running = False

if __name__ == "__main__":
    main()