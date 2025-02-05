import asyncio
import streamlit as st
from typing import Dict, Any
import logging

from hola.server.config import ConnectionConfig, SocketType
from hola.client.client import OptimizationClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def connect_client(socket_type: str, host: str, port: int, ipc_path: str) -> OptimizationClient:
    """Create and connect a client to the HOLA server."""
    config = ConnectionConfig(
        socket_type=SocketType(socket_type),
        host=host,
        port=port,
        ipc_path=ipc_path
    )

    client = OptimizationClient(config)
    await client.start()
    return client

def create_objective_config() -> Dict[str, Dict[str, Any]]:
    """Create objective configuration UI."""
    st.subheader("Objectives Configuration")

    objectives = {}
    num_objectives = st.number_input("Number of objectives", min_value=1, value=1)

    for i in range(num_objectives):
        st.write(f"Objective {i+1}")
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(f"Name #{i+1}", key=f"obj_name_{i}")
            target = st.number_input(f"Target #{i+1}", key=f"obj_target_{i}")
            direction = st.selectbox(
                f"Direction #{i+1}",
                options=["minimize", "maximize"],
                key=f"obj_dir_{i}"
            )

        with col2:
            limit = st.number_input(f"Limit #{i+1}", key=f"obj_limit_{i}")
            priority = st.number_input(
                f"Priority #{i+1}",
                min_value=0.0,
                value=1.0,
                key=f"obj_priority_{i}"
            )

        if name:
            objectives[name] = {
                "target": target,
                "limit": limit,
                "direction": direction,
                "priority": priority
            }

    return objectives

def create_parameter_config() -> Dict[str, Dict[str, Any]]:
    """Create parameter configuration UI."""
    st.subheader("Parameters Configuration")

    parameters = {}
    num_params = st.number_input("Number of parameters", min_value=1, value=1)

    for i in range(num_params):
        st.write(f"Parameter {i+1}")
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input(f"Name #{i+1}", key=f"param_name_{i}")
            param_type = st.selectbox(
                f"Type #{i+1}",
                options=["continuous", "integer", "categorical", "lattice"],
                key=f"param_type_{i}"
            )

        with col2:
            if param_type == "categorical":
                categories = st.text_input(
                    f"Categories #{i+1} (comma-separated)",
                    key=f"param_cats_{i}"
                )
                config = {
                    "type": "categorical",
                    "categories": tuple(cat.strip() for cat in categories.split(","))
                }
            else:
                min_val = st.number_input(f"Min #{i+1}", key=f"param_min_{i}")
                max_val = st.number_input(f"Max #{i+1}", key=f"param_max_{i}")
                config = {
                    "type": param_type,
                    "min": min_val,
                    "max": max_val
                }

                if param_type == "continuous":
                    scale = st.selectbox(
                        f"Scale #{i+1}",
                        options=["linear", "log"],
                        key=f"param_scale_{i}"
                    )
                    config["scale"] = scale
                elif param_type == "lattice":
                    num_values = st.number_input(
                        f"Number of values #{i+1}",
                        min_value=2,
                        value=10,
                        key=f"param_num_values_{i}"
                    )
                    config["num_values"] = num_values

        if name:
            parameters[name] = config

    return parameters

async def initialize_optimization(
    client: OptimizationClient,
    objectives_config: Dict[str, Dict[str, Any]],
    parameters_config: Dict[str, Dict[str, Any]]
) -> None:
    """Initialize the optimization process."""
    response = await client.initialize(
        objectives_config=objectives_config,
        parameters_config=parameters_config
    )
    return response

async def monitor_optimization(client: OptimizationClient, status_placeholder):
    """Monitor optimization progress."""
    while True:
        status = await client.get_status()
        status_placeholder.write("Current Status:")
        status_placeholder.json({"total_evaluations": status.status.total_evaluations})
        if status.status.best_result:
            status_placeholder.write("Best Result:")
            status_placeholder.json({
                "parameters": status.status.best_result.parameters,
                "objectives": status.status.best_result.objectives
            })
        await asyncio.sleep(1)

def main():
    st.title("HOLA Client")

    # Server Connection Settings
    st.header("Server Connection")
    col1, col2 = st.columns(2)

    with col1:
        socket_type = st.selectbox("Socket Type", options=["TCP", "IPC"])
        if socket_type == "TCP":
            host = st.text_input("Host", value="localhost")
            port = st.number_input("Port", value=5555)
            ipc_path = ""
        else:
            host = ""
            port = 0
            ipc_path = st.text_input("IPC Path", value="/tmp/hola-optimizer")

    # Connect button
    if "client" not in st.session_state:
        if st.button("Connect to Server"):
            client = asyncio.run(connect_client(socket_type.lower(), host, port, ipc_path))
            st.session_state.client = client
            st.success("Connected to server!")

    if "client" in st.session_state:
        # Create tabs for different operations
        tab1, tab2, tab3 = st.tabs(["Initialize", "Control", "Monitor"])

        with tab1:
            objectives_config = create_objective_config()
            parameters_config = create_parameter_config()

            if st.button("Initialize Optimization"):
                response = asyncio.run(initialize_optimization(
                    st.session_state.client,
                    objectives_config,
                    parameters_config
                ))
                st.success("Optimization initialized!")

        with tab2:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Pause"):
                    asyncio.run(st.session_state.client.pause())
                    st.success("Optimization paused")

            with col2:
                if st.button("Resume"):
                    asyncio.run(st.session_state.client.resume())
                    st.success("Optimization resumed")

            # Add buttons for other control operations
            if st.button("Update Objectives"):
                asyncio.run(st.session_state.client.update_objectives(objectives_config))
                st.success("Objectives updated")

            if st.button("Update Parameters"):
                asyncio.run(st.session_state.client.update_parameters(parameters_config))
                st.success("Parameters updated")

        with tab3:
            status_placeholder = st.empty()
            if st.button("Start Monitoring"):
                try:
                    asyncio.run(monitor_optimization(st.session_state.client, status_placeholder))
                except KeyboardInterrupt:
                    st.warning("Monitoring stopped")

if __name__ == "__main__":
    main()