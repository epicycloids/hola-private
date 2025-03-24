"""
Monitor component for HOLA distributed optimization system.

This module implements a monitor that connects to the optimization scheduler
to retrieve and display real-time optimization data.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import msgspec
import pandas as pd
import zmq

from hola.core.leaderboard import Trial


class OptimizationMonitor:
    """
    Monitor for the distributed optimization process.

    Connects to the scheduler via ZMQ to retrieve optimization data
    for display in dashboards and UIs.
    """

    def __init__(self, scheduler_address: str = "tcp://localhost:5555"):
        """
        Initialize the optimization monitor.

        Args:
            scheduler_address: ZMQ address of the scheduler.
        """
        self.scheduler_address = scheduler_address
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(scheduler_address)

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the optimization.

        Returns:
            Dict containing status information:
            - active_workers: Number of active workers
            - total_evaluations: Total number of evaluations
            - best_objectives: Best objectives found so far
        """
        try:
            request = msgspec.json.encode({"tag": "status"})
            self.socket.send(request)

            response = self.socket.recv()
            status_data = msgspec.json.decode(response)

            # Handle potential error responses
            if isinstance(status_data, dict) and "error" in status_data:
                print(f"Warning: Error in status response: {status_data['error']}")
                return {
                    "active_workers": 0,
                    "total_evaluations": 0,
                    "best_objectives": None,
                    "is_running": False,
                    "error": status_data["error"]
                }

            # Process valid response
            return {
                "active_workers": status_data.get("active_workers", 0),
                "total_evaluations": status_data.get("total_evaluations", 0),
                "best_objectives": status_data.get("best_objectives"),
                "is_running": status_data.get("active_workers", 0) > 0
            }
        except Exception as e:
            print(f"Error getting status: {e}")
            return {
                "active_workers": 0,
                "total_evaluations": 0,
                "best_objectives": None,
                "is_running": False,
                "error": str(e)
            }

    def get_trials_dataframe(self, ranked_only: bool = True) -> pd.DataFrame:
        """
        Get a DataFrame of trials from the optimization process.

        Args:
            ranked_only: If True, only include ranked trials.
                        If False, include all trials.

        Returns:
            DataFrame containing trial information
        """
        try:
            request = msgspec.json.encode({
                "tag": "get_trials",
                "ranked_only": ranked_only
            })
            self.socket.send(request)

            response = self.socket.recv()
            trials_data = msgspec.json.decode(response)

            # Convert to DataFrame
            if trials_data and "trials" in trials_data:
                df = pd.DataFrame(trials_data["trials"])
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error getting trials dataframe: {e}")
            return pd.DataFrame()

    def get_all_trials_dataframe(self) -> pd.DataFrame:
        """
        Get a DataFrame of all trials from the optimization process.

        Returns:
            DataFrame containing all trial information
        """
        return self.get_trials_dataframe(ranked_only=False)

    def get_trials_metadata(self) -> pd.DataFrame:
        """
        Get metadata for trials as a DataFrame.

        Returns:
            DataFrame with trial IDs as index and metadata as columns
        """
        try:
            request = msgspec.json.encode({"tag": "get_metadata"})
            self.socket.send(request)

            response = self.socket.recv()
            metadata = msgspec.json.decode(response)

            # Convert to DataFrame
            if metadata and "metadata" in metadata:
                df = pd.DataFrame(metadata["metadata"])
                if not df.empty and "trial_id" in df.columns:
                    df.set_index("trial_id", inplace=True)
                return df
            return pd.DataFrame()
        except Exception as e:
            print(f"Error getting trials metadata: {e}")
            return pd.DataFrame()

    def get_top_k_trials(self, k: int = 1) -> List[Dict[str, Any]]:
        """
        Get the k best trials from the optimization process.

        Args:
            k: Number of trials to return

        Returns:
            List of up to k best trials
        """
        try:
            request = msgspec.json.encode({
                "tag": "get_top_k",
                "k": k
            })
            self.socket.send(request)

            response = self.socket.recv()
            top_k_data = msgspec.json.decode(response)

            if top_k_data and "trials" in top_k_data:
                return top_k_data["trials"]
            return []
        except Exception as e:
            print(f"Error getting top k trials: {e}")
            return []

    def is_multi_group(self) -> bool:
        """
        Check if this is a multi-objective optimization using multiple comparison groups.

        Returns:
            True if using multiple objective comparison groups, False otherwise
        """
        try:
            request = msgspec.json.encode({"tag": "is_multi_group"})
            self.socket.send(request)

            response = self.socket.recv()
            data = msgspec.json.decode(response)

            return data.get("is_multi_group", False)
        except Exception as e:
            print(f"Error checking multi group status: {e}")
            return False

    def close(self):
        """Close the connection to the scheduler."""
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()
        if hasattr(self, 'context') and self.context:
            self.context.term()