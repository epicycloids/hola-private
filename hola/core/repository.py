from abc import ABC, abstractmethod
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import numpy as np
import msgspec

from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName


class Trial(msgspec.Struct, frozen=True):
    """
    Immutable record of a single optimization trial.

    Contains all information about a trial including its unique identifier,
    the parameter values used, the objective values achieved, whether
    the parameter values are considered feasible under current constraints,
    and any associated metadata.
    """

    trial_id: int
    """Unique identifier for the trial."""

    objectives: dict[ObjectiveName, float]
    """Dictionary mapping objective names to their achieved values."""

    parameters: dict[ParameterName, Any]
    """Dictionary mapping parameter names to their trial values."""

    is_feasible: bool = True
    """Whether the parameter values satisfy current constraints."""

    metadata: Dict[str, Any] = {}
    """Additional metadata about the trial, such as sampler information."""


class TrialRepository(ABC):
    """
    Abstract base class for trial storage repositories.

    Repositories are responsible for persisting and retrieving trial data.
    """

    @abstractmethod
    def add_trial(self, trial: Trial) -> None:
        """
        Add a trial to the repository.

        :param trial: The trial to add
        """
        pass

    @abstractmethod
    def get_trial(self, trial_id: int) -> Optional[Trial]:
        """
        Retrieve a specific trial by ID.

        :param trial_id: ID of the trial to retrieve
        :return: The requested trial, or None if not found
        """
        pass

    @abstractmethod
    def get_all_trials(self) -> List[Trial]:
        """
        Retrieve all trials from the repository.

        :return: List of all trials
        """
        pass

    @abstractmethod
    def get_feasible_trials(self) -> List[Trial]:
        """
        Retrieve all feasible trials.

        :return: List of feasible trials
        """
        pass

    @abstractmethod
    def get_trial_ids(self) -> Set[int]:
        """
        Get the set of all trial IDs in the repository.

        :return: Set of trial IDs
        """
        pass

    @abstractmethod
    def get_dataframe(self, ranked_only: bool = False) -> pd.DataFrame:
        """
        Convert repository trials to a DataFrame.

        :param ranked_only: If True, include only trials marked as ranked
        :return: DataFrame of trials
        """
        pass


class MemoryTrialRepository(TrialRepository):
    """
    In-memory implementation of TrialRepository.

    Stores trials in a dictionary with no persistence.
    """

    def __init__(self):
        self._trials: Dict[int, Trial] = {}
        self._ranked_ids: Set[int] = set()

    def add_trial(self, trial: Trial, is_ranked: bool = False) -> None:
        """
        Add a trial to the repository.

        :param trial: The trial to add
        :param is_ranked: Whether the trial is included in ranking
        """
        self._trials[trial.trial_id] = trial
        if is_ranked:
            self._ranked_ids.add(trial.trial_id)

    def get_trial(self, trial_id: int) -> Optional[Trial]:
        """
        Retrieve a specific trial by ID.

        :param trial_id: ID of the trial to retrieve
        :return: The requested trial, or None if not found
        """
        return self._trials.get(trial_id)

    def get_all_trials(self) -> List[Trial]:
        """
        Retrieve all trials from the repository.

        :return: List of all trials
        """
        return list(self._trials.values())

    def get_feasible_trials(self) -> List[Trial]:
        """
        Retrieve all feasible trials.

        :return: List of feasible trials
        """
        return [t for t in self._trials.values() if t.is_feasible]

    def get_trial_ids(self) -> Set[int]:
        """
        Get the set of all trial IDs in the repository.

        :return: Set of trial IDs
        """
        return set(self._trials.keys())

    def set_ranked(self, trial_id: int, is_ranked: bool = True) -> None:
        """
        Mark a trial as ranked or unranked.

        :param trial_id: ID of the trial
        :param is_ranked: Whether the trial is included in ranking
        """
        if is_ranked:
            self._ranked_ids.add(trial_id)
        elif trial_id in self._ranked_ids:
            self._ranked_ids.remove(trial_id)

    def get_dataframe(self, ranked_only: bool = False) -> pd.DataFrame:
        """
        Convert repository trials to a DataFrame.

        :param ranked_only: If True, include only trials marked as ranked
        :return: DataFrame of trials
        """
        data_rows = []

        # Determine which trials to process
        trial_ids = self._ranked_ids if ranked_only else self._trials.keys()

        # Process each trial
        for tid in trial_ids:
            if tid not in self._trials:
                continue

            trial = self._trials[tid]

            # Basic information for the trial
            row_data = {
                "Trial": tid,
                **{str(k): v for k, v in trial.parameters.items()},
                **{str(k): v for k, v in trial.objectives.items()},
                "Is Ranked": tid in self._ranked_ids,
                "Is Feasible": trial.is_feasible
            }

            data_rows.append(row_data)

        # Create and return DataFrame
        if not data_rows:
            # Return empty DataFrame with appropriate columns
            columns = ["Trial", "Is Ranked", "Is Feasible"]
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(data_rows)


class SQLiteTrialRepository(TrialRepository):
    """
    SQLite-based implementation of TrialRepository.

    Stores trials in a SQLite database for persistence and shared access.
    """

    def __init__(self, db_path: str):
        """
        Initialize SQLite repository.

        :param db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema if needed."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Create tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id INTEGER PRIMARY KEY,
                    is_feasible BOOLEAN,
                    is_ranked BOOLEAN DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS parameters (
                    trial_id INTEGER,
                    name TEXT,
                    value TEXT,
                    FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS objectives (
                    trial_id INTEGER,
                    name TEXT,
                    value REAL,
                    FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS metadata (
                    trial_id INTEGER,
                    key TEXT,
                    value TEXT,
                    FOREIGN KEY (trial_id) REFERENCES trials(trial_id) ON DELETE CASCADE
                )
            ''')

            # Create indexes for faster queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_parameters_trial_id ON parameters(trial_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_objectives_trial_id ON objectives(trial_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metadata_trial_id ON metadata(trial_id)')

    def _serialize_value(self, value: Any) -> str:
        """
        Serialize a value to store in SQLite.

        :param value: Value to serialize
        :return: JSON string representation
        """
        try:
            return json.dumps(value)
        except (TypeError, ValueError, OverflowError):
            # Fall back to string representation for complex objects
            return str(value)

    def _deserialize_value(self, value_str: str, name: str) -> Any:
        """
        Deserialize a value from SQLite storage.

        :param value_str: Serialized value string
        :param name: Parameter or objective name
        :return: Deserialized value
        """
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            # Return as string if not valid JSON
            return value_str

    def add_trial(self, trial: Trial, is_ranked: bool = False) -> None:
        """
        Add a trial to the repository.

        :param trial: The trial to add
        :param is_ranked: Whether the trial is included in ranking
        """
        with sqlite3.connect(self.db_path) as conn:
            # Use a transaction to ensure atomicity
            conn.execute("INSERT OR REPLACE INTO trials VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                       (trial.trial_id, trial.is_feasible, is_ranked))

            # Clear existing data first if replacing
            conn.execute("DELETE FROM parameters WHERE trial_id = ?", (trial.trial_id,))
            conn.execute("DELETE FROM objectives WHERE trial_id = ?", (trial.trial_id,))
            conn.execute("DELETE FROM metadata WHERE trial_id = ?", (trial.trial_id,))

            # Insert parameters
            params = [(trial.trial_id, str(name), self._serialize_value(value))
                     for name, value in trial.parameters.items()]
            conn.executemany("INSERT INTO parameters VALUES (?, ?, ?)", params)

            # Insert objectives
            objectives = [(trial.trial_id, str(name), float(value))
                         for name, value in trial.objectives.items()]
            conn.executemany("INSERT INTO objectives VALUES (?, ?, ?)", objectives)

            # Insert metadata
            if trial.metadata:
                metadata = [(trial.trial_id, str(key), self._serialize_value(value))
                           for key, value in trial.metadata.items()]
                conn.executemany("INSERT INTO metadata VALUES (?, ?, ?)", metadata)

    def set_ranked(self, trial_id: int, is_ranked: bool = True) -> None:
        """
        Mark a trial as ranked or unranked.

        :param trial_id: ID of the trial
        :param is_ranked: Whether the trial is included in ranking
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE trials SET is_ranked = ? WHERE trial_id = ?",
                       (is_ranked, trial_id))

    def get_trial(self, trial_id: int) -> Optional[Trial]:
        """
        Retrieve a specific trial by ID.

        :param trial_id: ID of the trial to retrieve
        :return: The requested trial, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            # Check if trial exists
            cursor = conn.execute("SELECT is_feasible FROM trials WHERE trial_id = ?", (trial_id,))
            result = cursor.fetchone()
            if not result:
                return None

            is_feasible = bool(result[0])

            # Get parameters
            cursor = conn.execute(
                "SELECT name, value FROM parameters WHERE trial_id = ?",
                (trial_id,)
            )
            parameters = {
                ParameterName(row[0]): self._deserialize_value(row[1], row[0])
                for row in cursor.fetchall()
            }

            # Get objectives
            cursor = conn.execute(
                "SELECT name, value FROM objectives WHERE trial_id = ?",
                (trial_id,)
            )
            objectives = {
                ObjectiveName(row[0]): float(row[1])
                for row in cursor.fetchall()
            }

            # Get metadata
            cursor = conn.execute(
                "SELECT key, value FROM metadata WHERE trial_id = ?",
                (trial_id,)
            )
            metadata = {
                row[0]: self._deserialize_value(row[1], row[0])
                for row in cursor.fetchall()
            }

            return Trial(
                trial_id=trial_id,
                objectives=objectives,
                parameters=parameters,
                is_feasible=is_feasible,
                metadata=metadata
            )

    def get_all_trials(self) -> List[Trial]:
        """
        Retrieve all trials from the repository.

        :return: List of all trials
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT trial_id FROM trials")
            trial_ids = [row[0] for row in cursor.fetchall()]

        return [self.get_trial(tid) for tid in trial_ids if self.get_trial(tid) is not None]

    def get_feasible_trials(self) -> List[Trial]:
        """
        Retrieve all feasible trials.

        :return: List of feasible trials
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT trial_id FROM trials WHERE is_feasible = 1")
            trial_ids = [row[0] for row in cursor.fetchall()]

        return [self.get_trial(tid) for tid in trial_ids if self.get_trial(tid) is not None]

    def get_trial_ids(self) -> Set[int]:
        """
        Get the set of all trial IDs in the repository.

        :return: Set of trial IDs
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT trial_id FROM trials")
            return {row[0] for row in cursor.fetchall()}

    def get_dataframe(self, ranked_only: bool = False) -> pd.DataFrame:
        """
        Convert repository trials to a DataFrame.

        :param ranked_only: If True, include only trials marked as ranked
        :return: DataFrame of trials
        """
        with sqlite3.connect(self.db_path) as conn:
            # Build the query based on ranked_only flag
            query = """
                SELECT t.trial_id, t.is_feasible, t.is_ranked
                FROM trials t
            """

            if ranked_only:
                query += " WHERE t.is_ranked = 1"

            cursor = conn.execute(query)
            rows = cursor.fetchall()

            if not rows:
                return pd.DataFrame(columns=["Trial", "Is Feasible", "Is Ranked"])

            data_rows = []
            for trial_id, is_feasible, is_ranked in rows:
                # Get trial details
                trial = self.get_trial(trial_id)
                if not trial:
                    continue

                # Build row data
                row_data = {
                    "Trial": trial_id,
                    "Is Feasible": bool(is_feasible),
                    "Is Ranked": bool(is_ranked),
                    **{str(k): v for k, v in trial.parameters.items()},
                    **{str(k): v for k, v in trial.objectives.items()}
                }

                data_rows.append(row_data)

            return pd.DataFrame(data_rows)