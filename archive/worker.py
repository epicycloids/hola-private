"""HOLA optimization client utilities.

This module provides a client interface for communicating with HOLA optimization
servers. It handles:

- Parameter sampling requests
- Reporting objective values
- Error handling and validation

Example:
    >>> # Create a worker connected to local server
    >>> worker = Worker()
    >>>
    >>> # Get initial parameter sample
    >>> params = worker.get_param_sample()
    >>> print(params)
    {'learning_rate': 0.001, 'batch_size': 0.7}
    >>>
    >>> # Report results and get next sample
    >>> next_params = worker.report_sim_result(
    ...     objectives={'accuracy': 0.95, 'training_time': 120},
    ...     params=params
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import requests
from requests.exceptions import RequestException

from hola.objective import ObjectiveName
from hola.params import ParamName
from hola.utils import BaseConfig

# Type definitions
ServerParams = dict[ParamName, float]
ObjectiveValues = dict[ObjectiveName, float]

# Constants
DEFAULT_HOST: Final[str] = "http://localhost"
DEFAULT_PORT: Final[int] = 8675
REPORT_ENDPOINT: Final[str] = "report_request"


class ServerResponse(BaseConfig):
    """Server response containing parameter samples and objective values.

    This class handles validation and serialization of data exchanged with
    the optimization server.

    Example:
        >>> # Create from server response
        >>> result = ServerResponse.from_response(response)
        >>> print(result.params)
        {'learning_rate': 0.001}
        >>>
        >>> # Create for sending results
        >>> result = ServerResponse(
        ...     params={'batch_size': 32},
        ...     objectives={'accuracy': 0.95}
        ... )
        >>> response = requests.post(url, json=result.model_dump())

    :param params: Parameter name to value mapping
    :type params: dict[str, float]
    :param objectives: Objective name to value mapping
    :type objectives: dict[str, float]
    """

    params: ServerParams
    objectives: ObjectiveValues

    @classmethod
    def from_response(cls, response: requests.Response) -> ServerResponse:
        """Create a ServerResponse from HTTP response.

        :param response: Server response to parse
        :type response: requests.Response
        :return: Validated optimization result
        :rtype: ServerResponse
        :raises ValueError: If response contains invalid data
        """
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError(f"Expected dict response, got {type(data)}")
            return cls.model_validate(data)
        except Exception as e:
            raise ValueError(f"Invalid server response: {str(e)}") from e


@dataclass
class Worker:
    """Client for communicating with a HOLA optimization server.

    Handles parameter sampling requests and reporting of objective values
    while providing robust error handling and validation.

    Example:
        >>> # Connect to local server
        >>> worker = Worker()
        >>>
        >>> try:
        ...     # Get parameter sample
        ...     params = worker.get_param_sample()
        ...
        ...     # Run simulation/training
        ...     results = run_simulation(params)
        ...
        ...     # Report results and get next sample
        ...     next_params = worker.report_sim_result(
        ...         objectives=results
        ...     )
        ... except ConnectionError:
        ...     print("Server connection failed")
        ... except RequestException as e:
        ...     print(f"Server error: {e}")

    :param server_url: Base URL of HOLA server
    :type server_url: str
    :param port: Server port number
    :type port: int
    """

    server_url: str = DEFAULT_HOST
    port: int = DEFAULT_PORT

    def __post_init__(self) -> None:
        """Initialize the request URL."""
        self.rep_req_url = f"{self.server_url}:{self.port}/{REPORT_ENDPOINT}"

    def sample(self) -> ServerParams:
        """Get a new parameter sample from the server.

        :return: Parameter name to value mapping
        :rtype: dict[str, float]
        :raises ConnectionError: If server connection fails
        :raises RequestException: If server returns error response
        :raises ValueError: If server response is invalid
        """
        try:
            response = requests.get(self.rep_req_url)
            response.raise_for_status()
            result = ServerResponse.from_response(response)
            return result.params
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to HOLA server at {self.rep_req_url}") from e
        except RequestException as e:
            raise RequestException(f"Error getting parameter sample: {str(e)}") from e

    def report_sim_result(
        self,
        objectives: ObjectiveValues | None = None,
        params: ServerParams | None = None,
    ) -> ServerParams:
        """Report simulation results and get next parameter sample.

        Results can include objective values, parameter values, or both.
        At least one must be provided.

        Example:
            >>> # Report just objective values
            >>> next_params = worker.report_sim_result(
            ...     objectives={'accuracy': 0.95}
            ... )
            >>>
            >>> # Report both objectives and parameters
            >>> next_params = worker.report_sim_result(
            ...     objectives={'accuracy': 0.95},
            ...     params={'learning_rate': 0.001}
            ... )

        :param objectives: Objective name to value mapping
        :type objectives: dict[str, float] | None
        :param params: Parameter name to value mapping
        :type params: dict[str, float] | None
        :return: Next parameter sample
        :rtype: dict[str, float]
        :raises ConnectionError: If server connection fails
        :raises RequestException: If server returns error response
        :raises ValueError: If no data provided or response invalid
        """
        if objectives is None and params is None:
            raise ValueError("Must provide either objectives or params")

        request_data = ServerResponse(params=params or {}, objectives=objectives or {})

        try:
            response = requests.post(self.rep_req_url, json=request_data.model_dump())
            response.raise_for_status()
            result = ServerResponse.from_response(response)
            return result.params
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to HOLA server at {self.rep_req_url}") from e
        except RequestException as e:
            raise RequestException(f"Error reporting simulation result: {str(e)}") from e

    def __repr__(self) -> str:
        """Return string representation.

        :return: String representation
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}("
            f"server_url='{self.server_url}', "
            f"port={self.port}"
            f")"
        )