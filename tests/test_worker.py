"""Tests for HOLA optimization client functionality."""

import json
from unittest.mock import Mock, patch

import pytest
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError

from hola.worker import DEFAULT_HOST, DEFAULT_PORT, REPORT_ENDPOINT, ServerResponse, Worker


@pytest.fixture
def mock_response() -> Mock:
    """Create a mock requests.Response object."""
    response = Mock(spec=requests.Response)
    response.status_code = 200
    return response


@pytest.fixture
def test_server_url() -> str:
    """Server URL for testing."""
    return "http://testserver"


@pytest.fixture
def test_port() -> int:
    """Port number for testing."""
    return 8000


@pytest.fixture
def sample_params() -> dict[str, float]:
    """Sample parameter values."""
    return {"learning_rate": 0.001, "batch_size": 0.7}


@pytest.fixture
def sample_objectives() -> dict[str, float]:
    """Sample objective values."""
    return {"accuracy": 0.95, "training_time": 120.0}


@pytest.fixture
def worker(test_server_url: str, test_port: int) -> Worker:
    """Create a worker instance for testing."""
    return Worker(server_url=test_server_url, port=test_port)


class TestOptimizationResult:
    """Tests for OptimizationResult class."""

    def test_valid_response_parsing(
        self, mock_response: Mock, sample_params: dict[str, float]
    ) -> None:
        """Test parsing of valid server response."""
        mock_response.json.return_value = {"params": sample_params, "objectives": {}}

        result = ServerResponse.from_response(mock_response)
        assert result.params == sample_params
        assert result.objectives == {}

    def test_invalid_json_response(self, mock_response: Mock) -> None:
        """Test handling of invalid JSON response."""
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)

        with pytest.raises(ValueError, match="Invalid server response"):
            ServerResponse.from_response(mock_response)

    def test_invalid_response_type(self, mock_response: Mock) -> None:
        """Test handling of non-dict response."""
        mock_response.json.return_value = ["not", "a", "dict"]

        with pytest.raises(ValueError, match="Expected dict response"):
            ServerResponse.from_response(mock_response)

    def test_missing_fields(self, mock_response: Mock) -> None:
        """Test handling of response with missing required fields."""
        mock_response.json.return_value = {}  # Missing required fields

        with pytest.raises(ValueError):
            ServerResponse.from_response(mock_response)


class TestWorkerInitialization:
    """Tests for Worker initialization."""

    def test_default_initialization(self) -> None:
        """Test worker initialization with default values."""
        worker = Worker()
        assert worker.server_url == DEFAULT_HOST
        assert worker.port == DEFAULT_PORT
        assert worker.rep_req_url == f"{DEFAULT_HOST}:{DEFAULT_PORT}/{REPORT_ENDPOINT}"

    def test_custom_initialization(self, test_server_url: str, test_port: int) -> None:
        """Test worker initialization with custom values."""
        worker = Worker(server_url=test_server_url, port=test_port)
        assert worker.server_url == test_server_url
        assert worker.port == test_port
        assert worker.rep_req_url == f"{test_server_url}:{test_port}/{REPORT_ENDPOINT}"

    def test_repr(self, worker: Worker) -> None:
        """Test string representation."""
        expected = f"Worker(server_url='{worker.server_url}', port={worker.port})"
        assert repr(worker) == expected


class TestParameterSampling:
    """Tests for parameter sampling functionality."""

    def test_successful_sampling(
        self, worker: Worker, mock_response: Mock, sample_params: dict[str, float]
    ) -> None:
        """Test successful parameter sampling."""
        mock_response.json.return_value = {"params": sample_params, "objectives": {}}

        with patch("requests.get", return_value=mock_response):
            params = worker.sample()
            assert params == sample_params

    def test_connection_error(self, worker: Worker) -> None:
        """Test handling of connection error."""
        with patch("requests.get", side_effect=RequestsConnectionError()):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                worker.sample()

    def test_server_error(self, worker: Worker, mock_response: Mock) -> None:
        """Test handling of server error."""
        mock_response.raise_for_status.side_effect = HTTPError("500 Server Error")

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(requests.exceptions.RequestException):
                worker.sample()


class TestResultReporting:
    """Tests for simulation result reporting."""

    def test_successful_reporting(
        self,
        worker: Worker,
        mock_response: Mock,
        sample_params: dict[str, float],
        sample_objectives: dict[str, float],
    ) -> None:
        """Test successful result reporting."""
        mock_response.json.return_value = {"params": sample_params, "objectives": {}}

        with patch("requests.post", return_value=mock_response) as mock_post:
            next_params = worker.report_sim_result(
                objectives=sample_objectives, params=sample_params
            )

            # Check response handling
            assert next_params == sample_params

            # Check request data
            mock_post.assert_called_once()
            request_data = mock_post.call_args[1]["json"]
            assert request_data["params"] == sample_params
            assert request_data["objectives"] == sample_objectives

    def test_missing_data(self, worker: Worker) -> None:
        """Test error when no data provided."""
        with pytest.raises(ValueError, match="Must provide either objectives or params"):
            worker.report_sim_result()

    def test_partial_data(
        self, worker: Worker, mock_response: Mock, sample_objectives: dict[str, float]
    ) -> None:
        """Test reporting with only objectives."""
        mock_response.json.return_value = {"params": {}, "objectives": {}}

        with patch("requests.post", return_value=mock_response) as mock_post:
            worker.report_sim_result(objectives=sample_objectives)

            request_data = mock_post.call_args[1]["json"]
            assert request_data["objectives"] == sample_objectives
            assert request_data["params"] == {}

    def test_connection_error(self, worker: Worker, sample_objectives: dict[str, float]) -> None:
        """Test handling of connection error during reporting."""
        with patch("requests.post", side_effect=RequestsConnectionError()):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                worker.report_sim_result(objectives=sample_objectives)

    def test_server_error(
        self, worker: Worker, mock_response: Mock, sample_objectives: dict[str, float]
    ) -> None:
        """Test handling of server error during reporting."""
        mock_response.raise_for_status.side_effect = HTTPError("400 Bad Request")

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(
                requests.RequestException, match="Error reporting simulation result"
            ):
                worker.report_sim_result(objectives=sample_objectives)


if __name__ == "__main__":
    pytest.main([__file__])
