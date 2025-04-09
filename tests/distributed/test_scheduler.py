"""Tests for hola.distributed.scheduler (WorkerState and SchedulerProcess)."""

import time
import pytest
from unittest.mock import MagicMock, patch
import zmq
import logging
import msgspec
# Need OptimizationCoordinator for spec in mock_coordinator
from hola.core.coordinator import OptimizationCoordinator

# Assuming the tests directory is at the same level as hola/
from hola.distributed.scheduler import WorkerState, SchedulerProcess
# Import necessary message types
from hola.distributed.messages import ParameterName, HeartbeatRequest, HeartbeatResponse, GetSuggestionRequest, GetSuggestionResponse, SubmitResultRequest, SubmitResultResponse, Result, ShutdownRequest, StatusRequest, StatusResponse, GetTrialsRequest, GetTrialsResponse, GetMetadataRequest, GetMetadataResponse, GetTopKRequest, GetTopKResponse, IsMultiGroupRequest, IsMultiGroupResponse


# --- Tests for WorkerState ---

def test_worker_state_initialization():
    """Test the initial state of a WorkerState instance."""
    worker_id = 1
    start_time_before = time.time()
    state = WorkerState(worker_id)
    start_time_after = time.time()

    assert state.worker_id == worker_id
    assert state.current_parameters is None
    assert state.start_time is None
    assert start_time_before <= state.last_heartbeat <= start_time_after
    assert state.retry_count == 0

def test_worker_state_assign_parameters():
    """Test assigning parameters to a worker."""
    state = WorkerState(1)
    params: dict[ParameterName, float] = {"x": 1.0, "y": 2.0}

    time.sleep(0.01) # Ensure time progresses slightly
    time_before_assign = time.time()
    state.assign_parameters(params)
    time_after_assign = time.time()

    assert state.current_parameters == params
    assert state.start_time is not None
    assert time_before_assign <= state.start_time <= time_after_assign
    assert state.last_heartbeat == pytest.approx(state.start_time)
    assert state.retry_count == 0 # Should not change on assignment

def test_worker_state_update_heartbeat():
    """Test updating the heartbeat timestamp."""
    state = WorkerState(1)
    initial_heartbeat = state.last_heartbeat

    state.assign_parameters({"p": 1})
    assign_heartbeat = state.last_heartbeat
    assert assign_heartbeat > initial_heartbeat

    time.sleep(0.01)
    time_before_update = time.time()
    state.update_heartbeat()
    time_after_update = time.time()

    assert state.last_heartbeat > assign_heartbeat
    assert time_before_update <= state.last_heartbeat <= time_after_update

def test_worker_state_is_timed_out():
    """Test the timeout logic."""
    state = WorkerState(1)
    timeout_seconds = 0.1

    # Worker hasn't started, should not be timed out
    assert not state.is_timed_out(timeout_seconds)

    # Assign parameters (starts the timer)
    state.assign_parameters({"p": 1})
    assert not state.is_timed_out(timeout_seconds) # Should not be timed out immediately

    # Wait for less than the timeout period
    time.sleep(timeout_seconds / 2)
    assert not state.is_timed_out(timeout_seconds)

    # Update heartbeat
    state.update_heartbeat()
    time.sleep(timeout_seconds / 2)
    assert not state.is_timed_out(timeout_seconds)

    # Wait for more than the timeout period since the last heartbeat
    time.sleep(timeout_seconds * 1.1)
    assert state.is_timed_out(timeout_seconds)

# --- Tests for SchedulerProcess ---

# --- Placeholder for SchedulerProcess Tests ---
# Add tests for SchedulerProcess here later, likely requiring extensive mocking.

# Mock Coordinator for testing SchedulerProcess
@pytest.fixture
def mock_coordinator():
    mock = MagicMock(spec=OptimizationCoordinator)
    mock.get_total_evaluations.return_value = 0
    mock.suggest_parameters.return_value = ([{"p1": 1.0}], None) # Example suggestion
    mock.get_best_trial.return_value = None # Initially no best trial
    # Add other necessary methods/attributes as needed for tests
    return mock

# Mock ZMQ for SchedulerProcess tests (might need refinement)
@pytest.fixture
def mock_zmq_scheduler():
    with patch('zmq.Context', autospec=True) as mock_context_cls, \
         patch('zmq.Poller', autospec=True) as mock_poller_cls:

        mock_context_instance = mock_context_cls.return_value
        mock_main_socket = MagicMock(spec=zmq.Socket)
        mock_heartbeat_socket = MagicMock(spec=zmq.Socket)
        # Configure socket creation
        # Use a dictionary to map type to a list of sockets to return
        socket_map = {
            zmq.REP: [mock_main_socket, mock_heartbeat_socket]
        }
        # Need to handle potential list index errors if called more times than expected
        socket_map_copy = {k: list(v) for k, v in socket_map.items()} # copy the list
        def socket_side_effect(socket_type):
            if socket_type in socket_map_copy and socket_map_copy[socket_type]:
                 return socket_map_copy[socket_type].pop(0)
            raise ValueError(f"Unexpected socket type requested or list empty: {socket_type}")
        mock_context_instance.socket.side_effect = socket_side_effect

        mock_poller_instance = mock_poller_cls.return_value
        # Default poll simulates no messages
        mock_poller_instance.poll.return_value = {}

        yield {
            "context_cls": mock_context_cls, # yield class mock too
            "context": mock_context_instance,
            "main_socket": mock_main_socket,
            "heartbeat_socket": mock_heartbeat_socket,
            "poller": mock_poller_instance
        }

# Mock threading.Thread for SchedulerProcess
@pytest.fixture
def mock_threading_scheduler():
     with patch('hola.distributed.scheduler.threading.Thread', autospec=True) as mock_thread_cls:
        mock_thread_instance = mock_thread_cls.return_value
        # Prevent thread from actually starting
        mock_thread_instance.start = MagicMock()
        yield mock_thread_cls # yield the class mock

# Fixture for setup_logging specific to scheduler module
@pytest.fixture
def mock_setup_logging_fixture_scheduler():
    with patch('hola.distributed.scheduler.setup_logging', autospec=True) as mock_setup:
        mock_logger = MagicMock(spec=logging.Logger)
        mock_setup.return_value = mock_logger
        yield mock_logger, mock_setup # yield logger instance and setup mock

# Basic Initialization Test
def test_scheduler_process_init(mock_coordinator, mock_setup_logging_fixture_scheduler, mock_threading_scheduler):
    """Test SchedulerProcess initialization."""
    mock_logger, mock_setup = mock_setup_logging_fixture_scheduler
    mock_thread_cls = mock_threading_scheduler

    with patch('hola.distributed.scheduler.os.makedirs') as mock_makedirs, \
         patch.object(SchedulerProcess, 'save_readme') as mock_save_readme: # Mock save_readme

        scheduler = SchedulerProcess(mock_coordinator, save_dir="test_results")

        assert scheduler.coordinator is mock_coordinator
        assert scheduler.logger is mock_logger
        mock_setup.assert_called_once_with("Scheduler")
        assert scheduler.save_dir.startswith("test_results/run_")
        assert mock_makedirs.call_count >= 2 # Base dir and run dir
        mock_save_readme.assert_called_once()
        assert not scheduler.running
        # Check that the timeout thread was NOT started yet (only in run)
        mock_thread_cls.assert_not_called()

# Test receiving a heartbeat
def test_scheduler_run_heartbeat_known_worker(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler, # Provides dict with context, sockets, poller mocks
    mock_threading_scheduler
):
    """Test processing a heartbeat from a known worker."""
    mock_logger, _ = mock_setup_logging_fixture_scheduler
    mocks = mock_zmq_scheduler
    mock_thread_cls = mock_threading_scheduler

    # Prepare scheduler and add a known worker
    worker_id = 101
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'): # Mock irrelevant parts of init
        scheduler = SchedulerProcess(mock_coordinator)
        scheduler.logger = mock_logger # Assign mock logger
        # Add worker state manually
        known_worker_state = WorkerState(worker_id)
        scheduler.workers[worker_id] = known_worker_state
        initial_heartbeat_time = known_worker_state.last_heartbeat

    # Configure mocks for the run loop
    # 1. Poller returns the heartbeat socket as ready
    # 2. Heartbeat socket receives a HeartbeatRequest
    # 3. Poller raises an exception or returns {} repeatedly after first message
    heartbeat_req_msg = HeartbeatRequest(worker_id=worker_id)
    # Encode using msgpack, as that's likely what ZMQ uses
    encoded_heartbeat_req = msgspec.msgpack.encode(heartbeat_req_msg)

    # Simulate one message, then stop polling effectively
    mocks["poller"].poll.side_effect = [
        {mocks["heartbeat_socket"]: zmq.POLLIN}, # First poll: message ready
        {} # Subsequent polls: nothing ready (loop should check self.running)
    ]
    # We also need to ensure scheduler.running becomes False eventually
    # Patching `self.running = False` directly is tricky.
    # Instead, let's mock the shutdown logic to set it.
    # For this specific test, let's just stop after one iteration.
    # Modify the side effect to keep returning empty after the first call.
    poll_results = [{mocks["heartbeat_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results:
            return poll_results.pop(0)
        # After the first call returns the message, subsequent calls return empty dict
        # This still relies on the `while self.running` loop terminating some other way.
        # For a more robust test, we should mock a ShutdownRequest.
        # For now, let's limit the run duration or explicitly set running=False.
        scheduler.running = False # Force stop after first poll cycle completes
        return {}
    mocks["poller"].poll.side_effect = poll_side_effect

    mocks["heartbeat_socket"].recv.return_value = encoded_heartbeat_req

    # --- Run the scheduler briefly ---
    # Patch sleep and the background thread method
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts') as mock_check_timeouts, \
         patch.object(scheduler, 'save_coordinator_state') as mock_save_state:

        scheduler.run() # Should start, process one poll, then exit

    # --- Assertions ---
    # Check timeout thread started
    mock_thread_cls.assert_called_once_with(target=mock_check_timeouts)
    mock_thread_instance = mock_thread_cls.return_value
    mock_thread_instance.start.assert_called_once()

    # Check heartbeat socket interactions
    mocks["heartbeat_socket"].recv.assert_called_once()
    expected_response = HeartbeatResponse(success=True)
    # Expect response encoded with msgpack
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["heartbeat_socket"].send.assert_called_once_with(encoded_expected_response)

    # Check main socket was not used
    mocks["main_socket"].recv.assert_not_called()
    mocks["main_socket"].send.assert_not_called()

    # Check worker state was updated
    assert known_worker_state.last_heartbeat > initial_heartbeat_time

    # Check logging (optional, basic check)
    mock_logger.debug.assert_any_call(f"Received heartbeat from worker {worker_id}")

    # Check cleanup
    mock_save_state.assert_called_once() # Should save state on exit
    mocks["main_socket"].close.assert_called_once()
    mocks["heartbeat_socket"].close.assert_called_once()
    mocks["context"].term.assert_called_once()

# Test GetSuggestion from a new worker
def test_scheduler_run_get_suggestion_new_worker(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test processing GetSuggestionRequest from a new worker."""
    mock_logger, _ = mock_setup_logging_fixture_scheduler
    mocks = mock_zmq_scheduler
    mock_thread_cls = mock_threading_scheduler

    worker_id = 202
    suggestion = {"param_a": 10, "param_b": -5.5}

    # Configure coordinator mock
    mock_coordinator.suggest_parameters.return_value = ([suggestion], None)

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'): # Mock irrelevant parts of init
        scheduler = SchedulerProcess(mock_coordinator)
        scheduler.logger = mock_logger
        assert worker_id not in scheduler.workers # Ensure worker is new

    # Configure mocks for the run loop
    # 1. Poller returns the main socket as ready
    # 2. Main socket receives a GetSuggestionRequest
    # 3. Poller returns nothing (to stop the loop)
    get_suggestion_req_msg = GetSuggestionRequest(worker_id=worker_id)
    encoded_req = msgspec.msgpack.encode(get_suggestion_req_msg)

    # Define the side effect for polling
    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results:
            return poll_results.pop(0)
        scheduler.running = False # Stop loop after first message
        return {}
    mocks["poller"].poll.side_effect = poll_side_effect

    mocks["main_socket"].recv.return_value = encoded_req

    # --- Run the scheduler briefly ---
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts') as mock_check_timeouts, \
         patch.object(scheduler, 'save_coordinator_state') as mock_save_state:

        scheduler.run()

    # --- Assertions ---
    # Check timeout thread started
    mock_thread_cls.assert_called_once_with(target=mock_check_timeouts)
    mock_thread_instance = mock_thread_cls.return_value
    mock_thread_instance.start.assert_called_once()

    # Check main socket interactions
    mocks["main_socket"].recv.assert_called_once()
    mock_coordinator.suggest_parameters.assert_called_once_with(n_samples=1)
    expected_response = GetSuggestionResponse(parameters=suggestion)
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)

    # Check heartbeat socket was not used
    mocks["heartbeat_socket"].recv.assert_not_called()
    mocks["heartbeat_socket"].send.assert_not_called()

    # Check worker state was created and suggestion assigned
    assert worker_id in scheduler.workers
    new_worker_state = scheduler.workers[worker_id]
    assert isinstance(new_worker_state, WorkerState)
    assert new_worker_state.current_parameters == suggestion
    assert new_worker_state.start_time is not None

    # Check logging
    mock_logger.info.assert_any_call(f"First contact from worker {worker_id}")
    mock_logger.info.assert_any_call(f"New worker registered: {worker_id}. Total active workers: 1")

    # Check cleanup
    mock_save_state.assert_called_once()
    mocks["main_socket"].close.assert_called_once()
    mocks["heartbeat_socket"].close.assert_called_once()
    mocks["context"].term.assert_called_once()

# Test GetSuggestion from an existing worker
def test_scheduler_run_get_suggestion_existing_worker(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test processing GetSuggestionRequest from an existing worker."""
    mock_logger, _ = mock_setup_logging_fixture_scheduler
    mocks = mock_zmq_scheduler
    mock_thread_cls = mock_threading_scheduler

    worker_id = 303
    suggestion = {"p_exist": 42.0}

    # Configure coordinator mock
    mock_coordinator.suggest_parameters.return_value = ([suggestion], None)

    # Prepare scheduler and add the worker beforehand
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator)
        scheduler.logger = mock_logger
        existing_worker_state = WorkerState(worker_id)
        scheduler.workers[worker_id] = existing_worker_state
        # Mock the update_heartbeat method to check it's called
        existing_worker_state.update_heartbeat = MagicMock()

    # Configure mocks for the run loop
    get_suggestion_req_msg = GetSuggestionRequest(worker_id=worker_id)
    encoded_req = msgspec.msgpack.encode(get_suggestion_req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results:
            return poll_results.pop(0)
        scheduler.running = False
        return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(scheduler, 'save_coordinator_state'):
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    mock_coordinator.suggest_parameters.assert_called_once_with(n_samples=1)
    expected_response = GetSuggestionResponse(parameters=suggestion)
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)

    # Check worker state was updated (heartbeat and params)
    existing_worker_state.update_heartbeat.assert_called_once()
    assert existing_worker_state.current_parameters == suggestion
    assert existing_worker_state.start_time is not None
    # Check logs don't contain 'New worker registered' or 'First contact'
    for log_call in mock_logger.info.call_args_list:
        assert "New worker registered" not in log_call[0][0]
        assert "First contact from worker" not in log_call[0][0]

# Test GetSuggestion using the retry queue
def test_scheduler_run_get_suggestion_retry_queue(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test GetSuggestionRequest when retry queue has items."""
    mock_logger, _ = mock_setup_logging_fixture_scheduler
    mocks = mock_zmq_scheduler
    worker_id = 404
    retry_suggestion = {"retry_param": True}

    # Prepare scheduler with item in retry queue
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator)
        scheduler.logger = mock_logger
        scheduler.retry_queue.append(retry_suggestion)

    # Configure mocks for the run loop
    get_suggestion_req_msg = GetSuggestionRequest(worker_id=worker_id)
    encoded_req = msgspec.msgpack.encode(get_suggestion_req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results:
            return poll_results.pop(0)
        scheduler.running = False
        return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(scheduler, 'save_coordinator_state'):
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    # Coordinator should NOT be called for suggestions
    mock_coordinator.suggest_parameters.assert_not_called()
    # Response should contain the item from the retry queue
    expected_response = GetSuggestionResponse(parameters=retry_suggestion)
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)
    # Check retry queue is now empty
    assert not scheduler.retry_queue
    # Check worker state was created and assigned the retry suggestion
    assert worker_id in scheduler.workers
    assert scheduler.workers[worker_id].current_parameters == retry_suggestion
    mock_logger.info.assert_any_call(f"Assigning retry parameters to worker {worker_id}: {retry_suggestion}")

# Test GetSuggestion when coordinator has no more suggestions
def test_scheduler_run_get_suggestion_no_more(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test GetSuggestionRequest when coordinator returns no suggestions."""
    mocks = mock_zmq_scheduler
    worker_id = 505

    # Configure coordinator mock to return nothing
    mock_coordinator.suggest_parameters.return_value = ([], None)

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator)

    # Configure mocks for the run loop
    get_suggestion_req_msg = GetSuggestionRequest(worker_id=worker_id)
    encoded_req = msgspec.msgpack.encode(get_suggestion_req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results:
            return poll_results.pop(0)
        scheduler.running = False
        return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(scheduler, 'save_coordinator_state'):
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    mock_coordinator.suggest_parameters.assert_called_once_with(n_samples=1)
    # Response should indicate no parameters
    expected_response = GetSuggestionResponse(parameters=None)
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)
    # Check worker state was created but has no assigned parameters
    assert worker_id in scheduler.workers
    assert scheduler.workers[worker_id].current_parameters is None

# Test SubmitResult from a known worker
def test_scheduler_run_submit_result(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test processing SubmitResultRequest from a known worker."""
    mock_logger, _ = mock_setup_logging_fixture_scheduler
    mocks = mock_zmq_scheduler
    worker_id = 606
    params = {"p_submit": 1.0}
    objectives = {"o_submit": 100.0}
    result = Result(parameters=params, objectives=objectives)

    # Mock coordinator methods
    # Let record_evaluation return a mock trial object (or just None if not needed)
    mock_trial = MagicMock()
    mock_trial.trial_id = 999
    mock_coordinator.record_evaluation.return_value = mock_trial
    # Simulate this result becoming the new best
    mock_coordinator.get_best_trial.return_value = mock_trial
    mock_coordinator.get_total_evaluations.return_value = 5 # Assume not hitting save interval yet

    # Prepare scheduler and add worker state (simulating worker has a task)
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator, save_interval=10) # Explicit save interval
        scheduler.logger = mock_logger
        worker_state = WorkerState(worker_id)
        worker_state.assign_parameters(params) # Worker is processing this task
        # Mock update_heartbeat to check it
        worker_state.update_heartbeat = MagicMock()
        scheduler.workers[worker_id] = worker_state
        assert worker_state.current_parameters is not None # Pre-condition

    # Configure mocks for the run loop
    submit_req_msg = SubmitResultRequest(worker_id=worker_id, result=result)
    encoded_req = msgspec.msgpack.encode(submit_req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results:
            return poll_results.pop(0)
        scheduler.running = False
        return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(scheduler, 'save_coordinator_state') as mock_save_state: # Monitor save state call
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    # Check coordinator was called
    mock_coordinator.record_evaluation.assert_called_once_with(params, objectives, metadata={"source": "worker"})
    mock_coordinator.get_best_trial.assert_called_once()
    # Check response sent
    expected_response = SubmitResultResponse(success=True, is_best=True) # is_best=True based on mock
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)

    # Check worker state was cleared and heartbeat updated
    assert worker_state.current_parameters is None
    assert worker_state.retry_count == 0 # Should be reset
    worker_state.update_heartbeat.assert_called_once()

    # Check save_coordinator_state *was* called (due to final save on exit)
    mock_save_state.assert_called_once()

    # Check logging
    mock_logger.info.assert_any_call(
        f"Best trial found: objectives={objectives}, "
        f"total_evaluations={mock_coordinator.get_total_evaluations()}"
    )

# Test SubmitResult triggering save interval
def test_scheduler_run_submit_result_triggers_save(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test SubmitResultRequest triggers save_coordinator_state on interval."""
    mocks = mock_zmq_scheduler
    worker_id = 707
    params = {"p_save": 1.0}
    objectives = {"o_save": 200.0}
    result = Result(parameters=params, objectives=objectives)
    save_interval = 5
    # Set counts so the difference meets the interval
    eval_count_before = 0
    eval_count_after = 5 # (5 - 0 >= 5) is True

    # Mock coordinator methods
    mock_trial = MagicMock()
    mock_trial.trial_id = 1000
    mock_coordinator.record_evaluation.return_value = mock_trial
    mock_coordinator.get_best_trial.return_value = mock_trial
    # Simulate eval count reaching save interval
    mock_coordinator.get_total_evaluations.return_value = eval_count_after

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator, save_interval=save_interval)
        scheduler.last_save_count = eval_count_before # Set last save count
        worker_state = WorkerState(worker_id)
        worker_state.assign_parameters(params)
        scheduler.workers[worker_id] = worker_state

    # Configure mocks for the run loop
    submit_req_msg = SubmitResultRequest(worker_id=worker_id, result=result)
    encoded_req = msgspec.msgpack.encode(submit_req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results:
            return poll_results.pop(0)
        scheduler.running = False
        return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly, mocking save_coordinator_state
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(SchedulerProcess, 'save_coordinator_state') as mock_save_state: # Patch on class
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    mock_coordinator.record_evaluation.assert_called_once()
    expected_response = SubmitResultResponse(success=True, is_best=True)
    mocks["main_socket"].send.assert_called_once_with(msgspec.msgpack.encode(expected_response))

    # Check that save_coordinator_state was called *twice*
    # (once for interval, once for final save on exit)
    assert mock_save_state.call_count == 2
    # Check last_save_count was updated
    assert scheduler.last_save_count == eval_count_after

# Test ShutdownRequest
def test_scheduler_run_shutdown_request(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test processing ShutdownRequest."""
    mocks = mock_zmq_scheduler
    mock_thread_cls = mock_threading_scheduler

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator)

    # Configure mocks for the run loop
    shutdown_req_msg = ShutdownRequest()
    encoded_req = msgspec.msgpack.encode(shutdown_req_msg)

    # Poller returns main socket, then stops
    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results:
            return poll_results.pop(0)
        # Loop should terminate naturally because self.running becomes False
        return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly, mocking save_coordinator_state
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts') as mock_check_timeouts, \
         patch.object(SchedulerProcess, 'save_coordinator_state') as mock_save_state:

        scheduler.run() # Should process shutdown and exit

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    # Check response sent
    expected_response = SubmitResultResponse(success=True) # Shutdown response
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)

    # Check that running flag is now False (implicitly tested by loop termination)
    # assert not scheduler.running # Can't check reliably after process might exit

    # Check save_coordinator_state was called *twice*
    # (once *before* sending response in Shutdown handler, once for final save)
    assert mock_save_state.call_count == 2

    # Check timeout thread was started
    mock_thread_cls.assert_called_once_with(target=mock_check_timeouts)
    mock_thread_instance = mock_thread_cls.return_value
    mock_thread_instance.start.assert_called_once()

    # Check cleanup
    mocks["main_socket"].close.assert_called_once()
    mocks["heartbeat_socket"].close.assert_called_once()
    mocks["context"].term.assert_called_once()

# Test StatusRequest
def test_scheduler_run_status_request(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test processing StatusRequest."""
    mocks = mock_zmq_scheduler
    total_evals = 123
    best_objectives = {"obj": 1.0}

    # Configure coordinator mock
    mock_coordinator.get_total_evaluations.return_value = total_evals
    mock_best_trial = MagicMock()
    mock_best_trial.objectives = best_objectives
    mock_coordinator.get_best_trial.return_value = mock_best_trial

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator)

    # Configure mocks for the run loop
    req_msg = StatusRequest()
    encoded_req = msgspec.msgpack.encode(req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results: return poll_results.pop(0)
        scheduler.running = False; return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(SchedulerProcess, 'save_coordinator_state'):
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    mock_coordinator.get_total_evaluations.assert_called_once()
    mock_coordinator.get_best_trial.assert_called_once()
    expected_response = StatusResponse(
        active_workers=0, # No workers added in this test
        total_evaluations=total_evals,
        best_objectives=best_objectives
    )
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)

# Test GetTrialsRequest
def test_scheduler_run_get_trials_request(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test processing GetTrialsRequest."""
    mocks = mock_zmq_scheduler
    mock_trials_data = [
        {"trial_id": 1, "parameters": {"x": 1}, "objectives": {"y": 2}},
        {"trial_id": 2, "parameters": {"x": 3}, "objectives": {"y": 4}},
    ]

    # Mock the method that returns the DataFrame, make it return a list directly
    # to avoid needing pandas and mocking its methods.
    # This requires the tested code to handle list return, or adjust the mock.
    # The code currently calls: df.reset_index().to_dict(orient="records")
    # Let's mock the DataFrame and its methods instead.
    mock_df = MagicMock()
    mock_df.reset_index.return_value.to_dict.return_value = mock_trials_data
    mock_coordinator.get_trials_dataframe.return_value = mock_df
    mock_coordinator.get_all_trials_dataframe.return_value = mock_df # Mock both for simplicity

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator)

    # Configure mocks for the run loop (test ranked_only=True)
    req_msg = GetTrialsRequest(ranked_only=True)
    encoded_req = msgspec.msgpack.encode(req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results: return poll_results.pop(0)
        scheduler.running = False; return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(SchedulerProcess, 'save_coordinator_state'):
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    mock_coordinator.get_trials_dataframe.assert_called_once_with(ranked_only=True)
    mock_coordinator.get_all_trials_dataframe.assert_not_called()
    expected_response = GetTrialsResponse(trials=mock_trials_data)
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)

# Test GetMetadataRequest
def test_scheduler_run_get_metadata_request(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test processing GetMetadataRequest."""
    mocks = mock_zmq_scheduler
    trial_ids_req = [1, 3]
    # Mock the DataFrame and iterrows
    mock_metadata_df = MagicMock()
    # Simulate iterrows yielding index and data (as mocked Series/dict)
    mock_row1 = MagicMock()
    mock_row1.to_dict.return_value = {"colA": "val1", "colB": 10}
    mock_row2 = MagicMock()
    mock_row2.to_dict.return_value = {"colA": "val3", "colB": 30}
    mock_metadata_df.iterrows.return_value = [
        (1, mock_row1),
        (3, mock_row2),
    ]
    mock_coordinator.get_trials_metadata.return_value = mock_metadata_df

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator)

    # Configure mocks for the run loop
    req_msg = GetMetadataRequest(trial_ids=trial_ids_req)
    encoded_req = msgspec.msgpack.encode(req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results: return poll_results.pop(0)
        scheduler.running = False; return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(SchedulerProcess, 'save_coordinator_state'):
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    mock_coordinator.get_trials_metadata.assert_called_once_with(trial_ids=trial_ids_req)
    # Check the constructed list (includes trial_id)
    expected_metadata = [
        {"colA": "val1", "colB": 10, "trial_id": 1},
        {"colA": "val3", "colB": 30, "trial_id": 3},
    ]
    expected_response = GetMetadataResponse(metadata=expected_metadata)
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)

# Test GetTopKRequest
def test_scheduler_run_get_top_k_request(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test processing GetTopKRequest."""
    mocks = mock_zmq_scheduler
    k = 2
    # Mock Trial objects with necessary attributes
    mock_trial1 = MagicMock()
    mock_trial1.trial_id = 10
    mock_trial1.parameters = {"x":1}
    mock_trial1.objectives = {"y":11}
    mock_trial1.is_feasible = True
    mock_trial2 = MagicMock()
    mock_trial2.trial_id = 5
    mock_trial2.parameters = {"x":2}
    mock_trial2.objectives = {"y":5}
    mock_trial2.is_feasible = False # Test with infeasible too
    mock_top_trials = [mock_trial1, mock_trial2]
    mock_coordinator.get_top_k_trials.return_value = mock_top_trials

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator)

    # Configure mocks for the run loop
    req_msg = GetTopKRequest(k=k)
    encoded_req = msgspec.msgpack.encode(req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results: return poll_results.pop(0)
        scheduler.running = False; return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(SchedulerProcess, 'save_coordinator_state'):
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    mock_coordinator.get_top_k_trials.assert_called_once_with(k=k)
    # Check the constructed list of dicts
    expected_trials_dict = [
        {"trial_id": 10, "parameters": {"x":1}, "objectives": {"y":11}, "is_feasible": True},
        {"trial_id": 5, "parameters": {"x":2}, "objectives": {"y":5}, "is_feasible": False},
    ]
    expected_response = GetTopKResponse(trials=expected_trials_dict)
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)

# Test IsMultiGroupRequest
def test_scheduler_run_is_multi_group_request(
    mock_coordinator,
    mock_setup_logging_fixture_scheduler,
    mock_zmq_scheduler,
    mock_threading_scheduler
):
    """Test processing IsMultiGroupRequest."""
    mocks = mock_zmq_scheduler
    is_multi = True
    mock_coordinator.is_multi_group.return_value = is_multi

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(mock_coordinator)

    # Configure mocks for the run loop
    req_msg = IsMultiGroupRequest()
    encoded_req = msgspec.msgpack.encode(req_msg)

    poll_results = [{mocks["main_socket"]: zmq.POLLIN}]
    def poll_side_effect(*args, **kwargs):
        if poll_results: return poll_results.pop(0)
        scheduler.running = False; return {}
    mocks["poller"].poll.side_effect = poll_side_effect
    mocks["main_socket"].recv.return_value = encoded_req

    # Run the scheduler briefly
    with patch('hola.distributed.scheduler.time.sleep'), \
         patch.object(scheduler, 'check_worker_timeouts'), \
         patch.object(SchedulerProcess, 'save_coordinator_state'):
        scheduler.run()

    # Assertions
    mocks["main_socket"].recv.assert_called_once()
    mock_coordinator.is_multi_group.assert_called_once()
    expected_response = IsMultiGroupResponse(is_multi_group=is_multi)
    encoded_expected_response = msgspec.msgpack.encode(expected_response)
    mocks["main_socket"].send.assert_called_once_with(encoded_expected_response)

# Test check_worker_timeouts logic
def test_scheduler_check_worker_timeouts(
    mock_coordinator, # Needed for SchedulerProcess init
    mock_setup_logging_fixture_scheduler
):
    """Test the check_worker_timeouts method directly."""
    mock_logger, _ = mock_setup_logging_fixture_scheduler
    timeout_seconds = 60.0
    max_retries = 2
    base_mock_time = time.time() # Get a base time

    # Prepare scheduler
    with patch('hola.distributed.scheduler.os.makedirs'), \
         patch.object(SchedulerProcess, 'save_readme'):
        scheduler = SchedulerProcess(
            mock_coordinator,
            worker_timeout_seconds=timeout_seconds,
            max_retries=max_retries
        )
        scheduler.logger = mock_logger

    # --- Create Worker States and Assign ---
    worker_ok_id = 1
    worker_timed_out_retry_id = 2
    worker_timed_out_max_retry_id = 3
    worker_timed_out_no_params_id = 4

    params_to_retry = {"retry": 1}
    params_max_retry = {"max": 2}

    # Use a consistent mock time for state creation and checking
    current_mock_time = base_mock_time + timeout_seconds * 2
    with patch('hola.distributed.scheduler.time.time', return_value=current_mock_time):
        # Create states using the current mocked time
        state_ok = WorkerState(worker_ok_id)
        state_ok.assign_parameters({"ok": 0})
        state_ok.last_heartbeat = current_mock_time - (timeout_seconds / 2) # Keep OK worker recent

        state_retry = WorkerState(worker_timed_out_retry_id)
        state_retry.assign_parameters(params_to_retry)
        state_retry.retry_count = 1 # Below max_retries
        state_retry.last_heartbeat = current_mock_time - (timeout_seconds * 1.5) # Make timed out

        state_max_retry = WorkerState(worker_timed_out_max_retry_id)
        state_max_retry.assign_parameters(params_max_retry)
        state_max_retry.retry_count = max_retries # At max_retries
        state_max_retry.last_heartbeat = current_mock_time - (timeout_seconds * 1.5) # Make timed out

        state_no_params = WorkerState(worker_timed_out_no_params_id)
        state_no_params.assign_parameters({"temp":1}) # Assign then clear
        state_no_params.current_parameters = None
        state_no_params.last_heartbeat = current_mock_time - (timeout_seconds * 1.5) # Make timed out

        scheduler.workers = {
            worker_ok_id: state_ok,
            worker_timed_out_retry_id: state_retry,
            worker_timed_out_max_retry_id: state_max_retry,
            worker_timed_out_no_params_id: state_no_params,
        }

        # --- Call the method directly (still within the time patch) ---
        scheduler.check_worker_timeouts()

    # --- Assertions ---
    # Check retry queue
    assert scheduler.retry_queue == [params_to_retry]

    # Check remaining workers
    assert list(scheduler.workers.keys()) == [worker_ok_id]
    assert worker_ok_id in scheduler.workers
    assert worker_timed_out_retry_id not in scheduler.workers
    assert worker_timed_out_max_retry_id not in scheduler.workers
    assert worker_timed_out_no_params_id not in scheduler.workers

    # Check logs (use assert_any_call for flexibility)
    mock_logger.info.assert_any_call(
        f"Worker {worker_timed_out_retry_id} timed out. Queueing parameters for retry: {params_to_retry}"
    )
    mock_logger.warning.assert_any_call(
        f"Parameters from worker {worker_timed_out_max_retry_id} exceeded max retries: {params_max_retry}"
    )
    mock_logger.info.assert_any_call(f"Removing timed out worker: {worker_timed_out_retry_id}")
    mock_logger.info.assert_any_call(f"Removing timed out worker: {worker_timed_out_max_retry_id}")
    mock_logger.info.assert_any_call(f"Removing timed out worker: {worker_timed_out_no_params_id}")


# TODO: Add tests for error handling in request processing
# TODO: Add tests for check_worker_timeouts logic (INCOMPLETE - MOCKING ISSUES)
# TODO: Add tests for save_coordinator_state file operations
# TODO: Add more comprehensive tests for SchedulerProcess: