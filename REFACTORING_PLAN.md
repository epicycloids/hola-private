# Refactoring Plan: test.py -> hola/distributed

This plan outlines the steps to refactor the distributed components from `test.py` into a dedicated `hola/distributed` package and create a runnable example script.

- [x] **1. Create Directory Structure:**
    - **Goal:** Establish the necessary directories for the new package structure.
    - Create `hola/distributed/` for the distributed system components (scheduler, worker, server, messages, utils).
    - Create `examples/` at the root level for the runnable example script (`run_distributed.py`).
    ```
    hola/
    ├── core/
    │   └── ... (existing files)
    ├── distributed/      <-- NEW
    │   ├── __init__.py
    │   ├── messages.py
    │   ├── scheduler.py
    │   ├── worker.py
    │   ├── server.py
    │   └── utils.py
    ├── __init__.py
    └── ...
    examples/             <-- NEW
    └── run_distributed.py
    optimization_results/
    logs/
    README.md
    REFACTORING_PLAN.md
    test.py               <-- To be deleted later
    ```

- [x] **2. Move Message Definitions:**
    - **Goal:** Centralize all message structures (`msgspec.Struct`) used for communication.
    - Create `hola/distributed/messages.py`.
    - Move all `msgspec.Struct` definitions from `test.py` into this file. This includes:
        - ZMQ core types: `ParameterSet`, `Result`.
        - ZMQ request types: `GetSuggestionRequest`, `SubmitResultRequest`, `HeartbeatRequest`, `ShutdownRequest`, `StatusRequest`, `GetTrialsRequest`, `GetMetadataRequest`, `GetTopKRequest`, `IsMultiGroupRequest`.
        - ZMQ response types: `GetSuggestionResponse`, `SubmitResultResponse`, `HeartbeatResponse`, `StatusResponse`, `GetTrialsResponse`, `GetMetadataResponse`, `GetTopKResponse`, `IsMultiGroupResponse`.
        - REST API types: `RESTGetSuggestionResponse`, `RESTSubmitResult`, `RESTSubmitResponse`, `RESTHeartbeatRequest`, `RESTHeartbeatResponse`, `RESTGetTrialsResponse`, `RESTGetMetadataResponse`, `RESTGetTopKResponse`, `RESTIsMultiGroupResponse`.
    - Move the `Message` union type definition (`Message = GetSuggestionRequest | ...`) into this file.
    - Ensure necessary imports are added at the top of `messages.py`: `msgspec`, `typing` (`Any`, `List`, `Dict`, `Optional`, `Union`), and potentially `hola.core.parameters.ParameterName`, `hola.core.objectives.ObjectiveName`.

- [x] **3. Move Utility Functions:**
    - **Goal:** Isolate general utility functions used by the distributed components.
    - Create `hola/distributed/utils.py`.
    - Move the `setup_logging` function from `test.py` into this file.
    - Ensure necessary imports are added at the top of `utils.py`: `logging`, `sys`, `os`, `datetime`.

- [x] **4. Move Scheduler Logic:**
    - **Goal:** Encapsulate the central scheduler logic in its own module.
    - Create `hola/distributed/scheduler.py`.
    - Move the `WorkerState` class from `test.py` into this file.
    - Move the `SchedulerProcess` class from `test.py` into this file.
    - Update imports at the top of `scheduler.py`:
        - Standard libraries: `logging`, `threading`, `time`, `os`, `msgspec`, `zmq`, `datetime`, `typing` (`Any`, `Callable`, `List`, `Dict`, `Optional`, `Union`).
        - Internal package imports: `from .messages import ...` (needed message types), `from .utils import setup_logging`, `from hola.core.coordinator import OptimizationCoordinator`, `from hola.core.parameters import ParameterName`, `from hola.core.objectives import ObjectiveName`, `from .messages import Result`.

- [x] **5. Move Worker Logic:**
    - **Goal:** Isolate the worker implementation.
    - Create `hola/distributed/worker.py`.
    - Move the `LocalWorker` class from `test.py` into this file.
    - Update imports at the top of `worker.py`:
        - Standard libraries: `logging`, `threading`, `time`, `msgspec`, `zmq`, `typing` (`Any`, `Callable`, `Dict`, `Optional`).
        - Internal package imports: `from .messages import ...` (needed message types), `from .utils import setup_logging`, `from hola.core.objectives import ObjectiveName`, `from hola.core.parameters import ParameterName`, `from .messages import Result`.

- [x] **6. Move Server Logic:**
    - **Goal:** Separate the REST API server implementation.
    - Create `hola/distributed/server.py`.
    - Move the `Server` class from `test.py` into this file.
    - Update imports at the top of `server.py`:
        - Standard libraries: `logging`, `threading`, `time`, `msgspec`, `zmq`, `uvicorn`, `fastapi` (`FastAPI`, `Request`), `typing` (`Any`, `List`, `Dict`, `Optional`).
        - Internal package imports: `from .messages import ...` (both ZMQ and REST message types needed), `from .utils import setup_logging`.

- [x] **7. Create Example Script:**
    - **Goal:** Provide a standalone script demonstrating how to use the distributed components.
    - Create `examples/run_distributed.py`.
    - Copy the entire `if __name__ == "__main__":` block content from `test.py` into this new file.
    - Move the helper functions `spawn_local_worker` and `shutdown_system` from `test.py` into `examples/run_distributed.py` (as they are specific to setting up and running this example).
    - Update all imports within `examples/run_distributed.py` to point to the new module locations:
        - `from hola.core.coordinator import OptimizationCoordinator`
        - `from hola.core.samplers import SobolSampler, ClippedGaussianMixtureSampler, ExploreExploitSampler`
        - `from hola.distributed.scheduler import SchedulerProcess`
        - `from hola.distributed.server import Server`
        - `from hola.distributed.worker import LocalWorker`
        - `from hola.distributed.utils import setup_logging`
        - `from hola.distributed.messages import StatusRequest, Message, SubmitResultResponse, ShutdownRequest, ...` (add any other message types needed by the moved functions)
        - Standard libraries: `multiprocessing as mp`, `os`, `random`, `time`, `zmq`, `msgspec`, `numpy as np`.
        - `from typing import Callable, Dict` (for type hints in moved functions).

- [x] **8. Create `__init__.py` files:**
    - **Goal:** Ensure the new directories are treated as Python packages.
    - Create an empty file `hola/distributed/__init__.py`.
    - Verify that `hola/core/__init__.py` and `hola/__init__.py` already exist (they likely do).

- [ ] **9. Refinement and Cleanup (Post-Move):**
    - **Goal:** Address remaining inconsistencies, potential improvements, and documentation after the structural changes are complete.

    **Phase 1: Testing Foundation (Partially Complete)**
    - [x] **9.1 Setup Testing Framework:**
        - Chosen: `pytest`.
        - Structure: `tests/distributed/` created.
        - Configured via `pyproject.toml`.
    - [x] **9.2 Test `messages.py`:**
        - Basic serialization/deserialization tests added and passing.
        - TODO: Add tests for edge cases if necessary.
    - [x] **9.3 Test `utils.py` (`setup_logging`):**
        - Tests added using mocking, passing.
    - [x] **9.4 Test `WorkerState` (`scheduler.py`):**
        - Tests added for initialization and methods, passing.
    - [ ] **9.5 Test `LocalWorker` (`worker.py`): POSTPONED**
        - Initial tests written but commented out due to complex mocking issues (threading/ZMQ interaction).
        - **Remaining Work:** Revisit mocking strategy, implement tests for:
            - Basic run loop success path.
            - Loop termination (no suggestions).
            - Error handling (ZMQ errors, eval function errors, max errors).
            - `send_heartbeats` logic.
            - `evaluate_parameters` edge cases.
    - [ ] **9.6 Test `SchedulerProcess` (`scheduler.py`): PARTIALLY COMPLETE**
        - Tests added and passing for:
            - Initialization.
            - Basic message handling (`Heartbeat`, `GetSuggestion`, `SubmitResult`, `Shutdown`, `Status`, `GetTrials`, `GetMetadata`, `GetTopK`, `IsMultiGroup`).
        - **Remaining Work:**
            - Test error handling during message processing (e.g., decode errors, coordinator errors).
            - Test `check_worker_timeouts` thoroughly (POSTPONED due to mocking difficulty).
            - Test `save_coordinator_state` file interactions.
    - [x] **9.7 Test `Server` (`server.py`):**
        - Tests added for all endpoints using `TestClient`, passing.
        - **Remaining Work:** Add tests for error conditions (e.g., ZMQ errors returned to client, bad request data variations).

    **Phase 2: Code Style and Readability (Not Started)**
    - [ ] **9.8 Docstring Style and Content:**
        - Review/update docstrings in `hola/distributed/*` for reST compliance (`:param:`, `:return:`, etc.).
        - Add module-level docstrings.
    - [ ] **9.9 Type Hinting:**
        - Review/improve type hints in `hola/distributed/*`.
    - [ ] **9.10 Code Formatting and Linting:**
        - Run `black`, `isort`, `flake8` on `hola/distributed/*`.
    - [ ] **9.10a Remove Temporary Inline Comments:**
        - Remove developer comments added during refactoring.

    **Phase 3: Configuration and API (Not Started)**
    - [ ] **9.11 Configuration Management:**
        - Refactor hardcoded values (ZMQ addresses, ports, timeouts, etc.) into config objects.
    - [ ] **9.12 Convenience Entrypoint (`run_local_distributed`):**
        - Implement the simplified local runner function.
    - [ ] **9.13 API Review (REST vs ZMQ Messages):**
        - Evaluate potential for reducing message struct duplication.
    - [ ] **9.14 Error Handling Review:**
        - Perform a holistic review of error handling and logging consistency.

    **Phase 4: Final Checks (Not Started)**
    - [ ] **9.15 Final Import Check:**
        - Verify all imports after refinements.
    - [ ] **9.16 Example Script Update:**
        - Update `examples/run_distributed.py` to use configuration objects, etc.

- [ ] **10. Delete Original File:**
    - **Goal:** Remove the old monolithic file once the refactoring is complete and verified.
    - Delete `test.py` after addressing remaining TODOs.