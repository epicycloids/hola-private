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

- [ ] **4. Move Scheduler Logic:**
    - **Goal:** Encapsulate the central scheduler logic in its own module.
    - Create `hola/distributed/scheduler.py`.
    - Move the `WorkerState` class from `test.py` into this file.
    - Move the `SchedulerProcess` class from `test.py` into this file.
    - Update imports at the top of `scheduler.py`:
        - Standard libraries: `logging`, `threading`, `time`, `os`, `msgspec`, `zmq`, `datetime`, `typing` (`Any`, `Callable`, `List`, `Dict`, `Optional`, `Union`).
        - Internal package imports: `from .messages import ...` (needed message types), `from .utils import setup_logging`, `from hola.core.coordinator import OptimizationCoordinator`, `from hola.core.parameters import ParameterName`, `from hola.core.objectives import ObjectiveName`, `from .messages import Result`.

- [ ] **5. Move Worker Logic:**
    - **Goal:** Isolate the worker implementation.
    - Create `hola/distributed/worker.py`.
    - Move the `LocalWorker` class from `test.py` into this file.
    - Update imports at the top of `worker.py`:
        - Standard libraries: `logging`, `threading`, `time`, `msgspec`, `zmq`, `typing` (`Any`, `Callable`, `Dict`, `Optional`).
        - Internal package imports: `from .messages import ...` (needed message types), `from .utils import setup_logging`, `from hola.core.objectives import ObjectiveName`, `from hola.core.parameters import ParameterName`, `from .messages import Result`.

- [ ] **6. Move Server Logic:**
    - **Goal:** Separate the REST API server implementation.
    - Create `hola/distributed/server.py`.
    - Move the `Server` class from `test.py` into this file.
    - Update imports at the top of `server.py`:
        - Standard libraries: `logging`, `threading`, `time`, `msgspec`, `zmq`, `uvicorn`, `fastapi` (`FastAPI`, `Request`), `typing` (`Any`, `List`, `Dict`, `Optional`).
        - Internal package imports: `from .messages import ...` (both ZMQ and REST message types needed), `from .utils import setup_logging`.

- [ ] **7. Create Example Script:**
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
    - **Unit Testing:** Implement comprehensive unit tests for the new distributed components (`scheduler`, `worker`, `server`, `messages`, `utils`) to ensure correctness and prevent regressions.
    - **Docstring Style:** Ensure all docstrings in the new modules adhere to the reStructuredText (reST) format compatible with Sphinx, matching the style of the existing `hola.core` modules.
    - **Convenience Entrypoint:** Create a high-level function (e.g., `run_local_distributed`) that simplifies running an optimization locally using the distributed components. This function should accept the objective function, parameter/objective configuration, number of workers, and manage the setup/teardown of the scheduler and local workers.
    - **Redundancy (REST vs ZMQ Messages):** Review the duplicated message definitions in `messages.py`. Consider if REST responses can directly use or adapt ZMQ structs to reduce code.
    - **Worker ID for REST:** Evaluate the temporary negative `worker_id` assignment in `Server`. Consider if a more robust UUID or registration mechanism is needed.
    - **Configuration:** Identify hardcoded values (ZMQ addresses, ports, timeouts, save intervals) in `scheduler.py`, `worker.py`, `server.py`, and `run_distributed.py`. Plan to move these into configuration objects (e.g., dataclasses, Pydantic models) for better flexibility.
    - **Error Handling:** Review error handling loops (like `MAX_CONSECUTIVE_ERRORS`) for consistency and robustness.
    - **Imports:** Perform a final check of all relative and absolute imports across all modified and new files.
    - **Docstrings (Module/Class Level):** Add module-level docstrings to the new files (`messages.py`, `scheduler.py`, etc.). Review and update existing docstrings for classes and functions that were moved.

- [ ] **10. Delete Original File:**
    - **Goal:** Remove the old monolithic file once the refactoring is complete and verified.
    - After successfully moving all components and verifying the example script runs correctly, delete the original `test.py` file.