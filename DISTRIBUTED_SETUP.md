# Distributed Optimization Setup

This guide explains how to run the optimization server on one machine and connect to it from other machines on the same network.

## Server Setup

1. Make sure you have all the required dependencies installed:
   ```
   poetry install
   ```

2. Run the optimization server:
   ```
   poetry run python test.py
   ```

   This starts:
   - The ZMQ scheduler on ports 5555 and 5556
   - The HTTP REST API server on port 8000
   - Local worker processes on the same machine

3. The server will be available to all machines on your network at the server's IP address.

## Remote Worker Options

There are two ways to connect remote machines to the optimization server:

### Option 1: ZMQ Remote Worker (Recommended)

This method connects directly to the ZMQ scheduler using the same protocol as local workers, providing better performance and more robust communication.

1. On the remote machine, install the required dependencies:
   ```
   pip install zmq msgspec
   ```

2. Copy the `remote_worker.py` script to the remote machine.

3. Run the remote worker script, replacing `SERVER_IP` with the IP address of the server machine:
   ```
   python remote_worker.py --host SERVER_IP
   ```

4. Additional options:
   ```
   --port PORT              Server main port number (default: 5555)
   --heartbeat-port PORT    Server heartbeat port number (default: 5556)
   --worker-id ID           Specify a worker ID (default: random integer)
   ```

### Option 2: REST API Client

This method uses the HTTP REST API for communication, which is simpler but may have higher overhead.

1. On the client machine, install the required dependencies:
   ```
   pip install requests
   ```

2. Copy the `rest_client.py` script to the client machine.

3. Run the client script, replacing `SERVER_IP` with the IP address of the server machine:
   ```
   python rest_client.py --host SERVER_IP --port 8000
   ```

## Customizing the Worker

You can modify either script to use your own evaluation function:

### For ZMQ Remote Worker:

Modify the `example_evaluation_fn` in `remote_worker.py`:

```python
def your_evaluation_fn(x: float, y: float, ...other params...) -> dict[str, float]:
    # Your evaluation code here
    return {
        "objective1": value1,
        "objective2": value2,
        # ... other objectives
    }

# Then update the worker creation:
worker = RemoteWorker(
    worker_id=worker_id,
    evaluation_fn=your_evaluation_fn,  # Use your function here
    host=args.host,
    main_port=args.port,
    heartbeat_port=args.heartbeat_port
)
```

### For REST API Client:

Modify the evaluation code in `rest_client.py`:

```python
# Replace the objective calculations with your evaluation code:
objectives = your_evaluation_function(params)
```

## Security Considerations

This setup is intended for use on trusted local networks only. The server exposes endpoints without authentication, so it should not be exposed to the public internet without adding proper security measures.

## Viewing Results

The server saves optimization results in the `optimization_results` directory. Each run creates a timestamped subdirectory with:

- `coordinator_state.json`: The complete optimizer state
- `README.txt`: Information about the optimization run