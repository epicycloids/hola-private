# HOLA Optimization Monitor

The HOLA Optimization Monitor is a Streamlit-based dashboard that provides real-time monitoring of distributed optimization jobs. It connects to a running OptimizationServer via ZMQ and displays information about the optimization progress, active workers, and visualization of optimization results.

## Features

- **Real-time monitoring** of optimization progress
- **Worker status** tracking showing active workers and their current jobs
- **Job tracking** showing parameters being evaluated and job status
- **Optimization metrics** including total trials, feasible trials, and best trial information
- **Interactive visualizations** of objective values and parameter spaces
- **Pareto front visualization** for multi-objective optimization

## Usage

### Running with the Example Script

The easiest way to get started is to use the provided example script:

```bash
python -m hola.distributed.zmq_monitor_example --workers 4 --evals 50
```

This will:
1. Start an OptimizationServer
2. Launch 4 worker processes
3. Start the monitoring dashboard
4. Automatically open your browser to view the dashboard

### Command-line Options

The example script supports the following options:

- `--workers N`: Set the number of worker processes (default: 4)
- `--evals N`: Set the maximum number of evaluations (default: 50)
- `--no-browser`: Don't automatically open the browser

### Running the Monitor Standalone

You can also run the monitor standalone to connect to an existing optimization server:

```bash
streamlit run hola/distributed/monitor.py -- --zmq_endpoint tcp://localhost:5555
```

Replace `tcp://localhost:5555` with the appropriate ZMQ endpoint for your server.

## Dashboard Features

### Connection Settings

The dashboard's sidebar allows you to:
- Specify the server ZMQ endpoint
- Connect to the server
- Configure auto-refresh settings
- Adjust display limits

### Dashboard Tabs

The dashboard is organized into four tabs:

1. **Overview**: Shows server status, uptime, connected workers, and overall progress
2. **Workers & Jobs**: Displays worker status and active job information
3. **Optimization Results**: Shows optimization metrics, best trial found, and recent trial data
4. **Visualizations**: Provides interactive visualizations of objectives and parameters

## Integration with Your Application

To integrate the monitor with your own distributed optimization application:

1. Make sure your OptimizationServer is configured with a ZMQ TCP endpoint:
   ```python
   server_config = ServerConfig(
       zmq_tcp_endpoint="tcp://127.0.0.1:5555",
       # other configuration options...
   )
   server = OptimizationServer(scheduler, config=server_config)
   ```

2. Start the monitor in a separate process or terminal:
   ```python
   from multiprocessing import Process
   from hola.distributed.monitor import monitor_process

   # Start monitor in separate process
   monitor_proc = Process(
       target=monitor_process,
       args=("tcp://127.0.0.1:5555", True)  # endpoint, open_browser
   )
   monitor_proc.start()
   ```

   Or run from the command line:
   ```bash
   streamlit run hola/distributed/monitor.py -- --zmq_endpoint tcp://127.0.0.1:5555
   ```

## Requirements

The monitor requires:
- ZMQ
- Streamlit
- Pandas
- Plotly
- NumPy

These are typically installed with the HOLA package.