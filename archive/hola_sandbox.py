import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import zmq
import zmq.asyncio
from uuid import uuid4
import numpy as np
from multiprocessing import Lock
import psutil
from collections import deque
from datetime import datetime, timezone
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class Evaluation:
    parameters: Dict
    objectives: Dict
    timestamp: float
    metadata: Optional[Dict] = None

class ThreadSafeParetoCube:
    def __init__(self, groups: List[str]):
        self._lock = Lock()
        self.groups = groups
        self.evaluations = []

    def add(self, eval: Evaluation) -> None:
        with self._lock:
            self.evaluations.append(eval)
            # Sort by first objective just for demo
            self.evaluations.sort(key=lambda x: x.objectives['obj1'])

    def get_best(self, n: int = 1) -> List[Evaluation]:
        with self._lock:
            return self.evaluations[:n]

class SimpleSampler:
    def __init__(self, bounds: Dict[str, tuple]):
        self.bounds = bounds

    def sample(self, n_samples: int = 1) -> List[Dict]:
        samples = []
        for _ in range(n_samples):
            sample = {
                param: np.random.uniform(low, high)
                for param, (low, high) in self.bounds.items()
            }
            samples.append(sample)
        return samples

class OptimizationServer:
    def __init__(self, transport: str = "tcp", host: str = "*", port: int = 5555):
        # Core optimization state
        self.evaluations = ThreadSafeParetoCube(groups=["default"])
        self.sampler = SimpleSampler({
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0)
        })

        # Setup async ZMQ communication
        self.context = zmq.asyncio.Context()
        self.router = self.context.socket(zmq.ROUTER)

        if transport == "tcp":
            self.router.bind(f"tcp://{host}:{port}")
        elif transport == "ipc":
            self.router.bind("ipc:///tmp/hyperopt")

        self.worker_stats = {}
        self.worker_timeseries = {}  # Store metrics over time per worker
        self.start_time = datetime.now(timezone.utc)

        print(f"Server started on {transport}://{host}:{port}")

    def update_worker_stats(self, worker_id, stats):
        """Add debug prints"""
        print(f"Updating stats for worker {worker_id}")
        print(f"Stats: {stats}")

        if worker_id not in self.worker_timeseries:
            self.worker_timeseries[worker_id] = WorkerTimeSeriesMetrics()

        ts = self.worker_timeseries[worker_id]
        ts.timestamps.append(datetime.fromisoformat(stats['timestamp']))
        ts.cpu_percents.append(stats['cpu_percent'])
        ts.utilizations.append(stats['metrics'].get('utilization', 0))
        ts.states.append(stats['metrics'].get('current_state', 'unknown'))
        ts.eval_times.append(stats['metrics'].get('avg_eval_time', 0))
        ts.wait_times.append(stats['metrics'].get('avg_wait_time', 0))

    def plot_worker_metrics(self, output_file: str = "worker_metrics.html"):
        """Create interactive plotly visualization of worker metrics"""
        print("\nPlotting metrics:")
        for worker_id, metrics in self.worker_timeseries.items():
            print(f"\nWorker {worker_id}:")
            print(f"Number of datapoints: {len(metrics.timestamps)}")
            print(f"CPU percents: {metrics.cpu_percents[:5]}...")  # First 5 values
            print(f"Utilizations: {metrics.utilizations[:5]}...")

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('CPU Usage %', 'Worker Utilization %', 'Evaluation vs Wait Times'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Some nice colors

        # Plot metrics for each worker
        for i, (worker_id, metrics) in enumerate(self.worker_timeseries.items()):
            worker_name = worker_id.decode()[:8]
            color = colors[i % len(colors)]

            # Convert timestamps to relative seconds from start
            rel_times = [(t - self.start_time).total_seconds() for t in metrics.timestamps]

            # CPU Usage
            fig.add_trace(
                go.Scatter(
                    x=rel_times,
                    y=metrics.cpu_percents,
                    name=f'Worker {worker_name} CPU',
                    line_color=color
                ),
                row=1, col=1
            )

            # Utilization
            fig.add_trace(
                go.Scatter(
                    x=rel_times,
                    y=[u * 100 for u in metrics.utilizations],
                    name=f'Worker {worker_name} Utilization',
                    line_color=color
                ),
                row=2, col=1
            )

            # Eval vs Wait times
            fig.add_trace(
                go.Scatter(
                    x=rel_times,
                    y=metrics.eval_times,
                    name=f'Worker {worker_name} Eval Time',
                    line_color=color
                ),
                row=3, col=1
            )

            # Wait times with dotted line
            rgba_color = f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.5,)}'
            fig.add_trace(
                go.Scatter(
                    x=rel_times,
                    y=metrics.wait_times,
                    name=f'Worker {worker_name} Wait Time',
                    line_color=rgba_color,
                    line_dash='dot'  # Changed this line
                ),
                row=3, col=1
            )

        # Update layout
        fig.update_layout(
            height=900,
            title_text="Worker Metrics Over Time",
            showlegend=True,
            xaxis3_title="Time (seconds)",
            yaxis_title="CPU %",
            yaxis2_title="Utilization %",
            yaxis3_title="Time (seconds)"
        )

        # Save interactive HTML
        fig.write_html(output_file)

    async def serve(self):
        """Asynchronous server loop"""
        while True:
            try:
                identity, _, message = await self.router.recv_multipart()
                message = json.loads(message)

                if 'worker_stats' in message:
                    self.worker_stats[identity] = message['worker_stats']
                    self.update_worker_stats(identity, message['worker_stats'])

                if message["type"] == "get_samples":
                    samples = self.sampler.sample(message["n_samples"])
                    response = {"samples": samples}
                    await self.router.send_multipart([
                        identity,
                        b"",
                        json.dumps(response).encode()
                    ])

                elif message["type"] == "add_evaluation":
                    eval = Evaluation(**message["evaluation"])
                    self.evaluations.add(eval)
                    print(f"Got evaluation: {eval}")
                    response = {"status": "ok"}
                    await self.router.send_multipart([
                        identity,
                        b"",
                        json.dumps(response).encode()
                    ])
            except Exception as e:
                print(f"Server error: {e}")

    async def get_optimization_stats(self):
        stats = {
            'n_workers': len(self.worker_stats),
            'worker_stats': self.worker_stats,
            'avg_utilization': np.mean([
                stats['metrics'].get('utilization', 0)
                for stats in self.worker_stats.values()
            ]),
            'total_evaluations': len(self.evaluations.evaluations),
            'evaluations_per_second': len(self.evaluations.evaluations) /
                (time.time() - self.start_time)
        }
        return stats

class WorkerMetrics:
    def __init__(self, window_size: int = 100):
        self.eval_times = deque(maxlen=window_size)
        self.wait_times = deque(maxlen=window_size)
        self.last_eval_end = None
        self.current_state = "idle"

    def start_evaluation(self):
        now = time.time()
        self.current_state = "evaluating"
        if self.last_eval_end is not None:
            self.wait_times.append(now - self.last_eval_end)

    def end_evaluation(self):
        now = time.time()
        self.current_state = "idle"
        self.last_eval_end = now

    def get_stats(self):
        if not self.eval_times or not self.wait_times:
            return {}

        return {
            'avg_eval_time': sum(self.eval_times) / len(self.eval_times),
            'avg_wait_time': sum(self.wait_times) / len(self.wait_times),
            'utilization': sum(self.eval_times) / (sum(self.eval_times) + sum(self.wait_times)),
            'current_state': self.current_state
        }

@dataclass
class WorkerTimeSeriesMetrics:
    timestamps: List[datetime] = field(default_factory=list)
    cpu_percents: List[float] = field(default_factory=list)
    utilizations: List[float] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    eval_times: List[float] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list)

class Worker:
    def __init__(self, objective_fn,
                 transport: str = "tcp",
                 host: str = "localhost",
                 port: int = 5555):
        self.objective_fn = objective_fn
        self.worker_id = str(uuid4()).encode()

        self.context = zmq.asyncio.Context()
        self.dealer = self.context.socket(zmq.DEALER)
        self.dealer.setsockopt(zmq.IDENTITY, self.worker_id)

        if transport == "tcp":
            self.dealer.connect(f"tcp://{host}:{port}")
        elif transport == "ipc":
            self.dealer.connect("ipc:///tmp/hyperopt")

        self.metrics = WorkerMetrics()
        self.process = psutil.Process()

        print(f"Worker {self.worker_id} connected")

    async def run(self):
        while True:
            try:
                eval_start = time.time()
                self.metrics.start_evaluation()

                print(f"Worker {self.worker_id} requesting samples")

                # Request parameters
                await self.dealer.send_multipart([
                    b"",
                    json.dumps({
                        "type": "get_samples",
                        "n_samples": 1,
                        "worker_stats": self.get_worker_stats()
                    }).encode()
                ])

                _, response = await self.dealer.recv_multipart()
                params = json.loads(response)["samples"][0]

                # Evaluate
                try:
                    result = await self._evaluate(params)
                    eval_time = time.time() - eval_start
                    self.metrics.eval_times.append(eval_time)
                    self.metrics.end_evaluation()

                    print(f"Worker {self.worker_id} completed evaluation")

                    await self.dealer.send_multipart([
                        b"",
                        json.dumps({
                            "type": "add_evaluation",
                            "evaluation": {
                                "parameters": params,
                                "objectives": result,
                                "timestamp": time.time(),
                                "metadata": {
                                    "eval_time": eval_time,
                                    "worker_stats": self.get_worker_stats()
                                }
                            }
                        }).encode()
                    ])
                    await self.dealer.recv_multipart()

                except Exception as e:
                    print(f"Worker {self.worker_id} evaluation error: {e}")
                    self.metrics.end_evaluation()

            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")

    def get_worker_stats(self):
        return {
            'metrics': self.metrics.get_stats(),
            'cpu_percent': self.process.cpu_percent(),
            'memory_percent': self.process.memory_percent(),
            'pid': self.process.pid,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

    async def _evaluate(self, params):
        if asyncio.iscoroutinefunction(self.objective_fn):
            return await self.objective_fn(params)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, self.objective_fn, params
            )

async def optimize(objective_fn, n_workers: int = 2, n_iterations: int = 10,
                  distributed: bool = False, host: str = "localhost",
                  port: int = 5555):
    transport = "tcp" if distributed else "ipc"
    server = OptimizationServer(
        transport=transport,
        host="*" if distributed else None,
        port=port
    )
    server_task = asyncio.create_task(server.serve())
    stats_task = asyncio.create_task(print_stats(server))

    worker_tasks = []
    for _ in range(n_workers):
        worker = Worker(
            objective_fn=objective_fn,
            transport=transport,
            host=host if distributed else None,
            port=port
        )
        task = asyncio.create_task(worker.run())
        worker_tasks.append(task)

    # Let it run for a while then cancel
    await asyncio.sleep(n_iterations)
    for task in worker_tasks:
        task.cancel()
    stats_task.cancel()
    server_task.cancel()

    # Generate visualization
    server.plot_worker_metrics()

async def print_stats(server):
    """Simple monitoring coroutine"""
    while True:
        print("\nCurrent Statistics:")
        print(f"Total evaluations: {len(server.evaluations.evaluations)}")
        print("\nWorker Stats:")
        for worker_id, stats in server.worker_stats.items():
            print(f"\nWorker {worker_id.decode()[:8]}...")
            if 'metrics' in stats:
                metrics = stats['metrics']
                print(f"  State: {metrics.get('current_state', 'unknown')}")
                print(f"  Utilization: {metrics.get('utilization', 0):.2%}")
                print(f"  CPU: {stats.get('cpu_percent', 0)}%")
        await asyncio.sleep(2)


# Example usage:
def dummy_objective(params):
    """Dummy objective function with variable runtime"""
    # Runtime varies based on parameter values to simulate different computation costs
    runtime = abs(params['x'] + params['y']) * 0.1 + 0.1
    time.sleep(runtime)
    x, y = params['x'], params['y']
    return {
        'obj1': x*x + y*y,
        'obj2': (x-1)*(x-1) + (y-1)*(y-1)
    }

if __name__ == "__main__":
    asyncio.run(optimize(
        objective_fn=dummy_objective,
        n_workers=3,
        n_iterations=20,  # Run longer to see statistics
        distributed=False
    ))