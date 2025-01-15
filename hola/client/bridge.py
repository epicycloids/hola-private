import asyncio
import json
import logging

from websockets.asyncio.server import ServerConnection, serve

from hola.client.client import Client
from hola.server.messages.base import OptimizationStatus


class WebSocketBridge:
    """Bridge between HOLA Client and WebSocket for UI."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.hola_client: Client | None = None
        self.server = None
        self.connections: set[ServerConnection] = set()
        self.logger = logging.getLogger("hola.websocket")

    def _serialize_status(self, status: OptimizationStatus) -> str:
        """Convert status to JSON string."""
        return json.dumps(
            {
                "type": "status",
                "data": {
                    "current_best": status.current_best,
                    "total_evaluations": status.total_evaluations,
                    "active_workers": [str(w) for w in status.active_workers],
                    "pareto_fronts": status.pareto_fronts,
                    "recent_evaluations": [
                        {"parameters": eval.parameters, "objectives": eval.objectives}
                        for eval in status.recent_evaluations
                    ],
                },
            }
        )

    async def _broadcast_status(self, status: OptimizationStatus) -> None:
        """Broadcast status update to all connected clients."""
        if not self.connections:
            return

        message = self._serialize_status(status)
        connections = self.connections.copy()
        await asyncio.gather(*(self._send_to_client(conn, message) for conn in connections))

    async def _send_to_client(self, connection: ServerConnection, message: str) -> None:
        """Send a message to a client with error handling."""
        try:
            await connection.send(message)
        except Exception as e:
            self.logger.error(f"Failed to send to client: {e}")
            self.connections.discard(connection)

    async def _handle_client_message(self, connection: ServerConnection, message: str) -> None:
        """Handle an incoming message from a client."""
        if not self.hola_client:
            return

        try:
            data = json.loads(message)
            match data["type"]:
                case "pause":
                    await self.hola_client.pause()
                    await connection.send(json.dumps({"type": "pause_response", "success": True}))
                case "resume":
                    await self.hola_client.resume()
                    await connection.send(json.dumps({"type": "resume_response", "success": True}))
                case "update_parameters":
                    await self.hola_client.update_parameter_config(data["config"])
                    await connection.send(
                        json.dumps({"type": "update_parameters_response", "success": True})
                    )
                case "update_objectives":
                    await self.hola_client.update_objective_config(data["config"])
                    await connection.send(
                        json.dumps({"type": "update_objectives_response", "success": True})
                    )
                case _:
                    self.logger.warning(f"Unknown message type: {data['type']}")

        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
            try:
                await connection.send(json.dumps({"type": "error", "message": str(e)}))
            except Exception:
                self.logger.error("Failed to send error message to client")
                self.connections.discard(connection)

    async def _handle_connection(self, connection: ServerConnection) -> None:
        """Handle a WebSocket connection."""
        self.connections.add(connection)
        self.logger.info(f"Client connected (total: {len(self.connections)})")

        try:
            # Send initial status
            if self.hola_client:
                status = await self.hola_client.get_status()
                await self._send_to_client(connection, self._serialize_status(status))

            # Handle incoming messages
            async for message in connection:
                await self._handle_client_message(connection, message)

        except Exception as e:
            self.logger.error(f"Connection handler error: {e}")
        finally:
            self.connections.discard(connection)
            self.logger.info(f"Client disconnected (remaining: {len(self.connections)})")

    async def start(self) -> None:
        """Start the WebSocket server and HOLA client."""
        # Initialize HOLA client with status callback
        self.hola_client = Client(status_callback=self._broadcast_status)
        await self.hola_client.start()

        # Start WebSocket server
        async with serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,  # Keep connections alive
            ping_timeout=20,
            close_timeout=10,
        ) as server:
            self.server = server
            self.logger.info(f"WebSocket bridge running on ws://{self.host}:{self.port}")
            # Run until cancelled
            await asyncio.Future()

    async def stop(self) -> None:
        """Stop the bridge and clean up resources."""
        if self.hola_client:
            await self.hola_client.stop()

        # Close all connections
        if self.connections:
            await asyncio.gather(*(connection.close() for connection in self.connections))
            self.connections.clear()
