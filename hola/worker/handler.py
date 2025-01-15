import time
from datetime import datetime, timezone

from hola.server.messages.server import (
    OK,
    ConfigUpdated,
    Error,
    SampleResponse,
    ServerMessage,
    StatusUpdate,
)
from hola.server.messages.worker import ErrorReport, Evaluation, EvaluationReport, WorkerMessage
from hola.worker.base import WorkerState


class ServerMessageHandler:
    """Handles incoming server messages and produces worker responses."""

    async def handle_message(
        self,
        message: ServerMessage,
        state: WorkerState,
    ) -> WorkerMessage | None:
        """Handle a message from the server and return a response if needed."""
        try:
            match message:
                case SampleResponse():
                    return await self._handle_sample_response(message, state)
                case StatusUpdate():
                    state.logger.debug(f"Received status update: {message.status}")
                    return None
                case ConfigUpdated():
                    state.logger.info(
                        f"Configuration updated: {message.message or 'No details provided'}"
                    )
                    return None
                case Error():
                    state.logger.error(
                        f"Received error from server: {message.message or 'No details provided'}"
                    )
                    return None
                case OK():
                    return None
                case _:
                    state.logger.warning(f"Received unexpected message type: {type(message)}")
                    return None

        except Exception as e:
            state.logger.error(f"Error handling message: {e}")
            return ErrorReport(
                error=str(e),
                timestamp=datetime.now(timezone.utc).isoformat(),
                worker_id=state.worker_id,
            )

    async def _handle_sample_response(
        self,
        response: SampleResponse,
        state: WorkerState,
    ) -> WorkerMessage:
        """Handle a sample response containing parameters to evaluate."""
        for params in response.samples:
            try:
                state.logger.info(f"Evaluating parameters: {params}")
                result = await state.evaluate(params)

                evaluation = Evaluation(parameters=params, objectives=result)

                return EvaluationReport(
                    evaluation=evaluation,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    worker_id=state.worker_id,
                )

            except Exception as e:
                state.logger.error(f"Evaluation failed: {e}")
                return ErrorReport(
                    error=str(e),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    worker_id=state.worker_id,
                    parameters=params,
                )
