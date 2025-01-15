from typing import Final

from hola.server.base import ExecutionState
from hola.server.messages.server import OK, Error, SampleResponse, ServerMessage
from hola.server.messages.worker import ErrorReport, EvaluationReport, SampleRequest, WorkerMessage


class WorkerMessageHandler:

    OPTIMIZATION_PAUSED: Final[str] = "Optimization is currently paused"
    NO_PARAMETERS: Final[str] = "No parameters available for evaluation"
    UNEXPECTED_MESSAGE: Final[str] = "Received unexpected message type from worker"

    async def handle_message(
        self, message: WorkerMessage, execution_state: ExecutionState
    ) -> ServerMessage:
        try:
            match message:
                case SampleRequest():
                    return await self._handle_sample_request(message, execution_state)
                case EvaluationReport():
                    return await self._handle_evaluation_report(message, execution_state)
                case ErrorReport():
                    return await self._handle_error_report(message, execution_state)
                case _:
                    execution_state.logger.warning(f"{self.UNEXPECTED_MESSAGE}: {type(message)}")
                    return Error(self.UNEXPECTED_MESSAGE)
        except Exception as e:
            execution_state.logger.error(f"Error handling worker message: {e}")
            return Error(str(e))

    async def _handle_sample_request(
        self, message: SampleRequest, execution_state: ExecutionState
    ) -> SampleResponse | Error:
        # Check if optimization is accepting new evaluations
        if not execution_state.coordinator.is_active():
            return Error(self.OPTIMIZATION_PAUSED)

        # Get suggested parameters to evaluate
        suggested_params = await execution_state.coordinator.suggest_parameters(message.n_samples)
        if not suggested_params:
            return Error(self.NO_PARAMETERS)

        return SampleResponse(samples=suggested_params)

    async def _handle_evaluation_report(
        self, message: EvaluationReport, execution_state: ExecutionState
    ) -> ServerMessage:
        try:
            await execution_state.coordinator.record_evaluation(
                message.evaluation,
                message.timestamp,
                worker_id=message.worker_id,
            )
            return OK()
        except Exception as e:
            error_msg = f"Failed to record evaluation from worker {message.worker_id}: {e}"
            execution_state.logger.error(error_msg)
            return Error(error_msg)

    async def _handle_error_report(
        self, message: ErrorReport, execution_state: ExecutionState
    ) -> ServerMessage:
        if message.parameters is not None:
            try:
                await execution_state.coordinator.record_failed_evaluation(
                    message.error, message.parameters, message.timestamp, message.worker_id
                )
            except Exception as e:
                error_msg = (
                    f"Failed to record failed evaluation from worker {message.worker_id}: {e}"
                )
                execution_state.logger.error(error_msg)
                return Error(error_msg)

        execution_state.logger.error(
            f"Worker {message.worker_id} reported error: {message.error}",
            extra={
                "worker_id": str(message.worker_id),
                "error": message.error,
                "parameters": message.parameters,
            },
        )

        return OK()
