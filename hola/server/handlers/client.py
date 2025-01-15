from datetime import datetime, timezone
from typing import Final

from hola.server.base import ExecutionState
from hola.server.messages.client import (
    ClientMessage,
    LoadStateRequest,
    PauseRequest,
    ResumeRequest,
    SaveStateRequest,
    StatusRequest,
    UpdateObjectiveConfigRequest,
    UpdateParameterConfigRequest,
)
from hola.server.messages.server import OK, ConfigUpdated, Error, ServerMessage, StatusUpdate
from hola.server.status import OptimizationStatus


class ClientMessageHandler:

    UNEXPECTED_MESSAGE: Final[str] = "Received unexpected message type from client"
    PAUSE_SUCCESS: Final[str] = "Optimization paused by client request"
    RESUME_SUCCESS: Final[str] = "Optimization resumed by client request"
    OBJECTIVE_UPDATE_SUCCESS: Final[str] = "Objective configuration updated successfully"
    PARAMETER_UPDATE_SUCCESS: Final[str] = "Parameter configuration updated successfully"
    SAVE_SUCCESS: Final[str] = "Optimization state saved successfully"
    LOAD_SUCCESS: Final[str] = "Optimization state loaded successfully"

    async def handle_message(
        self, message: ClientMessage, execution_state: ExecutionState
    ) -> ServerMessage:
        try:
            match message:
                case StatusRequest():
                    return await self._handle_status_request(execution_state)
                case PauseRequest():
                    return await self._handle_pause_request(execution_state)
                case ResumeRequest():
                    return await self._handle_resume_request(execution_state)
                case UpdateObjectiveConfigRequest():
                    return await self._handle_update_objective_config_request(
                        message, execution_state
                    )
                case UpdateParameterConfigRequest():
                    return await self._handle_update_parameter_config_request(
                        message, execution_state
                    )
                case SaveStateRequest():
                    return await self._handle_save_state(message, execution_state)
                case LoadStateRequest():
                    return await self._handle_load_state(message, execution_state)
                case _:
                    execution_state.logger.warning(f"{self.UNEXPECTED_MESSAGE}: {type(message)}")
                    return Error(self.UNEXPECTED_MESSAGE)
        except Exception as e:
            execution_state.logger.error(f"Error handling client message: {e}", exc_info=True)
            return Error(str(e))

    async def _handle_status_request(self, execution_state: ExecutionState) -> StatusUpdate:
        try:
            opt_state = await execution_state.coordinator.get_state()
            worker_statuses = execution_state.worker_status.get_all_statuses()
            active_workers = execution_state.worker_status.get_active_workers()

            status = OptimizationStatus(
                timestamp=datetime.now(timezone.utc),
                optimization_state=opt_state,
                worker_status=worker_statuses,
                num_active_workers=len(active_workers),
            )

            return StatusUpdate(status=status)
        except Exception as e:
            execution_state.logger.error(f"Failed to get optimization status: {e}", exc_info=True)
            raise

    async def _handle_pause_request(self, execution_state: ExecutionState) -> ServerMessage:
        try:
            await execution_state.coordinator.pause()
            execution_state.logger.info(self.PAUSE_SUCCESS)
            return OK()
        except Exception as e:
            error_msg = f"Failed to pause optimization: {e}"
            execution_state.logger.error(error_msg, exc_info=True)
            return Error(error_msg)

    async def _handle_resume_request(self, execution_state: ExecutionState) -> ServerMessage:
        try:
            await execution_state.coordinator.resume()
            execution_state.logger.info(self.RESUME_SUCCESS)
            return OK()
        except Exception as e:
            error_msg = f"Failed to resume optimization: {e}"
            execution_state.logger.error(error_msg, exc_info=True)
            return Error(error_msg)

    async def _handle_update_objective_config_request(
        self, message: UpdateObjectiveConfigRequest, execution_state: ExecutionState
    ) -> ServerMessage:
        try:
            await execution_state.coordinator.update_objective_config(message.new_config)
            execution_state.logger.info(
                "Objective configuration updated", extra={"num_objectives": len(message.new_config)}
            )
            return ConfigUpdated(self.OBJECTIVE_UPDATE_SUCCESS)
        except Exception as e:
            error_msg = f"Failed to update objective configuration: {e}"
            execution_state.logger.error(error_msg, exc_info=True)
            return Error(error_msg)

    async def _handle_update_parameter_config_request(
        self, message: UpdateParameterConfigRequest, execution_state: ExecutionState
    ) -> ServerMessage:
        try:
            await execution_state.coordinator.update_parameter_config(message.new_config)
            execution_state.logger.info(
                "Parameter configuration updated", extra={"num_parameters": len(message.new_config)}
            )
            return ConfigUpdated(self.PARAMETER_UPDATE_SUCCESS)
        except Exception as e:
            error_msg = f"Failed to update parameter configuration: {e}"
            execution_state.logger.error(error_msg, exc_info=True)
            return Error(error_msg)

    async def _handle_save_state(
        self, message: SaveStateRequest, execution_state: ExecutionState
    ) -> ServerMessage:
        try:
            await execution_state.coordinator.save_state(message.filepath)
            execution_state.logger.info(
                "Optimization state saved", extra={"filepath": message.filepath}
            )
            return ConfigUpdated(self.SAVE_SUCCESS)
        except Exception as e:
            error_msg = f"Failed to save optimization state: {e}"
            execution_state.logger.error(error_msg)
            return Error(error_msg)

    async def _handle_load_state(
        self, message: LoadStateRequest, execution_state: ExecutionState
    ) -> ServerMessage:
        try:
            await execution_state.coordinator.load_state(message.filepath)
            execution_state.logger.info(
                "Optimization state loaded", extra={"filepath": message.filepath}
            )
            return ConfigUpdated(self.LOAD_SUCCESS)
        except Exception as e:
            error_msg = f"Failed to load optimization state: {e}"
            execution_state.logger.error(error_msg)
            return Error(error_msg)
