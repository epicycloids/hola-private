import logging
from typing import Any, Final

from hola.server.messages.server import OK, ConfigUpdated, Error, ServerMessage, StatusUpdate


class ServerMessageHandler:

    # Status messages
    CONFIG_UPDATED: Final[str] = "Configuration updated: {}"
    SERVER_ERROR: Final[str] = "Received error from server: {}"
    UNEXPECTED_MESSAGE: Final[str] = "Received unexpected message type: {}"

    async def handle_message(
        self,
        message: ServerMessage,
        logger: logging.Logger,
    ) -> Any:
        try:
            match message:
                case StatusUpdate():
                    return message.status
                case ConfigUpdated():
                    details = message.message or "No details provided"
                    logger.info(self.CONFIG_UPDATED.format(details))
                    return True
                case Error():
                    details = message.message or "No details provided"
                    logger.error(self.SERVER_ERROR.format(details))
                    return False
                case OK():
                    return True
                case _:
                    logger.warning(self.UNEXPECTED_MESSAGE.format(type(message)))
                    return None

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            return None
