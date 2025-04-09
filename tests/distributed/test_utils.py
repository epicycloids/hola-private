"""Tests for hola.distributed.utils."""

import logging
import os
import sys
from unittest.mock import patch, MagicMock, call

import pytest

# Assuming the tests directory is at the same level as hola/
# Adjust imports if the structure is different
from hola.distributed.utils import setup_logging

# Define a fixed path for testing log directory creation
TEST_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

@pytest.fixture(autouse=True)
def reset_loggers():
    """Fixture to reset logging state before each test."""
    # Shutdown existing handlers
    logging.shutdown()
    # Force clear the dictionary of existing loggers
    logging.Logger.manager.loggerDict.clear()
    yield # Run the test

@patch('hola.distributed.utils.os.path.abspath')
@patch('hola.distributed.utils.os.makedirs')
@patch('hola.distributed.utils.logging.FileHandler')
@patch('hola.distributed.utils.logging.StreamHandler')
def test_setup_logging_basic(mock_stream_handler, mock_file_handler, mock_makedirs, mock_abspath, caplog):
    """Test basic logger setup: name, level, console and file handlers."""
    # Mock abspath to control the log directory calculation
    # Assuming utils.py is in hola/distributed/, abspath(__file__) -> .../hola/distributed/utils.py
    # dirname(...) -> .../hola/distributed/
    # dirname(...) -> .../hola/
    # dirname(...) -> .../ (project root)
    mock_abspath.return_value = "/fake/path/to/hola/distributed/utils.py"
    project_root = "/fake/path/to"
    expected_log_dir = os.path.join(project_root, 'logs')

    mock_stream_instance = MagicMock()
    mock_stream_handler.return_value = mock_stream_instance

    mock_file_instance = MagicMock()
    mock_file_handler.return_value = mock_file_instance

    logger_name = "TestLogger"
    logger = setup_logging(logger_name, level=logging.DEBUG)

    assert logger.name == logger_name
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 2

    # Check console handler setup
    mock_stream_handler.assert_called_once_with(sys.stdout)
    assert mock_stream_instance in logger.handlers
    # Check if formatter was set (difficult to check exact formatter instance)
    assert mock_stream_instance.setFormatter.call_count == 1

    # Check directory creation
    mock_makedirs.assert_called_once_with(expected_log_dir, exist_ok=True)

    # Check file handler setup
    # Check that the filename contains the logger name and has a .log extension
    file_handler_call_args = mock_file_handler.call_args[0][0]
    assert file_handler_call_args.startswith(os.path.join(expected_log_dir, f'{logger_name}_'))
    assert file_handler_call_args.endswith('.log')
    assert mock_file_instance in logger.handlers
    assert mock_file_instance.setFormatter.call_count == 1

@patch('hola.distributed.utils.os.path.abspath')
@patch('hola.distributed.utils.os.makedirs')
@patch('hola.distributed.utils.logging.FileHandler')
@patch('hola.distributed.utils.logging.StreamHandler')
def test_setup_logging_default_level(mock_stream_handler, mock_file_handler, mock_makedirs, mock_abspath):
    """Test logger setup with the default INFO level."""
    mock_abspath.return_value = "/fake/path/to/hola/distributed/utils.py"
    logger_name = "InfoLogger"
    logger = setup_logging(logger_name) # Use default level

    assert logger.name == logger_name
    assert logger.level == logging.INFO # Default level
    assert len(logger.handlers) == 2

@patch('hola.distributed.utils.os.path.abspath')
@patch('hola.distributed.utils.os.makedirs')
@patch('hola.distributed.utils.logging.FileHandler')
@patch('hola.distributed.utils.logging.StreamHandler')
def test_setup_logging_existing_handlers(mock_stream_handler, mock_file_handler, mock_makedirs, mock_abspath):
    """Test that handlers are not added if the logger already has them."""
    mock_abspath.return_value = "/fake/path/to/hola/distributed/utils.py"
    logger_name = "ExistingHandlerLogger"

    # Setup logger first time
    logger1 = setup_logging(logger_name)
    assert len(logger1.handlers) == 2
    mock_stream_handler.reset_mock()
    mock_file_handler.reset_mock()
    mock_makedirs.reset_mock()

    # Call setup again for the same logger name
    logger2 = setup_logging(logger_name)

    # Assert logger is the same instance
    assert logger1 is logger2
    # Check that the mocks *were* called again on the second run
    mock_stream_handler.assert_called_once()
    # File handler setup might not be called if makedirs failed, but in this test it shouldn't
    mock_file_handler.assert_called_once()
    mock_makedirs.assert_called_once()

@patch('hola.distributed.utils.os.path.abspath')
@patch('hola.distributed.utils.os.makedirs', side_effect=OSError("Permission denied"))
@patch('hola.distributed.utils.logging.FileHandler') # Mock to prevent actual file creation
@patch('hola.distributed.utils.logging.StreamHandler')
def test_setup_logging_makedirs_fails(mock_stream_handler, mock_file_handler, mock_makedirs, mock_abspath, caplog):
    """Test that console logging still works if os.makedirs fails."""
    mock_abspath.return_value = "/fake/path/to/hola/distributed/utils.py"
    project_root = "/fake/path/to"
    expected_log_dir = os.path.join(project_root, 'logs')

    mock_stream_instance = MagicMock()
    mock_stream_handler.return_value = mock_stream_instance
    # Set a level on the mock handler instance to avoid TypeError during logging
    mock_stream_instance.level = logging.INFO # Or any valid level

    logger_name = "MakeDirsFailLogger"
    with caplog.at_level(logging.ERROR):
        logger = setup_logging(logger_name)

    assert logger.name == logger_name
    assert len(logger.handlers) == 1 # Only stream handler should be added
    assert mock_stream_instance in logger.handlers
    mock_makedirs.assert_called_once_with(expected_log_dir, exist_ok=True)
    mock_file_handler.assert_not_called() # File handler should not be created

    # Check that an error was logged
    assert "Failed to set up file logging" in caplog.text
    assert "Permission denied" in caplog.text

@patch('hola.distributed.utils.os.path.abspath')
@patch('hola.distributed.utils.os.makedirs')
@patch('hola.distributed.utils.logging.FileHandler', side_effect=OSError("Cannot write file"))
@patch('hola.distributed.utils.logging.StreamHandler')
def test_setup_logging_filehandler_fails(mock_stream_handler, mock_file_handler, mock_makedirs, mock_abspath, caplog):
    """Test that console logging still works if FileHandler fails."""
    mock_abspath.return_value = "/fake/path/to/hola/distributed/utils.py"
    project_root = "/fake/path/to"
    expected_log_dir = os.path.join(project_root, 'logs')

    mock_stream_instance = MagicMock()
    mock_stream_handler.return_value = mock_stream_instance
    # Set a level on the mock handler instance to avoid TypeError during logging
    mock_stream_instance.level = logging.INFO # Or any valid level

    logger_name = "FileHandlerFailLogger"
    with caplog.at_level(logging.ERROR):
        logger = setup_logging(logger_name)

    assert logger.name == logger_name
    assert len(logger.handlers) == 1 # Only stream handler should be added
    assert mock_stream_instance in logger.handlers
    mock_makedirs.assert_called_once_with(expected_log_dir, exist_ok=True)
    # File handler should have been called (attempted)
    assert mock_file_handler.call_count == 1

    # Check that an error was logged
    assert "Failed to set up file logging" in caplog.text
    assert "Cannot write file" in caplog.text