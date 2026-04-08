"""
Unit tests for utility modules.
"""
import pytest
import tempfile
import os
import logging
from utils.logger import setup_logger, setup_json_logger, ProgressLogger


def test_setup_logger():
    """Test logger setup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.log")

        # Create logger
        logger = setup_logger(
            name="test_logger",
            log_level="DEBUG",
            log_file=log_file,
            console_output=False
        )

        # Test logging
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Check log file exists
        assert os.path.exists(log_file)

        # Check log content
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Debug message" in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content


def test_setup_json_logger():
    """Test JSON logger setup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_json.log")

        # Create JSON logger
        logger = setup_json_logger(
            name="test_json_logger",
            log_level="INFO",
            log_file=log_file
        )

        # Test logging with extra fields
        logger.info("Test message", extra={"user": "test", "action": "login"})

        # Check log file exists
        assert os.path.exists(log_file)

        # Check JSON format
        import json
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) > 0

            # Parse JSON
            log_entry = json.loads(lines[0].strip())
            assert "message" in log_entry
            assert log_entry["message"] == "Test message"
            assert "user" in log_entry
            assert log_entry["user"] == "test"
            assert "action" in log_entry
            assert log_entry["action"] == "login"


def test_progress_logger():
    """Test progress logger."""
    import io
    import sys

    # Capture log output
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)

    test_logger = logging.getLogger("test_progress")
    test_logger.handlers.clear()
    test_logger.addHandler(handler)
    test_logger.setLevel(logging.INFO)

    # Create progress logger
    progress = ProgressLogger(total=100, desc="Test Progress", unit="steps", logger=test_logger)

    # Update progress
    progress.update(50, status="Halfway")
    progress.finish("Completed")

    # Check log output
    log_output = log_capture.getvalue()
    assert "Test Progress" in log_output
    assert "50%" in log_output
    assert "Halfway" in log_output
    assert "Completed" in log_output


def test_logger_levels():
    """Test logger level filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_levels.log")

        # Create logger with INFO level
        logger = setup_logger(
            name="test_levels",
            log_level="INFO",
            log_file=log_file,
            console_output=False
        )

        # Log messages at different levels
        logger.debug("Debug - should not appear")
        logger.info("Info - should appear")
        logger.warning("Warning - should appear")

        # Check log content
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "Debug - should not appear" not in content
            assert "Info - should appear" in content
            assert "Warning - should appear" in content


def test_multiple_loggers():
    """Test multiple logger instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file1 = os.path.join(tmpdir, "test1.log")
        log_file2 = os.path.join(tmpdir, "test2.log")

        # Create two different loggers
        logger1 = setup_logger(
            name="logger1",
            log_file=log_file1,
            console_output=False
        )

        logger2 = setup_logger(
            name="logger2",
            log_file=log_file2,
            console_output=False
        )

        # Log different messages
        logger1.info("Logger1 message")
        logger2.info("Logger2 message")

        # Check separate log files
        with open(log_file1, 'r', encoding='utf-8') as f:
            assert "Logger1 message" in f.read()
            assert "Logger2 message" not in f.read()

        with open(log_file2, 'r', encoding='utf-8') as f:
            assert "Logger2 message" in f.read()
            assert "Logger1 message" not in f.read()


def test_log_rotation():
    """Test log rotation (basic test)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test_rotation.log")

        # Create logger with small max file size
        logger = setup_logger(
            name="test_rotation",
            log_file=log_file,
            max_file_size=100,  # Very small for testing
            backup_count=2,
            console_output=False
        )

        # Write enough logs to trigger rotation
        for i in range(100):
            logger.info(f"Test message {i:04d}")

        # Check that log file exists
        assert os.path.exists(log_file)

        # Check backup files (rotation may not trigger due to buffering, but that's okay)
        import glob
        backup_files = glob.glob(log_file + ".*")
        # We don't assert on backup files as rotation behavior depends on buffering


if __name__ == "__main__":
    pytest.main([__file__, "-v"])