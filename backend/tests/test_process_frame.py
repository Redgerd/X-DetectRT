import numpy as np
import pytest
from unittest.mock import patch
from backend.core.celery.stream_worker import publish_frame 

@pytest.fixture
def test_frame():
    """Generate a random 640x480 test frame."""
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

@patch("core.celery.socket_worker.publish_frame.delay")  # Mock Celery task execution
def test_publish_frame(mock_publish, test_frame):
    camera_id = 1

    # Call the Celery task
    result = publish_frame(camera_id, test_frame)

    # Verify Celery task was called correctly
    mock_publish.assert_called_once_with(camera_id, test_frame)

    print("Test Result:", result)
