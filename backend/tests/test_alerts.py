import pytest
import json
import redis
from httpx import AsyncClient
from fastapi import FastAPI
from api.alerts.routes import router

app = FastAPI()
app.include_router(router)

# Redis test setup
@pytest.fixture(scope="function")
def test_redis():
    """Mock Redis client for testing."""
    redis_client = redis.Redis(host="localhost", port=6379, db=1, decode_responses=True)
    yield redis_client
    redis_client.flushdb()  # Cleanup after test

@pytest.mark.asyncio
async def test_create_alert(test_redis):
    """Tests creating an alert via the API and checks Redis publishing."""
    
    alert_data = {
        "camera_id": 1,
        "timestamp": "26 02 2024",
        "is_acknowledged": False,
        "file_path": "https://youtu.be/_oTgwjM6mBU?si=u-7zZ84pi9qLb7K8"
    }

    async with AsyncClient(app=app, base_url="http://127.0.0.1:8000") as client:
        response = await client.post("/alerts/", json=alert_data)
    
    assert response.status_code == 200
    response_json = response.json()
    
    assert response_json["camera_id"] == 1
    assert response_json["timestamp"] == "26 02 2024"
    assert response_json["is_acknowledged"] is False
    assert response_json["file_path"] == "https://youtu.be/_oTgwjM6mBU?si=u-7zZ84pi9qLb7K8"

    # Check if the alert was published in Redis
    pubsub = test_redis.pubsub()
    pubsub.subscribe("camera_alerts:1")

    message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
    assert message is not None
    alert_from_redis = json.loads(message["data"])
    assert alert_from_redis["camera_id"] == 1
    assert alert_from_redis["timestamp"] == "26 02 2024"
    assert alert_from_redis["is_acknowledged"] is False
    assert alert_from_redis["file_path"] == "https://youtu.be/_oTgwjM6mBU?si=u-7zZ84pi9qLb7K8"