import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from core.database import get_db, SessionLocal
from main import app
from models.users import Users
import bcrypt

client = TestClient(app)

@pytest.fixture(scope="function")
def test_db():
    """Provides a clean database session for each test."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()

@pytest.fixture
def test_user(test_db):
    """Creates a test user in the database after clearing previous data."""
    test_db.query(Users).delete()  # Ensure no duplicates

    hashed_password = bcrypt.hashpw("testpassword".encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    user = Users(
        username="testuser",
        email="test@example.com",
        hashed_password=hashed_password,
        is_admin=False
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)

    return user



def test_register_user():
    """Test successful user registration."""
    user_data = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "password123"
    }
    response = client.post("/users/register", json=user_data)

    assert response.status_code == 200
    assert response.json()["message"] == "User registered successfully"
    assert "user_id" in response.json()


def test_register_duplicate_user(test_user):
    """Test registering with an existing username or email."""
    user_data = {
        "username": test_user.username,  # Same username
        "email": "newemail@example.com",  # Different email
        "password": "anotherpassword"
    }
    response = client.post("/users/register", json=user_data)

    assert response.status_code == 400
    assert response.json()["detail"] == "Username or email already exists"


def test_login_user(test_user):
    """Test login with valid credentials."""
    credentials = {
        "username": test_user.username,
        "password": "testpassword"
    }
    response = client.post("/users/auth/login", json=credentials)

    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"


def test_login_invalid_password(test_user):
    """Test login with an incorrect password."""
    response = client.post("/users/auth/login", json={
        "username": test_user.username,
        "password": "wrongpassword"
    })

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"


def test_login_non_existent_user():
    """Test login with a non-existent user."""
    response = client.post("/users/auth/login", json={
        "username": "nonexistent",
        "password": "password123"
    })

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password"