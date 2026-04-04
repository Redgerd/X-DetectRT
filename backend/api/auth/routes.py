from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from core.database import get_db
from models.users import Users
from api.auth.security import create_access_token
from api.auth.schemas import UserLoginSchema, TokenSchema, UserCreateSchema, GoogleAuthSchema
import bcrypt
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/login", response_model=TokenSchema)
def login(user_credentials: UserLoginSchema, db: Session = Depends(get_db)):
    user = db.query(Users).filter(Users.username == user_credentials.username).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    if not bcrypt.checkpw(user_credentials.password.encode("utf-8"), user.hashed_password.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    # Generate a token with role-based expiry
    token = create_access_token(user.id, user.username, user.role.value == "admin")

    return {"access_token": token, "token_type": "bearer"}


@router.post("/register")
def register(user_data: UserCreateSchema, db: Session = Depends(get_db)):
    existing_user = db.query(Users).filter(
        (Users.username == user_data.username) | (Users.email == user_data.email)
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username or email already exists"
        )

    hashed_password = bcrypt.hashpw(user_data.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    new_user = Users(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully", "user_id": new_user.id}

@router.post("/token", response_model=TokenSchema)
def form_data_login(form_data: OAuth2PasswordRequestForm=Depends(), db: Session = Depends(get_db)):
    """
    Form-data Login for Access Token
    """

    return login(user_credentials=UserLoginSchema(username=form_data.username, password=form_data.password), db=db)


@router.post("/google", response_model=TokenSchema)
def google_login(payload: GoogleAuthSchema, db: Session = Depends(get_db)):
    """
    Google login - frontend handles Google auth via Firebase, sends email to backend.
    Backend just adds user to database and returns JWT token.
    """
    # Frontend sends email after successful Firebase Google sign-in
    email = payload.id_token
    
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required")
    
    # Extract username from email
    name = email.split("@")[0] if email else "User"
    
    # Check if user exists
    user = db.query(Users).filter(Users.email == email).first()
    
    if not user:
        # Create new user
        user = Users(
            username=name,
            email=email,
            hashed_password="",
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    token = create_access_token(user.id, user.username, user.role.value == "admin")
    return {"access_token": token, "token_type": "bearer"}


@router.post("/firebase/register", response_model=TokenSchema)
def firebase_register(payload: GoogleAuthSchema, db: Session = Depends(get_db)):
    """
    Firebase-based registration - frontend handles Firebase auth, backend just saves user.
    """
    # For Firebase, we receive the email in the id_token field (reusing the schema)
    email = payload.id_token
    
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required")
    
    # Extract name from email if not provided
    name = email.split("@")[0] if email else "User"
    
    # Check if user already exists
    existing_user = db.query(Users).filter(Users.email == email).first()
    
    if existing_user:
        # User already exists, return token
        token = create_access_token(existing_user.id, existing_user.username, existing_user.role.value == "admin")
        return {"access_token": token, "token_type": "bearer"}
    
    # Create new user
    user = Users(
        username=name,
        email=email,
        hashed_password=""  # Firebase-managed password
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    token = create_access_token(user.id, user.username, user.role.value == "admin")
    return {"access_token": token, "token_type": "bearer"}


@router.post("/firebase/login", response_model=TokenSchema)
def firebase_login(payload: GoogleAuthSchema, db: Session = Depends(get_db)):
    """
    Firebase-based login - frontend handles Firebase auth, backend just verifies and returns token.
    """
    email = payload.id_token
    
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email is required")
    
    # Check if user exists in database
    user = db.query(Users).filter(Users.email == email).first()
    
    if not user:
        # Create new user on first login
        name = email.split("@")[0] if email else "User"
        user = Users(
            username=name,
            email=email,
            hashed_password=""
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    
    token = create_access_token(user.id, user.username, user.role.value == "admin")
    return {"access_token": token, "token_type": "bearer"}


@router.post("/logout")
def logout():
    """
    Stateless logout endpoint. Clients should delete stored JWT.
    """
    return {"message": "Logged out"}