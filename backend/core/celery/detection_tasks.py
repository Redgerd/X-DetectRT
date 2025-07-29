import os
import logging
from celery import shared_task
from celery.signals import worker_process_init
from tensorflow.keras.models import load_model

# Global variable for model
xception_model = None

# Path to model
MODEL_PATH = os.path.join("backend", "ml_models", "XceptionNet.keras")

# Logger
logger = logging.getLogger(__name__)

# Load model when worker starts
@worker_process_init.connect
def load_xception_model(**kwargs):
    global xception_model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return

        logger.info("Loading XceptionNet model...")
        xception_model = load_model(MODEL_PATH)
        logger.info("XceptionNet model loaded successfully.")
    except Exception as e:
        logger.exception(f"Failed to load XceptionNet model: {e}")

# Used to test if the model is loaded properly
def confirm_model_loaded():
    global xception_model
    return xception_model is not None

# Placeholder for detection task
@shared_task(name="backend.core.celery.detection_tasks.perform_detection")
def perform_detection(input_data):
    global xception_model
    if xception_model is None:
        raise RuntimeError("Model not loaded")
    
    # Add actual prediction logic here using input_data
    # prediction = xception_model.predict(...)
    return {"status": "success", "message": "Model is loaded and ready"}

