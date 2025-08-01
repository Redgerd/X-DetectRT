import os
import logging
from celery import shared_task
from celery.signals import worker_process_init
from tensorflow.keras.models import load_model

# Global variable for model
xception_model = None

# Path to model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "ml_models", "XceptionNet.keras"))

# Logger
logger = logging.getLogger(__name__)

# # Load model when worker starts
# @worker_process_init.connect
# def load_xception_model(**kwargs):
#     global xception_model
#     try:
#         # Debug: Print current working directory and model path
#         logger.info(f"Current working directory: {os.getcwd()}")
#         logger.info(f"Model path: {MODEL_PATH}")
#         logger.info(f"Absolute model path: {os.path.abspath(MODEL_PATH)}")
        
#         if not os.path.exists(MODEL_PATH):
#             logger.error(f"Model file not found at {MODEL_PATH}")
#             # List contents of ml_models directory if it exists
#             ml_models_dir = os.path.dirname(MODEL_PATH)
#             if os.path.exists(ml_models_dir):
#                 logger.info(f"Contents of {ml_models_dir}: {os.listdir(ml_models_dir)}")
#             else:
#                 logger.error(f"ml_models directory not found at {ml_models_dir}")
#             return

#         logger.info("Loading XceptionNet model...")
#         xception_model = load_model(MODEL_PATH, compile=False)
#         logger.info("XceptionNet model loaded successfully.")
#     except Exception as e:
#         logger.exception(f"Failed to load XceptionNet model: {e}")


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

