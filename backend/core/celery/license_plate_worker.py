import os
import cv2
import time
import redis
import logging
import numpy as np
import re
from config import settings
from ultralytics import YOLO
from datetime import datetime
from celery import Celery, group
from models.cameras import Camera
from models.license_detection import License
from sqlalchemy.orm import Session
from core.database import SessionLocal
from api.alerts.schemas import AlertBase
from api.alerts.routes import create_alert
from api.license_plate.ocr_instance import ocr

# celery worker for processing video feeds for license plate detection
license_plate_worker_app = Celery('license_plate_worker', broker=settings.REDIS_URL, backend=settings.REDIS_URL)
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '8'

# Initialize OCR and CLAHE globally
ocr_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def unsharp_mask(image: np.ndarray, kernel_size=(5, 5), sigma=1.0, amount=0.5) -> np.ndarray:
    """
    Sharpens an image using the unsharp mask technique.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened

def lp_image_processing(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses the license plate image for better detection results.
    """
    # convert to grayscale
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # double the size
    processed_img = cv2.resize(processed_img, (2*processed_img.shape[1], 2*processed_img.shape[0]))
    # apply CLAHE
    processed_img = ocr_clahe.apply(processed_img)
    # sharpen
    processed_img = unsharp_mask(processed_img)
    return processed_img

def apply_lp_ocr_rules(license_plate: str, class_name: str) -> tuple[bool, str]:
    """
    Applies rules to the extracted license plate number to make it more readable.
    Trims known regional keywords like 'ICT-ISLAMABAD', 'PUNJAB', 'SINDH', etc.
    Logs the original and cleaned license plate text.
    """
    
    # Remove spaces and convert to uppercase
    text = license_plate.replace(' ', '').upper()
    # List of known region keywords to trim
    banned_keywords = ['ICTISLAMABAD', 'ICT-', 'PUNJAB', 'SINDH', 'KPK', 'BALOCHISTAN', 'AJK', 'GILGIT', 'ISLAMABAD']
    
    # Remove banned keywords if found at the start
    for keyword in banned_keywords:
        if text.startswith(keyword):
            text = text[len(keyword):]
            break  # Trim only the first matching keyword
    
    # Remove any remaining leading hyphens
    text = text.lstrip('-')

    # Define patterns for different vehicle classes
    car_bus_pattern = r'^[A-Z]{2,3}-?\d{3,4}[A-Z]?$'
    motorcycle_pattern = r'^[A-Z]{2,3}-?\d{3,4}[A-Z]?$'

    if class_name in ("car", "bus"):
        if re.match(car_bus_pattern, text):
            logging.info(f"License plate matched pattern for class '{class_name}'")
            return True, text
        
    if class_name == "motorcycle":
        if re.match(motorcycle_pattern, text):
            logging.info(f"License plate matched pattern for class '{class_name}'")
            return True, text

    logging.info(f"License plate did not match any pattern for class '{class_name}'")
    return False, text

def license_plate_ocr(plate_img: np.ndarray, class_name: str) -> tuple[str, float, bool]:
    """
    Extracts the license plate number from a cropped image.
    """
    # preprocess the image
    preprocessed_image = lp_image_processing(plate_img)
    
    # perform OCR
    lp_results = ocr.ocr(preprocessed_image, cls=True)

    license_plate_number = ""
    confidence_scores = []

    if len(lp_results) == 0:
        return "", 0.0, False
    
    for lp_res in lp_results:
        if lp_res is None:
            continue
        for line in lp_res:
            license_plate_number += line[1][0]
            confidence_scores.append(int(float(line[1][1]) * 100))

    valid, license_plate_number = apply_lp_ocr_rules(license_plate_number, class_name)
    average_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

    if license_plate_number == "":
        return "", 0.0, False
    return license_plate_number, average_confidence, valid

## ===== General Video Processing ===== ##

@license_plate_worker_app.task
def process_feed(camera_id: int):
    """
    Process a single feed source: capture frames, detect license plates, and publish annotated frames if that websocket is open.
    This function is run as a Celery task.
    Args:
        camera_id (int): The ID of the camera to process.
    """
    try:
        # get all cameras
        db: Session = SessionLocal()
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        db.close()

        if not camera:
            logging.error(f"Camera {camera_id} not found.")
            return {"error": "Camera not found"}

        # Only open capture if detect_intrusions is False
        if camera.detect_intrusions:
            logging.info(f"Camera {camera_id} has intrusion detection enabled. Skipping license plate processing.")
            return {"status": "Skipped due to intrusion detection enabled"}

        # Initialize Redis license plate detection flag
        redis_client = redis.from_url(settings.REDIS_URL)
        redis_client.set(f"camera_{camera_id}_license_plate_flag", "False")

        cap = open_capture(camera.url, camera_id, max_tries=10, timeout=6)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        stop_check_counter = 300

        # load yolo model for license plate detection
        model = YOLO(model="./yolo-models/yolo-license-plates.pt")  # Replace with your license plate model path
        logging.info(f"Loaded license plate detection model for camera {camera_id}.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame from camera {camera_id}. Attempting to reopen capture object...")
                cap = open_capture(camera.url, camera_id, max_tries=10, timeout=6)
                continue

            annotated_frame = preprocess_frame(frame, camera)

            # Process frame with license plate detection model
            results = model.predict(annotated_frame, verbose=False)
            
            license_plate_detected = False
            for res in results:
                for detection in res.boxes:
                    if detection.conf < 0.57:  # Confidence threshold
                        continue
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    
                    # Extract license plate region
                    plate_region = annotated_frame[y1:y2, x1:x2]
                    
                    # Perform OCR on the license plate
                    lp_number, confidence, valid = license_plate_ocr(plate_region, "car")  # Assuming car for now
                    # If not valid, retry as motorcycle
                    if not valid:
                        lp_number, confidence, valid = license_plate_ocr(plate_region, "motorcycle")
                    
                    # Draw bounding box and text
                    color = (0, 255, 0) if valid else (0, 0, 255)
                    annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    if lp_number:
                        label = f"{lp_number} ({confidence:.2f})"
                        cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 4, lineType=cv2.LINE_AA)
                        cv2.putText(annotated_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, lineType=cv2.LINE_AA)
                        license_plate_detected = True

            # only publish the frame if the websocket is open
            if redis_client.get(f"camera_{camera_id}_websocket_active") == b"True":
                publish_frame(camera_id, annotated_frame)

            # handle license plate detection event if detected
            if license_plate_detected and lp_number and valid:
                # Redis key: unique per camera and plate number
                redis_key = f"camera_{camera_id}_lp_{lp_number}"
                
                if not redis_client.exists(redis_key):
                    handle_license_plate_event(camera_id, lp_number, annotated_frame)
                    redis_client.set(redis_key, "1", ex=60)  # 1 minute TTL
                else:
                    logging.info(f"Plate {lp_number} already handled recently for camera {camera_id}. Skipping DB insert.")

            
            stop_check_counter -= 1
            if stop_check_counter <= 0:
                stop_check_counter = 300
                camera_running = redis_client.get(f"camera_{camera_id}_running")
                global_cameras_running = redis_client.get("license_plate_workers_running")

                if camera_running == b"False" or global_cameras_running == b"False":
                    logging.info(f"Stopping license plate detection for camera {camera_id}.")
                    break
                
        cap.release()

    except Exception as e:
        logging.exception(f"Error processing feed for camera {camera_id}: {e}")
    finally:
        redis_client.close()
        logging.info(f"Stopped license plate detection for camera {camera_id}")

    return {"status": "License plate detection stopped."}

def preprocess_frame(frame, camera: Camera):
    """
    Preprocess the frame (resize, crop, etc.) based on the camera's settings.
    """
    if camera.crop_region:
        crop_region = eval(camera.crop_region)
        frame = frame[crop_region[0][1]:crop_region[1][1], crop_region[0][0]:crop_region[1][0]]

    if camera.resize_dims:
        resize_dims = eval(camera.resize_dims)
        frame = cv2.resize(frame, resize_dims)

    return frame

# redis client for publishing frames
redis_client_ws = redis.from_url(settings.REDIS_URL)

def publish_frame(camera_id: int, annotated_frame: np.ndarray):
    """
    Publish the annotated frame to Redis.
    """
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    global redis_client_ws
    redis_client_ws.publish(f"camera_{camera_id}", buffer.tobytes())

def open_capture(url:str, camera_id:int, max_tries:int=10, timeout:int=6):
    """
    Reopen video capture object if failed
    """
    for attempt in range(0, max_tries):
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            logging.info(f"Video Capture object for Camera {camera_id} successfully created.")
            return cap
        else:
            logging.error(f"Attempt {attempt} of starting capture for Camera {camera_id} failed.")
            cap.release()
            time.sleep(timeout)
    logging.error(f"Failed to create Capture object for Camera {camera_id}")
    raise Exception(f"Failed to create Capture object for Camera {camera_id}")

## ====== Handling License Plate Detection Logic ===== ##

def handle_license_plate_event(camera_id: int, license_number: str, frame: np.ndarray = None):
    """
    Handle a license plate detection event: create a database record and save the frame.
    """
    logging.info(f"License plate detected for camera {camera_id}: {license_number}")

    file_path = None

    if frame is not None:
        # Save the frame to a file
        timestamp = int(datetime.now().timestamp())
        file_path = f"/app/alert_images/license_plates/license_plate_{camera_id}_{timestamp}.jpg"
        cv2.imwrite(file_path, frame)

    # Create database record
    db = SessionLocal()
    try:
        new_license = License(
            camera_id=camera_id,
            license_number=license_number,
            timestamp=datetime.now(),
            file_path=file_path
        )
        db.add(new_license)
        db.commit()
        db.refresh(new_license)
    except Exception as e:
        logging.error(f"Error saving license plate detection to database: {e}")
        db.rollback()
    finally:
        db.close()

    # Create alert for the detection
    # alert_data = AlertBase(
    #     camera_id=camera_id, 
    #     timestamp=str(datetime.now().replace(microsecond=0)), 
    #     is_acknowledged=False, 
    #     file_path=file_path
    # )
    # db = SessionLocal()
    # create_alert(alert_data, db)
    # db.close()

## ====== Celery Tasks for Starting and Stopping License Plate Workers ===== ##

@license_plate_worker_app.task
def start_license_plate_worker(camera_id: int):
    """
    Start the license plate detection worker for a specific camera.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set(f"license_plate_worker_{camera_id}_running", "True")
    redis_client.close()

    license_plate_worker_app.send_task(
        "core.celery.license_plate_worker.process_feed",
        args=[camera_id],
        queue="license_plate_tasks"
    )
    return f"License plate detection worker started for camera {camera_id}"

@license_plate_worker_app.task
def start_all_license_plate_workers(camera_ids: list):
    """
    Start license plate detection workers for all cameras.
    """
    os.makedirs("/app/alert_images", exist_ok=True)

    if not camera_ids or len(camera_ids) == 0:
        logging.warning("No cameras found to start license plate detection workers.")
        return "No cameras found"
    
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set("license_plate_workers_running", "True")
    redis_client.close()

    tasks = group(start_license_plate_worker.s(camera_id) for camera_id in camera_ids)
    result = tasks.apply_async(queue="license_plate_tasks", priority=10)
    return result

@license_plate_worker_app.task
def stop_license_plate_worker(camera_id: int):
    """
    Stop the license plate detection worker for a specific camera.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set(f"license_plate_worker_{camera_id}_running", "False")
    redis_client.close()
    return f"License plate detection worker stopping for camera {camera_id}..."

@license_plate_worker_app.task
def stop_all_license_plate_workers():
    """
    Stop all license plate detection workers.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set("license_plate_workers_running", "False")
    redis_client.close()
    return "All license plate detection workers stopping..."
