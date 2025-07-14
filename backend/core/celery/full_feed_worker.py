import os
import ast
import cv2
import time
import redis
import logging
import numpy as np
from config import settings
from ultralytics import YOLO
from datetime import datetime
from celery import Celery, group
from models.cameras import Camera
from sqlalchemy.orm import Session
from core.database import SessionLocal
from api.alerts.schemas import AlertBase
from api.alerts.routes import create_alert
from shapely.geometry import Polygon, MultiPolygon


# celery worker for processing video feeds
full_feed_worker_app = Celery('unified_worker', broker=settings.REDIS_URL, backend=settings.REDIS_URL)
# remove the many h264 & rtsp warnings that get logged
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '8'


## ===== General Video Processing ===== ##

@full_feed_worker_app.task
def process_feed(camera_id: int):
    """
    Process a single feed source: capture frames, detect intrusions, and publish annotated frames if that websocket is open.
    This function is run as a Celery task.
    Args:
        camera_id (int): The ID of the camera to process.
    """
    try:
        # get all polygons and the camera
        threshold_polygons, line_points, camera = update_polygons_and_camera(camera_id)

        if not camera:
            logging.error(f"Camera {camera_id} not found.")
            return {"error": "Camera not found"}
        
        # Initialize Redis intrusion flag
        redis_client = redis.from_url(settings.REDIS_URL)
        redis_client.set(f"camera_{camera_id}_intrusion_flag", "False")

        # Only open capture and process frames if detect_intrusions is True
        if not camera.detect_intrusions:
            logging.info(f"Camera {camera_id} detect_intrusions is False. Skipping capture and processing.")
            return {"status": "Intrusion detection disabled for this camera."}

        cap = open_capture(camera.url, camera_id, max_tries=10, timeout=6)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logging.info('Camera Url:' + camera.url)

        stop_check_counter = 300

        # load yolo and move to GPU
        model = YOLO(model="./yolo-models/yolov8n.pt")
        model.to("cuda:0")
        logging.info(f"Loaded YOLO model for camera {camera_id}.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame from camera {camera_id}. Attempting to reopen capture object...")
                cap = open_capture(camera.url, camera_id, max_tries=10, timeout=6)
                continue

            annotated_frame = preprocess_frame(frame, camera)

            redis_client.get(f"camera_{camera_id}_intrusion_flag")

            intrusion_detected = False
            results = model.predict(annotated_frame, classes=[0], verbose=False)

            for res in results:
                for detection in res.boxes:
                    if detection.conf < 0.30:
                        continue
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    bbox = detection.xyxy[0]

                    # if intrusion is detected, raise flag and draw red box, put text
                    if centroid_near_line(bbox, threshold_polygons, camera.detection_threshold) or box_intersects_line(bbox, threshold_polygons):
                        intrusion_detected = True
                        annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for intrusion
                    else:
                        annotated_frame = cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for no intrusion

            if settings.SHOW_INTRUSION_LINES == "True":
                annotated_frame = cv2.polylines(annotated_frame, line_points, True, (0,0,255), 1)
            
            redis_intrusion_flag = redis_client.get(f"camera_{camera_id}_intrusion_flag")
            if settings.SHOW_INTRUSION_FLAG == "True" and redis_intrusion_flag == b"True":
                annotated_frame = cv2.putText(annotated_frame, "Intrusion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # only publish the frame if the websocket is open
            if redis_client.get(f"camera_{camera_id}_websocket_active") == b"True":
                publish_frame(camera_id, annotated_frame)

            # handle intrusion event if detected, and the flag is not already set (to avoid duplicate alerts)
            if intrusion_detected and redis_intrusion_flag == b"False":
                handle_intrusion_event(camera_id, annotated_frame)
            
            stop_check_counter -= 1
            if stop_check_counter <= 0:
                stop_check_counter = 300
                camera_running = redis_client.get(f"feed_worker_{camera_id}_running")
                global_cameras_running = redis_client.get("feed_workers_running")

                threshold_polygons, line_points, camera = update_polygons_and_camera(camera_id)
                if threshold_polygons is None and line_points is None:
                    raise Exception(f"Camera {camera_id} not found.")

                if threshold_polygons is None:
                    logging.error(f"Failed to get polygons for camera {camera_id}.")
                    break

                if camera_running == b"False" or global_cameras_running == b"False":
                    logging.info(f"Stopping feed processing for camera {camera_id}.")
                    break
                
        cap.release()

    except Exception as e:
        logging.exception(f"Error processing feed for camera {camera_id}: {e}")
    finally:
        redis_client.set(f"feed_worker_{camera_id}_running", "False")
        redis_client.set(f"camera_{camera_id}_intrusion_flag", "False")

        redis_client.close()
        logging.info(f"Stopped feed processing for camera {camera_id}")

    return {"status": "Feed processing stopped."}

def update_polygons_and_camera(camera_id: int):
    """
    Get the polygons for a specific camera.

    Args:
        camera_id (int) : The id number of the camera needed

    Returns:
        Polygons (shapely.geometry.MultiPolygon) : The Polygons which are needed to detect intrusions
        line_points (list[numpy.array]) : The points of the polygons as needed by cv2.polylines() function
        camera (Camera) : The updated camera info
    """
    with SessionLocal() as db:
        camera = db.query(Camera).filter(Camera.id == camera_id).first()

    if not camera:
        logging.error(f"Camera {camera_id} not found.")
        return (None, None, None)
    
    try:
        temp = ast.literal_eval(camera.lines)
        polygons = MultiPolygon([Polygon(poly) for poly in temp])
        line_points = [ np.array(polygon.exterior.coords, dtype=np.int32).reshape((-1, 1, 2)) for polygon in polygons.geoms ]
        return polygons, line_points, camera
    except Exception as e:
        logging.error(f"Error creating polygon/line threshold for camera {camera_id}: {e}")
        return (MultiPolygon(), [], camera)


@full_feed_worker_app.task
def process_feed_without_model(camera_id: int):
    """
    Process a single feed source without using a model.
    This function is run as a Celery task.
    Args:
        camera_id (int): The ID of the camera to process.
    """
    try:
        # get relevant cameras
        db: Session = SessionLocal()
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        db.close()

        if not camera:
            logging.error(f"Camera {camera_id} not found.")
            return {"error": "Camera not found"}
        
        # Only open capture and process frames if detect_intrusions is True
        if not camera.detect_intrusions:
            logging.info(f"Camera {camera_id} detect_intrusions is False. Skipping capture and processing.")
            return {"status": "Intrusion detection disabled for this camera."}

        # cap = cv2.VideoCapture(camera.url)
        cap = open_capture(camera.url, camera_id, max_tries=10, timeout=6)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        redis_client = redis.from_url(settings.REDIS_URL)

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame from camera {camera_id}. Attempting to reopen capture object...")
                cap = open_capture(camera.url, camera_id, max_tries=10, timeout=6)
                continue

            frame = preprocess_frame(frame, camera)

            if redis_client.get(f"camera_{camera_id}_websocket_active") == b"True":
                publish_frame(camera_id, frame)

    except Exception as e:
        logging.exception(f"Error processing feed for camera {camera_id}: {e}")
    finally:
        redis_client.close()
        cap.release()
        logging.info(f"Released VideoCapture object for camera {camera_id}")

    return {"status": "Feed processing stopped."}


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
    # logging.info(f"Published frame for camera {camera_id}")


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



## ====== Handling Intrusion Logic ===== ##

@full_feed_worker_app.task
def set_intrusion_flag(camera_id: int):
    """
    Unset the intrusion flag for a camera.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set(f"camera_{camera_id}_intrusion_flag", "True")
    redis_client.close()
    logging.info(f"Set intrusion flag for camera {camera_id}")
    full_feed_worker_app.send_task('core.celery.full_feed_worker.unset_intrusion_flag', args=[camera_id], queue="feed_tasks", countdown=5)



@full_feed_worker_app.task
def unset_intrusion_flag(camera_id: int):
    """
    Unset the intrusion flag for a camera.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set(f"camera_{camera_id}_intrusion_flag", "False")
    redis_client.close()
    logging.info(f"Unset intrusion flag for camera {camera_id}")


def handle_intrusion_event(camera_id: int, frame: np.ndarray = None):
    """
    Handle an intrusion event: create an alert and set a timer to reset the intrusion flag.
    Save the frame where the initial intrusion was detected to later display in alert.
    """
    logging.warning(f"Intrusion detected for camera {camera_id}!!!")

    file_path = None

    if frame is not None:
        # Save the frame to a file
        timestamp = int(datetime.now().timestamp())
        file_path = f"/app/alert_images/intrusion_{camera_id}_{timestamp}.jpg"
        cv2.imwrite(file_path, frame)


    alert_data = AlertBase(camera_id=camera_id, timestamp=str(datetime.now().replace(microsecond=0)), 
                           is_acknowledged=False, file_path=file_path)
    db = SessionLocal()
    create_alert(alert_data, db)
    db.close()

    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set(f"camera_{camera_id}_intrusion_flag", "True")
    redis_client.close()
    full_feed_worker_app.send_task('core.celery.full_feed_worker.unset_intrusion_flag', args=[camera_id], queue="feed_tasks", countdown=settings.INTRUSION_FLAG_DURATION)


def centroid_near_line(bounding_box: tuple, region:MultiPolygon, threshold:float=5) -> bool:
    """
    Determine if a centroid is near or has crossed a region.
    """
    x1, y1, x2, y2 = bounding_box
    box_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return region.distance(box_polygon.centroid) <= threshold


def box_intersects_line(bounding_box: tuple, region:MultiPolygon) -> bool:
    """
    Determines if a bounding box intersects with a threshold region.
    """
    x1, y1, x2, y2 = bounding_box
    box_polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return box_polygon.intersects(region)



## ====== Celery Tasks for Starting and Stopping Feed Workers ===== ##




@full_feed_worker_app.task
def start_feed_worker(camera_id: int):
    """
    Start the feed worker for a specific camera.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set(f"feed_worker_{camera_id}_running", "True")
    redis_client.close()

    db = SessionLocal()
    camera = db.query(Camera).filter(Camera.id == camera_id).first()
    db.close()

    if camera.detect_intrusions:
        full_feed_worker_app.send_task(
            "core.celery.full_feed_worker.process_feed",
            args=[camera_id],
            queue="feed_tasks"
        )
    else:
        full_feed_worker_app.send_task(
            "core.celery.full_feed_worker.process_feed_without_model",
            args=[camera_id],
            queue="feed_tasks"
        )
    return f"Feed worker started for camera {camera_id}"


@full_feed_worker_app.task
def start_all_feed_workers(camera_ids: list):
    """
    Start feed workers for all cameras.
    """

    os.makedirs("/app/alert_images", exist_ok=True)

    if not camera_ids or len(camera_ids) == 0:
        logging.warning("No cameras found to start feed workers.")
        return "No cameras found"
    
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set("feed_workers_running", "True")
    redis_client.close()

    tasks = group(start_feed_worker.s(camera_id) for camera_id in camera_ids)
    result = tasks.apply_async(queue="feed_tasks", priority=10)
    return result


@full_feed_worker_app.task
def stop_feed_worker(camera_id: int):
    """
    Stop the feed worker for a specific camera.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set("feed_workers_running", "False")
    return f"Feed worker stopping for camera {camera_id}..."


@full_feed_worker_app.task
def stop_all_feed_workers():
    """
    Stop all feed workers.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set("feed_workers_running", "False")
    redis_client.close()
    return "All feed workers stopping..."