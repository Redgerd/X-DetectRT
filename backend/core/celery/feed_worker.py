import cv2
import redis
import logging
from typing import List
from config import settings
from models.cameras import Camera
from sqlalchemy.orm import Session
from core.database import SessionLocal
from celery import Celery, signals, group

feed_worker_app = Celery('feed_worker', broker=settings.REDIS_URL, backend=settings.REDIS_URL)
feed_worker_app.conf.update(
    broker_transport_options={'visibility_timeout': 3600},
    worker_heartbeat=60,
)

worker_id = None
capture_objects = {}


@signals.worker_ready.connect # automatically trigger task on worker startup
def on_feed_worker_startup(**kwargs):
    """
    Initialize the feed workers on Celery feed_worker startup.
    """

    # assume that default point of entry for this task is worker startup
    try:
        worker_name = kwargs['sender'].hostname.split('@')[1]
    except Exception as e:
        worker_name = on_feed_worker_startup.request.hostname.split('@')[1]

    # worker name will be celery@feed_worker(num)
    global celery_worker_id
    celery_worker_id = int(worker_name.split('feed_worker')[-1])

    start_camera_id = (celery_worker_id - 1) * 10 + 1
    end_camera_id = start_camera_id + 9
    
    logging.info(f"Feed Worker {celery_worker_id} will process cameras {start_camera_id}-{end_camera_id}")

    db : Session = SessionLocal()    
    worker_cameras = db.query(Camera).filter(Camera.id >= start_camera_id, Camera.id <= end_camera_id).all()
    db.close()

    if not worker_cameras:
        logging.warning(f"No cameras found for worker {celery_worker_id}.")
        return
    
    redis_client = redis.from_url(settings.REDIS_URL)
    for camera in worker_cameras:
        redis_client.set(f"camera_{camera.id}_intrusion_flag", "False") 

    # if celery_worker_id == 1:
    #     time.sleep(5) # wait for all workers to start before starting the processing of every worker
    #     start_all_feed_workers.apply_async(queue='feed_tasks', priority=5)

    return {"status": "Feed worker initialized."}


# main task - processes every camera assigned to this worker
@feed_worker_app.task
def process_cameras(worker_id: int):
    """
    Starting point to processing cameras, for manual restarts
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    global_stop_flag = redis_client.get("feed_workers_running")
    individual_stop_flag = redis_client.get(f"feed_worker_{worker_id}_running")
    redis_client.close()

    start_camera_id = (worker_id - 1) * 10 + 1
    end_camera_id = start_camera_id + 9

    while global_stop_flag == b"True" and individual_stop_flag == b"True":
        db = SessionLocal()
        worker_cameras = db.query(Camera).filter(Camera.id >= start_camera_id, Camera.id <= end_camera_id).all()
        update_cameras_for_feed_workers(worker_cameras)
        db.close()

        for _ in range(10): # run for a batch of frames per camera, then update the cameras list
            for camera in worker_cameras:
                    capture_video_frames(camera)

        redis_client = redis.from_url(settings.REDIS_URL)
        global_stop_flag = redis_client.get("feed_workers_running")
        individual_stop_flag = redis_client.get(f"feed_worker_{worker_id}_running")
        redis_client.close()

    release_capture_objects()
    logging.info("Feed worker stopped. Capture objects released.")

    return {"status": "Feed worker stopped."}


# captures a single frame and sends it to model worker
from .model_worker import process_frame

def capture_video_frames(camera: Camera):
    """
    Capture frames from the video source (URL) using OpenCV and send them to a Celery queue.
    """
    if camera.id in capture_objects:
        cap = capture_objects[camera.id]
    else:
        logging.info(f"Creating new VideoCapture object for camera {camera.id}")
        cap = cv2.VideoCapture(camera.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 5)
        if not cap.isOpened():
            logging.error(f"Could not open video stream for camera {camera.id} at URL {camera.url}")
            return
        redis_client = redis.from_url(settings.REDIS_URL)
        redis_client.set(f"camera_{camera.id}_intrusion_flag", "False")
        redis_client.close()
        capture_objects[camera.id] = cap

    # read the frame
    # logging.info(f"Reading frame from camera {camera.id}, path {camera.url}...")
    ret, frame = cap.read()
    frame = cv2.convertScaleAbs(frame)

    if not ret:
        logging.warning(f"Failed to read frame from camera {camera.id}, URL {camera.url}")
        return

    frame = preprocess_frame(frame, camera)
    process_frame.apply_async(args=[camera.id, frame.tolist()], queue='model_tasks')


def preprocess_frame(frame, camera: Camera):
    """
    Preprocess the frame (resize, crop, etc.) based on the camera's settings (e.g., resize_dims, crop_region).
    Resize dimensions are strings in format "(1280, 720)" and crop region is in format "((0,0), (1280,720))".
    Crop before resize.
    """

    if camera.crop_region:
        crop_region = eval(camera.crop_region)
        frame = frame[crop_region[0][1]:crop_region[1][1], crop_region[0][0]:crop_region]
    
    if camera.resize_dims:
        resize_dims = eval(camera.resize_dims) # default is "(640,480)"
        frame = cv2.resize(frame, resize_dims) # resize to 640x480

    return frame


# high priority task to update the cameras list
def update_cameras_for_feed_workers(cameras: List[Camera]):
    """
    Updates the cameras list for the feed workers.
    """
    try:
        global capture_objects

        # release the capture objects for cameras that are not in the new list
        for camera_id in capture_objects.keys():
            if camera_id not in [c.id for c in cameras]:
                logging.info(f"Releasing VideoCapture object for camera {camera_id}")
                capture_objects[camera_id].release()
                del capture_objects[camera_id]

    except Exception as e:
        logging.exception(e)

# ========== feed worker starting tasks ========== #

@feed_worker_app.task
def start_all_feed_workers():
    """
    Task to start the ALL feed workers simultaneously.
    """
    logging.info("Starting all feed workers...")

    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set("feed_workers_running", "True")
    redis_client.close()

    start_jobs_group = group(start_feed_worker.s(i+1) for i in range(settings.FEED_WORKERS))
    start_jobs_group.apply_async(queue='feed_tasks', priority=10)

    logging.info("All feed workers started!")
    return {"status": "Feed workers started."}

@feed_worker_app.task
def start_feed_worker(worker_id: int):
    """
    Start a single feed worker by setting its individual running flag.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set(f"feed_worker_{worker_id}_running", "True")
    redis_client.close()

    process_cameras.apply_async(queue='feed_tasks', args=[worker_id], priority=10)

    logging.info(f"Feed worker {worker_id} started!")
    return {"status": f"Feed worker {worker_id} started."}


# ========== feed worker stopping tasks ========== #

@feed_worker_app.task
def stop_all_feed_workers():
    """
    Task to stop the ALL feed workers simultaneously.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set("feed_workers_running", "False")
    redis_client.close()

    logging.info("All feed workers stopped!")
    return {"status": "Feed workers stopped."}


@feed_worker_app.task
def stop_feed_worker(worker_id: int):
    """
    Task to stop a single feed worker by setting its individual stop flag.
    """
    redis_client = redis.from_url(settings.REDIS_URL)
    redis_client.set(f"feed_worker_{worker_id}_running", "False")
    redis_client.close()

    logging.info(f"Feed worker {worker_id} stopped!")
    return {"status": f"Feed worker {worker_id} stopped."}


def release_capture_objects():
    """
    Release all the capture objects when done.
    """
    for camera_id, cap in capture_objects.items():
        logging.info(f"Releasing VideoCapture object for camera {camera_id}")
        cap.release()
