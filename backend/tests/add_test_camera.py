from core.database import SessionLocal
from models.cameras import Camera

def add_test_camera():
    db = SessionLocal()
    try:
        camera = Camera(
            url='file:///d:/Smart-Campus/alert_images/dummy.mp4',
            location='Test Location',
            detection_threshold=30,
            resize_dims='(640, 480)'
        )
        db.add(camera)
        db.commit()
        print(f'Created camera with ID: {camera.id}')
    finally:
        db.close()

if __name__ == "__main__":
    add_test_camera() 