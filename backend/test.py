import cv2

cap = cv2.VideoCapture("")
if not cap.isOpened():
    print("Unable to open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to retrieve frame, retrying...")
        continue

    cv2.imshow("test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
