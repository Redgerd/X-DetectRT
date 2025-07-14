import cv2
from dotenv import load_dotenv
import os
import numpy as np  # Make sure numpy is imported for filling polygons

load_dotenv(override=True)

# Initialize video capture
cap = cv2.VideoCapture("data/sample2.mp4")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

def mouse_click(event, x, y, flags, params):
    global frame, points, all_polygons

    # add a point to the polygon and draw it
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        frame = cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        if len(points) > 1:
            frame = cv2.line(frame, points[-1], points[-2], (0, 255, 0), 2)

        print(f"Point added: ({x}, {y})")
        cv2.imshow("Frame", frame)  # Update display with intermediate lines

    elif event == cv2.EVENT_RBUTTONDOWN:
        # close the polygon and print the points in MultiPolygon format
        if len(points) >= 3:  # ensure valid polygon
            print(f"Polygon created: {points}")
            
            frame = cv2.line(frame, points[-1], points[0], (0, 128, 0), 2)

            polygon_mask = np.zeros_like(frame, dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [np.array(points)], (128, 0, 0)) 
            alpha = 0.75
            frame = cv2.addWeighted(frame, 1, polygon_mask, alpha, 0)

            # Add the polygon to the memory of all polygons
            all_polygons.append(points)

        else:
            print("Not enough points to form a polygon.")
        
        points = []  # Reset points for the next polygon

        # Display the updated frame after drawing the polygon
        cv2.imshow("Frame", frame)

# Function to redraw all polygons from memory
def redraw_polygons():
    global frame, original_frame, all_polygons, resize_dims
    # Clear the frame and redraw all polygons
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to the first frame for a fresh start
    frame = original_frame
    if not ret:
        print("Error: Could not read frame.")
        exit()

    # redraw all previous polygons in memory
    for polygon in all_polygons:
        frame = cv2.polylines(frame, [np.array(polygon)], isClosed=True, color=(0, 128, 0), thickness=1)
        polygon_mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [np.array(polygon)], (128, 0, 0))
        alpha = 0.75
        frame = cv2.addWeighted(frame, 1, polygon_mask, alpha, 0)

    cv2.imshow("Frame", frame)

# load the first frame
ret, frame = cap.read()
while not ret:
    print("Error: Could not read frame.")
    ret, frame = cap.read()
    continue

# resize the frame if required
resize_dims = eval(os.getenv("FEED_DIMS", "(640,480)"))
frame = cv2.resize(frame, resize_dims)
original_frame = frame

print(  "Left click to draw polygons\n" \
        "Right click to create/close polygon after drawing 3 points.\n" \
        "Press 'x' to remove last drawn polygon\n" \
        "Press 'q' to quit\n\n")

# global variables
points = []
all_polygons = []

# display image initially
cv2.imshow("Frame", frame)

# add mouse callback
cv2.setMouseCallback("Frame", mouse_click)

# keypress functions
while True:
    key = cv2.waitKey(0)  # Wait indefinitely for a key press

    if key == ord('x'):  # If 'x' is pressed, clear the last polygon
        if all_polygons:
            all_polygons.pop()  # Remove the last polygon
            print("Last polygon removed.")
            redraw_polygons()  # Redraw all remaining polygons
        else:
            print("No polygons to remove.")

    elif key == ord('q'):  # If 'q' is pressed, exit
        print("Exiting...")
        print(f"Final polygons (without spaces): {all_polygons.__str__().replace(' ', '')}")
        cap.release()
        break

cv2.destroyAllWindows()