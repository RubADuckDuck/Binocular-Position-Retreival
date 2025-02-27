import cv2
import numpy as np

hMin = 37
sMin = 116 
vMin = 137 
hMax = 163
sMax = 255
vMax = 255

# Define the lower and upper bounds of the color you want to track in HSV
lower_color = np.array([hMin, sMin, vMin])  # Adjust these values for the color to track
upper_color = np.array([hMax, sMax, vMax])  # Adjust these values for the color to track

def find_color_coordinates(frame):
    """
    This function receives a frame, processes it to find the largest region of the specified color,
    and returns the coordinates of the center of this region.

    Args:
    frame: The input frame in which to detect the color.

    Returns:
    (center_x, center_y): Coordinates of the center of the largest detected color region.
    None: If no color region is detected.
    """
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the specified color
    color_mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    
    # Apply a series of dilations and erosions to remove any small blobs left in the mask
    color_mask = cv2.erode(color_mask, None, iterations=2)
    color_mask = cv2.dilate(color_mask, None, iterations=2)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour in the mask
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        return (center_x, center_y)
    
    return None

# Open the video capture and set resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get the coordinates of the color region
    coordinates = find_color_coordinates(frame)
    
    if coordinates:
        center_x, center_y = coordinates
        # Draw a circle at the center of the detected color region
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Display the resulting frame
    cv2.imshow('Tracked Color', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
