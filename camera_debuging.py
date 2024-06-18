import cv2
import numpy as np
import math

class Camera:
    def __init__(self, camera_id, im_width, im_height):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise ValueError(f"Camera {camera_id} could not be opened.")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, im_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, im_height)
        
        print(f"Webcam {camera_id} opened successfully with resolution {im_width}x{im_height}. Press 'q' to quit.")

    def capture(self):
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to capture image from camera {self.camera_id}.")
        return frame

    def release(self):
        self.cap.release() 



# marker settign 
color_to_find = (0, 132, 96)  # This is in RGB
threshold = 70
circle_radius = 10  # Radius of the circle to draw 

circle_color = (0,255,0)

# camera setting
screen_w = 1920 
screen_h = 1080

wfov_degree = 30  
hfov_degree = 30 

wfov_rad = wfov_degree / 180 * math.pi 
hfov_rad = hfov_degree / 180 * math.pi 

fx_focal_length = screen_w / (2 * math.tan(wfov_rad / 2))
fy_focal_length = screen_h / (2 * math.tan(hfov_rad / 2)) 

# positioned at - 1,0,0 
camera1 = Camera(0, screen_w, screen_h) 
camera1_global_coord = np.array([-1, 0, 0])

# positioned at + 1,0,0
camera2 = Camera(1, screen_w, screen_h)   
camera2_global_coord = np.array([+1, 0, 0])

camera1.capture()
camera2.capture()
