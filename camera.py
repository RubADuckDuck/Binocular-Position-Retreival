import cv2

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
