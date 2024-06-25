import cv2
import numpy as np

def find_color_in_img(img, color, threshold=20):
    # Convert the image to the same color space as the target color
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]

    # Calculate the absolute difference between the image and the target color
    diff = cv2.absdiff(hsv_img, target_color_hsv)
    diff = np.sum(diff, axis=2)

    # Find the coordinates of the pixel with the minimum difference
    min_diff_index = np.unravel_index(np.argmin(diff), diff.shape)

    # Check if the minimum difference is within the threshold
    if diff[min_diff_index] < threshold:
        return [min_diff_index[1], min_diff_index[0]]  # Note: OpenCV uses (x, y) format
    else:
        return None 

def find_color_in_img_mean(img, color, threshold=1):
    # Convert the image to the same color space as the target color
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    target_color_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]

    # Calculate the absolute difference between the image and the target color
    diff = cv2.absdiff(hsv_img, target_color_hsv)
    diff = np.sum(diff, axis=2) 

    # Find the points where the difference is under the threshold
    diff_under_thresh_true = diff < threshold

    if np.sum(diff_under_thresh_true) > 0: 
        # Get the list of indexes where the condition is true
        y_indices, x_indices = np.where(diff_under_thresh_true)
        
        # Calculate the mean coordinates
        mean_x = np.mean(x_indices)
        mean_y = np.mean(y_indices)
        
        weighted_average_index = [int(mean_x), int(mean_y)]
        return weighted_average_index
         
    else: 
        # Point doesn't exist 
        return None 

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

def main():
    color_to_find = (0, 132, 96)  # This is in RGB
    threshold = 70
    circle_color = (0, 255, 0)  # Green color for the circle in BGR
    circle_radius = 10  # Radius of the circle to draw

    cam1 = Camera(0, 640, 480)
    max_frames_per_sec = 15
    delay = int(1000 / max_frames_per_sec)

    while True:
        cur_img = cam1.capture()
        # position = find_color_in_img(cur_img, color_to_find, threshold)
        position = find_color_in_img_mean(cur_img, color_to_find, threshold)

        if position is not None:
            # Draw circle on the detected point
            cv2.circle(cur_img, (position[0], position[1]), circle_radius, circle_color, 2)

        # Show the image
        cv2.imshow('Webcam Feed', cur_img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cam1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
