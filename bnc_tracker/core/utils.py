import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time 
import math


def find_color_in_img_mean(img, color, threshold=1, blur_ksize=(5, 5), blur_sigma=0):

    # Apply Gaussian blur to the image
    blurred_img = cv2.GaussianBlur(img, blur_ksize, blur_sigma)
    
    # Convert the image to the same color space as the target color
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
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
    
def live_plot_bnc_result(ax, 
                         cam1_position, 
                         cam2_position, 
                         coord_3d_cam1_centered, 
                         coord_3d_cam2_centered, 
                         intersection_point):
    ax.cla()

    depth = 30
    box = 10
    
    ax.set_xlim([-box, box])
    ax.set_ylim([-box, box])
    ax.set_zlim([-5, depth])

    ax.scatter(*cam1_position, color='blue', label='Camera 1')
    ax.scatter(*cam2_position, color='red', label='Camera 2')

    len_coef = 20

    ax.scatter(cam1_position[0], cam1_position[2], cam1_position[1], color='blue', label='Camera 1')
    ax.scatter(cam2_position[0], cam2_position[2], cam2_position[1], color='red', label='Camera 2')

    ax.plot([cam1_position[0], cam1_position[0] + len_coef*coord_3d_cam1_centered[0]], 
            [cam1_position[2], cam1_position[2] + len_coef*coord_3d_cam1_centered[2]], 
            [- cam1_position[1], -(cam1_position[1] + len_coef*coord_3d_cam1_centered[1])], 
            color='blue')

    ax.plot([cam2_position[0], cam2_position[0] + len_coef*coord_3d_cam2_centered[0]], 
            [cam2_position[2], cam2_position[2] + len_coef*coord_3d_cam2_centered[2]], 
            [- cam2_position[1], -(cam2_position[1] + len_coef*coord_3d_cam2_centered[1])], 
            color='red')

    ax.scatter(intersection_point[0], intersection_point[2], - intersection_point[1], color='green', label='Intersection Point')

    ax.scatter(0, 0, 0, color='black', label='Global Center')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    plt.draw()
    plt.pause(0.01)

def get_rotation_matrix_y(angle_rad):
    """
    Generate a rotation matrix for rotating around the Y-axis (XZ plane rotation).

    :param angle_rad: The angle to rotate by, in radians
    :return: The rotation matrix (3x3 numpy array)
    """
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    rotation_matrix = np.array([
        [cos_angle, 0, sin_angle],
        [0, 1, 0],
        [-sin_angle, 0, cos_angle]
    ])
    
    return rotation_matrix

hMin = 84
sMin = 113 
vMin = 158 
hMax = 179
sMax = 255
vMax = 255


# Define the lower and upper bounds of the color you want to track in HSV
lower_color = np.array([hMin, sMin, vMin])  # Adjust these values for the color to track
upper_color = np.array([hMax, sMax, vMax])  # Adjust these values for the color to track

def find_color_coordinates_hsv(frame):
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
        
        return [center_x, center_y]
    
    return None
