from bnc_tracker.core.utils import live_plot_bnc_result, find_color_coordinates_hsv, get_rotation_matrix_y
from bnc_tracker.core.camera import Camera


import math
import cv2 
import numpy as np 

import matplotlib.pyplot as plt
##from mpl_tooqlkits.mplot3d import Axes3D 

import asyncio
import mmap
import struct

import logging
import time 
import keyboard

# Define the size of the shared memory segment
shm_size = 1024


class Timer: 
    def __init__(self): 
        self.prev_time = time.time()
        self.cur_time = time.time()

    def update_cur_time(self):
        self.cur_time = time.time() 

    def update_prev_time(self,):
        self.prev_time = time.time()

    def track(self,): 
        self.update_cur_time() 
        
        duration = self.cur_time - self.prev_time 
        # print(duration) 

        self.prev_time = self.cur_time
        return duration 
    
frame_timer = Timer()
print_timer = Timer()


# Setup logging
logging.basicConfig(level=logging.INFO)


red = (190, 108, 112)
blue = (31, 117, 210)
yellow = (194, 211, 84)
green = (45, 127, 79)
purple = (159, 147, 245)

do_plot = True


def calculate_intersection_of_ray(p1, d1, p2, d2):
    """
    Calculate the intersection (closest point) of two rays in 3D space.

    :param p1: Starting point of the first ray (numpy array)
    :param d1: Direction vector of the first ray (numpy array)
    :param p2: Starting point of the second ray (numpy array)
    :param d2: Direction vector of the second ray (numpy array)
    :return: Intersection point (numpy array)
    """
    # Ensure direction vectors are normalized
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    
    # Compute the cross product of the direction vectors
    cross_d1_d2 = np.cross(d1, d2)
    denom = np.dot(cross_d1_d2, cross_d1_d2)
    
    # If the cross product is zero, the rays are parallel
    if denom == 0:
        return None
    
    # Compute the difference between the start points
    dp = p2 - p1
    
    # Compute the parameters of the closest points on the lines
    t1 = np.dot(np.cross(dp, d2), cross_d1_d2) / denom
    t2 = np.dot(np.cross(dp, d1), cross_d1_d2) / denom
    
    # Compute the closest points on the lines
    closest_point1 = p1 + t1 * d1
    closest_point2 = p2 + t2 * d2
    
    # The intersection point is the midpoint of the closest points
    intersection_point = (closest_point1 + closest_point2) / 2
    
    return intersection_point

def img_2d_to_3d(img_coord, fx, fy, wfov, hfov): 
    u = img_coord[0]
    v = img_coord[1] 

    x = u / fx - math.tan(wfov / 2)
    y = v / fy - math.tan(hfov / 2) 
    z = 1 

    viewport_3d_coord = np.array([x, y, z])  

    return viewport_3d_coord



def process_camera_feed():
    color_to_find = blue  # This is in RGB
    threshold = 20
    circle_radius = 10  # Radius of the circle to draw 
    circle_color = (0, 255, 0)

    screen_w = 1920
    screen_h = 1080

    wfov_degree = 110
    hfov_degree = 30 

    camera_rotation_degree = 0
    camera_rotation_rad = camera_rotation_degree / 180 * math.pi
    
    # calculate rotation matrix in prior 
    camera_rot_mat : np.ndarray = get_rotation_matrix_y(angle_rad=camera_rotation_rad)

    wfov_rad = wfov_degree / 180 * math.pi 
    hfov_rad = hfov_degree / 180 * math.pi 

    fx_focal_length = screen_w / (2 * math.tan(wfov_rad / 2))
    fy_focal_length = screen_h / (2 * math.tan(hfov_rad / 2)) 

    camera1 = Camera(0, screen_w, screen_h) 
    camera1_global_coord = np.array([-1, 0, 0])
    camera2 = Camera(1, screen_w, screen_h)   
    camera2_global_coord = np.array([1, 0, 0]) 



    max_frames_per_sec = 60
    delay = int(1000 / max_frames_per_sec) 

    current_frame = 0

    if do_plot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Create a named shared memory file
    with mmap.mmap(-1, shm_size, tagname='Local\\memfile', access=mmap.ACCESS_WRITE) as mm:

        while True:
            cur_img1 = camera1.capture()
            cur_img2 = camera2.capture()

            position_cam1 = find_color_coordinates_hsv(cur_img1)
            position_cam2 = find_color_coordinates_hsv(cur_img2)

            # Handle key inputs using keyboard module
            if keyboard.is_pressed('e'):
                # exchange camera positions
                temp = camera1_global_coord 
                camera1_global_coord = camera2_global_coord
                camera2_global_coord = temp 


            elif keyboard.is_pressed('w'):
                # invert angle
                camera_rotation_degree = - camera_rotation_degree
                camera_rotation_rad = camera_rotation_degree / 180 * math.pi
                camera_rot_mat = get_rotation_matrix_y(angle_rad=camera_rotation_rad) 
                print(camera_rotation_degree)

            elif keyboard.is_pressed('u'):  # up arrow key
                # increase angle
                camera_rotation_degree += 1  # Increment by 1 degree
                camera_rotation_rad = camera_rotation_degree / 180 * math.pi
                camera_rot_mat = get_rotation_matrix_y(angle_rad=camera_rotation_rad)
                print(camera_rotation_degree)

            elif keyboard.is_pressed('d'):  # down arrow key
                # decrease angle 
                camera_rotation_degree -= 1  # Decrement by 1 degree
                camera_rotation_rad = camera_rotation_degree / 180 * math.pi
                camera_rot_mat = get_rotation_matrix_y(angle_rad=camera_rotation_rad)
                print(camera_rotation_degree)

            elif keyboard.is_pressed('q'):
                break
            

            if position_cam1 is not None and position_cam2 is not None:
                if do_plot:
                    cv2.circle(cur_img1, (position_cam1[0], position_cam1[1]), circle_radius, circle_color, 2)
                    cv2.circle(cur_img2, (position_cam2[0], position_cam2[1]), circle_radius, circle_color, 2)

                coord_3d_cam1_centered = img_2d_to_3d(position_cam1, fx=fx_focal_length, fy=fy_focal_length, wfov=wfov_rad, hfov=hfov_rad)
                coord_3d_cam2_centered = img_2d_to_3d(position_cam2, fx=fx_focal_length, fy=fy_focal_length, wfov=wfov_rad, hfov=hfov_rad) 
 
                if camera_rotation_rad != 0: 
                    coord_3d_cam1_centered = np.dot(camera_rot_mat, coord_3d_cam1_centered) 
                    coord_3d_cam2_centered = np.dot(camera_rot_mat.T, coord_3d_cam2_centered) 

                cam1_position = camera1_global_coord
                cam2_position = camera2_global_coord

                intersection_point = calculate_intersection_of_ray(cam1_position, coord_3d_cam1_centered, cam2_position, coord_3d_cam2_centered)

                if intersection_point is not None:
                    # print(f"Intersection Point: {intersection_point}")

                    # Send the intersection point via WebSocket
                    message = f"{intersection_point[0]},{intersection_point[1]},{intersection_point[2]}"
                    message = message.encode()
                    # Send the message
                    # Write data to shared memory 
                    mm.seek(0) 
                    # Pack the 3 coordinate as floats 
                    data = struct.pack('fff', intersection_point[0], intersection_point[1], intersection_point[2]) 
                    mm.write(data) 
                    # Ensure data is written by 'flushing'? 
                    mm.flush()

                    if do_plot:
                        # live plot 
                        live_plot_bnc_result(ax=ax, 
                                            cam1_position=cam1_position, 
                                            cam2_position=cam2_position, 
                                            coord_3d_cam1_centered=coord_3d_cam1_centered,
                                            coord_3d_cam2_centered=coord_3d_cam2_centered, 
                                            intersection_point=intersection_point)

            if do_plot:
                cv2.imshow('Webcam Feed Cam1', cur_img1)
                cv2.imshow('Webcam Feed Cam2', cur_img2) 

            # check time 
            frame_duration = frame_timer.track() 
            fps = 1 / frame_duration 
            # print(f'Current_frames: {fps}')

            
            if 0xFF == ord('q'):
                break

    camera1.release()
    camera2.release()

    cv2.destroyAllWindows()

    plt.ioff()
    plt.show()


def main(): 
    process_camera_feed() 

main()