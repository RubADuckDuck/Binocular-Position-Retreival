import math
from camera import Camera
from utils import find_color_in_img_mean
import cv2 
import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import websockets


red = (189, 81, 88)
blue = (31, 117, 210)
yellow = (194, 211, 84)
green = (45, 127, 79)



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


def main(): 

    # marker settign q
    color_to_find = blue  # This is in RGB
    threshold = 50
    circle_radius = 10  # Radius of the circle to draw 

    circle_color = (0,255,0)

    # camera setting
    screen_w = 1920
    screen_h = 1080

    wfov_degree = 70
    hfov_degree = 30 

    wfov_rad = wfov_degree / 180 * math.pi 
    hfov_rad = hfov_degree / 180 * math.pi 

    fx_focal_length = screen_w / (2 * math.tan(wfov_rad / 2))
    fy_focal_length = screen_h / (2 * math.tan(hfov_rad / 2)) 

    # positioned at - 1,0,0 
    camera1 = Camera(0, screen_w, screen_h) 
    camera1_global_coord = np.array([1, 0, 0])

    # positioned at + 1,0,0
    camera2 = Camera(1, screen_w, screen_h)   
    camera2_global_coord = np.array([-1, 0, 0])

    max_frames_per_sec = 30
    delay = int(1000 / max_frames_per_sec)

    # Initialize 3D plot
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while True:
        cur_img1 = camera1.capture()
        cur_img2 = camera2.capture()

        position_cam1 = find_color_in_img_mean(cur_img1, color_to_find, threshold)
        position_cam2 = find_color_in_img_mean(cur_img2, color_to_find, threshold)

        if position_cam1 is not None and position_cam2 is not None:
            cv2.circle(cur_img1, (position_cam1[0], position_cam1[1]), circle_radius, circle_color, 2)
            cv2.circle(cur_img2, (position_cam2[0], position_cam2[1]), circle_radius, circle_color, 2)

            coord_3d_cam1_centered = img_2d_to_3d(position_cam1, fx=fx_focal_length, fy=fy_focal_length, wfov=wfov_rad, hfov=hfov_rad)
            coord_3d_cam2_centered = img_2d_to_3d(position_cam2, fx=fx_focal_length, fy=fy_focal_length, wfov=wfov_rad, hfov=hfov_rad) 

            print(f'\
                    cam1 dir: {coord_3d_cam1_centered}\n\
                    cam2 dir: {coord_3d_cam2_centered}\n\
                ')

            

            cam1_position = camera1_global_coord
            cam2_position = camera2_global_coord

            intersection_point = calculate_intersection_of_ray(cam1_position, coord_3d_cam1_centered, cam2_position, coord_3d_cam2_centered)

            if intersection_point is not None:
                print(f"Intersection Point: {intersection_point}")

                # Clear previous plot
                ax.cla()

                depth = 30
                box = 10
                
                # Set fixed plot size again after clearing
                ax.set_xlim([-box, box])
                ax.set_ylim([-box, box])
                ax.set_zlim([-5, depth])

                # Plot camera positions
                ax.scatter(*cam1_position, color='blue', label='Camera 1')
                ax.scatter(*cam2_position, color='red', label='Camera 2')

                len_coef = 20

                # Plot camera positions with new axes orientation (swap y and z)
                ax.scatter(cam1_position[0], cam1_position[2], cam1_position[1], color='blue', label='Camera 1')
                ax.scatter(cam2_position[0], cam2_position[2], cam2_position[1], color='red', label='Camera 2')

                # Plot rays with new axes orientation (swap y and z)
                ax.plot([cam1_position[0], cam1_position[0] + len_coef*coord_3d_cam1_centered[0]], 
                        [cam1_position[2], cam1_position[2] + len_coef*coord_3d_cam1_centered[2]], 
                        [- cam1_position[1], -(cam1_position[1] + len_coef*coord_3d_cam1_centered[1])], 
                        color='blue')

                ax.plot([cam2_position[0], cam2_position[0] + len_coef*coord_3d_cam2_centered[0]], 
                        [cam2_position[2], cam2_position[2] + len_coef*coord_3d_cam2_centered[2]], 
                        [- cam2_position[1], -(cam2_position[1] + len_coef*coord_3d_cam2_centered[1])], 
                        color='red')

                # Plot intersection point with new axes orientation (swap y and z)
                ax.scatter(intersection_point[0], intersection_point[2], - intersection_point[1], color='green', label='Intersection Point')

                # Plot global center
                ax.scatter(0, 0, 0, color='black', label='Global Center')

                # Labels and legend
                ax.set_xlabel('X axis')
                ax.set_ylabel('Y axis')
                ax.set_zlabel('Z axis')
                ax.legend()

                plt.draw()
                plt.pause(0.01)

        cv2.imshow('Webcam Feed Cam1', cur_img1)
        cv2.imshow('Webcam Feed Cam2', cur_img2)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    camera1.release()
    camera2.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()