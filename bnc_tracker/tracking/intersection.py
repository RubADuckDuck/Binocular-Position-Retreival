import numpy as np

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