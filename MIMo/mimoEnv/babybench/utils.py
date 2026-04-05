import numpy as np


def angle_between_vectors(v1, v2):
    if np.linalg.norm(v1)==0 or np.linalg.norm(v2)==0:
        return np.inf
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle = np.arccos(dot_product)
    return np.degrees(angle)

def angle_between_vector_and_rotation(vector, rotation_matrix):
    if np.linalg.norm(vector)==0:
        return np.inf
    # Normalize the vector
    vector = vector / np.linalg.norm(vector)
    # The forward direction is the third column of the rotation matrix
    forward_direction = rotation_matrix[:, 2]
    # Compute the dot product
    dot_product = np.dot(vector, forward_direction)
    # Clamp the dot product to [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    # Compute the angle in radians
    angle = np.arccos(dot_product)
    # Convert to degrees
    angle_degrees = np.degrees(angle)
    return angle_degrees
