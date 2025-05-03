import numpy as np
import SimpleITK as sitk


def sitk_jacobian_determinant(deformation: np.ndarray, transpose_displacements=True):
    '''
    deformation - 3, X, Y, Z, 3
    '''
    deformation = np.transpose(deformation, [1,2,3,0])
    print("SITK deformation shape:", deformation.shape)
    if transpose_displacements:
        deformation = deformation[:, :, :, [2,1,0]]
    #print(deformation[350, 200, 200, :])
    sitk_displacement_field = sitk.GetImageFromArray(deformation, isVector=True)
    jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
    jacobian_det_np_arr = sitk.GetArrayFromImage(jacobian_det_volume)
    #n_count = np.sum(jacobian_det_np_arr < 0)
    return jacobian_det_np_arr



def surrounding_points(coord: tuple, deformation: np.ndarray, jacobian_det: np.ndarray):
    """
    Print out the surrounding points of a specific coordinate in the deformation field
    along with their displacement vectors and Jacobian determinants.
    
    Parameters:
    - coord: The coordinate of the point (z, y, x).
    - deformation: The deformation field (3D vector field).
    - jacobian_det: The Jacobian determinant volume.
    """
    curr_coord = coord
    # Get the coordinates of the surrounding points
    curr_coord_up = (curr_coord[0], curr_coord[1] - 1, curr_coord[2])
    curr_coord_down = (curr_coord[0], curr_coord[1] + 1, curr_coord[2])
    curr_coord_left = (curr_coord[0], curr_coord[1], curr_coord[2] - 1)
    curr_coord_right = (curr_coord[0], curr_coord[1], curr_coord[2] + 1)
    #curr_coord_prev = (curr_coord[0] - 1, curr_coord[1], curr_coord[2])
    #curr_coord_next = (curr_coord[0] + 1, curr_coord[1], curr_coord[2])

    # Get the displacement vectors
    curr_vector = deformation[:, curr_coord[0], curr_coord[1], curr_coord[2]]
    left_vector = deformation[:, curr_coord_left[0], curr_coord_left[1], curr_coord_left[2]]
    right_vector = deformation[:, curr_coord_right[0], curr_coord_right[1], curr_coord_right[2]]
    up_vector = deformation[:, curr_coord_up[0], curr_coord_up[1], curr_coord_up[2]]
    down_vector = deformation[:, curr_coord_down[0], curr_coord_down[1], curr_coord_down[2]]
    #prev_vector = deformation[:, curr_coord_prev[0], curr_coord_prev[1], curr_coord_prev[2]]
    #next_vector = deformation[:, curr_coord_next[0], curr_coord_next[1], curr_coord_next[2]]

    # Get the jacobian determinants
    curr_det = jacobian_det[curr_coord[0], curr_coord[1], curr_coord[2]]
    left_det = jacobian_det[curr_coord_left[0], curr_coord_left[1], curr_coord_left[2]]
    right_det = jacobian_det[curr_coord_right[0], curr_coord_right[1], curr_coord_right[2]]
    up_det = jacobian_det[curr_coord_up[0], curr_coord_up[1], curr_coord_up[2]]
    down_det = jacobian_det[curr_coord_down[0], curr_coord_down[1], curr_coord_down[2]]
    #prev_det = jacobian_det[curr_coord_prev[0], curr_coord_prev[1], curr_coord_prev[2]]
    #next_det = jacobian_det[curr_coord_next[0], curr_coord_next[1], curr_coord_next[2]]

    # Print out information
    print("Current point:", curr_coord)
    print("Displacement vectors (z, y, x)")
    print("CURRENT point - Displacement vector at", curr_coord, ":\t\t", curr_vector)
    print("\t\tNew position:", curr_coord + curr_vector)
    print("\tLeft displacement vector at", curr_coord_left, ":\t\t\t", left_vector)
    print("\t\tNew position:", curr_coord_left + left_vector)
    print("\tRight displacement vector at", curr_coord_right, ":\t\t\t", right_vector)
    print("\t\tNew position:", curr_coord_right + right_vector)
    print("\tUp displacement vector at", curr_coord_up, ":\t\t\t", up_vector)
    print("\t\tNew position:", curr_coord_up + up_vector)
    print("\tDown displacement vector at", curr_coord_down, ":\t\t\t", down_vector)
    print("\t\tNew position:", curr_coord_down + down_vector)
    #print("\tPrevious section displacement vector at", curr_coord_prev, ":\t", prev_vector)
    #print("\t\tNew position:", curr_coord_prev + prev_vector)
    #print("\tNext section displacement vector at", curr_coord_next, ":\t\t", next_vector)
    #print("\t\tNew position:", curr_coord_next + next_vector)

    print("\nDeterminants")
    print("CURRENT point - Jacobian determinant at", curr_coord, ":\t\t", curr_det)
    print("\tLeft Jacobian determinant at", curr_coord_left, ":\t\t\t", left_det)
    print("\tRight Jacobian determinant at", curr_coord_right, ":\t\t\t", right_det)
    print("\tUp Jacobian determinant at", curr_coord_up, ":\t\t\t", up_det)
    print("\tDown Jacobian determinant at", curr_coord_down, ":\t\t\t", down_det)
    #print("\tPrevious section Jacobian determinant at", curr_coord_prev, ":\t", prev_det)
    #print("\tNext section Jacobian determinant at", curr_coord_next, ":\t\t", next_det)
    