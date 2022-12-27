import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import numbers

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


# Note: I want to make .npz files, just like bob and spot in the data/bob_and_spot/samples folder. These two .npz files
# contain a dictionary with keys 'pos' and 'neg'. These contain numpy arrays with shape (num_points, 4) where num_points
# is the number of points where the values are positive/negative and the first three points correspond to the location
# of the point and the fourth value the sdf values.


def exp_min(sdf_1, sdf_2, k):
    """"
    This is the exponential smooth min function from https://www.iquilezles.org/www/articles/smin/smin.htm
    NOTE: in their function they use 2**(-k*a) while we use e**(-k*a)
    """
    return -np.log(np.exp(-k * sdf_1) + np.exp(-k * sdf_2) / k)


def pow_min(sdf_1, sdf_2, k):
    """"
    This is the power smooth min function from https://www.iquilezles.org/www/articles/smin/smin.htm
    """
    a = sdf_1 ** k
    b = sdf_2 ** k
    return ((a * b) / (a + b)) ** k


def root_min(sdf_1, sdf_2, k):
    """"
    This is the root smooth min function from https://www.iquilezles.org/www/articles/smin/smin.htm
    """
    return 0.5 * (sdf_1 + sdf_2 - np.sqrt((sdf_1 - sdf_2) ** 2 + k))


def poly_min(sdf_1, sdf_2, k):
    """"
    This is the polynomial smooth min function from https://www.iquilezles.org/www/articles/smin/smin.htm
    """
    return np.minimum(sdf_1, sdf_2) - 0.25 * (np.maximum(k - np.abs(sdf_1 - sdf_2), 0.0) / k) ** 2 * k


def sdf_concatenate(sdf_1, sdf_2, k, method="numpy"):
    """"
    This function takes as input two vectors with sdf values and calculates a smooth minimum of the sdf values.
    This is done in order to combine to SDFs into one (as the minimization operation applied on two SDFs results in a
    new SDF). The choices of the (smooth) minimizers are:
        - numpy:    this just uses np.minimum.
        - exp:      this uses the exp_min function
        - pow:      this uses the pow_min function
        - root:     this uses the root_min function
        - poly:     this uses the poly_min function.
    The input parameter 'k' of sdf_concatenate corresponds to the value of k in exp_min, pow_min, root_min, and poly_min
    """
    if method == "numpy":
        return np.minimum(sdf_1, sdf_2)
    elif method == "exp":
        return exp_min(sdf_1, sdf_2, k)
    elif method == "pow":
        return pow_min(sdf_1, sdf_2, k)
    elif method == "root":
        return root_min(sdf_1, sdf_2, k)
    elif method == "poly":
        return poly_min(sdf_1, sdf_2, k)
    else:
        raise ValueError("The given method does not exist.")


class RotationOperator2D:
    def __init__(self, angle):
        self.angle = angle
        self.R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    def apply(self, xy):
        """

        Args:
            xy: (n, 2) numpy array

        Returns:
            rotated version of the numpy array
        """
        return np.matmul(self.R, xy.transpose((1, 0))).transpose((1, 0))


def create_rotation_operator(angles):
    """"
    This function creates a 3D rotation function with the three angles given in 'angles'
    """
    if isinstance(angles, numbers.Number):
        return RotationOperator2D(angles)
    elif len(angles) == 3:
        return R.from_euler('xyz', angles)
    else:
        raise ValueError("The input should be a list with 3 angles in 3D or a number for an angle in 1D.")


def init_samples(num_points, dim=3):
    """"
    This function creates a (num_points**3, 4) tensor. Here the rows correspond to positions in the grid. The first
    column corresponds to the x coordinate of the point, the second column to its y coordinate, and the third column
    to its z coordinate. The last column of the tensor will later on be modified to contain the SDF value at the
    specific point.
    """

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1] * dim
    voxel_size = 2.0 / (num_points - 1)

    overall_index = np.arange(0, num_points ** dim, 1)
    samples = np.zeros((num_points ** dim, dim + 1))

    # transform first dim columns
    # to be the x, y, z index
    for i in range(dim):
        k = dim - i - 1
        samples[:, k] = overall_index.astype(int).copy()
        for j in range(i):
            samples[:, k] = samples[:, k] // num_points
        samples[:, k] = samples[:, k] % num_points

        # samples[:, 2] = overall_index % num_points
        # samples[:, 1] = (overall_index.astype(int) // num_points) % num_points
        # samples[:, 0] = ((overall_index.astype(int) // num_points) // num_points) % num_points

    # Transform first dim columns
    samples = samples.astype(float)
    samples[:, :dim] = (samples[:, :dim] * voxel_size) + voxel_origin

    # # transform first 3 columns
    # # to be the x, y, z coordinate
    # samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    # samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    # samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples


def projection_dist_to_plane(x, a, b):
    """"
    The projection of x onto the plane described by a^T x + b = 0
    """
    return np.abs(x @ a - b) / np.linalg.norm(a)


def create_plane(a, b, num_points):
    """"
    This function creates a (num_points**3, 4) tensor. Here the rows correspond to positions in the grid. The first
    column corresponds to the x coordinate of the point, the second column to its y coordinate, and the third column
    to its z coordinate. The last column of the tensor will contain the distance from the point to the plane described
    by a^Tx + b =0.
    """
    samples = init_samples(num_points)
    samples[:, 3] = projection_dist_to_plane(samples[:, 0:3], a, b)
    return samples


def create_square(center, edge_length, angles, num_points, dim=3):
    """
    The square is created by concatenating the SDFs of 6 planes. This square has center 'center', edges of length
    edge_length, and is rotated via the 3D rotation matrix (created via create_rotation_operator(angles)). The num_points
    input determines the number of points sampled uniformly along each axis of the spatial domain [-1, 1]^3.
    """

    # Need a minus sign for the calculation of the distance later on
    rotation_operation = create_rotation_operator(-angles)

    # half_edge_length
    half_edge_length = edge_length / 2

    # Get the samples
    samples = init_samples(num_points, dim)

    # Samples in different coordinates (you transform the coordinates such that you have a box with the center at the
    # origin and it is not rotated at all)
    samples_box_at_center = np.zeros_like(samples)
    samples_box_at_center[:, 0:dim] = rotation_operation.apply(samples[:, 0:dim] - center)

    # Getting the indices that are outside 'box' range (so their absolute value are bigger than half_edge_length)
    num_coord_outside_box = np.sum(np.abs(samples_box_at_center[:, 0:dim]) > half_edge_length, axis=1)
    indices_0 = (num_coord_outside_box == 0)
    indices_1 = (num_coord_outside_box == 1)
    indices_2 = (num_coord_outside_box == 2)

    # In case we want cubes:
    if dim == 3:
        indices_3 = (num_coord_outside_box == 3)

        samples[indices_0, 3] = -(half_edge_length - np.linalg.norm(samples_box_at_center[indices_0, 0:3], ord=np.inf,
                                                                    axis=1))
        samples[indices_1, 3] = np.linalg.norm(samples_box_at_center[indices_1, 0:3], ord=np.inf, axis=1) - half_edge_length
        samples[indices_2, 3] = np.linalg.norm(np.sort(np.abs(samples_box_at_center[indices_2, 0:3]))[:, 1:3], axis=1) - \
                                               np.linalg.norm(np.array([half_edge_length, half_edge_length]))
        samples[indices_3, 3] = np.linalg.norm(np.abs(samples_box_at_center[indices_3, 0:3]) -
                                               np.array([half_edge_length, half_edge_length, half_edge_length]), 2, axis=1)
    elif dim == 2:
        # In case we want 2-dimensional squares
        samples[indices_0, 2] = -(half_edge_length - np.linalg.norm(samples_box_at_center[indices_0, 0:2], ord=np.inf,
                                                                    axis=1))
        samples[indices_1, 2] = np.linalg.norm(samples_box_at_center[indices_1, 0:2], ord=np.inf,
                                               axis=1) - half_edge_length
        samples[indices_2, 2] = np.linalg.norm(np.abs(samples_box_at_center[indices_2, 0:2]) -
                                               np.array([half_edge_length, half_edge_length]), 2, axis=1)
    else:
        raise ValueError("The code currently only supports 2d squares and 3d cubes")

    return samples


def save_shape(directory_name, filename, samples):
    """"
    This function saves the SDF samples into an .npz file. It does so by splitting it into positive and negative
    SDF values.
    """

    save_directory = os.path.join(directory_name)
    if not os.path.isdir(save_directory):
        os.makedirs(save_directory)

    pos = samples[(samples[:, -1] > 0), :]
    neg = samples[(samples[:, -1] <= 0), :]

    np.savez(os.path.join(save_directory, filename), pos=pos, neg=neg)


def check_for_compatible_sdf_samples(samples_1, samples_2):
    """"
    This function checks whether two SDF sample tensors can be combined to form a new SDF
    """
    if not samples_1.shape == samples_2.shape:
        raise ValueError('The two provided samples np.arrays do not have the same shape')
    elif not np.linalg.norm(samples_1[:, 0:-1] - samples_2[:, 0:-1]) == 0:
        raise ValueError('The sdf samples are not taken at the same locations')


def combine_sdf_of_two_samples(samples_1, samples_2, k, method="numpy"):
    """"
    This function combines two SDFs into one. More precisely, it makes sure that the samples_1 and samples_2 objects are
    combined into one tensor that represent the combined SDF. Here the rows correspond to data points in the grid. The
    first three datapoints indicate the location in the grid and the fourth column indicates the corresponding SDF
    value. To combine the two SDFs a (smooth) minimizer has to be chosen. The choices of the (smooth) minimzers are:
        - numpy:    this just uses np.minimum.
        - exp:      this uses the exp_min function
        - pow:      this uses the pow_min function
        - root:     this uses the root_min function
        - poly:     this uses the poly_min function.
    The input parameter 'k' of sdf_concatenate corresponds to the value of k in exp_min, pow_min, root_min, and poly_min
    """
    check_for_compatible_sdf_samples(samples_1, samples_2)
    new_samples = np.zeros_like(samples_1)
    new_samples[:, 3] = sdf_concatenate(samples_1[:, 3], samples_2[:, 3], k, method)
    return new_samples


def combine_sdf(sample_list, k, method="numpy"):
    """"
    See previous function, but now for combining multiple SDFs.
    """
    if len(sample_list) > 2:
        new_sample_list = list(sample_list[2:])
        new_sample_list.insert(0, combine_sdf_of_two_samples(sample_list[0], sample_list[1], k, method))
        return combine_sdf(new_sample_list, k, method)
    elif len(sample_list) <= 1:
        raise ValueError('The input should at least have two samples')
    else:
        return combine_sdf_of_two_samples(sample_list[0], sample_list[1], k, method)


def add_inner_line(samples, a, b, k, method):
    """"
    This function adds a line in the inner part of on object. E.g. assume you have a sphere of radius 0.5 and centered
    at the origin. A plane a^T + b = 0 may cut through this sphere. The 'add_inner_line' function makes sure that we
    get an SDF that adds the wall in the inside of the sphere created by this cutting.
    """
    dist_to_plane = projection_dist_to_plane(samples[:, 0:3], a, b)
    new_samples = np.zeros_like(samples)
    indices = (samples[:, 3] <= 0)
    new_samples[indices, 3] = - sdf_concatenate(-new_samples[indices, 3], dist_to_plane[indices], k, method)
    return new_samples


if __name__ == "__main__":

    import math

    # Define the properties of the squares
    center_1 = np.array([0.0, 0.0, 0.0])
    center_2 = np.array([0.0, 0.0, 0.0])
    edge_length_1 = 0.5
    edge_length_2 = 0.5
    angles_1 = np.array([0.0, math.pi / 2, math.pi / 4])
    angles_2 = -np.array([math.pi / 2, 0.0, math.pi / 4])
    num_points_1 = 64
    num_points_2 = 64

    # Creating the squares
    samples_square_1 = create_square(center_1, edge_length_1, angles_1, num_points_1)
    samples_square_2 = create_square(center_2, edge_length_2, angles_2, num_points_2)

    # Saving the shape
    directory_name = os.path.join("..")
    save_shape(directory_name, "Square_1", samples_square_1)
    save_shape(directory_name, "Square_2", samples_square_2)

    # # Creating and saving the corresponding mesh
    # from deep_sdf.mesh import convert_sdf_samples_to_ply
    #
    # N1 = round(samples_square_1.shape[0] ** (1.0 / 3.0))
    # convert_sdf_samples_to_ply(samples_square_1[..., -1].reshape(N1, N1, N1), np.zeros(3), 2 / (N1 - 1),
    #                            "../data/Squares/SurfaceSamples/Squares/samples/Square_1" + ".ply")
    # N2 = round(samples_square_2.shape[0] ** (1.0 / 3.0))
    # convert_sdf_samples_to_ply(samples_square_2[..., -1].reshape(N2, N2, N2), np.zeros(3), 2 / (N2 - 1),
    #                            "../data/Squares/SurfaceSamples/Squares/samples/Square_2" + ".ply")

    # Test 2D shape maker
    samples_square_3 = create_square(center_1[:2], edge_length_1, angles_1[0], 256, dim=2)
    samples_square_4 = create_square(center_2[:2] + 0.2, edge_length_2, math.pi/4, 256, dim=2)
    save_shape(directory_name, "Square_3", samples_square_3)
    save_shape(directory_name, "Square_4", samples_square_4)

    # Use Matplotlib to show the 2D squares
    N3 = round(samples_square_3.shape[0] ** (1.0 / 2.0))
    fig1, ax1 = plt.subplots()
    ax1.imshow(samples_square_3[..., -1].reshape(N3, N3) < 0.0, origin='lower')
    fig1.savefig("../picture_of_Square_3")
    N4 = round(samples_square_4.shape[0] ** (1.0 / 2.0))

    fig2, ax2 = plt.subplots()
    ax2.imshow(samples_square_4[..., -1].reshape(N4, N4) < 0.0, origin='lower')
    fig2.savefig("../picture_of_Square_4")

    # Plot the SDF values
    fig3, ax3 = plt.subplots()
    N3 = round(samples_square_3.shape[0] ** (1.0 / 2.0))
    ax3.imshow(samples_square_3[..., -1].reshape(N3, N3), origin='lower')
    fig3.savefig("../picture_of_SDF_of_Square_3")

    fig4, ax4 = plt.subplots()
    N4 = round(samples_square_4.shape[0] ** (1.0 / 2.0))
    ax4.imshow(samples_square_4[..., -1].reshape(N4, N4), origin='lower')
    fig4.savefig("../picture_of_SDF_of_Square_4")

    # Contour plots
    fig4, ax4 = plt.subplots()
    N4 = round(samples_square_3.shape[0] ** (1.0 / 2.0))
    ax4.contour(samples_square_3[..., -1].reshape(N4, N4), origin='lower')
    fig4.savefig("../picture_of_level_sets_of_SDF_of_Square_3")

    fig4, ax4 = plt.subplots()
    N4 = round(samples_square_4.shape[0] ** (1.0 / 2.0))
    ax4.contour(samples_square_4[..., -1].reshape(N4, N4), origin='lower')
    fig4.savefig("../picture_of_level_sets_of_SDF_of_Square_4")