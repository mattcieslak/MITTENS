#!/usr/bin/env python
import numpy as np
from scipy.io.matlab import loadmat
from sklearn.metrics import pairwise_distances
import os
_ROOT = os.path.abspath(os.path.dirname(__file__))

lps_neighbor_shifts = {
 'a': np.array([ 0, -1,  0]),
 'ai': np.array([ 0, -1, -1]),
 'as': np.array([ 0, -1,  1]),
 'i': np.array([ 0,  0, -1]),
 'l': np.array([1, 0, 0]),
 'la': np.array([ 1, -1,  0]),
 'lai': np.array([ 1, -1, -1]),
 'las': np.array([ 1, -1,  1]),
 'li': np.array([ 1,  0, -1]),
 'lp': np.array([1, 1, 0]),
 'lpi': np.array([ 1,  1, -1]),
 'lps': np.array([1, 1, 1]),
 'ls': np.array([1, 0, 1]),
 'p': np.array([0, 1, 0]),
 'pi': np.array([ 0,  1, -1]),
 'ps': np.array([0, 1, 1]),
 'r': np.array([-1,  0,  0]),
 'ra': np.array([-1, -1,  0]),
 'rai': np.array([-1, -1, -1]),
 'ras': np.array([-1, -1,  1]),
 'ri': np.array([-1,  0, -1]),
 'rp': np.array([-1,  1,  0]),
 'rpi': np.array([-1,  1, -1]),
 'rps': np.array([-1,  1,  1]),
 'rs': np.array([-1,  0,  1]),
 's': np.array([0, 0, 1])}




neighbor_names = sorted(lps_neighbor_shifts.keys())


ras_neighbor_shifts = {
 'a': np.array([0, 1, 0]),
 'ai': np.array([ 0,  1, -1]),
 'as': np.array([0, 1, 1]),
 'i': np.array([ 0,  0, -1]),
 'l': np.array([-1,  0,  0]),
 'la': np.array([-1,  1,  0]),
 'lai': np.array([-1,  1, -1]),
 'las': np.array([-1,  1,  1]),
 'li': np.array([-1,  0, -1]),
 'lp': np.array([-1, -1,  0]),
 'lpi': np.array([-1, -1, -1]),
 'lps': np.array([-1, -1,  1]),
 'ls': np.array([-1,  0,  1]),
 'p': np.array([ 0, -1,  0]),
 'pi': np.array([ 0, -1, -1]),
 'ps': np.array([ 0, -1,  1]),
 'r': np.array([1, 0, 0]),
 'ra': np.array([1, 1, 0]),
 'rai': np.array([ 1,  1, -1]),
 'ras': np.array([1, 1, 1]),
 'ri': np.array([ 1,  0, -1]),
 'rp': np.array([ 1, -1,  0]),
 'rpi': np.array([ 1, -1, -1]),
 'rps': np.array([ 1, -1,  1]),
 'rs': np.array([1, 0, 1]),
 's': np.array([0, 0, 1])}


def get_dsi_studio_ODF_geometry(odf_key):
    """
    Returns the default DSI studio odf vertices and odf faces for a 
    specified odf resolution
    
    Parameters:
    -----------
    odf_key:str
        Must be 'odf4', 'odf5', 'odf6', 'odf8', 'odf12' or 'odf20'
        
    Returns:
    --------
    odf_vertices, odf_faces: np.ndarray
      odf_vertices is (n,3) coordinates of the coordinate on the unit sphere and
      odf_faces is an (m,3) array of triangles between ``odf_vertices``
      
    Note:
    ------
    Here are the properties of each odf resolution
    
    Resolution: odf4
    =====================
        Unique angles: 81
        N triangles: 160
        Angluar Resolution: 17.216 +- 1.119
    
    Resolution: odf5
    =====================
        Unique angles: 126
        N triangles: 250
        Angluar Resolution: 13.799 +- 0.741
    
    Resolution: odf6
    =====================
        Unique angles: 181
        N triangles: 360
        Angluar Resolution: 11.512 +- 0.635
    
    Resolution: odf8
    =====================
        Unique angles: 321
        N triangles: 640
        Angluar Resolution: 8.644 +- 0.562
    
    Resolution: odf12
    =====================
        Unique angles: 721
        N triangles: 1440
        Angluar Resolution: 5.767 +- 0.372
    
    Resolution: odf20
    =====================
        Unique angles: 2001
        N triangles: 4000
        Angluar Resolution: 3.462 +- 0.225
    
    """
    m = loadmat(os.path.join(_ROOT,"data/odfs.mat"))
    odf_vertices = m[odf_key + "_vertices"].T
    odf_faces = m[odf_key + "_faces"].T
    return odf_vertices, odf_faces

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi

def compute_angular_probability(odf_vertices, ANGLE_MAX):
    """ 
    Computes and returns a matrix where the (i,j) entry is the probability of
    taking a step in direction j after a step in direction i

    Parameters:
    ----------

    odf_vertices: vector of tuples that specify the odf directions 

    ANGLE_MAX:float that specifies the maximum allowed distance between two
    angles for one step to follow another 

    Returns: 
    ------- 
    angular_probabilities: a matrix of floats where the i,j th
    entry gives the probability of taking a step in direction j after a step in
    direction i

    The degree to which the similarity of angles dictate the probability can be
    controlled through ANGULAR_SIM_STRENGTH
    """

    ANGULAR_SIM_STRENGTH = 4
    angular_probabilities = np.zeros((len(odf_vertices), len(odf_vertices)))
    for i, angle_1 in enumerate(odf_vertices):
        for j, angle_2 in enumerate(odf_vertices):
            similarity = angle_between(angle_1,angle_2)
            if similarity >= ANGLE_MAX:
                angular_probabilities[i][j] = 0 
            else:
                score = (180+similarity)/(180-similarity)
                angular_probabilities[i][j] = (1./score)**ANGULAR_SIM_STRENGTH
        angular_probabilities[i] = angular_probabilities[i]/angular_probabilities[i].sum()
    return angular_probabilities

def get_transition_analysis_matrices(odf_order, angle_max, 
                        angle_weight="flat", angle_weighting_power=1.):
    """
    Convenience function that creates and returns all the necessary matrices
    for iodf1 and iodf2

    Parameters:
    -----------
    odf_order: "odf4", "odf6", "odf8" or "odf12"
      A DSI Studio ODF order

    angle_max: Maximum turning angle in degrees

    angle_weights: "flat" or "weighted"

    angle_weighting_order: int
      How steep should the angle weights be? Only used when angle_weights=="weighted"

    Returns:
    ---------
    odf_vertices: np.ndarray (N,3)
      Coordinates on the ODF sphere

    prob_angles_weighted: np.ndarray(N/2,N/2)
      Each i,j in this array is the probability of taking step j given that the
      last step was i. The rows sum to 1.

    """
    odf_vertices,  odf_faces = get_dsi_studio_ODF_geometry(odf_order)
    n_unique_vertices = odf_vertices.shape[0] // 2
    angle_diffs = pairwise_distances(odf_vertices,metric=angle_between)
    compatible_angles = angle_diffs < angle_max

    if angle_weight == "flat":
        prob_angles_weighted = \
            compatible_angles.astype(np.float) / compatible_angles.sum(1)[:,np.newaxis]
    elif angle_weight == "weighted":
        prob_angles_weighted = ((180-angle_diffs)/(180+angle_diffs))**angle_weighting_power
        # Zero out the illegal transitions
        prob_angles_weighted = prob_angles_weighted * compatible_angles
        prob_angles_weighted = prob_angles_weighted / prob_angles_weighted.sum(1)[:,np.newaxis]
    # Collapse to n unique by n unique matrix
    prob_angles_weighted = prob_angles_weighted[:n_unique_vertices, :n_unique_vertices] + prob_angles_weighted[n_unique_vertices:, :n_unique_vertices]
    return odf_vertices, np.asfortranarray(prob_angles_weighted)

def weight_transition_probabilities_by_odf(odf, weight_matrix):
    """
    Creates a matrix where i,j is the probability that angle j will be taken 
    after angle i, given the weights in odf.
    """
    prob_angles_weighted = np.tile(odf[:,np.newaxis],
            (weight_matrix.shape[1] // odf.shape[0], weight_matrix.shape[0])).T * weight_matrix
    return prob_angles_weighted / prob_angles_weighted.sum(1)[:,np.newaxis]

def compute_weights_as_neighbor_voxels(odfs, weight_matrix):
    """
    Creates a matrix where each row is a voxel and each column (j) contains the 
    probability of creating a trackable direction given you entered the voxel
    with direction j.

    Parameters:
    ------------
    odfs: np.ndarray (n voxels, n unique angles)
      odf data. MUST SUM TO 1 ACROSS ROWS

    weight matrix: np.ndarray (n unique angles, n unique angles)
      Conditional angle probabilities such as those returned by 
      ``get_transition_analysis_matrices``. ALL ROWS MUST SUM TO 1

    Returns:
    --------
    weights: np.ndarray (n voxels, n unique angles)
      matrix where i,j is the probability of creating a trackable step after
      entering voxel i by angle j

    """
    return np.dot(odfs, weight_matrix)

def get_area_3d(v11, v12, v21,v22,direction,step_size=0.5):
    ''' 3D  computation of the area in v1 from which a step of size STEPSIZE in direction direction will land in the area define by v2
    '''

    def overlap(min1, max1, min2, max2):
        return max(0, min(max1, max2) - max(min1, min2)), max(min1,min2), min(max1,max2)
    
    x_min = v21[0] - step_size*direction[0]
    x_max = v22[0] - step_size*direction[0]
    
    x_delta,x_start,x_end = overlap(v11[0],v12[0],x_min,x_max)
    y_min = v21[1] - step_size*direction[1]
    y_max = v22[1] - step_size*direction[1]
    
    y_delta,y_start,y_end = overlap(v11[1],v12[1],y_min,y_max)

    z_min = v21[2] - step_size*direction[2]
    z_max = v22[2] - step_size*direction[2]
    
    z_delta,z_start,z_end = overlap(v11[2],v12[2],z_min,z_max)
  
    return x_delta*y_delta*z_delta, [x_start, y_start, z_start],[x_end,y_end,z_end]
