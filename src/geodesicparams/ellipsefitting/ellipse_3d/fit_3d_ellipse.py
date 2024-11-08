from numpy import dot, cos, sin, cross, zeros, newaxis, arccos, array 
from numpy.linalg import norm, svd 
from mpmath import matrix, pi, linspace, findroot 

from ..guaranteed_AML_ellipse_fit.ellipse_estimates import compute_guaranteedellipse_estimate, parametric_rep
# - Rotate given points based on a starting and ending vector
# - Axis k and angle of rotation theta given by vectors n0,n1
#   P_rot = P*cos(theta) + (k x P)*sin(theta) + k*<k,P>*(1-cos(theta))
#-------------------------------------------------------------------------------
def rodrigues_rot(data_points, n0, n1):
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if data_points.ndim == 1:
        data_points = data_points[newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0 / norm(n0)
    n1 = n1 / norm(n1)
    k = cross(n0, n1)
    k = k/norm(k)
    theta = arccos(dot(n0, n1))
    
    # Compute rotated points
    data_rot = zeros((len(data_points),3))
    for i in range(len(data_points)):
        data_rot[i] = data_points[i] * cos(theta) + cross(k, data_points[i]) * sin(theta) + k * dot(k, data_points[i]) * (1 - cos(theta))

    return data_rot

def fit_ellipse_3d(data_points, nPoints):
    P_mean = data_points.mean(axis=0)
    P_centered = data_points - P_mean
    
    # Fitting plane by SVD for the mean-centered data
    U,s,V = svd(P_centered, full_matrices=False)
    
    # Normal vector of fitting plane is given by 3rd column in V
    # Note svd returns V^T, so we need to select 3rd row from V^T
    # normal on 3d plane
    normal = V[2,:]
    
    # Project points to coords X-Y in 2D plane
    P_xy = rodrigues_rot(P_centered, normal, [0,0,1])
    P_xy = matrix(P_xy[:, :2].T)

    # Use skimage EllipseModel to fit an ellipse to set of 2d points
    coeffs = compute_guaranteedellipse_estimate(P_xy) 
    # Generate n 2D points on the fitted elippse
    x, y = parametric_rep(coeffs) 
    
    #print(data_points[0, 0])
    def f(t1, t2):
        return x(t1) - P_xy[0, 0], y(t2) - P_xy[1, 0]

    start = findroot(f, (0, 0))
    theta_x = linspace(start[0], 2 * pi + start[0], nPoints)
    theta_y = linspace(start[1], 2 * pi + start[1], nPoints)

    x_list = [x(i) for i in theta_x]
    y_list = [y(i) for i in theta_y]

    xy = matrix(len(x_list), 2)
    xy[:, 0] = matrix(x_list)
    xy[:, 1] = matrix(y_list)

    # Convert the 2D generated points to the 3D space
    points = []
    for i in range(len(xy)):
        points.append([xy[i, 0], xy[i, 1], 0])
    points = array(points)
    ellipse_points_3d = rodrigues_rot(points, [0,0,1], normal) + P_mean
    
    return matrix(ellipse_points_3d), coeffs
