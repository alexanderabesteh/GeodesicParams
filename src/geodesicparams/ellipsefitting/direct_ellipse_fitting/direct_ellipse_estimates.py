from mpmath import matrix, norm
from ..helper_functions.normalise_2d_pts import norm_2d_pts
from .ellipse_fit import direct_ellipse_fit

def compute_directellipse_estimate(data_points):

    x = data_points
    n = x.cols

    x_new = matrix(3, n)

    x_new[0, :] = x[0, :]
    x_new[1, :] = x[1, :]
    x_new[2, :] = 1

    x, t = norm_2d_pts(x_new)
    norm_data = x

    theta = direct_ellipse_fit(norm_data)
    theta = theta / norm(theta)
    a = theta[0, 0]; b = theta[0, 1]; c = theta[0, 2]; d = theta[1, 0]; e = theta[1, 1]; f = theta[1, 2]
    C = matrix([[a, b / 2, d /2], [b/2, c, e/2], [d/2, e/2, f]])

    C = t.T * C * t
    aa = C[0, 0]
    bb = C[0, 1] * 2
    dd = C[0, 2] * 2
    cc = C[1, 1]
    ee = C[1, 2] * 2
    ff = C[2, 2]
    
    theta = matrix([aa, bb, cc, dd, ee, ff]).T
    theta = theta / norm(theta)

    return theta
