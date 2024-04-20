from mpmath import sqrt, cos, sin, atan2, matrix, norm
from ..helper_functions.normalise_2d_pts import norm_2d_pts
from ..direct_ellipse_fitting.ellipse_fit import direct_ellipse_fit
from .guaranteed_ellipse_fit import guaranteedEllipseFit

def compute_guaranteedellipse_estimate(data_points):

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

    theta = guaranteedEllipseFit(theta, x)
    theta = theta / norm(theta)

    a = theta[0]; b = theta[1]; c = theta[2]; d = theta[3]; e = theta[4]; f = theta[5]
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

def conv_coeffs_to_axis(coeffs):

    a, b, c, d, e, f = coeffs
    convert = lambda sign : - sqrt(2 * (a * e**2 + c * d**2 - b * d * e + (b**2 - 4 * a * c) * f) * ((a + c) + sign * sqrt((a - c)**2 + b**2))) / (b**2 - 4 * a * c)
    semi_maj = convert(1)
    semi_min = convert(-1)

    return semi_maj, semi_min

def conv_coeffs_to_init_pos(coeffs):
    a, b, c, d, e, f = coeffs
    cond = b**2 - 4 * a * c

    x_init = (2 * c * d - b * e) / cond
    y_init = (2 * a * e - b * d) / cond
    rot_angle = 1/2 * atan2(-b, c - a)

    return [x_init, y_init, rot_angle]

def parametric_rep(coeffs):
    a, b =  conv_coeffs_to_axis(coeffs)
    x_init, y_init, rot_angle = conv_coeffs_to_init_pos(coeffs)

    x = lambda t : a * cos(rot_angle) * cos(t) - b * sin(rot_angle) * sin(t) + x_init
    y = lambda t : a * sin(rot_angle) * cos(t) + b * cos(rot_angle) * sin(t) + y_init  

    return x, y

