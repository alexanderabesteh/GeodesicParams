#!/usr/bin/env python3
"""
Procedures for converting between different coordinates.

These coordinate conversions include Schwarzschild to cartesian, celestial to cartesian, real 
orbits to apparent orbits, and cartesian to polar coordinates.

"""
from mpmath import cos, sin, re, nstr


cos = vectorize(cos, "D")
sin = vectorize(sin, "D")

def conv_schwarzs_cart(r_list, theta_list, phi_list, digits):
    """


    Parameters
    ----------


    Returns
    -------


    """

    x = r_list * cos(phi_list) * sin(theta_list) 
    y = r_list * sin(phi_list) * sin(theta_list)
    z = r_list * cos(theta_list)
   
    for i in range(len(x)):
        x[i] = float(nstr(re(x[i]), digits))
        y[i] = float(nstr(re(y[i]), digits))
        z[i] = float(nstr(re(z[i]), digits))

    return x, y, z

def conv_celestial_to_cartesian(distance, ra, dec):
    """


    Parameters
    ----------


    Returns
    -------


    """

    x = (distance * cos(dec)) * cos(ra)
    y = (distance * cos(dec)) * sin(ra)
    z = distance * sin(dec)

    return x, y, z

def conv_real_to_apparent(x, y, orbit_elements):
    """


    Parameters
    ----------


    Returns
    -------


    """

    inc, arg_peri, node = orbit_elements

    a = cos(node) * cos(arg_peri) - sin(node) * sin(arg_peri) * cos(inc) 
    b = sin(node) * cos(arg_peri) + cos(node) * sin(arg_peri) * cos(inc)
    c = sin(arg_peri) * sin(inc)
    f = - cos(node) * sin(arg_peri) - sin(node) * cos(arg_peri) * cos(inc) 
    g = - sin(node) * sin(arg_peri) + cos(node) * cos(arg_peri) * cos(inc)
    h = cos(arg_peri) * sin(inc)

    x_apparent = b * x + g * y 
    y_apparent = a * x + f * y
    z_apparent = c * x + h * y

    return x_apparent, y_apparent, z_apparent

def compute_initials(x_init, y_init, z_init):
    """


    Parameters
    ----------


    Returns
    -------


    """
    
    r, theta, phi = symbols("r theta phi", positive = True)

    inits = list(solve([r * spcos(phi) * spsin(theta) - x_init, r * spsin(phi) * spsin(theta) - y_init, r * spcos(theta) - z_init], [r, theta, phi])[0])

    return inits
