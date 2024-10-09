#!/usr/bin/env python3
"""
Procedures for computing orbital elements in a two-body system.

These procedures include the true anomaly, eccentric anomaly, mean anomaly, mean motion,
orbital period, etc. A procedure for converting from the Newtonian parametrization of
a geodesic to its integrals of motion counterpart has also been implemented.

"""

from mpmath import acos, tan, atan, sqrt, sin, pi, cos

def true_anomaly(ecc, semi_maj):
    """
    Computes the true anomaly of an orbiting body as a function of the distance from the
    origin.  

    Parameters
    ----------
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.

    Returns
    -------
    callable
        A lambda function that returns the true anomaly as a function of the distance
        from the origin.

    """

    return lambda x : acos(((semi_maj * (1 - ecc**2) / x) - 1) / ecc)

def ecc_anomaly(ecc, semi_maj):
    """
    Computes the eccentric anomaly of an orbiting body as a function of the distance from the
    origin.

    Parameters
    ----------
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.

    Returns
    -------
    callable
        A lambda function that returns the eccentric anomaly as a function of the distance
        from the origin.

    """

    nu = true_anomaly(ecc, semi_maj)
    denom = sqrt((1 + ecc) / (1 - ecc))

    return lambda x : 2 * atan(tan(nu(x) / 2) / denom)

def mean_anomaly(ecc, semi_maj):
    """
    Computes the mean anomaly of an orbiting body as a function of the distance from the
    origin.

    Parameters
    ----------
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.

    Returns
    -------
    callable
        A lambda function that returns the mean anomaly as a function of the distance
        from the origin.

    """

    E = ecc_anomaly(ecc, semi_maj)

    return lambda x : E(x) - ecc * sin(E(x))

def mean_motion(ecc, semi_maj, mean0, t0):
    """
    Computes the mean motion of an orbiting body from a reference point with an initial mean
    anomaly <mean0> and an initial time <t0> at which <mean0> was taken.

    Parameters
    ----------
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.
    mean0 : float
        The initial mean anomaly at time <t0>. 
    t0 : float
        The initial time at which the mean anomaly <mean0> was taken.

    Returns
    -------
    callable
        A lambda function that returns the mean motion as a function of the distance
        from the origin.   

    """

    M = mean_anomaly(ecc, semi_maj)

    return lambda x, t1 : (M(x) - mean0) / (t1 - t0)

def orbital_elements(ecc, semi_maj, mean1, mean0, t1, t0):
    """
    Computes the orbital elements of a two body system, including the mean motion n, the
    standard gravitational parameter mu, and the orbital period P.

    This is done by using the change in mean anomaly / the change in time to compute the
    mean motion, which can then be used to compute the standard gravitational parameter and
    the orbital period.

    Parameters
    ----------
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.
    mean1 : float
        The mean anomaly at time <t1>.
    mean0 : float
        The initial mean anomaly at time <t0>. 
    t1 : float
        The time at which the mean anomaly <mean1> was taken.
    t0 : float
        The initial time at which the mean anomaly <mean0> was taken.

    Returns
    -------
    n : float
        The mean motion of the system.
    standard_grav : float
        The standard gravitational parameter.
    orbital_period : float
        The orbital period of the orbiting body.

    """

    n = mean_motion(ecc, semi_maj, mean0, t0)(mean1, t1)
    standard_grav = semi_maj**3 * n**2
    orbital_period = 2 * pi * sqrt(semi_maj**3 / standard_grav)

    return n, standard_grav, orbital_period

def convert_newtonian(ecc, semi_maj, orbit_inc, sign_ang, b_rot, eCharge, b_mass, grav_const, speed_light, perm):
    """
    Convert the Newtonian parametrization of geodesics involving the eccentricity <ecc>, the
    semi major axis <semi_maj>, and orbital inclination <orbit_inc> to its integrals of motion
    counterpart (specific energy, specific angular momentum, and carter constant).

    This procedure only works for space-times with a vanishing cosmological constant (
    Schwarzschild, Reissner-Nordstrom, Kerr, and Kerr-Newman).

    Parameters
    ----------
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.
    orbit_inc : float
        The orbital inclination of the geodesic.
    sign_ang : int
        An integer, either +1 if the orbit is prograde or -1 if the orbit is retrograde.
    b_rot : float
        The angular momentum of the central body, where rot >= 0.
    eCharge : float
        The electric charge of the central body, where eCharge >= 0.
    b_mass : float
        The mass of the central body, where b_mass > 0.
    grav_const : float
        The gravitational constant.
    speed_light : float
        The speed of light.
    perm : float
        The permitivity of free space.

    Returns
    -------
    energies : list
        A list containing the specific energies of the geodesic squared. The first element
        assumes the branch of the square root in the quadratic solution is positive, while
        the second assumes the branch of the square root is negative.
    ang_moms : list
        A list containing the specific angular momentums of the geodesic. The first element
        assumes the energy used in the computation took the positive branch for its solution,
        while the second assumes the negative branch of the square root was taken.
    carters : list
        A list containing the carter constants of the geodesic. The first element assumes the 
        energy and angular momentum used in the computation took the positive branch for its 
        solution, while the second assumes the negative branch of the square root was taken. 

    """

    # Initial parameters
    schwarz_rad = 2 * grav_const * b_mass / speed_light**2
    char_length = eCharge**2 * grav_const / (4 * pi * perm * speed_light**4)
    theta_min = (pi/2 - orbit_inc) / sign_ang
    nu_min2 = cos(theta_min)**2
    p = semi_maj * (1 - ecc**2)
    r_min = p * b_mass / (1 + ecc)
    r_max = p * b_mass / (1 - ecc)
    horizon = lambda r : r**2 - schwarz_rad * r + char_length
    
    # Functions used in the conversion procedure
    f_r = lambda r : r**4 + b_rot**2 * (r * (r + 2) + nu_min2 * horizon(r))
    g_r = lambda r : 2 * b_rot * r
    h_r = lambda r : r * (r - 2) + (nu_min2 / (1 - nu_min2)) * horizon(r)
    d_r = lambda r : (r**2 + b_rot**2 * nu_min2) * horizon(r)

    f1 = f_r(r_min); g1 = g_r(r_min); h1 = h_r(r_min); d1 = d_r(r_min)
    f2 = f_r(r_max); g2 = g_r(r_max); h2 = h_r(r_max); d2 = d_r(r_max)

    # Determinants of the matrices in []
    kappa =  d1 * h2 - h1 * d2
    epsilon = d1 * g2 - g1 * d2
    rho = f1 * h2 - h1 * f2
    eta = f1 * g2 - g1 * f2
    sigma = g1 * h2 - h1 * g2

    # Energies of the geodesic squared
    # Since the solution for the energy squared is quadratic,
    # there is a positive and negative branch for the square
    # root in the quadratic formula
    energy2 = lambda sign : (kappa * rho + 2*epsilon*sigma + sign * 2 * sqrt(sigma * (
        sigma*epsilon**2 + rho * epsilon * kappa - eta * kappa**2))) / (rho**2 + 4 * eta * sigma)
    energy2_pos = energy2(1)
    energy2_neg = energy2(-1)
    energies = [energy2_pos, energy2_neg]

    # Angular momentums of the geodesic
    # Computed using the positive and negative branches of
    # the energies
    ang_mom = lambda energy, sign_ang : -g1*sqrt(energy)/h1 + sign_ang * sqrt(
        (g1**2 * energy / h1**2) + (f1*energy - d1) / h1)
    ang_mom_pos = ang_mom(energy2_pos, sign_ang)
    ang_mom_neg = ang_mom(energy2_neg, sign_ang)
    ang_moms = [ang_mom_pos, ang_mom_neg]

    # Carter constants of the geodesic
    # Computed using the positive and negative branches of
    # the energies
    beta_pos = b_rot**2 * (1 - energy2_pos)
    beta_neg = b_rot**2 * (1 - energy2_neg)
    carter = lambda ang_mom, beta : nu_min2 * (beta + (ang_mom**2 / (1 - nu_min2)))
    carter_pos = carter(ang_mom_pos, beta_pos)
    carter_neg = carter(ang_mom_neg, beta_neg)
    carters = [carter_pos, carter_neg]

    return energies, ang_moms, carters
