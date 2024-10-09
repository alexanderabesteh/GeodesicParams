#!/usr/bin/env python3
"""


"""

from mpmath import sqrt, cos, sin, linspace, matrix, exp, re, nstr
from .ellipsefitting import fit_ellipse_3d, conv_coeffs_to_axis
from scipy.optimize import minimize
from .utilities import clear_directory, separate_zeros, eval_roots, inlist
from numpy import load, isfinite, inf
from .solvegeodesics import solve_geodesic_orbit, four_velocity, classify_spacetime, get_allowed_orbits 
#from corner import corner
from sympy import sin as spsin, cos as spcos, symbols, solve, degree, Poly, re as spre, pprint
from lmfit import minimize as lmminimize

def check_r_theta(theta):
    """


    Parameters
    ----------


    Returns
    -------


    """

    M, J, q_e, m, K, E, L = theta
    spacetime = classify_spacetime(q_e, J, cosmo, NUT)

    p_r, p_nu = four_velocity(M, J, q_e, cosmo, NUT, q_m, c, G, eps_0, 1, E, L, K, m)[0:2]
    deg_p = degree(p_r)
    pprint(p_r)
    pprint(p_nu)
    if J == 0:
        types, bounds, zeros_p = get_allowed_orbits(p_r, deg_p)
    else:
        types, bounds, zeros_p = get_allowed_orbits(p_r, deg_p, True)

    realNS, complexNS = separate_zeros(sorted(eval_roots(Poly(p_nu).all_roots()), key = lambda y : spre(y)))
    allowed_zeros = []

    if inlist("bound", types) != -1:
        orbit = types[inlist("bound", types)]
    else:
        orbit = types[0]

    for i in range(len(realNS)):
        if (realNS[i] >= -1 and realNS[i] <= 1):
            allowed_zeros.append(realNS[i])
    
    print(allowed_zeros)
    if len(realNS) == 0 or len(allowed_zeros) == 0 or inlist("transit", types) != -1:
        return 1e10
    else:
        return orbit

def log_likelihood_scipy(theta, x_real, y_real, z_real, ecc, semi_maj, config):
    """


    Parameters
    ----------


    Returns
    -------


    """

    workdir, date, digits = config

    M, J, q_e, m, K, E, L = theta

    #mu = G * (M + m)
  #  L = m * sqrt(mu * semi_maj * (1 - ecc**2))
   # E = -G * M * m / (2 * semi_maj)

    possible_orbit = check_r_theta(theta)
    if type(possible_orbit) == float:
        return 1e20

    inits = compute_initials(x_real[0], y_real[0], z_real[0])

    sol = solve_geodesic_orbit(M, J, q_e, cosmo, NUT, q_m, c, G, eps_0, 1, E, L, K, m, possible_orbit, config, [0] + inits, inits_dir)
    sol_r, sol_theta, sol_phi = sol
    
    if cosmo != 0:
        periodMatrix = load(workdir + "temp/rdata_" + date + ".npy", allow_pickle = True)[0]
        period = matrix(periodMatrix)[1, 0] / 2
    else:
        periodMatrix = load(workdir + "temp/rdata_" + date + ".npy", allow_pickle = True)[0]
        period = periodMatrix[0]

    mino = linspace(0, 2 * re(period), 500)

    r_list = []
    theta_list = []

    for i in mino:
        r_list.append(sol_r(i))
        theta_list.append(sol_theta(i))
    phi_list = sol_phi(mino)
    
    clear_directory(workdir + "temp/") 

    x_theo, y_theo, z_theo = conv_schwarzs_cart(r_list, theta_list, phi_list, digits)

    sum = 0
    for i in range(len(mino)):
        x_sum = x_real[i] - x_theo[i]
        y_sum = y_real[i] - y_theo[i]
        z_sum = z_real[i] - z_theo[i]

        sum += x_sum**2 + y_sum**2 + z_sum**2

    print("sum = ", -sum)
    return - sum * 1e-7

def log_likelihood_emcee(theta, x_real, y_real, z_real, E, L, config):
    """


    Parameters
    ----------


    Returns
    -------


    """
    
    workdir, date, digits = config

    M, J, q_e, m, K, L, E = theta
 
    inits = compute_initials(x_real[0], y_real[0], z_real[0])

    sol = solve_geodesic_orbit(M, J, q_e, cosmo, NUT, q_m, c, G, eps_0, 1, E, L, K, m, "bound", config, inits, inits_dir)
    sol_r, sol_theta, sol_phi = sol
   
    if cosmo != 0:
        periodMatrix = load(workdir + "temp/rdata_" + date + ".npy", allow_pickle = True)[0]
        period = matrix(periodMatrix)[1, 0] / 2
    else:
        periodMatrix = load(workdir + "temp/rdata_" + date + ".npy", allow_pickle = True)[0]
        period = periodMatrix[0]

    mino = linspace(0, 2 * period, 1000)
    clear_directory(workdir + "temp/") 

    r_list = []
    theta_list = []

    for i in mino:
        r_list.append(sol_r(i))
        theta_list.append(sol_theta(i))
    phi_list = sol_phi(mino)

    x_theo, y_theo, z_theo = conv_schwarzs_cart(r_list, theta_list, phi_list)

    sum = 0
    for i in range(len(mino)):
        x_sum = x_real[i] - x_theo[i]
        y_sum = y_real[i] - y_theo[i]
        z_sum = z_real[i] - z_theo[i]

        sum += x_sum**2 + y_sum**2 + z_sum**2

    return - sum

def log_priors(theta, E, L):
    """


    Parameters
    ----------


    Returns
    -------


    """

    M, J, q_e, m, K = theta
    
    if M <= 0 or J < 0 or K < 0 or m <= 0:
        return - inf
    
    possible = check_r_theta(theta, E)

    if not isfinite(possible):
        return - inf
    else:
        return 0.0 

def log_probability(theta, x, y, z, ecc, semi_maj, config):
    """


    Parameters
    ----------


    Returns
    -------


    """

    M, J, q_e, m, K, E = theta

    mu = G * (M + m)
    #E = - m * mu / (2 * semi_maj)#m * c**2
    #L = m * sqrt(mu * semi_maj * (1 - ecc**2))

    lp = log_priors(theta, E, L)
    
    if not isfinite(lp):
        return - inf
    return lp + log_likelihood_emcee(theta, x, y, z, L, config)
