#!/usr/bin/env python3
"""
A collection of procedures for computing the MCMC log probability function.

A SciPy log likelihood function is also used in order to compute the Max Likelihood 
Estimation of the parameters. This MLE is used in the MCMC log probability function.

"""

from mpmath import linspace, matrix, re
from numpy import load, isfinite, inf
#from sympy import degree, Poly, re as spre, pprint

from ..utilities import clear_directory #,separate_zeros, eval_roots, inlist
from ..solvegeodesics import solve_geodesic_orbit #,four_velocity, classify_spacetime, get_allowed_orbits 
from .coordinates import conv_schwarzs_to_cart, conv_cartesian_to_schwarzs
from .orbital_conversions import convert_newtonian

# Constants
# In units of solar masses, astronomical units, and years
gLightSpeed = 6.3283e4
gGravConst = 39.46
gPerm = 7.54125e-6
gNut = 0
gMagnetic = 0

"""
def check_r_theta(theta):


    Parameters
    ----------
    theta : 

    Returns
    -------



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


"""

def log_likelihood_scipy(theta, x_real, y_real, z_real, ecc, semi_maj, orbit_inc, sign_ang, 
                         init_dirs, config):
    """
    Computes the log likelihood of the parameters in <theta>.

    This function is used to produced the Max Likelihood Estimation using SciPy minimize. The
    MLE is then used for the MCMC analysis.

    Parameters
    ----------
    theta : list
        A list containing the specific energy squared E, specfic angular momentum
        L, specific carter constant K, electric charge of the larger body e_charge, angular 
        momentum of the larger body a, and cosmological constant cosmo. 
    x_real : list
        The real x data to be fitted with the geodesic model.
    y_real : list
        The real y data to be fitted with the geodesic model.
    z_real : list
        The real z data to be fitted with the geodesic model.
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.  
    orbit_inc : float
        The orbital inclination of the geodesic in radians.
    sign_ang : int
        An integer, either +1 if the orbit is prograde or -1 if the orbit is retrograde.       
    init_dirs : list
       A list containing two integers, either +1 or -1. The first element in the list
       represents the initial direction of the geodesic (-1 for towards the larger body, +1
       for away from the larger body) while the second element represents the initial
       direction of the polar motion (-1 for towards the southern hemisphere, and +1 for
       towards the northern hemisphere).
    config : list
        A list containing 3 elements: the first is a string containing the path to the 
        working directory, the second is a string containing the date, and the third is
        the number of digits to be used in the computation.

    Returns
    -------
    sum : float
        The log likelihood of the parameter space <theta>.

    """

    workdir, date, digits = config
   
    # Determine astrophysical parameters
    if len(theta) == 2:
        e_charge, b_rot = theta
        cosmo = 0

        energy, ang_mom, carter = convert_newtonian(ecc, semi_maj, orbit_inc, sign_ang, 
                        b_rot, e_charge, gGravConst, gLightSpeed, gPerm)
    else:
        energy, ang_mom, carter, e_charge, b_rot, cosmo = theta
    
    #possible_orbit = check_r_theta(theta)
    #if type(possible_orbit) == float:
    #    return 1e20

    # Initial r, theta, and phi coordinates
    inits = conv_cartesian_to_schwarzs(x_real[0], y_real[0], z_real[0])

    # Compute solutions to the geodesic equations
    sol = solve_geodesic_orbit(b_rot, e_charge, cosmo, gNut, gMagnetic, gLightSpeed, 
            gGravConst, gPerm, 1, energy, ang_mom, carter, "bound", config, inits, init_dirs)
    sol_r, sol_theta, sol_phi = sol

    # Load period matrix to determine a full revolution of the orbiting body
    if cosmo != 0:
        period_matrix = load(workdir + "temp/rdata_" + date + ".npy", allow_pickle = True)[0]
        period = matrix(period_matrix)[1, 0]
    else:
        period_matrix = load(workdir + "temp/rdata_" + date + ".npy", allow_pickle = True)[0]
        period = period_matrix[0]

    mino = linspace(0, 2 * re(period), 500)

    r_list = []
    theta_list = []

    # Obtain theoretical data from the geodesic model
    for i in mino:
        r_list.append(sol_r(i))
        theta_list.append(sol_theta(i))
    phi_list = sol_phi(mino)
   
    x_theo, y_theo, z_theo = conv_schwarzs_to_cart(r_list, theta_list, phi_list, digits)

    # Compute log likelihood
    sum = 0
    for i in range(len(mino)):
        x_sum = x_real[i] - x_theo[i]
        y_sum = y_real[i] - y_theo[i]
        z_sum = z_real[i] - z_theo[i]

        sum += x_sum**2 + y_sum**2 + z_sum**2

    sum = -sum

    # Clear temporary files
    clear_directory(workdir + "temp/") 

    return sum

def log_likelihood_emcee(theta, x_real, y_real, z_real, ecc, semi_maj, orbit_inc, sign_ang, 
                         init_dirs, config):
    """
    Computes the log likelihood of the parameters in <theta>.

    This function is called by log_probability_emcee in order to compute the log
    probability in the MCMC analysis.

    Parameters
    ----------
    theta : list
        A list containing the specific energy squared E, specfic angular momentum
        L, specific carter constant K, electric charge of the larger body e_charge, angular 
        momentum of the larger body a, and cosmological constant cosmo. 
    x_real : list
        The real x data to be fitted with the geodesic model.
    y_real : list
        The real y data to be fitted with the geodesic model.
    z_real : list
        The real z data to be fitted with the geodesic model.
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.  
    orbit_inc : float
        The orbital inclination of the geodesic in radians.
    sign_ang : int
        An integer, either +1 if the orbit is prograde or -1 if the orbit is retrograde.       
    init_dirs : list
       A list containing two integers, either +1 or -1. The first element in the list
       represents the initial direction of the geodesic (-1 for towards the larger body, +1
       for away from the larger body) while the second element represents the initial
       direction of the polar motion (-1 for towards the southern hemisphere, and +1 for
       towards the northern hemisphere).
    config : list
        A list containing 3 elements: the first is a string containing the path to the 
        working directory, the second is a string containing the date, and the third is
        the number of digits to be used in the computation.

    Returns
    -------
    sum : float
        The log likelihood of the parameter space <theta>.
        
    """
    
    workdir, date, digits = config

    # Determine astrophysical parameters
    if len(theta) == 2:
        e_charge, b_rot = theta
        cosmo = 0

        energy, ang_mom, carter = convert_newtonian(ecc, semi_maj, orbit_inc, sign_ang, 
                        b_rot, e_charge, gGravConst, gLightSpeed, gPerm)
    else:
        energy, ang_mom, carter, e_charge, b_rot, cosmo = theta
    
    # Initial r, theta, and phi coordinates
    inits = conv_cartesian_to_schwarzs(x_real[0], y_real[0], z_real[0])

    # Compute solutions to the geodesic equations
    sol = solve_geodesic_orbit(b_rot, e_charge, cosmo, gNut, gMagnetic, gLightSpeed, 
            gGravConst, gPerm, 1, energy, ang_mom, carter, "bound", config, inits, init_dirs)
    sol_r, sol_theta, sol_phi = sol
    
    # Load period matrix to determine a full revolution of the orbiting body 
    if cosmo != 0:
        period_matrix = load(workdir + "temp/rdata_" + date + ".npy", allow_pickle = True)[0]
        period = matrix(period_matrix)[1, 0]
    else:
        period_matrix = load(workdir + "temp/rdata_" + date + ".npy", allow_pickle = True)[0]
        period = period_matrix[0]

    mino = linspace(0, 2 * period, 500)
   
    r_list = []
    theta_list = []
  
    # Obtain theoretical data from the geodesic model
    for i in mino:
        r_list.append(sol_r(i))
        theta_list.append(sol_theta(i))
    phi_list = sol_phi(mino)
    
    x_theo, y_theo, z_theo = conv_schwarzs_to_cart(r_list, theta_list, phi_list, digits)

    # Compute log likelihood
    sum = 0
    for i in range(len(mino)):
        x_sum = x_real[i] - x_theo[i]
        y_sum = y_real[i] - y_theo[i]
        z_sum = z_real[i] - z_theo[i]

        sum += x_sum**2 + y_sum**2 + z_sum**2
   
    sum = - sum

    # Clear temporary files
    clear_directory(workdir + "temp/") 

    return sum

def gaussian_prior(param, mu, sigma):
    """
    Compute the log of a Gaussian prior of a parameter <param> with expected value <mu> 
    and standard deviation <sigma>.

    Parameters
    ----------
    param : float
        A parameter for which the Gaussian prior is to be computed.
    mu : float
        The expected value of <param>. 
    sigma : float
        The standard deviation of <param>.

    Returns
    -------
    float
        The log of the Gaussian prior of the parameter.

    """

    return -0.5 * (param - mu)**2/sigma**2


def log_priors(theta, means, stdevs, ecc, semi_maj, orbit_inc, sign_ang):
    """
    Compute the log of the priors function used in the MCMC analysis.

    A combination of uniform and gaussian priors are used when computing
    the priors function. 

    Parameters
    ----------
    theta : list
        A list containing the specific energy squared E, specfic angular momentum
        L, specific carter constant K, electric charge of the larger body e_charge, angular 
        momentum of the larger body a, and cosmological constant cosmo.
    means : list
        The expected values of the parameters in the parameter space <theta>.
    stdevs : list
        The standard deviations of the parameters in the parameter space <theta>. 
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.  
    orbit_inc : float
        The orbital inclination of the geodesic in radians.
    sign_ang : int
        An integer, either +1 if the orbit is prograde or -1 if the orbit is retrograde.       
    
    Returns
    -------
    ln_priors : float
        The log of the priors function.

    """

    # Determine astrophysical parameters
    if len(theta) == 2:
        e_charge, b_rot = theta
        cosmo = 0

        energy, ang_mom, carter = convert_newtonian(ecc, semi_maj, orbit_inc, sign_ang, 
                        b_rot, e_charge, gGravConst, gLightSpeed, gPerm)
        mod_theta = [energy, ang_mom, carter, e_charge, b_rot, cosmo]
    else:
        b_rot = theta[4]; carter = theta[2]
        mod_theta = theta
    
    if b_rot < 0 or carter < 0:
        return - inf

    #possible = check_r_theta(mod_theta)
    ln_priors = 0

    #if not isfinite(possible):
    #    return - inf

    for i in range(6):
        if i != 0:
            ln_priors += gaussian_prior(mod_theta[i], means[i], stdevs[i])

    return ln_priors

def log_probability_emcee(theta, means, stdevs, x_real, y_real, z_real, ecc, semi_maj, 
                          orbit_inc, sign_ang, init_dirs, config):
    """
    Compute the log probability function used in the MCMC analysis.

    The log of the posterior probability distribution is the sum of the log of the priors 
    function and log likelihood function.

    Parameters
    ----------
    theta : list
        A list containing the specific energy squared E, specfic angular momentum
        L, specific carter constant K, electric charge of the larger body e_charge, angular 
        momentum of the larger body a, and cosmological constant cosmo.
    means : list
        The expected values of the parameters in the parameter space <theta>.
    stdevs : list
        The standard deviations of the parameters in the parameter space <theta>.  
    x_real : list
        The real x data to be fitted with the geodesic model.
    y_real : list
        The real y data to be fitted with the geodesic model.
    z_real : list
        The real z data to be fitted with the geodesic model.
    ecc : float
        The eccentricity of the ellipse from 0 < ecc < 1.
    semi_maj : float
        The semi major axis of the ellipse, where semi_maj > 0.  
    orbit_inc : float
        The orbital inclination of the geodesic in radians.
    sign_ang : int
        An integer, either +1 if the orbit is prograde or -1 if the orbit is retrograde.       
    init_dirs : list
       A list containing two integers, either +1 or -1. The first element in the list
       represents the initial direction of the geodesic (-1 for towards the larger body, +1
       for away from the larger body) while the second element represents the initial
       direction of the polar motion (-1 for towards the southern hemisphere, and +1 for
       towards the northern hemisphere).
    config : list
        A list containing 3 elements: the first is a string containing the path to the 
        working directory, the second is a string containing the date, and the third is
        the number of digits to be used in the computation.

    Returns
    -------
    float
        The log probability of the parameter space <theta>.

    """
    
    lp = log_priors(theta, ecc, semi_maj, orbit_inc, sign_ang, means, stdevs)

    if not isfinite(lp):
        return - inf

    return lp + log_likelihood_emcee(theta, x_real, y_real, z_real, ecc, semi_maj, orbit_inc, 
                                     sign_ang, init_dirs, config)
