#!/usr/bin/env python3
"""
Given a sample of cartesian data points of an elliptic orbit in a two-body system, compute
the astrophysical parameters that model this orbit in Plebanski-Demianski spacetimes.

This is done by using an ellipse fitting model to fit the observed cartesian data points
to an ellipse. Then, synthetic data is generated from this model to improve the accuracy
of the results. Finally, Markov Chain Monte Carlo is used with Gaussian priors on an initial
approximation of these astrophysical parameters to fit the synthetic data to analytic
solutions to geodesic equations in Plebanski-Demianski spacetimes. The results are the
astrophysical parameters (with a high precision and accuracy) that generate a geodesic orbit
that models this elliptic orbit in Plebanski-Demianski spacetimes. Additional information
such as the orbital period, standard deviation, and mean motion is also computed.

References
----------

"""

from mpmath import sqrt, cos, sin
from numpy import percentile, diff, vectorize
from numpy.random import randn
from emcee import EnsembleSampler
from scipy.optimize import minimize

from ..utilities import find_next
from ..ellipsefitting import fit_ellipse_3d, conv_coeffs_to_axis
from .mcmc_analysis import log_likelihood_scipy, log_probability_emcee
from .orbital_conversions import orbital_elements, convert_newtonian, convert_parametric_to_newton

# Constants
# In units of solar masses, astronomical units, and years
gLightSpeed = 6.3283e4
gGravConst = 39.46
gPerm = 7.54125e-6
gNut = 0
gMagnetic = 0

cos = vectorize(cos, "D")
sin = vectorize(sin, "D")

def fit_geodesic_orbit(data_points, init_theta, means, stdevs, sign_ang, init_dirs, steps, config):
    """
    Given a sample of cartesian data points of an elliptic orbit in a two-body system, 
    compute the astrophysical parameters that model this orbit in Plebanski-Demianski 
    spacetimes.
   
    This is done by using an ellipse fitting model to fit the observed cartesian data points
    to an ellipse. Then, synthetic data is generated from this model to improve the accuracy
    of the results. Finally, Markov Chain Monte Carlo is used with Gaussian priors on an 
    initial approximation of these astrophysical parameters to fit the synthetic data to 
    analytic solutions to geodesic equations in Plebanski-Demianski spacetimes. The results 
    are the astrophysical parameters (with a high precision and accuracy) that generate a 
    geodesic orbit that models this elliptic orbit in Plebanski-Demianski spacetimes. 
    Additional information such as the orbital period, standard deviation, and mean motion 
    is also computed.

    Parameters
    ----------
    data_points : matrix
        A 4xN matrix, where N is the number of data points observed. The first row contains 
        the times at which the data points were taken, the second contains the x coordinates 
        of the data points, the third contains the y coordinates of the data points, and the
        fourth contains the z coordinates of the data points.
    init_theta : list
        A list containing the desired parameters to be fitted. For space-times with a
        vanishing cosmological constant, the list contains the specific electric charge 
        eCharge and the specific angular momentum b_rot of the larger body. For non-vanishing
        space-times, the list contains the specific energy squared E, specfic angular momentum
        L, specific carter constant K, electric charge of the larger body eCharge, angular 
        momentum of the larger body a, and cosmological constant cosmo.
    means : list
        The expected values of the parameters in the initial parameter space <init_theta>.
    stdevs : list
        The standard deviations of the parameters in the initial parameter space <init_theta>.   
           sign_ang : int
        An integer, either +1 if the orbit is prograde or -1 if the orbit is retrograde.
    sign_ang : int
        An integer, either +1 if the orbit is prograde or -1 if the orbit is retrograde.
    init_dirs : list
       A list containing two integers, either +1 or -1. The first element in the list
       represents the initial direction of the geodesic (-1 for towards the larger body, +1
       for away from the larger body) while the second element represents the initial
       direction of the polar motion (-1 for towards the southern hemisphere, and +1 for
       towards the northern hemisphere). 
    steps : int
        The number of MCMC steps used in the analysis. 
    config : list
        A list containing 3 elements: the first is a string containing the path to the 
        working directory, the second is a string containing the date, and the third is
        the number of digits to be used in the computation. 

    Returns
    -------
    results : list
        The fitted astrophysical parameters in the parameter space.
    uncertainties : list
        The uncertainties in the fitted astrophysical parameters based on the 16th, 50th,
        and 84th percentile.
    orbit_elements : list
        A list containing 3 elements: the first is the mean motion of the orbiting body, the
        second is the standard gravitational parameter, and the third is the orbital period 
        of the orbiting body.
    flat_samples : matrix
        A JxK matrix, where J is the number of samples (the parameter values for each walker
        at each step in the chain), and K is the number of parameters in the parameter space.

    """

    if len(init_theta) != 2 and len(init_theta) != 6:
        raise ValueError("Theta must be either 2 or 6 elements long.")
    
    # -------------- Ellipse Fitting --------------  
    # Generate synthetic data
    xyz, coeffs, inc = fit_ellipse_3d(data_points, 500)
   
    # Semi-major axis, semi-minor axis, and eccentricity
    a, b = conv_coeffs_to_axis(coeffs)
    ecc = sqrt(1 - b**2 / a**2)

    t = data_points[:, 0]
    x = xyz[:, 1]
    y = xyz[:, 2]
    z = xyz[:, 3]

    # Find approximations of the original data points using the synthetic data
    x1_close = find_next(data_points[0, 0], x)
    x2_close = find_next(data_points[0, 1], x)
    y1_close = find_next(data_points[1, 0], y)
    y2_close = find_next(data_points[1, 1], y)
    t1 = t[0]
    t2 = t[1]
        
    #theta = [eCharge, a]
    #theta = [E, L, K, echarge, a, cosmo]

    # 2 synthetic data points projected onto Kepler orbit
    x_t1, y_t1, x_t2, y_t2 = convert_parametric_to_newton(coeffs, x1_close, y1_close, 
                                                          x2_close, y2_close, ecc, a)

    orbit_elements = orbital_elements(ecc, a, x_t1, y_t1, x_t2, y_t2, t2, t1)

    #if check_vanishing:
    #    bounds = ((-inf, inf), (0, inf))
    #else:
    #    bounds = ((0, inf), (-inf, inf), (-inf, inf), (0, inf), (-inf, inf), (-inf, inf))
  
    # -------------- Model Fitting --------------
    # Set initial parameter space used in the log likelihood function
    if len(init_theta) == 2:
        e_charge, b_rot = init_theta
        cosmo = 0

        energy, ang_mom, carter = convert_newtonian(ecc, a, inc, sign_ang, 
                        b_rot, e_charge, gGravConst, gLightSpeed, gPerm)
        mod_theta = [energy, ang_mom, carter, e_charge, b_rot, cosmo]
    else:
        mod_theta = init_theta

    # Maximum Likelihood Estimate
    nll = lambda *args: log_likelihood_scipy(*args)
    soln = minimize(nll, mod_theta, args = (x, y, z, ecc, a, inc, sign_ang, init_dirs, config))

    # -------------- Emcee --------------
    # Initialize tiny Gaussian ball around MLE
    pos = soln.x + 1e-4 * randn(32, 3)
    nwalkers, ndim = pos.shape
    sampler = EnsembleSampler(nwalkers, ndim, log_probability_emcee, args = (x, y, z, ecc, 
                                        a, inc, sign_ang, init_dirs, means, stdevs, config))
    
    # Run MCMC 
    sampler.run_mcmc(pos, steps, progress = True)
    flat_samples = sampler.get_chain(discard = 100, thin = 15, flat = True)
  
    # -------------- Results --------------  
    results = []
    uncertainties = []

    # Compute the fitted parameters and uncertainties (in the fitting)
    for i in range(ndim):
        mcmc = percentile(flat_samples[:, i], [16, 50, 84])
        q = diff(mcmc)

        results.append(mcmc[1])
        uncertainties.append(q)

    return results, uncertainties, orbit_elements, flat_samples 
