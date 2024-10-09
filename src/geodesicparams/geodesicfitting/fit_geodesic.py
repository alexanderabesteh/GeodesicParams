#!/usr/bin/env python3
"""


"""

from mpmath import sqrt, cos, sin
from numpy import percentile, diff, vectorize, inf
from emcee import EnsembleSampler
from scipy.optimize import minimize

from ..ellipsefitting import fit_ellipse_3d, conv_coeffs_to_axis

c = 6.3283e4 #299792 #9.454e+12
G = 39.46#6.674e-20
eps_0 = 7.54125e-6#1.1510444157e-14
#c = 1
#G = 1
#eps_0 = 1
NUT = 0
inits_dir = [1, 1]

cos = vectorize(cos, "D")
sin = vectorize(sin, "D")

cosmo = 0
q_m = 0

def fit_geodesic_orbit(data_points, init_theta, steps, config):
    """


    Parameters
    ----------


    Returns
    -------


    """

    nPoints = 500

    xyz, coeffs = fit_ellipse_3d(data_points, nPoints)

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    a, b = conv_coeffs_to_axis(coeffs)
    ecc = sqrt(1 - b**2 / a**2)
  
    print("semi = ", a)
    print("ecc = ", ecc)
  #  init_theta = [1, 0.0001328, 0, 0.00005149, -13.78, 18.78]
    bounds = ((1e-24, inf), (0, inf), (0, inf), (1e-24, inf), (0, inf), (-inf, inf), (-inf, inf))
    nll = lambda *args: log_likelihood_scipy(*args)
    soln = minimize(nll, init_theta, args = (x, y, z, ecc, a, config), bounds = bounds, method = "Nelder-Mead")
    pos = soln.x
    print(pos)

    nwalkers, ndim = pos.shape
    sampler = EnsembleSampler(nwalkers, ndim, log_probability, args = (x, y, z, ecc, a, config))

    print("Running Markov Chain Monte Carlo ...")
    sampler.run_mcmc(pos, steps, skip_initial_state_check=True, progress = True)
    samples = sampler.get_chain(flat = True)
    
    results = []
    uncertainties = []

    for i in range(ndim):
        mcmc = percentile(samples[:, i], [16, 50, 84])
        q = diff(mcmc)

        results.append(mcmc)
        uncertainties.append(q)

    return results, uncertainties, samples
