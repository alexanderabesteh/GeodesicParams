from mpmath import sqrt, cos, sin, linspace, matrix, exp, re, nstr
from .ellipsefitting import fit_ellipse_3d, conv_coeffs_to_axis
from scipy.optimize import minimize
from numpy import percentile, diff, vectorize
from emcee import EnsembleSampler
from .utilities import clear_directory, separate_zeros, eval_roots, inlist
from numpy import load, isfinite, inf
from .solvegeodesics import solve_geodesic_orbit, four_velocity, classify_spacetime, get_allowed_orbits 
#from corner import corner
from sympy import sin as spsin, cos as spcos, symbols, solve, degree, Poly, re as spre, pprint
from lmfit import minimize as lmminimize

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

def conv_schwarzs_cart(r_list, theta_list, phi_list, digits):

    x = r_list * cos(phi_list) * sin(theta_list) 
    y = r_list * sin(phi_list) * sin(theta_list)
    z = r_list * cos(theta_list)
   
    for i in range(len(x)):
        x[i] = float(nstr(re(x[i]), digits))
        y[i] = float(nstr(re(y[i]), digits))
        z[i] = float(nstr(re(z[i]), digits))

    return x, y, z

def conv_celestial_to_cartesian(distance, ra, dec):

    x = (distance * cos(dec)) * cos(ra)
    y = (distance * cos(dec)) * sin(ra)
    z = distance * sin(dec)

    return x, y, z

def conv_real_to_apparent(x, y, orbit_elements):
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
    r, theta, phi = symbols("r theta phi", positive = True)

    inits = list(solve([r * spcos(phi) * spsin(theta) - x_init, r * spsin(phi) * spsin(theta) - y_init, r * spcos(theta) - z_init], [r, theta, phi])[0])

    return inits

def check_r_theta(theta):
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
    M, J, q_e, m, K = theta
    
    if M <= 0 or J < 0 or K < 0 or m <= 0:
        return - inf
    
    possible = check_r_theta(theta, E)

    if not isfinite(possible):
        return - inf
    else:
        return 0.0 

def log_probability(theta, x, y, z, ecc, semi_maj, config):
    M, J, q_e, m, K, E = theta

    mu = G * (M + m)
    #E = - m * mu / (2 * semi_maj)#m * c**2
    #L = m * sqrt(mu * semi_maj * (1 - ecc**2))

    lp = log_priors(theta, E, L)
    
    if not isfinite(lp):
        return - inf
    return lp + log_likelihood_emcee(theta, x, y, z, L, config)

def fit_geodesic_orbit(data_points, init_theta, steps, config):
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
