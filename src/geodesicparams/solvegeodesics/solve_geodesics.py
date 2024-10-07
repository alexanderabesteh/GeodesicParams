#!/usr/bin/env python3
"""
Top level procedure for computing the solutions functions for the r, theta, and phi
components of the geodesic equation in Plebanski-Demianski spacetimes.

As of now, solutions have been implemented for the following spacetimes:
Schwarzschild, Reissner Nordstrom, Kerr, Kerr-Newman, Schwarzschild-de Sitter,
Reissner Nordstrom-de Sitter, Kerr-de Sitter, and Kerr-Newman-de Sitter.
 
"""

from mpmath import mp
from numpy import vectorize
from sympy import symbols, degree, Poly, re, Eq, solve, sqrt, pprint

from .solve_eom import invert_eom, integrate_eom
from .general_relativity import get_allowed_orbits, convert_boundsinit, check_rinitials, check_thetainitials, classify_spacetime, four_velocity, check_orbit_types
from .substitution import convert_polynomial 
from ..utilities import inlist, eval_roots

def solve_geodesic_orbit(b_mass, rot, eCharge, cosmo, NUT, mCharge, speed_light, grav_const, perm, particle_light, energy, ang_mom, carter, p_mass, orbittype, config, 
                   initial_values = [], initial_directions = [], time = False, periodM = None):
    """
    Computes the r, theta, and phi coordinates of the geodesic equation as 
    a function of mino time.
    
    Parameters
    ----------
    b_mass : float
        A positive non-zero float representing the mass of the larger body.
    rot : float
        A postive float representing the rotation of the larger body.
    eCharge : float
        A float representing the electric charge of the larger body.
    cosmo: float
        A float representing the cosmological constant.
    NUT : float
        A float representing the NUT parameter.
    mCharge : float
        A float representing the magnetic charge of the larger body.
    speed_light : float
        A float representing the speed of light.
    grav_const : float
        A float representing the universal gravitational constant.
    perm : float
        A float representing the permittivity of free space.
    particle_light : integer
        An integer, either 1 (for timelike geodesics) or 0 (for null geodesics).
    energy : float
        A float representing the energy of the smaller body.
    ang_mom : float
        A float reperesenting the angular momentum of the smaller body.
    carter : float
        A float representing the carter constant.
    p_mass : float
        A positive float representing the mass of the smaller body.
    orbittype : string
        A string representing the type of orbit to be modelled.
    config : list
        A list containing either 2 or 3 elements: a string, the path to
        the working directory; a string, the current date (in any format);
        and an integer, the binary precision. The working directory and date strings
        are required, while the default binary precision is 53.
    initial_values : list, optional
        A list of floats representing the initial values: the initial mino time,
        r value, theta value, and phi value.
    initial_directions : list, optional
        A list of 2 integers, either +1 or -1, representing the initial directions
        of the r and theta components of the 4-velocity. -1 as the r value means the 
        smaller body initially moves towards the larger body, while +1 means the smaller
        body initially moves away.
    time : bool, optional
        A boolean encoding whether or not the coordinate time component of the geodesic
        equation should be computed (True meaning yes, False meanign no). NOTE: NOT 
        IMPLEMENTED, TBD.
    periodM : list, optional
        A list of two matrices, each 1x2 when the cosmological constant is 0 (cosmo = 0) or 
        2x4 when the cosmological constant isn't 0 (cosmo != 0), representing the period 
        matrices of the riemann surfaces defined by the r and theta components of the 
        4-velocity. When cosmo = 0, the first element of th
    
    Returns
    -------
    list 
        A list of 3 lambda functions: the r, theta, and phi components of the geodesic
        equation as a function of mino time. The arguments of these lambda functions can
        vary, however; they can either be an individual number, or a list of numbers. This
        depends on the spacetimes and specific coordinate:

        1. The phi coordinate, regardless of spacetime, always accepts a list of numbers 
        as an argument.
        2. For spacetimes with vanishing cosmological constants (cosmo = 0) both the r and theta
        coordinates accept individual numbers as arguments.
        3. For spacetimes with non-vanishing cosmological constants (cosmo != 0), the r coordinate
        accepts a list of numbers as an argument, while the theta coordinate accepts individual values
        as an argument.


        
    """ 

    # Catch wrong inputs
    if b_mass <= 0 or rot < 0 or p_mass <= 0 or (particle_light != 1 and particle_light != 0):
        raise Exception("Parameters are not feasible")
    if orbittype not in ["bound", "inner bound", "middle bound", "flyby", "terminating", "crossover bound", "transit", "terminating escape", "crossover flyby"]:
        raise Exception(f"Type of orbit {orbittype} is not allowed")

    if orbittype in ["transit"]:
        raise Exception(f"{orbittype} orbits have not been implemented. tbd")

    if len(config) == 3:
        workdir, date, prec = config
    elif len(config) == 2:
        workdir, date = config
        prec = 53
    else:
        raise Exception("Invalid declaration of config")

    # Set precision
    mp.prec = prec
    digits = mp.dps

    # Determine spacetime
    spacetime = classify_spacetime(eCharge, rot, cosmo, NUT)

    check_orbit_types(orbittype, particle_light, spacetime)

    p_r, p_nu, phi_integrands = four_velocity(b_mass, rot, eCharge, cosmo, NUT, mCharge, speed_light, grav_const, perm, particle_light, energy, ang_mom, carter, p_mass)[0:3]
    x, y, r = symbols("x y r")
    ############################# r Equation ################################
    # ----------- Define rhs of (dr/dgamma)**2 = p(r)
    deg_p = degree(p_r)

    # ----------- Check orbittype
    if rot == 0: #spacetime in ["Schwarzschild", "Reissner Nordstrom", "Schwarzschild-de Sitter", "Reissner Nordstrom-de Sitter"]:
        types, bounds, zeros_p = get_allowed_orbits(p_r, deg_p)
    else:
        types, bounds, zeros_p = get_allowed_orbits(p_r, deg_p, True)
    
    if inlist("transit", types) != -1: # or inlist("crossover flyby", types) != -1:
        raise Exception(f"{orbittype} orbits have not been implemented. tbd")
    elif inlist(orbittype, types) == -1:
        raise Exception(f"orbittype {orbittype} is not allowed")

    # ----------- Check initial values
    if len(initial_values) != 0 and len(initial_values) != 4:
        raise Exception(f"Invalid declaration of initial_values {initial_values}.")
    if len(initial_directions) != 0 and len(initial_directions) != 2:
        raise Exception(f"Invalid declaration of initial directions {initial_directions}")

    # Define initial directions
    if initial_directions == []:
        r_dir = 1
        theta_dir = 1
    else:
        r_dir = initial_directions[0]
        theta_dir = initial_directions[1]

    # Check initial r value
    if initial_values == []:
        init_r = []
    else:
        init_r = initial_values[0:1]
    init_r = check_rinitials(initial_values, orbittype, types, bounds)

    # ----------- Convert p to standard form
    # Output convert_polynomial: [converted_polynomial, integrand, substitution]
    # i.e. (dr/dgamma)**2 = integrand(y) * (converted_polynomial)
    p_converted = convert_polynomial(p_r, deg_p, zeros_p, bounds[inlist(orbittype, types)])
    zeros_converted = sorted(eval_roots(Poly(p_converted[0]).all_roots()), key = lambda y : re(y))
 
    # Apply substitution to bounds and initial_values
    eqn = Eq(x, p_converted[3])
    inverse_substitution = solve(eqn, y)[0]
    bounds_converted, init_converted = convert_boundsinit(bounds, init_r, inlist(orbittype, types), inverse_substitution, zeros_converted)
    dir_converted = - p_converted[4] * r_dir

    # ----------- Solution function
    sol_r = invert_eom(p_converted[0], zeros_converted, p_converted[1], init_converted, dir_converted, p_converted[2], periodM, digits, workdir + "temp/rdata_" + date)

    ############################# Phi Equation, r Integral ################################
    # Define rhs of dphi/dgamma = phir_integrand - phinu_integrand
    #phir_integrand = phi_integrands[0] / sqrt(p_converted[1].subs(y, inverse_substitution).subs(x, r)).simplify()
    phir_integrand = phi_integrands[0] / sqrt(p_converted[1]).simplify()

    if phir_integrand == 0:
        sol_phir = 0
    else:
        sol_phir = integrate_eom(p_converted[0], zeros_converted, p_converted[3], phir_integrand, workdir + "temp/rdata_" + date, digits)

    ############################# theta Equation ################################
    # ----------- Define rhs of (dnu/dgamma)**2 = p(nu), nu = cos(theta)
    deg_p = degree(p_nu)
    zeros_p = sorted(eval_roots(Poly(p_nu).all_roots()), key = lambda y : re(y))
    print("Zeros = ", zeros_p)

    if len(zeros_p) != deg_p:
        raise Exception("could not find all zeros of underlying polynomial")

    # ----------- Range of theta and initial values
    allowed_theta_inits, init_nu, bounds_nu = check_thetainitials(zeros_p, initial_values)

    # ----------- Convert p to standard form
    # Output convert_polynomial: [converted_polynomial, integrand, substitution]
    p_converted = convert_polynomial(p_nu, deg_p, zeros_p, bounds_nu)
    zeros_converted = sorted(eval_roots(Poly(p_converted[0]).all_roots()), key = lambda y : re(y))
 
    # Apply substitution to initial_values
    eqn = Eq(x, p_converted[3])
    inverse_substitution = solve(eqn, y)[0]
    bounds_converted, init_converted = convert_boundsinit(bounds_nu, init_nu, 0, inverse_substitution, zeros_converted)
    dir_converted = -p_converted[4] * (-theta_dir)
    
    # ----------- Determine solution olution function
    sol_nu = invert_eom(p_converted[0], zeros_converted, p_converted[1], init_converted, dir_converted, p_converted[2], periodM, digits, workdir + "temp/thetadata_" + date)

    # Backsubstitution theta = arccos(nu)
    if deg_p == 6:
        acos = vectorize(mp.acos, "D")
        sol_theta = lambda s : acos(sol_nu(s)) 
    else:
        sol_theta = lambda s : mp.acos(sol_nu(s))

    ############################# Phi Equation, theta Integral ################################
    # Define rhs of dphi/dgamma = phir_integrand - phinu_integrand
    phinu_integrand = phi_integrands[1] / sqrt(p_converted[1])#.simplify()
    sol_phinu = integrate_eom(p_converted[0], zeros_converted, p_converted[3], phinu_integrand, workdir + "temp/thetadata_" + date, digits)
    
    if initial_values == []:
        init_phi = 0
    else: 
        init_phi = initial_values[3]

    def sol_phi(s):
        if sol_phir == 0:
            phir = 0
        else:
            phir = list(sol_phir(s))

        phinu = list(sol_phinu(s))

        s = list(s)
       
        if phir == 0:
            result = [init_phi + re(phir - phinu[i]) for i in range(len(s))]
        else:
           result = [init_phi + re(phir[i] - phinu[i]) for i in range(len(phir))]

        if len(result) == 1:
            return result[0]
        else:
            return result

    return [sol_r, sol_theta, lambda s : sol_phi(s)]


def solve_geodesic_time():
    return 0

def solve_geodesic_worldline():

    return 0
