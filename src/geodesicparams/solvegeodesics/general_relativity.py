#!/usr/bin/env python
"""
A collection of functions related to the checking the initial values, orbits, and spacetimes
as well as defining the four-velocity of a space time.

The possible orbit types are the following:
"terminating", "terminating escape", "bound", "inner bound", middle bound", "flyby",
"crossover flyby", "transit" ("crossover flyby" and "transit" have not been implemented, tbd).

As of now, the following spacetimes have been implemented:
Schwarzschild, Reissner Nordstrom, Kerr, Kerr-Newman, Schwarzschild-de Sitter,
Reissner Nordstrom-de Sitter, Kerr-de Sitter, and Kerr-Newman-de Sitter.
 
"""

from sympy import Poly, limit, Symbol, cos, symbols, oo, pi, re

from ..utilities import inlist, separate_zeros, find_next, eval_roots

def get_allowed_orbits(polynomial, deg, isnegativeallowed = False):
    """
    Determine the possible orbit types from a given polynomial as well as
    their respective boundaries.
    
    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial to be converted.
    deg : integer
        The degree of <polynomial>.
    isnegativeallowed : boolean, optional
        If set to True, negative radial values are allowed, meaning that transit and
        crossover flyby orbits are possible, and terminating orbits are not possible.
        (transit and crossover flyby orbits have not been implemented yet, tbd).
        
    Returns
    -------
    types : list
        A list of strings representing the possible orbit types for <polynomial>.
    bounds : list
        A list containing the bounds for each orbit type.
    zeros : list
        A list of complex or real numbers representing the zeros of <polynomial>.

    """

    p = polynomial

    zeros = sorted(eval_roots(Poly(p).all_roots()), key = lambda y : re(y))
    print("Zeros = ", zeros)

    if len(zeros) != deg:
        raise Exception("could not find all zeros of underlying polynomial.")

    realNS, complexNS = separate_zeros(zeros)

   # if len(realNS) == 0:
    #    raise Exception("Underlying polynomial has only complex zeros: this case is not supported.")
    
    # Determine positive real zeros
    pos_zeros = []

    for i in range(len(realNS)):
        if realNS[i].evalf() > 0:
            pos_zeros.append(realNS[i])
    
    # ------ Possible orbit types
    k = len(pos_zeros)
    max_coeff = Poly(p).all_coeffs()[0]
    types = []
    bounds = []
    count = 0

    if k == 0:
        if isnegativeallowed:
            if len(realNS) > 0:
                types = ["crossover flyby"]
                bounds = [[0, oo]]
            else:
                types = ["transit"]
                bounds = [[0, oo]]
        else:
            types = ["terminating escape"]
            bounds = [[0, oo]]

    # ------ P to infinity for x to infinity
    if max_coeff.evalf() > 0:
        while k > 0:
            if (k % 2) == 0:
                if isnegativeallowed:
                    if len([x for x in zeros if x not in pos_zeros]) > 0:
                        types.append("crossover bound")
                    else:
                        types.append("crossover flyby")
                else:
                    types.append("terminating")
                bounds.append([0, pos_zeros[0]])
                k -= 1
                count += 1
            elif k > 1:
                if k == 3:
                    types.append("bound")
                else:
                    types.append("inner bound")
                bounds.append([pos_zeros[count], pos_zeros[count + 1]])
                k -= 2
                count += 2
            elif k == 1:
                types.append("flyby")
                bounds.append([pos_zeros[count], oo])
                k -= 1

    # ------ P to -infinity for x to infinity
    else:
        while k > 0:
            if (k % 2) == 1:
                if isnegativeallowed:
                    if len([x for x in zeros if x not in pos_zeros]) > 0:
                        types.append("crossover bound")
                    else:
                        types.append("crossover flyby")
                else:
                    types.append("terminating")
                bounds.append([0, pos_zeros[count]])
                k -= 1
                count += 1
            else:
                if k == 2:
                    types.append("bound")
                elif k == 4 and len(pos_zeros) >= 5:
                    types.append("middle bound")
                else:
                    types.append("inner bound")
                bounds.append([pos_zeros[count], pos_zeros[count + 1]])
                k -= 2
                count += 2

    print("bounds = ", bounds)
    print("Possible orbittypes: ", types)
#    if inlist(orbittype, types) == -1:
     #   raise Exception(f"orbittype {orbittype} is not allowed")
    return types, bounds, zeros

def convert_boundsinit(bounds, initial_values, position, substitution, zeros_subs):
    """
    Convert a set of boundaries and initial values by applying <substitution>.
    
    Parameters
    ----------
    bounds : list
        A list of boundaries for orbit types to be converted, where each boundary is a list
        containing a minimal and maximal value for each orbit type.
    initial_values : list
        A list of two elements, the first being the initial mino time, the second being
        a spacetime coordinate.
    position : integer
        An index such that bounds[position][0] < initial_values[1] < bounds[position][1].
    substitution : symbolic
        A symbolic statement, the substitution created by convert_polynomial.
    zeros_subs : list
        A list of complex or real numbers, the zeros converted by convert_polynomial.

    Returns
    -------
    bounds_converted : list
        A list containing the converted bounds for each orbit type.
    init_converted : list
        A list of containing the converted initial values.

    """

    bounds_converted = []
    x = Symbol("x")
    
    # Apply substitution to bounds
    for i in range(len(bounds)):
        try:
            element1 = limit(substitution, x, bounds[i][0])
        except:
            print("Numeric exception: division by zero.")
            element1 = oo
        if element1 != oo:
            element1 = find_next(element1, zeros_subs)
        try:
            element2 = limit(substitution, x, bounds[i][1])
        except:
            print("Numeric exception: division by zero.")
            element2 = oo
        if element2 != oo:
            element2 = find_next(element2, zeros_subs)
        bounds_converted.append([element1, element2])

    # Apply substitution to initial values
    if (initial_values[1] == bounds[position][0] or initial_values[1] == bounds[position][1]):
        try:
            limit(substitution, x, initial_values[1])
            init_converted = [initial_values[0], find_next(limit(substitution, x, initial_values[1]), bounds_converted[position])]
        except:
            print("Numeric exception: division by zero.")
            init_converted = [initial_values[0], oo]
    else:
        try:
            limit(substitution, x, initial_values[1])
            init_converted = [initial_values[0], limit(substitution, x, initial_values[1])]
        except:
            print("Numeric exception: division by zero.")
            init_converted = [initial_values[0], oo]

    print("converted bounds = ", bounds_converted)
    return bounds_converted, init_converted

def check_rinitials(initial_values, orbittype, types, bounds):
    """
    Checks or sets the initial values for the r motion as necessairy.
    
    Parameters
    ----------
    initial_values : list
        A list of two elements, the first being the initial mino time, the second being the
        initial r coordinate value.
    orbittype : string
        A string encoding the orbit to be computed.
    types : list
        A list of strings representing the possible orbit types.
    bounds : list
        A list of boundaries for each orbit type, where each boundary is a list
        containing a minimal and maximal value for each orbit type.

    Returns
    -------
    initial_values : list
        A list of containing the initial values: if they were correct/not empty, they 
        were returned unchanged. Otherwise, they were set to the default values: 
        0 as the first element, then either the maximal r value if the orbit does not reach
        infinite, or the minimal r value.

    """

    # Error parameter
    eps = 10 ** (-6)

    # Set initial values to defaults if empty
    if len(initial_values) == 0:
        if orbittype in ["flyby", "crossover flyby", "terminating escape"]:
            return [0, bounds[inlist(orbittype, types)][0]]
        else:
            return [0, bounds[inlist(orbittype, types)][1] - eps]

    # Modify initial values to defaults if incorrect
    elif not (bounds[inlist(orbittype, types)][0] <= initial_values[1] and initial_values[1] <= bounds[inlist(orbittype, types)][1]):
        print(f"WARNING in check_rinitials: initial value {initial_values[1]} is not allowed for orbittype {orbittype}; set to default value.")
        if orbittype in ["flyby", "crossover flyby", "terminating escape"]:
            return [initial_values[0], bounds[inlist(orbittype, types)][0]]
        else:
            return [initial_values[0], bounds[inlist(orbittype, types)][1] - eps]
    else:
        return initial_values

def check_thetainitials(zeros, initial_values):
    """
    Checks or sets the initial values for the theta motion as necessairy.
    
    Parameters
    ----------
    zeros : list
        A list of complex or real numbers representing the roots of theta polynomial.
        
    initial_values : list
        A list of two elements, the first being the initial mino time, the second being the
        initial r coordinate value.

    Returns
    -------
    allowed_inits : list
        A list containing the zeros that could be initial values of nu = cos(theta).
    init_nu : list
        A list of containing the initial values: if they were correct/not empty, they 
        were returned unchanged. Otherwise, they were set to the default values: 
        0 as the first element, then pi/2 if possible or the maximal theta value in
        the nothern hemisphere.
    bounds_nu : list
        A list containing the extremal nu values for which init_nu[1] is bounded. Two
        sets of bounds are possible.
    """


    realNS, complexNS = separate_zeros(zeros)
    if len(realNS) == 0:
        raise ValueError("underlying polynomial has only complex zeros; this case is not supported.")
    
    allowed_inits = []
    eps = 10**(-6)

    # Check for real zeros in between -1 <= 0 <= 1
    for i in range(len(realNS)):
        if (realNS[i] >= -1 and realNS[i] <= 1):
            allowed_inits.append(realNS[i])

    if len(allowed_inits) == 0 or (len(allowed_inits) != 2 and len(allowed_inits) != 4):
        raise Exception("Bounds for theta motion are not allowed")

    # Determine bounds
    allowed_inits.sort()
    if len(allowed_inits) == 2:
        bounds_nu = [[allowed_inits[0], allowed_inits[1]]]
    else:
        bounds_nu = [[allowed_inits[0], allowed_inits[1]], [allowed_inits[2], allowed_inits[3]]]

    print("Allowed initial_values for nu=cos(theta) motion: ", bounds_nu[0])

    # Set default values
    if initial_values == []:
        init_nu = [0, allowed_inits[0]]
    # Check if initial values that were entered are correct
    else:
        cos_init = cos(initial_values[2])
        if len(allowed_inits) == 2 and (cos_init.evalf() < allowed_inits[0] or cos_init.evalf() > allowed_inits[1]):
            print(f"WARNING in check_thetainitials: initial value {initial_values[2]} for theta motion is not allowed; set to default value {allowed_inits[0]}")
            init_nu = [initial_values[0], allowed_inits[0] + eps]
        elif len(allowed_inits) == 4 and ((cos_init.evalf() < allowed_inits[0] or cos_init.evalf() > allowed_inits[1]) and (cos_init.evalf() < allowed_inits[2] or cos_init.evalf() > allowed_inits[3])):
            print(f"WARNING in check_thetainitials: initial value {initial_values[2]} for theta motion is not allowed; set to default value {allowed_inits[0]}")
            init_nu = [initial_values[0], allowed_inits[0] + eps]
        else:
            init_nu = [0, cos_init]

    return allowed_inits, init_nu, bounds_nu

def classify_spacetime(eCharge, rot, cosmo, NUT):
    """
    Classify a spacetime based on the parameters entered.
    
    Parameters
    ----------
    eCharge : float
        The electric charge on the larger body.
    rot : float
        The rotating parameter on the larger body.
    cosmo : float
        The cosmological constant.
    NUT : float
        The NUT parameter.
        
    Returns
    -------
    string
        The spacetime defined by the parameters entered.

    """

    # Determine space-time
    if NUT == 0:
        # Static space-times
        if rot == 0:
            # Schwarzschild
            if cosmo == 0 and eCharge == 0:
                return "Schwarzschild"
            # Reissner Nordstrom
            elif cosmo == 0:
                return "Reissner Nordstrom"
            # Schwarzschild-de Sitter
            elif eCharge == 0:
                return "Schwarzschild-de Sitter"
            # Reissner Nordstrom-de Sitter
            else:
                return "Reissner Nordstrom-de Sitter"
        # Stationary space-times
        else:
            # Kerr
            if cosmo == 0 and eCharge == 0:
                return "Kerr"
            # Kerr-Newman
            elif cosmo == 0:
                return "Kerr-Newman"
            # Kerr-de Sitter
            elif eCharge == 0:
                return "Kerr-de Sitter"
            # Kerr-Newman-de Sitter
            else:
                return "Kerr-Newman-de Sitter"
    else:
        raise ValueError("unknown space-time.")

def check_orbit_types(orbittype, particle_light, spacetime):
    """
    Check if an orbittype is possible in a given spacetime, raise exception if it is not.
    
    Parameters
    ----------
    orbittype : string
        A string representing the type of orbit to be modelled.
    particle_light : integer
        An integer, either 1 (for timelike geodesics) or 0 (for null geodesics).
    spacetime : string
        The spacetime to be checked against the orbittype.

    Returns
    -------
    None

    """

    if spacetime == "Schwarzschild":
        if orbittype in ["inner bound", "middle bound", "crossover bound", "transit", "crossover flyby"]:
            raise Exception(f"Orbit type {orbittype} is not possible in the Schwarzschild space-time.")
        if orbittype == "bound" and particle_light == 0:
            raise Exception(f"Orbit type {orbittype} is not possible for null geodesics in the Schwarzschild space-time.")
    elif spacetime == "Reissner Nordstrom":
        # Check orbittype
        if orbittype in ["middle bound", "terminating", "terminating escape", "crossover bound", "transit", "crossover flyby"]:
            raise Exception(f"Orbit type {orbittype} is not possible in the Reissner-Nordström space-time.")
    elif spacetime == "Schwarzschild-de Sitter":
        # Check orbittype
        if orbittype in ["middle bound", "crossover bound", "transit", "crossover flyby"]:
            raise Exception(f"Orbit type {orbittype} is not possible in the Schwarzschild-de Sitter space-time.")
    elif spacetime == "Reissner Nordstrom-de Sitter":
        # Check orbittype
        if orbittype in ["middle bound", "terminating", "terminating escape", "crossover bound", "transit", "crossover flyby"]:
            raise Exception(f"Orbit type {orbittype} is not possible in the Reissner-Nordström-de Sitter space-time.")

    # Axially symmetrical space-times
    elif spacetime == "Kerr":
        # Check orbit type
        if orbittype == "middle bound":
            raise Exception(f"Orbittype {orbittype} is not allowed in the Kerr space-time.")
    elif spacetime == "Kerr-Newman":
        # Check orbit type
        if orbittype == "middle bound":
            raise Exception(f"Orbittype {orbittype} is not allowed in the Kerr-Newman space-time.")

def four_velocity(b_mass, rot, eCharge, cosmo, NUT, mCharge, speed_light, grav_const, perm, particle_light, energy, ang_mom, carter, p_mass):
    """
    Compute the four velocity of a spacetime. The eigentime is also computed.
    
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

    Returns
    -------
    list
        The 4-velocity defined by the parameters entered. The phi and t coordinates are
        returned as a list, each containing two components: the r and theta components.
        The eigentime is also return as the 4 element.

    """

    r, nu = symbols("r nu")

    # Constants 
    schw_r = 2 * grav_const * b_mass / speed_light**2
    a = rot / b_mass
    rQ = eCharge**2 * grav_const / (4 * pi * perm * speed_light**4)

    # Account for particle mass
    ang_mom = ang_mom / p_mass
    energy = energy / p_mass
    carter = carter / p_mass**2
    particle_light = particle_light * p_mass**2
    
    spacetime = classify_spacetime(eCharge, rot, cosmo, NUT)

    if spacetime in ["Kerr", "Kerr-Newman"] or spacetime not in ["Kerr-de Sitter", "Kerr-Newman-de Sitter"]:
        chi = 1
    else:
        chi = 1 + (a**2 * cosmo) / 3

    # Simplifications
    p_r = energy * (r**2 + a**2 + NUT**2) - a * ang_mom
    t_nu = energy * (a * (1 - nu**2) + 2 * NUT * nu) - ang_mom

    # Horizon functions
    delta_r = ((1 - cosmo/3 * r**2 - cosmo * NUT**2) * (r**2 + a**2 - NUT**2) - schw_r * r + rQ + mCharge**2 - 4/3 * cosmo * NUT**2 * r**2).simplify().evalf()
    delta_nu = 1 + 1/3 * a * cosmo * nu**2 - 4/3 * cosmo * a  * NUT  * nu
   
    # Coordinate velocities
    r_vel = (chi**2 * p_r**2 - delta_r * (particle_light * r**2 + carter)).expand()
    nu_vel = (delta_nu * (1 - nu**2) * (carter - particle_light * (NUT - a * nu)**2) - chi**2 * t_nu**2).expand()

    if NUT == 0:
        nu_vel = nu_vel.subs(nu**6, 0)

    phi_vel = [chi**2 * (a * p_r / delta_r).simplify(), chi**2 * (t_nu / (delta_nu * (1 - nu**2))).simplify()]
    t_vel = [chi**2 * ((r**2 + a**2 + NUT**2) * p_r / delta_r).simplify(), chi**2 * ((a * (1 - nu**2) + 2 * NUT * nu) * t_nu / (delta_nu * (1 - nu**2))).simplify()]
    proper_vel = [r**2, ((NUT - a * nu)**2).simplify()] 

    return [r_vel, nu_vel, phi_vel, t_vel, proper_vel]

#def compute_phi_vals(spacetime, inverse_substituttion, datafile):
#
 #   if spacetime == "Schwarzschild":
