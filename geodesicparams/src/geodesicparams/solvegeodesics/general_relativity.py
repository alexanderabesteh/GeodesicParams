from sympy import Poly, limit, Symbol, cos, symbols, oo, pi, re
from ..utilities import inlist, separate_zeros, find_next, eval_roots

def get_allowed_orbits(polynomial, deg, isnegativeallowed = False):
    p = polynomial

    zeros = sorted(eval_roots(Poly(p).all_roots()), key = lambda y : re(y))
    print("Zeros = ", zeros)

    if len(zeros) != deg:
        raise Exception("could not find all zeros of underlying polynomial.")

    realNS, complexNS = separate_zeros(zeros)

   # if len(realNS) == 0:
    #    raise Exception("Underlying polynomial has only complex zeros: this case is not supported.")
    pos_zeros = []

    for i in range(len(realNS)):
        if realNS[i].evalf() > 0:
            pos_zeros.append(realNS[i])

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
    bounds_converted = []
    x = Symbol("x")

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
    eps = 10 ** (-6)

    if len(initial_values) == 0:
        if orbittype in ["flyby", "crossover flyby", "terminating escape"]:
            return [0, bounds[inlist(orbittype, types)][0]]
        else:
            return [0, bounds[inlist(orbittype, types)][1] - eps]

    elif not (bounds[inlist(orbittype, types)][0] <= initial_values[1] and initial_values[1] <= bounds[inlist(orbittype, types)][1]):
        print(f"WARNING in check_rinitials: initial value {initial_values[1]} is not allowed for orbittype {orbittype}; set to default value.")
        if orbittype in ["flyby", "crossover flyby", "terminating escape"]:
            return [initial_values[0], bounds[inlist(orbittype, types)][0]]
        else:
            return [initial_values[0], bounds[inlist(orbittype, types)][1] - eps]
    else:
        return initial_values

def check_thetainitials(zeros, initial_values):
    realNS, complexNS = separate_zeros(zeros)
    eps = 10**(-6)

    if len(realNS) == 0:
        raise ValueError("underlying polynomial has only complex zeros; this case is not supported.")
    allowed_inits = []

    for i in range(len(realNS)):
        if (realNS[i] >= -1 and realNS[i] <= 1):
            allowed_inits.append(realNS[i])

    if len(allowed_inits) == 0 or (len(allowed_inits) != 2 and len(allowed_inits) != 4):
        raise Exception("Bounds for theta motion are not allowed")

    allowed_inits.sort()
    if len(allowed_inits) == 2:
        bounds_nu = [[allowed_inits[0], allowed_inits[1]]]
    else:
        bounds_nu = [[allowed_inits[0], allowed_inits[1]], [allowed_inits[2], allowed_inits[3]]]

    print("Allowed initial_values for nu=cos(theta) motion: ", bounds_nu[0])
    if initial_values == []:
        init_nu = [0, allowed_inits[0]]
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
    r, nu = symbols("r nu")

    schw_r = 2 * grav_const * b_mass / speed_light**2
    a = rot / b_mass
    rQ = eCharge**2 * grav_const / (4 * pi * perm * speed_light**4)
    ang_mom = ang_mom / p_mass
    energy = energy / p_mass
    carter = carter / p_mass**2
    particle_light = particle_light * p_mass**2
    
    spacetime = classify_spacetime(eCharge, rot, cosmo, NUT)

    if spacetime in ["Kerr", "Kerr-Newman"] or spacetime not in ["Kerr-de Sitter", "Kerr-Newman-de Sitter"]:
        chi = 1
    else:
        chi = 1 + (a**2 * cosmo) / 3

    p_r = energy * (r**2 + a**2 + NUT**2) - a * ang_mom
    t_nu = energy * (a * (1 - nu**2) + 2 * NUT * nu) - ang_mom

    delta_r = ((1 - cosmo/3 * r**2 - cosmo * NUT**2) * (r**2 + a**2 - NUT**2) - schw_r * r + rQ + mCharge**2 - 4/3 * cosmo * NUT**2 * r**2).simplify().evalf()
    delta_nu = 1 + 1/3 * a * cosmo * nu**2 - 4/3 * cosmo * a  * NUT  * nu
    
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
