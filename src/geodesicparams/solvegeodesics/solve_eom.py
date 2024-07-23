from sympy import sympify, Poly, degree, sin as spsin, asin as spasin, sqrt as spsqrt, Symbol, lambdify, apart, together, solve, oo, collect, pprint, sympify, assemble_partfrac_list, apart_list
from ..utilities import separate_zeros, inlist
from mpmath import im, re, fabs, sign, matrix, exp, ln, quad, nint, log, mpc, chop, sqrt
from pickle import dump, load
from numpy import load as npload, save, array

from ..riemannsurfaces.riemann_funcs.elliptic_funcs import weierstrass_P, inverse_weierstrass_P, weierstrass_zeta, weierstrass_sigma
from ..riemannsurfaces.riemann_funcs.hyperelp_funcs import *
from ..riemannsurfaces.period_matrices.periods_genus1_first import periods_firstkind
from ..riemannsurfaces.period_matrices.periods_genus1_second import periods_secondkind
from ..riemannsurfaces.period_matrices.periods_genus2_first import periods, eval_period, int_genus2_first, set_period_globals_genus2
from ..riemannsurfaces.integrations.integrate_hyperelliptic import myint_genus2, int_genus2_complex, myint_genus2_second, int_genus2_complex_second
from ..riemannsurfaces.period_matrices.periods_genus2_second import periods_second

periods_inverse = 0
riemannM = 0

def invert_eom(polynomial, zeros, integrand, initial_values, int_sign, substitution, periodM, digits, datafile = None):
    p = polynomial
    sym = Poly(p).gen
    deg_p = degree(polynomial, sym)

    if deg_p == 5:
        if (sympify(1/integrand).is_polynomial()):
            if degree(Poly(1/integrand, sym), sym) == 2 and (1/integrand).coeff(sym, 0) == 0 and (1/integrand).coeff(sym, 1) == 0:
                if datafile == None:
                    return invert_hyperelliptic_first(zeros, 2, initial_values, int_sign, (1/(1/integrand).coeff(sym, 2)).evalf(), substitution, digits, periodM)
                else:
                    return invert_hyperelliptic_first(zeros, 2, initial_values, int_sign, (1/(1/integrand).coeff(sym, 2)).evalf(), substitution, digits, periodM, datafile)
            elif degree(Poly(1/integrand, sym), sym) == 0:
                if datafile == None:
                    return invert_hyperelliptic_first(zeros, 1, initial_values, int_sign, (1/(1/integrand).coeff(sym, 0)).evalf(), substitution, digits, periodM)
                else:
                    return invert_hyperelliptic_first(zeros, 1, initial_values, int_sign, (1/(1/integrand).coeff(sym, 0)).evalf(), substitution, digits, periodM, datafile)
            else:
                raise ValueError("Equations of motions of hyperelliptic type and second kind are not supported.")
        else:
            raise ValueError("Equations of motions of hyperelliptic type and third kind can not be inverted.")
    elif deg_p == 3:
        if (sympify(1/integrand)).is_polynomial(sym):
            if degree(Poly(1/integrand, sym), sym) == 0:
                if datafile == None:
                    return invert_elliptic_first(polynomial, initial_values, int_sign, integrand, substitution, periodM)
                else:
                    return invert_elliptic_first(polynomial, initial_values, int_sign, integrand, substitution, periodM, datafile)
            else:
                raise ValueError("Equations of motion elliptic type and second kind: tbd.")
        else:
            raise ValueError("Equations of motion of elliptic type and third kind cannot be inverted.")
    elif deg_p <= 2:
        if datafile == None:
            return invert_trigonometric_first(polynomial, initial_values, int_sign, integrand)
        else:
            return invert_trigonometric_first(polynomial, initial_values, int_sign, integrand, datafile)
    else:
        raise ValueError(f"Polynomial {polynomial} is not of the standard form needed by invert_eom.")

def integrate_eom(polynomial, zeros, substitution, integrand, datafile, digits):
    p = polynomial
    int_sym = list(integrand.free_symbols) #Poly((1 / integrand).simplify()).gen
    deg_p = degree(polynomial)

    if deg_p <= 2:
        s = Symbol("s")
        sol_nu, inits = load(open(f"{datafile}.pickle", "rb"))
        int_sym = int_sym[0]
        integrand_subs = lambdify(s, integrand.subs(int_sym, sol_nu).simplify(), "mpmath")
        res_func = lambda s : quad(integrand_subs, [inits[0], s])
    else:

        u = Symbol("u", positive = True)
        p_sym = Poly(p).gen
        int_sym = [i for i in int_sym if i not in [p_sym]][0]
        
        if deg_p == 5:
        #    integrand = integrand.subs(int_sym, u).simplify()
            integrand = integrand.subs(int_sym, substitution).subs(p_sym, u).simplify()
            top_int, bot_int = integrand.as_numer_denom()
            integrand = (top_int.expand() / bot_int).subs(u, u)
            #print(str(integrand))
            #integrand = sympify(str(integrand).replace("Abs", ""))
            #integrand = integrand.subs(list(integrand.free_symbols)[0], u)
        else:
            integrand = integrand.subs(int_sym, u).simplify()

        #pprint(integrand)
        #f_parfrac = apart(integrand, full = True).evalf()
        #pprint(f_parfrac)
        
        f_parfrac = assemble_partfrac_list(apart_list(integrand)).doit()#.doit()
       # pprint(f_parfrac)
      #  pprint(together(f_parfrac).simplify())
        poly_fracs = []
        rat_fracs = []
        rat_fracs_final = []
        
        for i in range(len(f_parfrac.args)):
            if sympify(f_parfrac.args[i]).is_polynomial(u):
                poly_fracs.append(f_parfrac.args[i])
            else:
                rat_fracs.append(f_parfrac.args[i])

        if deg_p == 5:
            for i in rat_fracs:
                rat_fracs_final.append(i.subs(u, p_sym))
            for i in range(len(poly_fracs)):
                poly_fracs[i] = poly_fracs[i].subs(u, p_sym)
        else:
            for i in range(len(rat_fracs)):
                rat = apart(rat_fracs[i].subs(u, substitution).simplify(), full = True).evalf()
                for j in range(len(rat.args)):
                    if sympify(rat.args[j]).is_polynomial(p_sym):
                        poly_fracs.append(rat.args[j])
                    else:
                        rat_fracs_final.append(rat.args[j].simplify())

            for i in range(len(rat_fracs_final)):
                top_frac, inv_frac = rat_fracs_final[i].as_numer_denom()
                coeff = inv_frac.coeff(p_sym, 1)
                new_inv_frac = (top_frac / coeff / rat_fracs_final[i]).simplify()
                rat_fracs_final[i] = (top_frac / coeff) * 1 / new_inv_frac
       # pprint(rat_fracs_final)
        #de = 0
        #for i in rat_fracs_final:
         #   de += i
        #tog = together(de.subs(p_sym, u)).simplify()
        #pprint(tog)
    """
  elif deg_p == 5:
        p_sym = Poly(p).gen
        int_sym = [i for i in int_sym if i not in [p_sym]][0]
        u = Symbol("u", positive = True)
        deg_p = degree(p, p_sym)
        integrand = integrand.subs(int_sym, substitution).simplify()
        integrand = integrand.subs(p_sym, u).simplify()
       # pprint(integrand) 
        f_parfrac = apart(integrand, full = True).evalf()
        #f_parfrac = assemble_partfrac_list(apart_list(integrand, u)).evalf()
        f_parfrac = f_parfrac.subs(u, p_sym)
        pprint(f_parfrac)
       # f_parfrac = apart(integrand, full = True).evalf()
        poly_fracs = []
        rat_fracs = []

        for i in range(len(f_parfrac.args)):
            #pprint(f_parfrac.args[i])
            if sympify(f_parfrac.args[i]).is_polynomial(p_sym):
                #pprint(f_parfrac.args[i].simplify())
                poly_fracs.append(f_parfrac.args[i])
            else:
                rat_fracs.append(f_parfrac.args[i])


    """

    if deg_p == 5:

        def res_hyp(s):
            poly_first = []
            poly_first_int = []

            poly_second = []
            poly_second_int = []
            res = []

            s = list(s)
            for i in range(len(poly_fracs)):
                if degree(poly_fracs[i].as_poly(p_sym), p_sym) > 1:
                    raise ValueError("Equations of motion of hyperelliptic type and second kind are not supported")
                elif degree(poly_fracs[i].as_poly(p_sym), p_sym) == 0:
                    poly_first.append(poly_fracs[i])
                    poly_first_int.append(integrate_hyperelliptic_first(1, datafile))

                elif degree(poly_fracs[i].as_poly(p_sym), p_sym) == 1:
                    poly_second.append(poly_fracs[i].coeff(p_sym, 1))
                    poly_second_int.append(integrate_hyperelliptic_first(2, datafile))

            if len(rat_fracs_final) > 0:
                periodMatrix, invert_data, eps = npload(f"{datafile}.npy", allow_pickle = True)

                eta, r1, r2 = compute_secondkind_periods(zeros, eps, periodMatrix, datafile, digits)
                
                integrations = [integrate_hyperelliptic_third(zeros, r1, r2, eta, i, datafile, digits)(s) for i in rat_fracs_final]
               # pprint(rat_fracs_final)
                #pprint(rat_fracs_final[3].as_numer_denom()[0])
                rat_res = [rat_fracs_final[0].as_numer_denom()[0] * integrations[0][i] for i in range(len(integrations[0]))]
                for j in range(1, len(integrations)):
                    rat_res = [rat_res[i] + rat_fracs_final[j].as_numer_denom()[0] * integrations[j][i] for i in range(len(integrations[j]))]
               # pprint(integrations)
               # print()
                length = len(rat_res)
            else:
                length = len(s)
            #print("")
          #  pprint(rat_res)
            for i in range(length):
                poly_res = 0
                for j in range(len(poly_first)):
                    poly_res += poly_first[j] * poly_first_int[j][i]

                for j in range(len(poly_second)):
                    poly_res += poly_second[j] * poly_second_int[j][i]

                if len(rat_fracs_final) == 0:
                    res.append(mpc(poly_res))
                else:
                    res.append(mpc((poly_res + rat_res[i]).evalf()))
            #print(res)
            if len(res) == 1:
                return res[0]
            else:
                return res

        return lambda s : res_hyp(s)

    elif deg_p == 3:
        periods, g2, g3, int_init, inits = npload(f"{datafile}.npy", allow_pickle = True)
        def res_elp(s):
            poly_first = []
            poly_second = []
            res = []

            s = list(s)
            for i in range(len(poly_fracs)):
                if degree(poly_fracs[i].as_poly(p_sym), p_sym) == 0:
                    poly_first.append(poly_fracs[i])
                elif degree(poly_fracs[i].as_poly(p_sym), p_sym) == 1:
                    poly_second.append(poly_fracs[i].coeff(p_sym, 1))

            integrations = [integrate_elliptic_third(i, datafile)(s) for i in rat_fracs_final]
            rat_res = integrations[0]

            for j in range(1, len(integrations)):
                rat_res = [rat_res[i] + integrations[j][i] for i in range(len(s))]

            for i in range(len(s)):
                poly_res = 0

                for j in range(len(poly_first)):
                    bounds_res = s[i] - inits[0]
                    poly_res += bounds_res * mpc(poly_first[j])

                for j in range(len(poly_second)):
                    zeta_bounds = weierstrass_zeta(s[i] - inits[0], periods[0], periods[1])
                    poly_res += zeta_bounds * mpc(poly_second[j])

                res.append(mpc(poly_res + rat_res[i].evalf()))
            if len(res) == 1:
                return res[0]
            else:
                return res

        return lambda s : res_elp(s)
    elif deg_p <= 2:
        def trig_res(s):
            res = [res_func(i) for i in s]

            if len(res) == 1:
                return res[0]
            else:
                return res

        return lambda s : trig_res(s)


def invert_trigonometric_first(polynomial, initial_values, int_sign, constant, datafile = None):
    p = polynomial
    sym = Poly(p).gen
    s = Symbol("s")
  
    coeff_2 = polynomial.coeff(sym, 2)
    coeff_0 = polynomial.coeff(sym, 0)

    root = spsqrt(-coeff_2/coeff_0)
    init_const = spasin(initial_values[1] * root)
    inverse = 1 / root * spsin(int_sign * spsqrt(-coeff_2) * constant * (s - initial_values[0] + init_const))

    sol_func = lambdify(s, inverse, "mpmath")
    if datafile != None:
        with open(f"{datafile}.pickle", "wb") as output_file:
            dump([inverse, initial_values], output_file)
        print(f"In invert_trigonometric_first: solution function saved to, {datafile}.pickle")
    return sol_func

def invert_elliptic_first(polynomial, initial_values, int_sign, constant, substitution, periodM = None, datafile = None):
    p = polynomial
    sym = Poly(p).gen
    g2 = - p.coeff(sym, 1)
    g3 = - p.coeff(sym, 0)

    if periodM == None:
        periodMatrix = periods_firstkind(g2, g3)
    else:
        periodMatrix = periodM
    print("periodMatrix = ", periodMatrix)

    if type(initial_values[1]) == oo:
        int_initial = 0
    else:
        int_initial = int_sign * inverse_weierstrass_P(initial_values[1], periodMatrix[0], periodMatrix[1])

        if sign(weierstrass_P(initial_values[0] - int_initial, periodMatrix[0], periodMatrix[1], 1)) != int_sign:
            int_initial *= -1

    if datafile != None:
        integrate_initial = int_initial
        initials = initial_values
        data = array([periodMatrix, g2, g3, integrate_initial, initials], dtype = object)
        save(datafile, data)
        print(f"In invert_elliptic_first: periodMatrix saved to, {datafile}.npy")
    return lambda s : substitution(re(weierstrass_P(sqrt(constant) * s - initial_values[0] - int_initial, periodMatrix[0], periodMatrix[1])))

def invert_hyperelliptic_first(zeros, physical_comp, initial_values, int_sign, constant, substitution, digits, periodM = None, datafile = None):
    global periods_inverse, riemannM 
    realNS, complexNS = separate_zeros(zeros)

    if periodM == None:
        print("Computing periods ...")
        periodMatrix = periods(realNS, complexNS, digits)
    else:
        periodMatrix = periodM

    periods_inverse, riemannM = set_period_globals_genus2(periodMatrix)

    print("periodMatrix = ", periodMatrix)
    
    omega1 = periodMatrix[0:2, 0:2]
    omega2 = periodMatrix[0:2, 2:4]
    
    m = omega2 * omega1.T - omega1 * omega2.T
    print("Legendre relation = ", m)

    if fabs(m[0, 1]) > 10**(-digits):
        eps = fabs(m[0, 1]) * 10
        print(f"WARNING in invert_hyperelliptic_first: accuracy reduced to {eps} due to Legendre relation.")
    else:
        eps = 10**(-digits + 1)

    if physical_comp == 1:
        print("WARNING in invert_hyperelliptic_first: case that physical component is the first has to be tested")

        if type(initial_values[1]) == oo:
            initNewton = 0; modified_init = spsqrt(constant) * initial_values[0]
        elif inlist(initial_values[1], realNS) >= 0:
            initNewton = -eval_period(inlist(initial_values[1], realNS), oo, realNS, zeros, periodMatrix, 1)
            modified_init = spsqrt(constant) * initial_values[0] + eval_period(inlist(initial_values[1], realNS), oo, realNS, zeros, periodMatrix, 0)
        else:
            k = inlist(initial_values[1], sorted(realNS + [initial_values[1]], key = lambda x : re(x)))
            if k == len(realNS):
                k = len(realNS) - 1
            h = int_genus2_first(zeros, initial_values[1], realNS[k], digits, periodMatrix)
            initNewton = -eval_period(k, oo, realNS, zeros, periodMatrix, 0) - h[1]
            modified_init = spsqrt(constant) * initial_values[0] + eval_period(k, oo, realNS, zeros, periodMatrix, 1) + h[0]
    else:
        if type(initial_values[1]) == oo:
            initNewton = 0; modified_init = spsqrt(constant) * initial_values[0]
        elif inlist(initial_values[1], realNS) >= 0:
            initNewton = -eval_period(inlist(initial_values[1], realNS), oo, realNS, zeros, periodMatrix, 0)
            modified_init = spsqrt(constant) * initial_values[0] + eval_period(inlist(initial_values[1], realNS), oo, realNS, zeros, periodMatrix, 1)
        else:
            k = inlist(initial_values[1], sorted(realNS + [initial_values[1]], key = lambda x : re(x)))
            if k == len(realNS):
                k = len(realNS) - 1
            h = int_genus2_first(zeros, initial_values[1], realNS[k], digits, periodMatrix)
            initNewton = -eval_period(k, oo, realNS, zeros, periodMatrix, 0) - h[0]
            modified_init = spsqrt(constant) * initial_values[0] + eval_period(k, oo, realNS, zeros, periodMatrix, 1) + h[1]
    print("Check initial value for Newton method ...")
    max = check_initNewton(physical_comp, initNewton, [spsqrt(constant) * initial_values[0], initial_values[1]], modified_init, eps)
    
    if datafile != None:
        initials_mod = [spsqrt(constant) * i for i in initial_values]
        if physical_comp == 1:
            invert_data = [initials_mod, [modified_init - spsqrt(constant) * initial_values[0], -initNewton], max]
        else:
            invert_data = [initials_mod, [-initNewton, modified_init - spsqrt(constant) * initial_values[0]], max]
        save(datafile, array([periodMatrix.tolist(), invert_data, eps], dtype = object))
        print(f"In invert_hyperelliptic_first: period matrix saved to, {datafile}.npy")
        sol = lambda affine_list : orbitdata(initial_values, modified_init, [spsqrt(constant) * i for i in affine_list], substitution, initNewton, eps, max, physical_comp, datafile)
    else:
        sol = lambda affine_list : orbitdata(initial_values, modified_init, [spsqrt(constant) * i for i in affine_list], substitution, initNewton, eps, max, physical_comp)
    return sol

def integrate_hyperelliptic_first(component, datafile):
    extended_orbitdata = npload(datafile + "_orbitdata.npy", allow_pickle = True)
    invert_data = npload(datafile + ".npy", allow_pickle = True)
    
    result = [extended_orbitdata[i][3][component - 1] + invert_data[1][1][component - 1] for i in range(len(extended_orbitdata))]
    return result

def check_initNewton(physical_comp, initNewton, initial_values, modified_init, eps):
    global periods_inverse, riemannM
    g = [1/2, 1/2]
    h = [0, 1/2]

    if physical_comp == 1:
        z = 1/2 * periods_inverse * matrix([initial_values[0] - modified_init, initNewton])
    else:
        z = 1/2 * periods_inverse * matrix([initNewton, initial_values[0] - modified_init])
                                        
    max = 5
    f = hyp_theta(z, riemannM, max)

    while ((fabs(re(f)) > eps/10 or fabs(im(f)) > eps/10) and max < 30):
        for m1 in range(-max - 1, max + 2):
            m = [m1, -max - 1]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j]) 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            f += exp(1j * pi * char_sum)

            m = [m1, max + 1]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j]) 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            f += exp(1j * pi * char_sum)

        for m2 in range(-max, max + 1):
            m = [-max - 1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j]) 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])
            f += exp(1j * pi * char_sum)

            m = [max + 1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM[i, j] * (m[j] + g[j]) 
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            f += exp(1j * pi * char_sum)
        #print("f = ", f)
        max += 1
    s1 = sigma1(z, riemannM, max)
    s2 = sigma2(z, riemannM, max)
    u = - ((s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0]) / (s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1]))

    if ((initial_values[1] == oo and fabs(u) < 1/eps*(10**(-4))) or (initial_values[1] < oo and fabs(u - initial_values[1]) > eps*10**4)):
        max += 1
        s1 = sigma1(z, riemannM, max)
        s2 = sigma2(z, riemannM, max)
        u = - ((s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0]) / (s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1]))
        if ((initial_values[1] == oo and fabs(u) < 1/eps) or (initial_values[1] < oo and fabs(u - initial_values[1]) > eps*10**4)):
            raise ValueError(f"u({initial_values[0]}) not close enough to u0 = {initial_values[1]} for x0 = {initNewton}") 
    print("Maximal summation index for Kleinian sigma function set to ", max)
    
    return max

def orbitdata(initial_values, modified_init, affine_list, substitution, initNewton, eps, minMax, physical_comp, datafile = None):
    global periods_inverse, riemannM

    init_coord = initial_values[1]
    pos0 = inlist(initial_values[0], affine_list)

    if pos0 == -1:
        raise ValueError(f"Initial value {initial_values[0]} is not contained in list of affine parameters")
    x = [initNewton]; coordinates = [init_coord]; subs_coord = [substitution(init_coord)]

    if physical_comp == 1:
        divisor = [[-modified_init, initNewton]]
    else:
        divisor = [[initNewton, -modified_init]]
    print("Computing " , len(affine_list) - pos0, " solution points from i = ", pos0 + 1, " to ", len(affine_list), " ...")

    for i in range(pos0 + 1, len(affine_list)):
        coord = solution(affine_list[i] - modified_init, x[-1], eps, minMax, physical_comp)
        count = 0

        while (len(coord) == 2 and re(coord[1]) < 1 and im(coord[1]) < 1 and count < 2):
            print("Try more iterations ...")
            coord = solution(affine_list[i] - modified_init, coord[0][-1], eps, minMax, physical_comp)
            count += 1

        if (len(coord) == 3 and im(coord[0]) < 100 * eps):
            coordinates.append(re(coord[0]))
            subs_coord.append(substitution(re(coord[0])))
            x.append(coord[1])
            divisor.append(coord[2])
        else:
            print(f"WARNING in orbitdata: solution point for {affine_list[i]} ({i}. element) could not be computed result was {coord[0]}")
            break

        if type(i/10) == int:
            print(" ... ")
    print("Computing ", pos0 + 1, " solution points from i = ", pos0 + 1, " to 0 ...")
    
    for i in range(pos0 - 1, -1, -1):
        coord = solution(affine_list[i] - modified_init, x[0], eps, minMax, physical_comp)
        count = 0
        
        while (len(coord) == 2 and re(coord[1]) < 1 and im(coord[1]) < 1 and count < 2):
            print("Try more iterations ...")
            coord = solution(affine_list[i] - modified_init, coord[0][-1], eps, minMax, physical_comp)
            count += 1
        
        if (len(coord) == 3 and im(coord[0]) < 100 * eps):
            coordinates.insert(0, re(coord[0]))
            subs_coord.insert(0, substitution(re(coord[0])))
            x.insert(0, coord[1])
            divisor.insert(0, coord[2])
        else:
            print(f"WARNING in orbitdata: solution point for {affine_list[i]} ({i}. element) could not be computed.")
            break

        if type(i/10) == int:
            print(" ... ")

    if datafile != None:
        extended_orbitdata = array([[subs_coord[j], coordinates[j], x[j], divisor[j], affine_list[j]] for j in range(len(coordinates))], dtype = object)
        save(datafile + "_orbitdata", extended_orbitdata)
        print(f"extended orbitdata saved in {datafile + '_orbitdata'}.npy")

    return subs_coord
 
def sigma_ln_numerical(x, y, omega1, omega3):
    value = chop(log(weierstrass_sigma(x-y, omega1, omega3) / weierstrass_sigma(x+y, omega1, omega3)))

    if im(value) != 0:
        value = chop(quad(lambda s : weierstrass_zeta(s-y, omega1, omega3) - weierstrass_zeta(s+y, omega1, omega3), [0, x]))

    return value

def sigma_ln(x, y, omega1, omega3):

    result = []
    eta = periods_secondkind(omega1, omega3)[0]

    c = sigma_ln_numerical(omega1, y, omega1, omega3) - sigma_ln_numerical(10**(-10), y, omega1, omega3)
    branch = nint(chop(-omega1/(pi) * (im(c)/omega1 + im(2 * eta * y/omega1))))
    switch = branch*pi*1j

    for i in x:

        sigmatilde1 = weierstrass_sigma(i-y, omega1, omega3) * exp(-eta * (i-y)**2/(2*omega1))
        sigmatilde2 = weierstrass_sigma(i+y, omega1, omega3) * exp(-eta * (i+y)**2/(2*omega1))
        
        value = log((sigmatilde1/sigmatilde2) * exp(switch * (i/omega1-1))) - switch * (i/omega1-1) - 2*eta*i * y/omega1
        result.append(value)

    if len(result) == 1:
        return result[0]
    else:
        return result

def integrate_elliptic_third(integrand, datafile):
    periods, g2, g3, int_init, inits = npload(f"{datafile}.npy", allow_pickle = True)
    mod_int = inits[0] + int_init
    coeff = integrand.as_numer_denom()[0]
    inv_integrand = 1 / integrand
    pole = solve(inv_integrand, Poly(inv_integrand).gen)
    v1 = inverse_weierstrass_P(pole[0], periods[0], periods[1])

    def result(s):

        s_int = [i - mod_int for i in s]

        log_res = sigma_ln(s_int, v1, periods[0], periods[1])
        log_res0 = sigma_ln([inits[0] - mod_int], v1, periods[0], periods[1])

        if not type(log_res) in [list, tuple]:
            log_res = [log_res]

        int_sol = lambda s, y : 1 / weierstrass_P(v1, periods[0], periods[1], 1) * (2 * (s - mod_int) * weierstrass_zeta(v1, periods[0], periods[1]) + y)

        int_res0 = int_sol(inits[0], log_res0)
        sol = [coeff * (int_sol(s[i], log_res[i]) - int_res0) for i in range(len(s))]
        return sol
    return lambda s : result(s)

def compute_secondkind_periods(zeros, eps, periodMatrix, datafile, digits):
    x = Symbol("x")
    
    periodMatrix = matrix(periodMatrix)

    omega1 = periodMatrix[0:2, 0:2]
    omega2 = periodMatrix[0:2, 2:4]

    p = (x - zeros[0]) * (x - zeros[1]) * (x - zeros[2]) * (x - zeros[3]) * (x - zeros[4])
    p = collect(p.expand(), x)

    coeffsP = [re(p.coeff(x, i)) for i in range(6)]

    r1 = [0, 1/4 * coeffsP[3], 1/2 * coeffsP[4], 3/4 * coeffsP[5]]
    r2 = [0, 0, 1/4 * coeffsP[5]]

    realNS, complexNS = separate_zeros(zeros)
    
    print("Computing second kind periods ...")
    secondkindperiods = periods_second(r1, r2, realNS, complexNS, digits)

    eta1 = secondkindperiods[0:2, 0:2]
    eta2 = secondkindperiods[0:2, 2:4]

    m = eta2 * eta1.T - eta1 * eta2.T
    print("Legendre relation for periods of second kind = ", m)

    if fabs(m[0, 1]) > eps / 10:
        eps = fabs(m[0, 1]) * 10
        print(f"WARNING in solve_hyperelliptic_third: accuracy further reduced to {eps} due to Legendre relation for periods of second kind")

    m = omega2 * eta1.T - omega1 * eta2.T
    print("Mixed Legendre relation = ", m)

    if fabs(m[0, 1]) > eps / 10 or fabs(m[0, 0] - pi/2 * 1j) > eps / 10:
        if fabs(m[0, 1]) > fabs(m[0, 0] - pi/2 * 1j):
            eps = fabs(m[0, 1]) * 10
        else:
            eps = fabs(m[0, 0] - pi/2 * 1j) * 10
        print(f"WARNING in solve_hyperelliptic_third: further reduced to {eps} due to relation between periods of first and second kind")

    save(datafile + "_secondkindperiods", array([secondkindperiods.tolist(), eps], dtype = object))
    print(f"Saved second kind period matrix to {datafile + '_secondkindperiods.npy'}")

    return secondkindperiods, r1, r2

def integrate_hyperelliptic_third(zeros, r1, r2, eta, integrand, datafile, digits):
    global periods_inverse, riemannM
    periodMatrix, invert_data, eps = npload(datafile + ".npy", allow_pickle = True)

    periodMatrix = matrix(periodMatrix)
    x = Symbol("x") 
    init = invert_data[0]
    s0 = invert_data[1]
    max = invert_data[2]

    periods_inverse, riemannM = set_period_globals_genus2(periodMatrix)

    inv_integrand = 1 / integrand
    pole = solve(inv_integrand, Poly(inv_integrand).gen)[0]

    p = (x - zeros[0]) * (x - zeros[1]) * (x - zeros[2]) * (x - zeros[3]) * (x - zeros[4])
    p = collect(p.expand(), x)

    coeffsP = [re(p.coeff(x, i)) for i in range(6)]
 
    if inlist(pole, zeros) >= 0:
        raise ValueError("Invalid use: integral is of second kind")
    print("Computing constants needed for solution hyperelliptic integral of third kind ...")

    realNS, complexNS = separate_zeros(zeros)

    k = inlist(pole, sorted([pole] + zeros, key = lambda x : re(x)))

    if k > inlist(realNS[-1], zeros) or k == 4: # pole on realNS[-1]..oo
        int_dz = myint_genus2(zeros,pole,realNS[-1],1,digits)
        int_2=2*myint_genus2_second(zeros,[0,0,1],pole,realNS[-1],1,digits)
        int_3=2*myint_genus2_second(zeros,[0,0,0,1],pole,realNS[-1],1,digits)
        inf1=eval_period(len(realNS) - 1,oo,realNS,zeros,periodMatrix,0)
        inf2=eval_period(len(realNS) - 1,oo,realNS,zeros,periodMatrix,1)
    # in the remaining cases there is at least one real zero > pole!
    elif k == 3:
        if im(zeros[3]) == 0:
            int_dz = myint_genus2(zeros,pole,zeros[3],1,digits)
            int_2=2*myint_genus2_second(zeros,[0,0,1],pole,zeros[3],1,digits)
            int_3=2*myint_genus2_second(zeros,[0,0,0,1],pole,zeros[3],1,digits)
            inf1=eval_period(inlist(zeros[3],realNS),oo,realNS,zeros,periodMatrix,0)
            inf2=eval_period(inlist(zeros[3],realNS),oo,realNS,zeros,periodMatrix,1)
        else: # cases ima2Per3 and ima4Per1
            raise ValueError("Case that the pole is located on a vertical branch cut is tbd")
        
    elif k == 2:
        if im(zeros[1]) == 0 and im(zeros[0]) == 0:
            int_dz = myint_genus2(zeros,pole,zeros[1],1,digits)
            int_2=2*myint_genus2_second(zeros,[0,0,1],pole,zeros[1],1,digits)
            int_3=2*myint_genus2_second(zeros,[0,0,0,1],pole,zeros[1],1,digits)
            inf1=eval_period(inlist(zeros[1],realNS),oo,realNS,zeros,periodMatrix,0)
            inf2=eval_period(inlist(zeros[1],realNS),oo,realNS,zeros,periodMatrix,1)

        elif im(zeros[2]) == 0 and im(zeros[3]) == 0:
            int_dz=myint_genus2(zeros,pole,zeros[2],1,digits)
            int_2=2*myint_genus2_second(zeros,[0,0,1],pole,zeros[2],1,digits)
            int_3=2*myint_genus2_second(zeros,[0,0,0,1],pole,zeros[2],1,digits)
            inf1=eval_period(inlist(zeros[2],realNS),oo,realNS,zeros,periodMatrix,0)
            inf2=eval_period(inlist(zeros[2],realNS),oo,realNS,zeros,periodMatrix,1)
        elif pole ==re(zeros[1]): # ima2Per2
            int_dz=myint_genus2(zeros,pole,realNS[0],1,digits)
            int_2=2*myint_genus2_second(zeros,[0,0,1],pole,realNS[0],1,digits)
            int_3=2*myint_genus2_second(zeros,[0,0,0,1],pole,realNS[0],1,digits)
            inf1=eval_period(0,oo,realNS,zeros,periodMatrix,0)
            inf2=eval_period(0,oo,realNS,zeros,periodMatrix,1)
        else: # ima4Per1 or ima4Per3
            r = 0
            for i in range(1, 7):
                r += re(coeffsP[i - 1])*x**(i-1)
            r = lambdify(x, r)
            int_dz=matrix([1j* quad(lambda x : 1/sqrt(-r(x)),[pole, re(zeros[1])]),
                           1j * quad(lambda x : x/sqrt(-r(x)),[pole, re(zeros[1])])])
            int_2=2*1j*quad(lambda x : x**2/sqrt(-r(x)),[pole, re(zeros[1])])
            int_3=2*1j*quad(lambda x : x**3/sqrt(-r(x)),[pole, re(zeros[1])])
            int_dz=int_dz+int_genus2_complex(zeros,re(zeros[1]),im(zeros[1]),0,1,digits)
            int_2=int_2+2*int_genus2_complex_second(zeros,[0,0,1],re(zeros[0]),
            fabs(im(zeros[0])),0,1,digits)
            int_3=int_3+2*int_genus2_complex_second(zeros,[0,0,0,1],re(zeros[0]),
            fabs(im(zeros[0])),0,1,digits)
            inf1=periodMatrix[0,2]-periodMatrix[0,0]
            inf2=periodMatrix[1,2]-periodMatrix[1,0]
            
    elif k == 1:
        if im(zeros[0]) == 0:
            int_dz=myint_genus2(zeros,pole,zeros[0],1,digits)
            int_2=2*myint_genus2_second(zeros,[0,0,1],pole,zeros[0],1,digits)
            int_3=2*myint_genus2_second(zeros,[0,0,0,1],pole,zeros[0],1,digits)
            inf1=eval_period(inlist(zeros[0],realNS),oo,realNS,zeros,periodMatrix,0)
            inf2=eval_period(inlist(zeros[0],realNS),oo,realNS,zeros,periodMatrix,1)
        else: # cases ima2Per1, ima4Per1, and Ima4Per3
            raise ValueError("Case that the pole is located on a vertical branch cut is tbd")
            
    elif k == 0:
        if im(zeros[0]) == 0 and im(zeros[1]) == 0:
            int_dz=myint_genus2(zeros,pole,zeros[0],1,digits)
            int_2=2*myint_genus2_second(zeros,[0,0,1],pole,zeros[0],1,digits)
            int_3=2*myint_genus2_second(zeros,[0,0,0,1],pole,zeros[0],1,digits)
            inf1=eval_period(0,oo,realNS,zeros,periodMatrix,0)
            inf2=eval_period(0,oo,realNS,zeros,periodMatrix,1)
        elif im(zeros[0]) != 0 and im(zeros[1]) != 0:
            r = 0
            for i in range(1, 7):
                r +=re(coeffsP[i - 1])*x**(i-1)
            r = lambdify(x, r)

            int_dz=matrix([1j*quad(lambda x : 1/sqrt(-r(x)),[pole, re(zeros[0])]),
                           1j*quad(lambda x : x/sqrt(-r(x)),[pole, re(zeros[0])])])
            +int_genus2_complex(zeros,re(zeros[0]),fabs(im(zeros[0])),0,1,digits)
            int_2=2*1j*quad(lambda x : x**2/sqrt(-r(x)), [pole, re(zeros[0])])
            +2*int_genus2_complex_second(zeros,[0,0,1],re(zeros[0]),fabs(im(zeros[0])),
            0,1,digits)
            int_3=2*1j*quad(lambda x : x**3/sqrt(-r(x)), [pole, re(zeros[0])])
            +2*int_genus2_complex_second(zeros,[0,0,0,1],re(zeros[0]),fabs(im(zeros[0])),
            0,1,digits)
            inf1=periodMatrix[0,2]-periodMatrix[0,0]
            inf2=periodMatrix[1,2]-periodMatrix[1,0]
        else:
            r = 0
            for i in range(1, 7):
                r +=re(coeffsP[i - 1])*x**(i-1)
            r = lambdify(x, r)

            int_dz=matrix([1j*quad(lambda x : 1/sqrt(-r(x)),[pole, zeros[0]]),
                                   1j*quad(lambda x : x/sqrt(-r(x)), [pole, zeros[0]])])
            +myint_genus2(zeros,zeros[0],re(zeros[1]),1,digits)
            +int_genus2_complex(zeros,re(zeros[1]),fabs(im(zeros[1])),1,1,digits)
            int_2=2*1j*quad(lambda x : x**2/sqrt(-r(x)),[pole, zeros[0]])
            +2*myint_genus2_second(zeros,[0,0,1],zeros[0],re(zeros[1]),1,digits)
            +2*int_genus2_complex_second(zeros,[0,0,1],re(zeros[1]),fabs(im(zeros[1])),
            1,1,digits)
            int_3=2*1j*quad(lambda x : x**3/sqrt(-r(x)),[pole, zeros[0]])
            +2*myint_genus2_second(zeros,[0,0,0,1],zeros[0],re(zeros[1]),1,digits)
            +2*int_genus2_complex_second(zeros,[0,0,0,1],re(zeros[1]),fabs(im(zeros[1])),
            1,1,digits)
            inf1=periodMatrix[0,2]-periodMatrix[0,0]
            inf2=periodMatrix[1,2]-periodMatrix[1,0]
    int_dr1 = r1[1] * 2 * int_dz[1] + r1[2] * int_2 + r1[3] * int_3 
    int_dr2 = r2[2] * int_2
    yi_inf = matrix([int_dz[0] + inf1, int_dz[1] + inf2])
    xi_inf = matrix([-int_dz[0] + inf1, -int_dz[1] + inf2])
    
    eta1 = eta[0:2, 0:2]

    kappa = 1/2 * eta1 * periods_inverse

    def sigma(z):
        inverse_sum1 = 0
        inverse_sum2 = 0

        for i in range(2):
            inverse_sum1 += periods_inverse[0, i] * mpc(z[i])
            inverse_sum2 += periods_inverse[1, i] * mpc(z[i])

        return hyp_theta([1/2 * inverse_sum1, 1/2 * inverse_sum2], riemannM, max)

    def result(s):
        extended_orbitdata = npload(datafile + "_orbitdata.npy", allow_pickle = True)
        divisor = [extended_orbitdata[i][3] for i in range(len(extended_orbitdata))]
        affine_list = [extended_orbitdata[i][4] for i in range(len(extended_orbitdata))]
        pos0 = inlist(init[0], affine_list)

        if pos0 == -1:
            raise ValueError("Position of initial value could not be located in list of affine parameters")

        print("Computing solution points ...")
        const = ((2 * matrix(divisor[pos0]).T * kappa * (xi_inf - yi_inf)) + 1/2 * ln(sigma([divisor[pos0][0] + 2 * xi_inf[0], divisor[pos0][1] + 2 * xi_inf[1]])
        / sigma([divisor[pos0][0] + 2 * yi_inf[0], divisor[pos0][1] + 2 * yi_inf[1]])))[0]
        vars = [sigma([divisor[i][0] + 2 * xi_inf[0], divisor[i][1] + 2 * xi_inf[1]]) / sigma([divisor[i][0] + 2 * yi_inf[0], divisor[i][1] + 2 * yi_inf[1]])
                for i in range(len(divisor))]
        log_vars = [(2 * matrix(divisor[i]).T * kappa * (xi_inf - yi_inf))[0] for i in range(len(divisor))]
        
        branch = 0
        res = [log_vars[pos0] + 1/2 * ln(vars[pos0]) - const - ((divisor[pos0][0] + s0[0]) * int_dr1 + (divisor[pos0][1] + s0[1]) * int_dr2)]

        for i in range(pos0 + 1, len(divisor)):
            if re(vars[i - 1]) < 0:
                if im(vars[i - 1]) > 0 and im(vars[i]) < 0:
                    branch += 1
                elif im(vars[i - 1]) < 0 and im(vars[i]) > 0:
                    branch -= 1
            res.append(log_vars[i] + 1/2 * ln(vars[i]) + (pi * 1j * branch) - const - ((divisor[i][0] + s0[0]) * int_dr1 + (divisor[i][1] + s0[1]) * int_dr2))

        branch = 0
        for i in range(pos0 - 1, -1, -1):
            if re(vars[i + 1]) < 0:
                if im(vars[i + 1]) > 0 and im(vars[i]) < 0:
                    branch += 1
                elif im(vars[i + 1]) < 0 and im(vars[i]) > 0:
                    branch -= 1
            res.insert(0, log_vars[i] + 1/2 * ln(vars[i]) + (pi * 1j * branch) - const - ((divisor[i][0] + s0[0]) * int_dr1 + (divisor[i][1] + s0[1]) * int_dr2))
        for i in range(len(res)):
            res[i] = res[i] / sqrt(re(((pole - zeros[0]) * (pole - zeros[1]) * (pole - zeros[2]) * (pole - zeros[3]) * (pole - zeros[4])).evalf()))
        return res

    return lambda s : result(s)

def solution(affineParameter, initNewton, eps, minMax, physical_comp):
    if physical_comp == 1:
        return solution_first(affineParameter, initNewton, eps, minMax)
    else:
        return solution_second(affineParameter, initNewton, eps, minMax)

def solution_first(affineParameter, initNewton, eps, minMax):
    global periods_inverse, riemannM

    x = [initNewton]
    zeroR = re(initNewton)
    zeroI = im(initNewton)

    affR = re(affineParameter)
    affI = im(affineParameter)

    perR = periods_inverse.apply(re)
    perI = periods_inverse.apply(im)

    p1 = perR[0, 0] * affR - perI[0, 0] * affI
    p2 = perR[0, 0] * affI + perI[0, 0] * affR
    p3 = perR[1, 0] * affR - perI[1, 0] * affI
    p4 = perR[1, 0] * affI + perI[1, 0] * affR

    zfirstR = 1/2 * (perR[0, 1] * zeroR - perI[0, 1] * zeroI + p1)
    zfirstI = 1/2 * (perR[0, 1] * zeroI + perI[0, 1] * zeroR + p2)
    zsecondR = 1/2 * (perR[1, 1] * zeroR - perI[1, 1] * zeroI + p3)
    zsecondI = 1/2 * (perR[1, 1] * zeroI + perI[1, 1] * zeroR + p4)

    z = [zfirstR + 1j * zfirstI, zsecondR + 1j * zsecondI]
    f = hyp_theta(z, riemannM, minMax)
    af = fabs(f)

    count = 0

    while af > eps and count < 30:
        a = 1/2 * (perR[0, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, minMax)
            - perI[0, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, minMax)
            + perR[1, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, minMax)
            - perI[1, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, minMax))
        c = 1/2 * (perR[0, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, minMax)
            + perI[0, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, minMax)
            + perR[1, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, minMax)
            + perI[1, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, minMax))
        
        b = -c; d = a
        det = a*d-b*c

        zeroR = zeroR-1/det*(d*re(f)-b*im(f))
        zeroI = zeroI-1/det*(-c*re(f)+a*im(f))
        zero = zeroR + 1j*zeroI
        x.append(zero)
        # update z and f
        zfirstR = 1/2*(perR[0, 1]*zeroR-perI[0, 1] * zeroI + p1)
        zfirstI = 1/2*(perR[0, 1]*zeroI+perI[0, 1] * zeroR + p2)
        zsecondR = 1/2*(perR[1, 1]*zeroR-perI[1, 1] * zeroI + p3)
        zsecondI = 1/2*(perR[1, 1]*zeroI+perI[1, 1] * zeroR + p4)
        z = [zfirstR + 1j * zfirstI, zsecondR + 1j * zsecondI]
        f = hyp_theta(z, riemannM, minMax)
        af = fabs(f)
        count += 1
        
    if af > eps:
        print("In solution: Iteration process stopped after 30 iterations.")
        print("hyp_theta(1/2 * omega1inv * (phi, initNewton)^t) = ", f)
        return [x, f]

    s1 = sigma1(z, riemannM, minMax)
    s2 = sigma2(z, riemannM, minMax)
    sol = - ((s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0]) / (s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1]))
    #print("sol = ", sol)
    return [sol, x[-1], [affineParameter, x[-1]]]


def solution_second(affineParameter, initNewton, eps, minMax):
    global periods_inverse, riemannM

    x = [initNewton]
    zeroR = re(initNewton)
    zeroI = im(initNewton)

    affR = re(affineParameter)
    affI = im(affineParameter)

    perR = periods_inverse.apply(re)
    perI = periods_inverse.apply(im)

    p1 = perR[0, 1] * affR - perI[0, 1] * affI
    p2 = perR[0, 1] * affI + perI[0, 1] * affR
    p3 = perR[1, 1] * affR - perI[1, 1] * affI
    p4 = perR[1, 1] * affI + perI[1, 1] * affR

    zfirstR = 1/2 * (perR[0, 0] * zeroR - perI[0, 0] * zeroI + p1)
    zfirstI = 1/2 * (perR[0, 0] * zeroI + perI[0, 0] * zeroR + p2)
    zsecondR = 1/2 * (perR[1, 0] * zeroR - perI[1, 0] * zeroI + p3)
    zsecondI = 1/2 * (perR[1, 0] * zeroI + perI[1, 0] * zeroR + p4)

    z = [zfirstR + 1j * zfirstI, zsecondR + 1j * zsecondI]
    f = hyp_theta(z, riemannM, minMax)
    af = fabs(f)

    count = 0

    while af > eps and count < 30:
        a = 1/2 * (perR[0, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, minMax)
            - perI[0 ,0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, minMax)
            + perR[1, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, minMax)
            - perI[1, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, minMax))
        c = 1/2 * (perR[0, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, minMax)
            + perI[0, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, minMax)
            + perR[1, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, minMax)
            + perI[1, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, minMax))
        b = -c; d = a
        det = a*d-b*c

        zeroR = zeroR-1/det*(d*re(f)-b*im(f))
        zeroI = zeroI-1/det*(-c*re(f)+a*im(f))
        zero = zeroR+ 1j*zeroI
        x.append(zero)
        # update z and f
        zfirstR = 1/2*(perR[0, 0]*zeroR-perI[0, 0] * zeroI + p1)
        zfirstI = 1/2*(perR[0, 0]*zeroI+perI[0, 0] * zeroR + p2)
        zsecondR = 1/2*(perR[1, 0]*zeroR-perI[1, 0] * zeroI + p3)
        zsecondI = 1/2*(perR[1, 0]*zeroI+perI[1, 0] * zeroR + p4)
        z = [zfirstR + 1j * zfirstI, zsecondR + 1j * zsecondI]
        f = hyp_theta(z, riemannM, minMax)
        af = fabs(f)
        count += 1

    if af > eps:
        print("In solution: Iteration process stopped after 30 iterations.")
        print("hyp_theta(1/2 * omega1inv * (phi, initNewton)^t) = ", f)
        return [x, f]

    s1 = sigma1(z, riemannM, minMax)
    s2 = sigma2(z, riemannM, minMax)
    sol = - ((s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0]) / (s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1]))

   # print("sol = ", sol)

    return [sol, x[-1], [x[-1], affineParameter]]
