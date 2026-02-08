#!/usr/bin/env python3
"""
Procedures for inverting and solving the equations of motion for a variety of spacetimes.

Equations of motion used by these procedures are required to be in standard form, which can
be achieved through convert_polynomial and solve_geodesic_orbit. Period matrices are required
for each procedure (except for when the equation of motion is a degree 2 polynomial), and are
either computed through the period_matrices modules in riemannsurfaces, or provided directly.

References
----------
[] E. Hackmann, Geodesic equations in black hole space-times with cosmological constant (2010).
[] T. mpmath development team, mpmath: a Python library for arbitrary-precision floating-point arithmetic
    (version 1.3.0) (2023), http://mpmath.org/.
[] A. Meurer, C. P. Smith, M. Paprocki, O. ˇCert´ık, S. B. Kirpichev, M. Rocklin, A. Kumar, S. Ivanov, J. K.
    Moore, S. Singh, T. Rathnayake, S. Vig, B. E. Granger, R. P. Muller, F. Bonazzi, H. Gupta, S. Vats, F. Johans-
    son, F. Pedregosa, M. J. Curry, A. R. Terrel, v. Rouˇcka, A. Saboo, I. Fernando, S. Kulal, R. Cimrman, and
    A. Scopatz, Sympy: symbolic computing in python, PeerJ Computer Science 3, e103 (2017).
[] A. Cieslik, E. Hackmann, and P. Mach, Kerr geodesics in terms of weierstrass elliptic functions, Physical Review D
    108, 10.1103/physrevd.108.024056 (2023).

"""

from pickle import dump, load

from mpmath import chop, exp, fabs, im, matrix, re, sign
from numpy import array, ndarray as np_ndarray, zeros as np_zeros
from numpy import load as npload
from numpy import save
from sympy import Poly, Symbol, apart
from sympy import asin as spasin
from sympy import collect, degree, lambdify, nsimplify, oo
from sympy import sin as spsin
from sympy import solve
from sympy import sqrt as spsqrt
from sympy import sympify
from torchquad import set_up_backend, GaussLegendre
from jax import jit, lax, vmap, ops
from jax.numpy import (
    array as jnp_array, 
    stack, 
    sum as jnp_sum, 
    complex128 as jnp_complex128, 
    dot, 
    float64 as jnp_float64, 
    save as jnp_save,
    real as jnp_real, 
    imag as jnp_im, 
    exp as jnp_exp, 
    log as jnp_log, 
    abs as jnp_abs, 
    sqrt as jnp_sqrt, 
    argmax, max as jnp_max, 
    logical_and as jnp_logical_and, 
    zeros_like as jnp_zeros_like, 
    zeros as jnp_zeros, 
    arange as jnp_arange, 
    ndarray as jnp_ndarray, 
    any as jnp_any, 
    where as jnp_where,
    logical_or as jnp_logical_or, 
    logical_not as jnp_logical_not,
    pi as jnp_pi, 
    sort as jnp_sort, 
    _bool as jnp_bool, 
    round as jnp_round
)
from torch import sqrt as t_sqrt

from ..riemannsurfaces.integrations.integrate_hyperelliptic import (
    eval_period,
    int_genus2_complex,
    int_genus2_complex_second,
    int_genus2_first,
    myint_genus2,
    myint_genus2_second,
)
from ..riemannsurfaces.period_matrices.periods_genus1_first import periods_firstkind
from ..riemannsurfaces.period_matrices.periods_genus1_second import periods_secondkind
from ..riemannsurfaces.period_matrices.periods_genus2_first import (
    periods,
    set_period_globals_genus2,
)
from ..riemannsurfaces.period_matrices.periods_genus2_second import periods_second
from ..riemannsurfaces.riemann_funcs.elliptic_funcs import (
    inverse_weierstrass_P,
    weierstrass_P,
    weierstrass_sigma,
    weierstrass_zeta,
)
from ..riemannsurfaces.riemann_funcs.hyperelp_funcs import *
from ..utilities import inlist, separate_zeros

# Global period matrices
periods_inverse = 0
riemannM = 0
set_up_backend("torch", data_type = "complex64", torch_enable_cuda = True)

def invert_eom(
    polynomial,
    zeros,
    integrand,
    initial_values,
    int_sign,
    substitution,
    digits,
    periodM=None,
    datafile=None,
):
    """
    Procedure for inverting equations of motion of trigonometric, elliptic, and
    hyperelliptic type for various spacetimes. Specifically

    The polynomial utilized is expected to be in standard form, converted by convert_polynomial.
    Unless provided, the period matrices have to be computed.

    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial in standard form.
    zeros : list
        A list of complex or real numbers representing the converted zeros of <polynomial>.
    integrand : symbolic
        The integrand in the equation above.
    initial_values : list
        The initial values converted by the same substitution used to convert <polynomial>.
    int_sign : int
        The sign of square root in the differential equation, either +1 or -1. The
        signs must have also been converted by the same substitution used to convert
        <polynomial>.
    substitution : callable
        The substitution used to convert <polynomial> as a callable lambda function.
    periodM : list, optional
        Either a 1x2 list for elliptic equations of motion, representing the period
        matrices of a genus 1 Riemann surface, or a 2x4 mpmath matrix, representing the
        period matrix of a genus 2 Riemann surface.
    digits : int
        The number of digits to be used in the computation.
    datafile : string, optional
        A file name to store information pertaining to the computation, such as the
        period matrix. This is optional, unlike solve_eom.

    Returns
    -------
    callable
        The solution function for the differential equation.

    """

    p = polynomial
    sym = Poly(p).gen
    deg_p = degree(polynomial, sym)

    # Hyperelliptic (genus 2)
    if deg_p == 5:
        if sympify(1 / integrand).is_polynomial():
            if (
                degree(Poly(1 / integrand, sym), sym) == 2
                and (1 / integrand).coeff(sym, 0) == 0
                and (1 / integrand).coeff(sym, 1) == 0
            ):
                if datafile == None:
                    return invert_hyperelliptic_first(
                        zeros,
                        2,
                        initial_values,
                        int_sign,
                        (1 / (1 / integrand).coeff(sym, 2)).evalf(),
                        substitution,
                        digits,
                        periodM,
                    )
                else:
                    return invert_hyperelliptic_first(
                        zeros,
                        2,
                        initial_values,
                        int_sign,
                        (1 / (1 / integrand).coeff(sym, 2)).evalf(),
                        substitution,
                        digits,
                        periodM,
                        datafile,
                    )
            elif degree(Poly(1 / integrand, sym), sym) == 0:
                if datafile == None:
                    return invert_hyperelliptic_first(
                        zeros,
                        1,
                        initial_values,
                        int_sign,
                        (1 / (1 / integrand).coeff(sym, 0)).evalf(),
                        substitution,
                        digits,
                        periodM,
                    )
                else:
                    return invert_hyperelliptic_first(
                        zeros,
                        1,
                        initial_values,
                        int_sign,
                        (1 / (1 / integrand).coeff(sym, 0)).evalf(),
                        substitution,
                        digits,
                        periodM,
                        datafile,
                    )
            else:
                raise ValueError(
                    "Equations of motions of hyperelliptic type and second kind are not supported."
                )
        else:
            raise ValueError(
                "Equations of motions of hyperelliptic type and third kind can not be inverted."
            )

    # Elliptic (genus 1)
    elif deg_p == 3:
        if (sympify(1 / integrand)).is_polynomial(sym):
            if degree(Poly(1 / integrand, sym), sym) == 0:
                if datafile == None:
                    return invert_elliptic_first(
                        polynomial,
                        initial_values,
                        int_sign,
                        integrand,
                        substitution,
                        periodM,
                    )
                else:
                    return invert_elliptic_first(
                        polynomial,
                        initial_values,
                        int_sign,
                        integrand,
                        substitution,
                        periodM,
                        datafile,
                    )
            else:
                raise ValueError(
                    "Equations of motion elliptic type and second kind: tbd."
                )
        else:
            raise ValueError(
                "Equations of motion of elliptic type and third kind cannot be inverted."
            )

    # Trigonometric
    elif deg_p <= 2:
        if datafile == None:
            return invert_trigonometric_first(
                polynomial, initial_values, int_sign, integrand
            )
        else:
            return invert_trigonometric_first(
                polynomial, initial_values, int_sign, integrand, datafile
            )
    else:
        raise ValueError(
            f"Polynomial {polynomial} is not of the standard form needed by invert_eom."
        )


def integrate_eom(polynomial, zeros, substitution, integrand, datafile, digits):
    """
    Procedure for integrating equations of motion of trigonometric, elliptic, and
    hyperelliptic type for various spacetimes. Specifically

    The polynomial utilized is expected to be in standard form, converted by convert_polynomial.
    Unless provided, the period matrices have to be computed.

    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial in standard form.
    zeros : list
        A list of complex or real numbers representing the converted zeros of <polynomial>.
    integrand : symbolic
        The integrand in the equation above (different from invert_eom).
    datafile : string
        A file name (including the ending, i.e .npy, etc) containing information about the
        computation, such as the period matrix. This is required for the integration.
    digits : int
        The number of decimal digits to be used in the computation.

    Returns
    -------
    callable
        The solution function for the integral.

    Notes
    -----
    1. Fix partial fraction decomposition bug in genus 2 equations of motion.

    """

    p = polynomial
    int_sym = list(integrand.free_symbols)  # Poly((1 / integrand).simplify()).gen
    deg_p = degree(polynomial)

    # Degree 2 or less
    if deg_p <= 2:
        s = Symbol("s")
        sol_nu, inits = load(open(f"{datafile}.pickle", "rb"))
        int_sym = int_sym[0]
        integrand_subs = lambdify(s, integrand.subs(int_sym, sol_nu).simplify(), "torch")
        gl = GaussLegendre()
        res_func = lambda s : gl.integrate(integrand_subs, dim=1, N=101, integration_domain=[[inits[0], s]], backend = "torch"),
    else:
        # Define symbols in the polynomial and integrand
        u = Symbol("u", positive=True)
        p_sym = Poly(p).gen
        int_sym = [i for i in int_sym if i not in [p_sym]][0]

        # Degree 5
        if deg_p == 5:
            integrand = integrand.subs(int_sym, substitution).subs(p_sym, u).simplify()
            top_int, bot_int = integrand.as_numer_denom()
            integrand = (top_int.expand() / bot_int).subs(u, u)
        # Degree 3
        else:
            integrand = integrand.subs(int_sym, u).simplify()

        # Bronstein partial fraction decomposition
        f_parfrac = apart(
            nsimplify(integrand, tolerance=10 ** (-digits)), full=True
        ).doit()
        poly_fracs = []
        rat_fracs = []
        rat_fracs_final = []

        # Separate rational and polynomial fractions
        for i in range(len(f_parfrac.args)):
            if sympify(f_parfrac.args[i]).is_polynomial(u):
                poly_fracs.append(f_parfrac.args[i])
            else:
                rat_fracs.append(f_parfrac.args[i])

        # Degree 5
        if deg_p == 5:
            for i in rat_fracs:
                rat_fracs_final.append(i.subs(u, p_sym))
            for i in range(len(poly_fracs)):
                poly_fracs[i] = poly_fracs[i].subs(u, p_sym)
        # Degree 3
        else:
            # Apply substitution to partial fractions
            for i in range(len(rat_fracs)):
                rat = apart(
                    rat_fracs[i].subs(u, substitution).simplify(), full=True
                ).evalf()
                for j in range(len(rat.args)):
                    if sympify(rat.args[j]).is_polynomial(p_sym):
                        poly_fracs.append(rat.args[j])
                    else:
                        rat_fracs_final.append(rat.args[j].simplify())

            # Fix rational fractions
            for i in range(len(rat_fracs_final)):
                top_frac, inv_frac = rat_fracs_final[i].as_numer_denom()
                coeff = inv_frac.coeff(p_sym, 1)
                new_inv_frac = (top_frac / coeff / rat_fracs_final[i]).simplify()
                rat_fracs_final[i] = (top_frac / coeff) * 1 / new_inv_frac

    # Solution procedures
    if deg_p == 5:

        @jit
        def res_hyp(s):
            s_array = jnp_array(s, dtype=jnp_complex128)
    
            poly_first = []
            poly_first_int = []
            poly_second = []
            poly_second_int = []
    
            for i in range(len(poly_fracs)):
                deg = degree(poly_fracs[i].as_poly(p_sym), p_sym)
                if deg > 1:
                    raise ValueError(
                    "Equations of motion of hyperelliptic type and second kind are not supported"
                    )
                elif deg == 0:
                    poly_first.append(jnp_complex128(poly_fracs[i]))
                    poly_first_int.append(integrate_hyperelliptic_first(1, datafile))
                elif deg == 1:
                    poly_second.append(jnp_complex128(poly_fracs[i].coeff(p_sym, 1)))
                    poly_second_int.append(integrate_hyperelliptic_first(2, datafile))
    
            poly_first_jax = jnp_array(poly_first, dtype=jnp_complex128) if poly_first else jnp_array([], dtype=jnp_complex128)
            poly_second_jax = jnp_array(poly_second, dtype=jnp_complex128) if poly_second else jnp_array([], dtype=jnp_complex128)
    
            if len(rat_fracs_final) > 0:
                periodMatrix, invert_data, eps = load(f"{datafile}.npy", allow_pickle=True)
                minMax = invert_data[2]
        
                eta, r1, r2 = compute_secondkind_periods(
                    zeros, eps, periodMatrix, datafile, digits, minMax
                )
        
                rat_coeffs = jnp_array([rf.as_numer_denom()[0] for rf in rat_fracs_final], dtype=jnp_complex128)
        
                @jit
                def compute_integrations():
                    integrations_list = []
                    for i in range(len(rat_fracs_final)):
                        integration_func = integrate_hyperelliptic_third(
                            zeros, r1, r2, eta, i, datafile, digits)
                        integrations_list.append(integration_func(s_array))
            
                    return stack(integrations_list, axis=0)  # shape: (num_rat_fracs, len(s))
        
                integrations = compute_integrations()
        
                rat_res = jnp_sum(rat_coeffs[:, None] * integrations, axis=0)
                length = len(s_array)
            else:
                rat_res = jnp_zeros_like(s_array, dtype=jnp_complex128)
                length = len(s_array)
    
            if len(poly_first_int) > 0:
                first_integrals = stack(poly_first_int, axis=0)  # shape: (num_first, len(s))
                first_contrib = jnp_sum(poly_first_jax[:, None] * first_integrals, axis=0)
            else:
                first_contrib = jnp_zeros_like(s_array, dtype=jnp_complex128)
    
            if len(poly_second_int) > 0:
                second_integrals = stack(poly_second_int, axis=0)  # shape: (num_second, len(s))
                second_contrib = jnp_sum(poly_second_jax[:, None] * second_integrals, axis=0)
            else:
                second_contrib = jnp_zeros_like(s_array, dtype=jnp_complex128)
    
            poly_res_total = first_contrib + second_contrib
            final_results = poly_res_total + rat_res
    
            if len(s_array) == 1:
                return complex(final_results[0])
            else:
                return [complex(x) for x in final_results]

        return lambda s: res_hyp(s)
    elif deg_p == 3:
        periods, g2, g3, int_init, inits = npload(f"{datafile}.npy", allow_pickle=True)

        @jit
        def res_elp(s):
            s_array = jnp_array(s, dtype=jnp_complex128)
    
            poly_first = []
            poly_second = []
    
            for i in range(len(poly_fracs)):
                deg = degree(poly_fracs[i].as_poly(p_sym), p_sym)
                if deg == 0:
                    poly_first.append(jnp_complex128(poly_fracs[i]))
                elif deg == 1:
                    poly_second.append(jnp_complex128(poly_fracs[i].coeff(p_sym, 1)))
    
            poly_first_jax = jnp_array(poly_first, dtype=jnp_complex128) if poly_first else jnp_array([], dtype=jnp_complex128)
            poly_second_jax = jnp_array(poly_second, dtype=jnp_complex128) if poly_second else jnp_array([], dtype=jnp_complex128)
    
            periods_array = jnp_array(periods, dtype=jnp_complex128)
            inits_array = jnp_array(inits, dtype=jnp_complex128)
            omega1 = periods_array[0]
            omega3 = periods_array[1]
            s0 = inits_array[0]
    
            if len(rat_fracs_final) > 0:
                integration_funcs = [integrate_elliptic_third(i, datafile) for i in rat_fracs_final]
        
                @jit
                def compute_rat_res():
                    integrations_stack = stack([func(s_array) for func in integration_funcs], axis=0)
                    return jnp_sum(integrations_stack, axis=0)
        
                rat_res = compute_rat_res()
            else:
                rat_res = jnp_zeros_like(s_array, dtype=jnp_complex128)
    
            if len(poly_first_jax) > 0:
                bounds = s_array - s0
                first_contrib = jnp_sum(poly_first_jax) * bounds  # Sum coefficients times bounds
            else:
                first_contrib = jnp_zeros_like(s_array, dtype=jnp_complex128)
    
            if len(poly_second_jax) > 0:
                zeta_args = s_array - s0
                zeta_vals = vmap(lambda x: weierstrass_zeta(x, omega1, omega3))(zeta_args)
                second_contrib = jnp_sum(poly_second_jax) * zeta_vals
            else:
                second_contrib = jnp_zeros_like(s_array, dtype=jnp_complex128)
    
            poly_res_total = first_contrib + second_contrib
            final_results = poly_res_total + rat_res
    
            if s_array.size == 1:
                return complex(final_results[0])
            else:
                return [complex(x) for x in final_results]

        return lambda s: res_elp(s)

    elif deg_p <= 2:

        # Solution function
        def trig_res(s):
            res = [res_func(i) for i in s]

            if len(res) == 1:
                return res[0]
            else:
                return res

        return lambda s: trig_res(s)


"""
Functions with invert_<something> and integrate_<something> have the same inputs as invert_eom
and integrate_eom respectively. 
"""


def invert_trigonometric_first(
    polynomial, initial_values, int_sign, constant, datafile=None
):
    """
    Inverts an equation of motion of trigonometric type (<= 2nd degree polynomial).

    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial in standard form.
    initial_values : list
        The initial values converted by the same substitution used to convert <polynomial>.
    int_sign : int
        The sign of square root in the differential equation, either +1 or -1. The
        signs must have also been converted by the same substitution used to convert
        <polynomial>.
    constant : real
       The constant in front of the differential equation.
    datafile : string, optional
        A file name to store information pertaining to the computation, such as the
        initial values. This is optional, unlike solve_eom.

    Returns
    -------
    callable
        The solution function for the differential equation.

    """

    p = polynomial
    sym = Poly(p).gen
    s = Symbol("s")

    coeff_2 = polynomial.coeff(sym, 2)
    coeff_0 = polynomial.coeff(sym, 0)

    # Invert differential equation
    root = spsqrt(-coeff_2 / coeff_0)
    init_const = spasin(initial_values[1] * root)
    inverse = (
        1
        / root
        * spsin(
            int_sign
            * spsqrt(-coeff_2)
            * constant
            * (s - initial_values[0] + init_const)
        )
    )

    sol_func = lambdify(s, inverse, "jax")

    # Save inverse polynomial and initial values to <datafile>
    if datafile != None:
        with open(f"{datafile}.pickle", "wb") as output_file:
            dump([inverse, initial_values], output_file)
        print(
            f"In invert_trigonometric_first: solution function saved to, {datafile}.pickle"
        )
    return sol_func


def invert_elliptic_first(
    polynomial,
    initial_values,
    int_sign,
    constant,
    substitution,
    periodM=None,
    datafile=None,
):
    """
    Inverts an equation of motion of elliptic type and first kind.

    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial in standard form.
    initial_values : list
        The initial values converted by the same substitution used to convert <polynomial>.
    int_sign : int
        The sign of square root in the differential equation, either +1 or -1. The
        signs must have also been converted by the same substitution used to convert
        <polynomial>.
    substitution : callable
        A lambda function that computes the substitution that casted <polynomial> into
        standard form.
    constant : real
       The constant in front of the differential equation.
    periodM : matrix, optional
        A 1x2 list representing the period matricex of a genus 1 Riemann surface.
    datafile : string, optional
        A file name to store information pertaining to the computation, such as the
        initial values. This is optional, unlike solve_eom.

    Returns
    -------
    callable
        The solution function for the differential equation.

    """

    p = polynomial
    sym = Poly(p).gen
    g2 = -p.coeff(sym, 1)
    g3 = -p.coeff(sym, 0)

    # Compute periods if necessairy
    if periodM == None:
        periodMatrix = periods_firstkind(g2, g3)
    else:
        periodMatrix = periodM
    print("periodMatrix = ", periodMatrix)

    # Compute initial values
    if type(initial_values[1]) == oo:
        int_initial = 0
    else:
        int_initial = int_sign * inverse_weierstrass_P(
            initial_values[1], periodMatrix[0], periodMatrix[1]
        )

        # Correct initial direction if necessairy
        if (
            sign(
                weierstrass_P(
                    initial_values[0] - int_initial, periodMatrix[0], periodMatrix[1], 1
                )
            )
            != int_sign
        ):
            int_initial *= -1

    # Save to datafile
    if datafile != None:
        integrate_initial = int_initial
        initials = initial_values
        data = array([periodMatrix, g2, g3, integrate_initial, initials], dtype=object)
        save(datafile, data)
        print(f"In invert_elliptic_first: periodMatrix saved to, {datafile}.npy")

    @jit
    def vectorized_computation(s_array):
        s_jax = jnp_array(s_array, dtype=jnp_complex128)
    
        args = jnp_sqrt(constant) * s_jax - initial_values[0] - int_initial
        weierstrass_results = weierstrass_P(args, periodMatrix[0], periodMatrix[1])
        real_results = jnp_real(weierstrass_results)
    
        return vmap(substitution)(real_results)

    return lambda s: vectorized_computation(s)


def invert_hyperelliptic_first(
    zeros,
    physical_comp,
    initial_values,
    int_sign,
    constant,
    substitution,
    digits,
    periodM=None,
    datafile=None,
):
    """
    Inverts an equation of motion of hyperelliptic type and first kind. The majority
    of the computation is performed by <orbitdata>.

    Parameters
    ----------
    polynomial : symbolic
        A symbolic statement representing the polynomial in standard form.
    zeros : list
        A list of complex or real numbers representing the converted zeros of <polynomial>.
    physical_comp : int
        An integer, either 1 or 0, representing the component of the theta divisor which
        corresponds to physical values (i.e. to the component of the vector of holomorphic
        differentials dz = [1/sqrt(P(z), z/sqrt(P(z))], where P(z) = <polynomial>).
    initial_values : list
        The initial values converted by the same substitution used to convert <polynomial>.
    int_sign : int
        The sign of square root in the differential equation, either +1 or -1. The
        signs must have also been converted by the same substitution used to convert
        <polynomial>. NOTE: this has yet to be implemented for hyperelliptic differential
        equations (i.e. this function).
    substitution : callable
        The substitution used to convert <polynomial> as a callable lambda function.
    periodM : list, optional
        A 2x4 mpmath matrix, representing the period matrix of a genus 2 Riemann surface.
        If not provided, it will be computed.
    digits : int
        The number of digits to be used in the computation.
    datafile : string, optional
        A file name to store information pertaining to the computation, such as the
        period matrix. This is optional, unlike solve_eom.

    Returns
    -------
    callable
        The solution function for the differential equation.

    """

    global periods_inverse, riemannM
    realNS, complexNS = separate_zeros(zeros)

    # Compute periods if necessairy
    if periodM == None:
        print("Computing periods ...")
        periodMatrix = periods(realNS, complexNS, digits)
    else:
        periodMatrix = periodM
    print("periodMatrix = ", periodMatrix)

    omega1 = periodMatrix[0:2, 0:2]
    omega2 = periodMatrix[0:2, 2:4]

    # Define the inverse of the first period matrix and the Riemann matrix tau
    periods_inverse, riemannM = set_period_globals_genus2(periodMatrix)

    # Compute Legendre relation
    m = omega2 * omega1.T - omega1 * omega2.T
    print("Legendre relation = ", m)

    # Check accuracy
    if fabs(m[0, 1]) > 10 ** (-digits):
        eps = fabs(m[0, 1]) * 10
        print(
            f"WARNING in invert_hyperelliptic_first: accuracy reduced to {eps} due to Legendre relation."
        )
    else:
        eps = 10 ** (-digits + 1)

    # Compute inital value for Newton method and integration constant
    if physical_comp == 1:
        print(
            "WARNING in invert_hyperelliptic_first: case that physical component is the first has to be tested"
        )

        if type(initial_values[1]) == oo:
            initNewton = 0
            modified_init = spsqrt(constant) * initial_values[0]
        # Initial value in real zeros
        elif inlist(initial_values[1], realNS) >= 0:
            initNewton = -eval_period(
                inlist(initial_values[1], realNS), oo, realNS, zeros, periodMatrix, 1
            )
            modified_init = spsqrt(constant) * initial_values[0] + eval_period(
                inlist(initial_values[1], realNS), oo, realNS, zeros, periodMatrix, 0
            )
        else:
            k = inlist(
                initial_values[1],
                sorted(realNS + [initial_values[1]], key=lambda x: re(x)),
            )

            # Correct index since the size of <realNS> was changed above when computing <k>
            if k == len(realNS):
                k = len(realNS) - 1

            h = int_genus2_first(
                zeros, initial_values[1], realNS[k], periodMatrix
            )
            initNewton = -eval_period(k, oo, realNS, zeros, periodMatrix, 0) - h[1]
            modified_init = (
                spsqrt(constant) * initial_values[0]
                + eval_period(k, oo, realNS, zeros, periodMatrix, 1)
                + h[0]
            )
    else:
        if type(initial_values[1]) == oo:
            initNewton = 0
            modified_init = spsqrt(constant) * initial_values[0]
        # Initial value in real zeros
        elif inlist(initial_values[1], realNS) >= 0:
            initNewton = -eval_period(
                inlist(initial_values[1], realNS), oo, realNS, zeros, periodMatrix, 0
            )
            modified_init = spsqrt(constant) * initial_values[0] + eval_period(
                inlist(initial_values[1], realNS), oo, realNS, zeros, periodMatrix, 1
            )
        else:
            k = inlist(
                initial_values[1],
                sorted(realNS + [initial_values[1]], key=lambda x: re(x)),
            )

            # Correct index since the size of <realNS> was changed above when computing <k>
            if k == len(realNS):
                k = len(realNS) - 1

            h = int_genus2_first(
                zeros, initial_values[1], realNS[k], periodMatrix
            )
            initNewton = -eval_period(k, oo, realNS, zeros, periodMatrix, 0) - h[0]
            modified_init = (
                spsqrt(constant) * initial_values[0]
                + eval_period(k, oo, realNS, zeros, periodMatrix, 1)
                + h[1]
            )

    print("Check initial value for Newton method ...")
    max = check_initNewton(
        physical_comp,
        initNewton,
        [spsqrt(constant) * initial_values[0], initial_values[1]],
        modified_init,
        eps,
    )

    # If <datafile> is provided, store initial values, period matrix, and accurary in <datafile>
    # Otherwise just compute solution
    if datafile != None:
        initials_mod = [spsqrt(constant) * i for i in initial_values]
        if physical_comp == 1:
            invert_data = [
                initials_mod,
                [modified_init - spsqrt(constant) * initial_values[0], -initNewton],
                max,
            ]
        else:
            invert_data = [
                initials_mod,
                [-initNewton, modified_init - spsqrt(constant) * initial_values[0]],
                max,
            ]
        # Use jnp.save instead of numpy save for JAX arrays
        invert_data_jax = jnp_array(invert_data, dtype=jnp_complex128)
        jnp_save(datafile, jnp_array([periodMatrix.tolist(), invert_data_jax, eps], dtype=object))
        print(f"In invert_hyperelliptic_first: period matrix saved to, {datafile}.npy")

        return lambda affine_list: orbitdata(
            initial_values,
            jnp_complex128(modified_init),
            jnp_array([spsqrt(constant) * i for i in affine_list], dtype=jnp_complex128),
            substitution,
            jnp_complex128(initNewton),
            jnp_float64(eps),
            max,
            physical_comp,
            datafile,
        )
    else:
        return lambda affine_list: orbitdata(
            initial_values,
            jnp_complex128(modified_init),
            jnp_array([spsqrt(constant) * i for i in affine_list], dtype=jnp_complex128),
            substitution,
            jnp_complex128(initNewton),
            jnp_float64(eps),
            max,
            physical_comp,
        )

def integrate_hyperelliptic_first(component, datafile):
    """
    Procedure for inverting equations of motion of trigonometric, elliptic, and
    hyperelliptic type for various spacetimes. Specifically

    Parameters
    ----------
    physical_comp : int
        An integer, either 1 or 0, representing the component of the theta divisor which
        corresponds to physical values (i.e. to the component of the vector of holomorphic
        differentials dz = [1/sqrt(P(z), z/sqrt(P(z))], where P(z) = <polynomial>).
    datafile : string
        The name of the file containing the data generated by <orbitdata> and
        <invert_hyperelliptic_first>

    Returns
    -------
    result : list
        The result of the integration.

    """
    # Load data (cannot be JIT compiled - I/O operations)
    extended_orbitdata = load(datafile + "_orbitdata.npy", allow_pickle=True)
    invert_data = load(datafile + ".npy", allow_pickle=True)
    
    # Extract the component value from invert_data
    component_val = invert_data[1][1][component - 1]
    
    # Convert to JAX arrays for GPU computation
    extended_orbitdata_jax = jnp_array(extended_orbitdata, dtype=jnp_complex128)
    component_val_jax = jnp_complex128(component_val)
    
    # JIT-compiled computation part
    @jit
    def compute_result(data, comp_val):
        # Extract the 3rd element's component-1 value from each row
        # Assuming data structure: [subs_coord, coordinates, x, divisor, affine_list]
        divisor_comps = data[:, 3, component - 1]  # Get component from divisor
        result = divisor_comps + comp_val
        return result
    
    # Compute result
    result_jax = compute_result(extended_orbitdata_jax, component_val_jax)
    
    # Convert back to Python list if needed
    return [complex(x) for x in result_jax]

def check_initNewton(physical_comp, initNewton, initial_values, modified_init, eps):
    """
    Determine the summation bound for computing the fourier series of the genus 2
    hyperelliptic theta function.

    Parameters
    ----------
    physical_comp : int
        An integer, either 1 or 0, representing the component of the theta divisor which
        corresponds to physical values (i.e. to the component of the vector of holomorphic
        differentials dz = [1/sqrt(P(z), z/sqrt(P(z))], where P(z) = <polynomial>).
    initNewton : complex
        The initial value of the Newton method.
    initial_values : list
        The initial values [gammain, xin] with gamma and x as in <invert_eom>.
    modified_init : complex
        A complex constant = <initial_values[1]> + integration constant (computed in
        <invert_eom>)
    eps : float
        The accuracy of the Newton method.

    Returns
    -------
    Max : int
        A natural number in the range 5 <= Max <= 30.

    """

    global periods_inverse, riemannM
    g = [1 / 2, 1 / 2]
    h = [0, 1 / 2]
    char = [g, h]

    if physical_comp == 1:
        z = (
            1
            / 2
            * periods_inverse
            * matrix([initial_values[0] - modified_init, initNewton])
        )
    else:
        z = (
            1
            / 2
            * periods_inverse
            * matrix([initNewton, initial_values[0] - modified_init])
        )

    max = 5
    f = hyp_theta_fourier(z, riemannM, char, minMax=max)

    # Determine summation bound
    while (fabs(re(f)) > eps / 10 or fabs(im(f)) > eps / 10) and max < 30:
        # Evaluate theta function
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
        # print("f = ", f)
        max += 1

    # Evaluate the derivatives of the sigma function
    s1 = hyp_theta_fourier(z, riemannM, char, [1], max)
    s2 = hyp_theta_fourier(z, riemannM, char, [2], max)
    u = -(
        (s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0])
        / (s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1])
    )

    # Check if u(initial_values[0]) is within the correct accuracy of initial_values[1] for the initial Newton method value
    if (initial_values[1] == oo and fabs(u) < 1 / eps * (10 ** (-4))) or (
        initial_values[1] < oo and fabs(u - initial_values[1]) > eps * 10**4
    ):
        max += 1
        s1 = hyp_theta_fourier(z, riemannM, char, [1, 0], max)
        s2 = hyp_theta_fourier(z, riemannM, char, [0, 1], max)
        u = -(
            (s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0])
            / (s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1])
        )
        if (initial_values[1] == oo and fabs(u) < 1 / eps) or (
            initial_values[1] < oo and fabs(u - initial_values[1]) > eps * 10**4
        ):
            raise ValueError(
                f"u({initial_values[0]}) not close enough to u0 = {initial_values[1]} for x0 = {initNewton}"
            )

    print("Maximal summation index for Kleinian sigma function set to ", max)

    return max

@jit
def orbitdata(
    initial_values,
    modified_init,
    affine_list,
    substitution,
    initNewton,
    eps,
    minMax,
    physical_comp,
    datafile=None,
):
    """
    JAX-optimized version of orbitdata.
    """
    global periods_inverse, riemannM
    
    # Convert to JAX types
    initNewton = jnp_complex128(initNewton)
    modified_init = jnp_complex128(modified_init)
    eps = jnp_float64(eps)
    affine_array = jnp_array(affine_list, dtype=jnp_complex128)
    
    init_coord = jnp_complex128(initial_values[1])
    init_gamma = jnp_complex128(initial_values[0])
    
    # Use existing inlist function
    pos0 = inlist(init_gamma, affine_list)
    
    # Pre-allocate arrays with fixed size
    max_len = len(affine_list)
    x_arr = jnp_zeros(max_len, dtype=jnp_complex128)
    coords_arr = jnp_zeros(max_len, dtype=jnp_complex128)
    subs_arr = jnp_zeros(max_len, dtype=jnp_complex128)
    valid_mask = jnp_zeros(max_len, dtype=jnp_bool)
    
    # Set initial position
    x_arr = x_arr.at[pos0].set(initNewton)
    coords_arr = coords_arr.at[pos0].set(init_coord)
    subs_arr = subs_arr.at[pos0].set(substitution(init_coord))
    valid_mask = valid_mask.at[pos0].set(True)
    
    # Process forward
    def forward_step(i, carry):
        x_arr, coords_arr, subs_arr, valid_mask, stop = carry
        current_idx = pos0 + i + 1
        
        def compute():
            # Get last valid x
            prev_idx = jnp_max(jnp_arange(max_len) * valid_mask)
            x_prev = x_arr[prev_idx]
            
            # Call solution
            coord = solution(affine_array[current_idx] - modified_init, x_prev, eps, minMax, physical_comp)
            
            # Check success
            result_len = jnp_array(len(coord))
            is_success = jnp_logical_and(
                result_len == 3,
                jnp_im(coord[0]) < 100 * eps
            )
            
            # Update if successful
            x_arr_new = lax.cond(
                is_success,
                lambda: x_arr.at[current_idx].set(coord[1]),
                lambda: x_arr
            )
            
            coords_arr_new = lax.cond(
                is_success,
                lambda: coords_arr.at[current_idx].set(jnp_real(coord[0])),
                lambda: coords_arr
            )
            
            subs_arr_new = lax.cond(
                is_success,
                lambda: subs_arr.at[current_idx].set(substitution(jnp_real(coord[0]))),
                lambda: subs_arr
            )
            
            valid_mask_new = lax.cond(
                is_success,
                lambda: valid_mask.at[current_idx].set(True),
                lambda: valid_mask
            )
            
            stop_new = jnp_logical_or(stop, jnp_logical_not(is_success))
            
            return (x_arr_new, coords_arr_new, subs_arr_new, valid_mask_new, stop_new)
        
        def skip():
            return (x_arr, coords_arr, subs_arr, valid_mask, stop)
        
        return lax.cond(
            jnp_logical_or(stop, current_idx >= max_len),
            skip,
            compute
        )
    
    # Process backward
    def backward_step(i, carry):
        x_arr, coords_arr, subs_arr, valid_mask, stop = carry
        current_idx = pos0 - i - 1
        
        def compute():
            # Get first valid x
            first_idx = argmax(valid_mask)
            x_prev = x_arr[first_idx]
            
            # Call solution
            coord = solution(affine_array[current_idx] - modified_init, x_prev, eps, minMax, physical_comp)
            
            # Check success
            result_len = jnp_array(len(coord))
            is_success = jnp_logical_and(
                result_len == 3,
                jnp_im(coord[0]) < 100 * eps
            )
            
            # Update if successful
            x_arr_new = lax.cond(
                is_success,
                lambda: x_arr.at[current_idx].set(coord[1]),
                lambda: x_arr
            )
            
            coords_arr_new = lax.cond(
                is_success,
                lambda: coords_arr.at[current_idx].set(jnp_real(coord[0])),
                lambda: coords_arr
            )
            
            subs_arr_new = lax.cond(
                is_success,
                lambda: subs_arr.at[current_idx].set(substitution(jnp_real(coord[0]))),
                lambda: subs_arr
            )
            
            valid_mask_new = lax.cond(
                is_success,
                lambda: valid_mask.at[current_idx].set(True),
                lambda: valid_mask
            )
            
            stop_new = jnp_logical_or(stop, jnp_logical_not(is_success))
            
            return (x_arr_new, coords_arr_new, subs_arr_new, valid_mask_new, stop_new)
        
        def skip():
            return (x_arr, coords_arr, subs_arr, valid_mask, stop)
        
        return lax.cond(
            jnp_logical_or(stop, current_idx < 0),
            skip,
            compute
        )
    
    # Run forward processing
    carry = (x_arr, coords_arr, subs_arr, valid_mask, False)
    forward_count = max_len - pos0 - 1
    for i in range(forward_count):
        carry = forward_step(i, carry)
    
    # Run backward processing
    backward_count = pos0
    for i in range(backward_count):
        carry = backward_step(i, carry)
    
    x_arr_final, coords_arr_final, subs_arr_final, valid_mask_final, _ = carry
    
    # Extract results
    valid_indices = jnp_arange(max_len)[valid_mask_final]
    valid_indices_sorted = jnp_sort(valid_indices)
    subs_coord_result = [complex(subs_arr_final[i]) for i in valid_indices_sorted]
    
    # Handle datafile saving outside JIT
    if datafile is not None:
        # Convert to numpy for saving
        subs_coord_np = array([complex(x) for x in subs_arr_final[valid_mask_final]])
        coords_np = array([complex(x) for x in coords_arr_final[valid_mask_final]])
        x_np = array([complex(x) for x in x_arr_final[valid_mask_final]])
        affine_np = array([complex(x) for x in affine_array[valid_mask_final]])
        
        # Create divisor array
        divisor_np = np_zeros((len(subs_coord_np), 2), dtype=complex)
        for j in range(len(subs_coord_np)):
            if physical_comp == 1:
                divisor_np[j] = [-modified_init, x_np[j]]
            else:
                divisor_np[j] = [x_np[j], -modified_init]
        
        extended_orbitdata = array([
            [subs_coord_np[j], coords_np[j], x_np[j], divisor_np[j], affine_np[j]]
            for j in range(len(subs_coord_np))
        ], dtype=object)
        
        save(datafile + "_orbitdata", extended_orbitdata)
    
    return subs_coord_result

def sigma_ln_numerical(x, y, omega1, omega3):
    """
    Computes the value of ln(weierstrass_sigma(x - y) / weierstrass_sigma(x + y)) numerically.

    When the imaginary part of the result is 0, then the result becomes
    integral(weierstrass_zeta(x - y) - weierstrass_zeta(x + y), from 0 to x).

    Parameters
    ----------
    x : complex
        A complex or real number.
    y : complex
        A potentially complex or real number.
    omega1 : complex
        The first half period in the period lattice.
    omega3 : complex
        The second half period in the period lattice.

    Returns
    -------
    value : complex
        The result of the evaluated function.

    """
    value = chop(jnp_log(
            weierstrass_sigma(x - y, omega1, omega3)
            / weierstrass_sigma(x + y, omega1, omega3)
        )
    )

    if im(value) != 0:
            gl = GaussLegendre()

    value = chop(gl.integrate(lambda s: weierstrass_zeta(s - y, omega1, omega3) - weierstrass_zeta(s + y, omega1, omega3), 
                              dim=1, N=101, integration_domain=[[0, x]], backend = "jax")) 

    return value

@jit
def sigma_ln(x, y, omega1, omega3):
    """
    Computes the value of ln(weierstrass_sigma(x-y) / weierstrass_sigma(x+y)) while
    accounting the change in branches, ensuring the function remains continuous.

    Parameters
    ----------
    x : list
        A list of numbers to be evaluated as a parameter.
    y : complex
        A potentially complex or real number.
    omega1 : complex
        The first half period in the period lattice.
    omega3 : complex
        The second half period in the period lattice.

    Returns
    -------
    value : complex
        The result of the evaluated function.

    """

    x_array = jnp_array(x, dtype=jnp_complex128) if not isinstance(x, jnp_ndarray) else x
    y = jnp_complex128(y)
    omega1 = jnp_complex128(omega1)
    omega3 = jnp_complex128(omega3)
    
    # Compute eta
    eta = periods_secondkind(omega1, omega3)[0]
    
    # Branch factor
    c = sigma_ln_numerical(omega1, y, omega1, omega3) - sigma_ln_numerical(
        1e-10, y, omega1, omega3
    )
    
    # Compute branch (using JAX-compatible rounding)
    branch_component = -omega1 / jnp_pi * (jnp_im(c) / omega1 + jnp_im(2 * eta * y / omega1))
    branch = jnp_round(branch_component)  # JAX equivalent of nint
    switch = branch * jnp_pi * 1j
    
    # Define function for single x value
    def compute_single(i):
        sigmatilde1 = weierstrass_sigma(i - y, omega1, omega3) * jnp_exp(
            -eta * (i - y) ** 2 / (2 * omega1)
        )
        sigmatilde2 = weierstrass_sigma(i + y, omega1, omega3) * jnp_exp(
            -eta * (i + y) ** 2 / (2 * omega1)
        )
        
        value = (
            jnp_log((sigmatilde1 / sigmatilde2) * jnp_exp(switch * (i / omega1 - 1)))
            - switch * (i / omega1 - 1)
            - 2 * eta * i * y / omega1
        )
        return chop(value)
    
    # Vectorize over all x values
    result = vmap(compute_single)(x_array)
    
    # Return scalar if single element, else array
    return lax.cond(
        x_array.size == 1,
        lambda: result[0],
        lambda: result
    )
   
def integrate_elliptic_third(integrand, datafile):
    """
    Computes the integral of 1 / (weierstrass(mino time) - pole)^n, where n is either 1 or 2
    (NOTE: n = 2 has not been implemented yet, tbd).

    Parameters
    ----------
    integrand : symbolic
        The integrand to be computed, containing a simple pole, or double pole (NOTE: double
        poles have yet to be implemented, tbd).
    datafile : string
        The name of the file containing the period matrix, elliptic invariants, initial
        values, etc.

    Returns
    -------
    callable
        The function that computes the integration, taking a list of values or a single
        value as input (list is faster for more points than single).

    """

    # Initial parameters
    periods, g2, g3, int_init, inits = npload(f"{datafile}.npy", allow_pickle=True)
    mod_int = inits[0] + int_init
    coeff = integrand.as_numer_denom()[0]

    # Locate the value v1 in the fundamental period parallelogram such that:
    # weierstrass_P(v1) = pole
    inv_integrand = 1 / integrand
    pole = solve(inv_integrand, Poly(inv_integrand).gen)
    v1 = inverse_weierstrass_P(pole[0], periods[0], periods[1])

    @jit
    def compute_single(s_val):
        """Compute result for a single value."""
        s_int = s_val - mod_int
        log_res = sigma_ln(s_int, v1, periods[0], periods[1])
        log_res0 = sigma_ln(inits[0] - mod_int, v1, periods[0], periods[1])
        
        int_sol_val = (1 / weierstrass_P(v1, periods[0], periods[1], 1)) * \
                     (2 * (s_val - mod_int) * weierstrass_zeta(v1, periods[0], periods[1]) + log_res)
        int_res0 = (1 / weierstrass_P(v1, periods[0], periods[1], 1)) * \
                  (2 * (inits[0] - mod_int) * weierstrass_zeta(v1, periods[0], periods[1]) + log_res0)
        
        return coeff * (int_sol_val - int_res0)
    
    # Vectorize using vmap
    compute_vectorized = jit(vmap(compute_single))
    
    def result(s):
        """Wrapper that handles both single values and lists."""
        if isinstance(s, (list, tuple, np_ndarray)):
            s_array = jnp_array(s, dtype=jnp_complex128)
            results = compute_vectorized(s_array)
            return [complex(x) for x in results]
        else:
            # Single value
            return complex(compute_single(jnp_complex128(s)))
    
    return lambda s: result(s)

def compute_secondkind_periods(zeros, eps, periodMatrix, datafile, digits, minMax=5):
    """
    Computes the period matrices of the second kind for a genus 2 Riemann surface (i.e the
    integral of the vector of canonical meromorphic differentials along the contours
    connecting the branch cuts and the contours looping around the branch cuts).

    Parameters
    ----------
    zeros : list
        A list of complex or real numbers representing the zeros of the polynomial defining
        the genus 2 Riemann surface.
    eps : float
        A small error epsilon to handle divergence.
    periodMatrix : matrix
        A 2x4 matrix representing the periods of the first kind (integral of the vector of
        canonical holomorphic diffentials along the same contours above).
    datafile : string
        A string containing the name of a file to store the periods and epsilon error.
    digits : int
        The number of digits to be used in the computation.
    minMax : natural
        A natural number from 0 <= minMax <= 30, the summation bound of the theta function.

    Returns
    -------
    secondkindperiods : matrix
        A 2x4 matrix containing the period matrices of the second kind.
    r1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    r2: list
        A list representing the coefficients of the second elment of the vector of
        canonical meromorphic differentials.

    """

    x = Symbol("x")

    periodMatrix = matrix(periodMatrix)

    omega1 = periodMatrix[0:2, 0:2]
    omega2 = periodMatrix[0:2, 2:4]

    p = (
        (x - zeros[0])
        * (x - zeros[1])
        * (x - zeros[2])
        * (x - zeros[3])
        * (x - zeros[4])
    )
    p = collect(p.expand(), x)

    coeffsP = [re(p.coeff(x, i)) for i in range(6)]

    # Meromorphic differential coefficients
    r1 = [0, 1 / 4 * coeffsP[3], 1 / 2 * coeffsP[4], 3 / 4 * coeffsP[5]]
    r2 = [0, 0, 1 / 4 * coeffsP[5]]

    realNS, complexNS = separate_zeros(zeros)

    print("Computing second kind periods ...")
    second_kind_periods = periods_second(r1, r2, realNS, complexNS, digits)

    eta1 = second_kind_periods[0:2, 0:2]
    eta2 = second_kind_periods[0:2, 2:4]

    m = eta2 * eta1.T - eta1 * eta2.T
    print("Legendre relation for periods of second kind = ", m)

    # Check accuracy
    if fabs(m[0, 1]) > eps / 10:
        eps = fabs(m[0, 1]) * 10
        print(
            f"WARNING in solve_hyperelliptic_third: accuracy further reduced to {eps} due to Legendre relation for periods of second kind"
        )

    m = omega2 * eta1.T - omega1 * eta2.T
    print("Mixed Legendre relation = ", m)

    if fabs(m[0, 1]) > eps / 10 or fabs(m[0, 0] - pi / 2 * 1j) > eps / 10:
        if fabs(m[0, 1]) > fabs(m[0, 0] - pi / 2 * 1j):
            eps = fabs(m[0, 1]) * 10
        else:
            eps = fabs(m[0, 0] - pi / 2 * 1j) * 10
        print(
            f"WARNING in solve_hyperelliptic_third: further reduced to {eps} due to relation between periods of first and second kind"
        )

    # Save second kind periods and accuracy
    save(
        datafile + "_secondkindperiods",
        array([second_kind_periods.tolist(), eps], dtype=object),
    )
    print(f"Saved second kind period matrix to {datafile + '_secondkindperiods.npy'}")

    return second_kind_periods, r1, r2


def integrate_hyperelliptic_third(zeros, r1, r2, eta, integrand, datafile, digits):
    """
    Integrates a hyperelliptic integral of the third kind.

    Parameters
    ----------
    zeros : list
        A list of complex or real numbers representing the zeros of the polynomial defining
        the genus 2 Riemann surface.
    r1 : list
        A list representing the coefficients of the first element of the vector of
        canonical meromorphic differentials.
    r2: list
        A list representing the coefficients of the second elment of the vector of
        canonical meromorphic differentials.
    eta : matrix
        A 2x2 matrix containing the first 2x2 part of the period matrix of the second kind.
    integrand : symbolic
        The integrand containg the pole.
    datafile : string
        The name of the file (not including the . extension that is storing the periods of
        the second kind).
    digits : int
        The number of digits to be used in the computation.

    Returns
    -------
    callable
        The solution function that performs the integration, taking in either a list of
        values or a single value (list of values is faster).

    """

    global periods_inverse, riemannM
    periodMatrix, invert_data, eps = npload(datafile + ".npy", allow_pickle=True)

    # Initial parameters
    periodMatrix = matrix(periodMatrix)
    x = Symbol("x")
    init = invert_data[0]
    s0 = invert_data[1]
    char = [[1 / 2, 1 / 2], [0, 1 / 2]]

    periods_inverse, riemannM = set_period_globals_genus2(periodMatrix)

    # Locate the pole
    inv_integrand = 1 / integrand
    pole = solve(inv_integrand, Poly(inv_integrand).gen)[0].evalf()

    p = (
        (x - zeros[0])
        * (x - zeros[1])
        * (x - zeros[2])
        * (x - zeros[3])
        * (x - zeros[4])
    )
    p = collect(p.expand(), x)

    coeffsP = [re(p.coeff(x, i)) for i in range(6)]

    # Check if integral is second kind
    if inlist(pole, zeros) >= 0:
        raise ValueError("Invalid use: integral is of second kind")

    print(
        "Computing constants needed for solution hyperelliptic integral of third kind ..."
    )

    realNS, complexNS = separate_zeros(zeros)

    k = inlist(pole, sorted([pole] + zeros, key=lambda x: re(x)))

    if k > inlist(realNS[-1], zeros) or k == 4:  # pole on realNS[-1]..oo
        int_dz = myint_genus2(zeros, pole, realNS[-1], 1)
        int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, realNS[-1], 1)
        int_3 = 2 * myint_genus2_second(
            zeros, [0, 0, 0, 1], pole, realNS[-1], 1 
        )
        inf1 = eval_period(len(realNS) - 1, oo, realNS, zeros, periodMatrix, 0)
        inf2 = eval_period(len(realNS) - 1, oo, realNS, zeros, periodMatrix, 1)
    # in the remaining cases there is at least one real zero > pole!
    elif k == 3:
        if im(zeros[3]) == 0:
            int_dz = myint_genus2(zeros, pole, zeros[3], 1)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[3], 1)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[3], 1 
            )
            inf1 = eval_period(
                inlist(zeros[3], realNS), oo, realNS, zeros, periodMatrix, 0
            )
            inf2 = eval_period(
                inlist(zeros[3], realNS), oo, realNS, zeros, periodMatrix, 1
            )
        else:  # cases ima2Per3 and ima4Per1
            raise ValueError(
                "Case that the pole is located on a vertical branch cut is tbd"
            )

    elif k == 2:
        if im(zeros[1]) == 0 and im(zeros[0]) == 0:
            int_dz = myint_genus2(zeros, pole, zeros[1], 1)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[1], 1)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[1], 1 
            )
            inf1 = eval_period(
                inlist(zeros[1], realNS), oo, realNS, zeros, periodMatrix, 0
            )
            inf2 = eval_period(
                inlist(zeros[1], realNS), oo, realNS, zeros, periodMatrix, 1
            )

        elif im(zeros[2]) == 0 and im(zeros[3]) == 0:
            int_dz = myint_genus2(zeros, pole, zeros[2], 1)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[2], 1)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[2], 1 
            )
            inf1 = eval_period(
                inlist(zeros[2], realNS), oo, realNS, zeros, periodMatrix, 0
            )
            inf2 = eval_period(
                inlist(zeros[2], realNS), oo, realNS, zeros, periodMatrix, 1
            )
        elif pole == re(zeros[1]):
            int_dz = myint_genus2(zeros, pole, realNS[0], 1)
            int_2 = 2 * myint_genus2_second(
                zeros, [0, 0, 1], pole, realNS[0], 1 
            )
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, realNS[0], 1 
            )
            inf1 = eval_period(0, oo, realNS, zeros, periodMatrix, 0)
            inf2 = eval_period(0, oo, realNS, zeros, periodMatrix, 1)
        else:  # ima4Per1 or ima4Per3
            r = 0
            for i in range(1, 7):
                r += re(coeffsP[i - 1]) * x ** (i - 1)
            r = lambdify(x, r, "torch")
            gl = GaussLegendre()
            int_dz = matrix(
                [
                    1j * gl.integrate(lambda x: 1 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[1])]], backend = "torch"),
                    1j * gl.integrate(lambda x: x / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[1])]], backend = "torch"),
                ]
            )
            int_2 = 2 * 1j * gl.integrate(lambda x: x**2 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[1])]], backend = "torch")
            int_3 = 2 * 1j * gl.integrate(lambda x: x**3 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[1])]], backend = "torch")
            int_dz += int_genus2_complex(
                zeros, re(zeros[1]), im(zeros[1]), 0, 1 
            )

            int_2 = int_2 + 2 * int_genus2_complex_second(
                zeros, [0, 0, 1], re(zeros[0]), fabs(im(zeros[0])), 0, 1 
            )

            int_3 += 2 * int_genus2_complex_second(
                zeros, [0, 0, 0, 1], re(zeros[0]), fabs(im(zeros[0])), 0, 1 
            )

            inf1 = periodMatrix[0, 2] - periodMatrix[0, 0]
            inf2 = periodMatrix[1, 2] - periodMatrix[1, 0]

    elif k == 1:
        if im(zeros[0]) == 0:
            int_dz = myint_genus2(zeros, pole, zeros[0], 1)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[0], 1)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[0], 1 
            )
            inf1 = eval_period(
                inlist(zeros[0], realNS), oo, realNS, zeros, periodMatrix, 0
            )
            inf2 = eval_period(
                inlist(zeros[0], realNS), oo, realNS, zeros, periodMatrix, 1
            )
        else:  # cases ima2Per1, ima4Per1, and Ima4Per3
            raise ValueError(
                "Case that the pole is located on a vertical branch cut is tbd"
            )

    elif k == 0:
        if im(zeros[0]) == 0 and im(zeros[1]) == 0:
            int_dz = myint_genus2(zeros, pole, zeros[0], 1)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[0], 1)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[0], 1 
            )
            inf1 = eval_period(0, oo, realNS, zeros, periodMatrix, 0)
            inf2 = eval_period(0, oo, realNS, zeros, periodMatrix, 1)
        elif im(zeros[0]) != 0 and im(zeros[1]) != 0:
            r = 0
            for i in range(1, 7):
                r += re(coeffsP[i - 1]) * x ** (i - 1)
            r = lambdify(x, r, "torch")
            gl = GaussLegendre()

            int_dz = matrix(
                [
                    1j * gl.integrate(lambda x: 1 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[0])]], backend = "torch"),
                    1j * gl.integrate(lambda x: x / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[0])]], backend = "torch"),
                ]
            )
            +int_genus2_complex(zeros, re(zeros[0]), fabs(im(zeros[0])), 0, 1)
            int_2 = (
                2 * 1j * gl.integrate(lambda x: x**2 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[0])]], backend = "torch")
                + 2 * int_genus2_complex_second(zeros, [0, 0, 1], re(zeros[0]), fabs(im(zeros[0])), 0, 1)
            )

            int_3 = (
                2 * 1j * gl.integrate(lambda x: x**3 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[0])]], backend = "torch")
                + 2 * int_genus2_complex_second(zeros, [0, 0, 0, 1], re(zeros[0]), fabs(im(zeros[0])), 0, 1)
            )

            inf1 = periodMatrix[0, 2] - periodMatrix[0, 0]
            inf2 = periodMatrix[1, 2] - periodMatrix[1, 0]
        else:
            r = 0
            for i in range(1, 7):
                r += re(coeffsP[i - 1]) * x ** (i - 1)
            r = lambdify(x, r, "torch")
            gl = GaussLegendre()

            int_dz = ( 
                matrix(
                    [
                        1j * gl.integrate(lambda x: 1 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, zeros[0]]], backend = "torch"),
                        1j * gl.integrate(lambda x: x / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, zeros[0]]], backend = "torch"),
                    ]
                )
                + myint_genus2(zeros, zeros[0], re(zeros[1]), 1)
                + int_genus2_complex(zeros, re(zeros[1]), fabs(im(zeros[1])), 1, 1)
            )

            int_2 = (
                2 * 1j * gl.integrate(lambda x: x**2 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, zeros[0]]], backend = "torch")
                + 2 * myint_genus2_second(zeros, [0, 0, 1], zeros[0], re(zeros[1]), 1)
                + 2 * int_genus2_complex_second(zeros, [0, 0, 1], re(zeros[1]), fabs(im(zeros[1])), 1, 1)
            )

            int_3 = (
                2 * 1j * gl.integrate(lambda x: x**3 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, zeros[0]]], backend = "torch")
                + 2 * myint_genus2_second(zeros, [0, 0, 0, 1], zeros[0], re(zeros[1]), 1)
                + 2 * int_genus2_complex_second(zeros, [0, 0, 0, 1], re(zeros[1]), fabs(im(zeros[1])), 1, 1 )
            )

            inf1 = periodMatrix[0, 2] - periodMatrix[0, 0]
            inf2 = periodMatrix[1, 2] - periodMatrix[1, 0]

    # Construct integrals of differentials
    int_dr1 = r1[1] * 2 * int_dz[1] + r1[2] * int_2 + r1[3] * int_3
    int_dr2 = r2[2] * int_2
    yi_inf = matrix([int_dz[0] + inf1, int_dz[1] + inf2])
    xi_inf = matrix([-int_dz[0] + inf1, -int_dz[1] + inf2])

    eta1 = eta[0:2, 0:2]
    kappa = 1 / 2 * eta1 * periods_inverse

    # Kleinian sigma function without exponential factor (may cause bad accuracy)
    @jit
    def sigma(z):
        z_array = jnp_array(z, dtype=jnp_complex128)
    
        inverse_sum = dot(periods_inverse, z_array)
    
        half_inverse_sum = 0.5 * inverse_sum
    
        return hyp_theta_fourier(half_inverse_sum, riemannM, char)

    # Solution function
    def result(s):
        extended_orbitdata = load(datafile + "_orbitdata.npy", allow_pickle=True)
    
        divisor = array([extended_orbitdata[i][3] for i in range(len(extended_orbitdata))])
        affine_list = array([extended_orbitdata[i][4] for i in range(len(extended_orbitdata))])
    
        pos0 = inlist(init[0], affine_list)
    
        if pos0 == -1:
            raise ValueError(
            "Position of initial value could not be located in list of affine parameters"
        )
    
        print("Computing solution points ...")
    
        # Convert to JAX arrays for GPU computation
        divisor_jax = jnp_array(divisor, dtype=jnp_complex128)
        xi_inf_jax = jnp_array(xi_inf, dtype=jnp_complex128)
        yi_inf_jax = jnp_array(yi_inf, dtype=jnp_complex128)
        kappa_jax = jnp_array(kappa, dtype=jnp_complex128)
        s0_jax = jnp_array(s0, dtype=jnp_complex128)
        int_dr1_jax = jnp_complex128(int_dr1)
        int_dr2_jax = jnp_complex128(int_dr2)
    
        @jit
        def compute_results():
            xi_terms = divisor_jax + 2 * xi_inf_jax
            yi_terms = divisor_jax + 2 * yi_inf_jax
        
            # Assuming sigma() works with vectorized inputs
            vars_vec = sigma(xi_terms) / sigma(yi_terms)
        
            # Vectorized log_vars computation
            diff_vec = xi_inf_jax - yi_inf_jax
            # Compute matrix multiplication for all points
            log_vars_vec = 2 * jnp_sum(divisor_jax * kappa_jax * diff_vec, axis=1)
        
            # Compute constant
            const = (
                (2 * divisor_jax[pos0] @ kappa_jax @ diff_vec) + 0.5 * jnp_log(
                    sigma(divisor_jax[pos0] + 2 * xi_inf_jax) 
                    / sigma(divisor_jax[pos0] + 2 * yi_inf_jax)
                )
            )
        
            # Pre-compute common terms for all points
            common_terms = log_vars_vec + 0.5 * jnp_log(vars_vec) - const
        
            # Compute dot product for all points
            dot_terms = jnp_sum((divisor_jax + s0_jax) * jnp_array([int_dr1_jax, int_dr2_jax]), axis=1)
        
            # Base results without branch tracking
            base_results = common_terms - dot_terms
        
            # BRANCH TRACKING - This is the critical sequential part
            def process_forward(carry, i):
                branch, results = carry
                current_idx = pos0 + i + 1
            
                # Branch condition
                re_prev = jnp_real(vars_vec[current_idx - 1])
                im_prev = jnp_im(vars_vec[current_idx - 1])
                im_curr = jnp_im(vars_vec[current_idx])
            
                branch_update = jnp_where(
                    re_prev < 0,
                    jnp_where(
                        jnp_logical_and(im_prev > 0, im_curr < 0),
                        branch + 1,
                    jnp_where(
                        jnp_logical_and(im_prev < 0, im_curr > 0),
                            branch - 1,
                            branch
                        )
                    ),
                    branch
                )
            
                # Add branch contribution
                result_with_branch = base_results[current_idx] + (jnp_pi * 1j * branch_update)
                results = ops.index_update(results, current_idx, result_with_branch)
            
                return (branch_update, results), None
        
            def process_backward(carry, i):
                branch, results = carry
                current_idx = pos0 - i - 1
            
                # Branch condition (looking at i+1)
                re_next = jnp_real(vars_vec[current_idx + 1])
                im_next = jnp_im(vars_vec[current_idx + 1])
                im_curr = jnp_im(vars_vec[current_idx])
            
                branch_update = jnp_where(
                    re_next < 0,
                    jnp_where(
                        jnp_logical_and(im_next > 0, im_curr < 0),
                        branch + 1,
                        jnp_where(
                            jnp_logical_and(im_next < 0, im_curr > 0),
                            branch - 1,
                            branch
                        )
                    ),
                    branch
                )
            
                # Add branch contribution
                result_with_branch = base_results[current_idx] + (jnp_pi * 1j * branch_update)
                results = ops.index_update(results, current_idx, result_with_branch)
            
                return (branch_update, results), None
        
            # Initialize results array
            results_array = jnp_zeros_like(base_results)
            results_array = results_array.at[pos0].set(base_results[pos0])
        
            # Process forward
            forward_count = len(divisor_jax) - pos0 - 1
            carry_forward = (0, results_array)
            for i in range(forward_count):
                carry_forward, _ = process_forward(carry_forward, i)
        
            # Process backward
            backward_count = pos0
            carry_backward = (0, carry_forward[1])
            for i in range(backward_count):
                carry_backward, _ = process_backward(carry_backward, i)
        
            final_results = carry_backward[1]
        
            # Final scaling (vectorized)
            # Assuming pole, zeros are defined elsewhere
            scaling_factor = jnp_sqrt(jnp_real(
                (pole - zeros[0]) * (pole - zeros[1]) * (pole - zeros[2]) * 
                (pole - zeros[3]) * (pole - zeros[4])
            ))
        
            scaled_results = final_results / scaling_factor
        
            return [complex(x) for x in scaled_results]
    
        return compute_results()

    return lambda s: result(s)

def solution(affineParameter, initNewton, eps, minMax, physical_comp):
    """
    Computes the unphysical component of the theta divisor.

    Parameters
    ----------
    affineParameter : complex
        The affine parameter gamma for which the solution for the hyperelliptic differential
        equation should be computed.
    initNewton : complex
        The initial value of the Newton method.
    eps : float
        The accuracy of the Newton method.
    minMax : int
        The summation bound for the hyperelliptic theta function.
    physical_comp : int
        An integer, either 1 or 0, representing the component of the theta divisor which
        corresponds to physical values (i.e. to the component of the vector of holomorphic
        differentials dz = [1/sqrt(P(z), z/sqrt(P(z))], where P(z) = <polynomial>).

    Returns
    -------
    list
        A list containing either two or three elements: if the Newton method failed, the list
        has two elements, where the first is a list of the steps taken by the iteration
        and the second the value of the theta function at the last iteration. If the iteration
        process was successful the list has three elements, where the first is the solution of
        the hyperelliptic differential equation at <affineParameter>, the second the redundant
        unphysical component of the theta divisor used to compute the first element,
        and the third (2*omega)*(the element of the thetadivisor used to compute the first
        element), where omega is the first 2x2 part of the period matrix of first kind.

    """

    if physical_comp == 1:
        return solution_first(affineParameter, initNewton, eps, minMax)
    else:
        return solution_second(affineParameter, initNewton, eps, minMax)

@jit
def solution_first(affineParameter: complex, initNewton: complex, eps: float, minMax: int) -> tuple:
    """
    Computes the unphysical component of the theta divisor.
    """
    # Convert to JAX arrays first
    affineParameter = jnp_complex128(affineParameter)
    initNewton = jnp_complex128(initNewton)
    eps = jnp_float64(eps)
    
    # Access global variables (assuming they're JAX arrays)
    global periods_inverse, riemannM
    
    # Convert periods_inverse to complex128 if needed
    if not isinstance(periods_inverse, jnp_ndarray):
        periods_inverse = jnp_array(periods_inverse, dtype=jnp_complex128)
    
    # Characteristic vector
    char = jnp_array([[0.5, 0.5], [0.0, 0.5]], dtype=jnp_complex128)
    
    # Initialize Newton iteration
    zero = initNewton
    aff = affineParameter
    
    # Extract real and imaginary parts
    zeroR = jnp_real(zero)
    zeroI = jnp_im(zero)
    affR = jnp_real(aff)
    affI = jnp_im(aff)
    
    # Extract period matrix components
    perR = jnp_real(periods_inverse)
    perI = jnp_im(periods_inverse)
    
    # Precompute p values
    p1 = perR[0, 0] * affR - perI[0, 0] * affI
    p2 = perR[0, 0] * affI + perI[0, 0] * affR
    p3 = perR[1, 0] * affR - perI[1, 0] * affI
    p4 = perR[1, 0] * affI + perI[1, 0] * affR
    
    # Define function for one Newton iteration
    def newton_step(carry, _):
        zeroR, zeroI, count = carry
        
        # Compute z
        zfirstR = 0.5 * (perR[0, 1] * zeroR - perI[0, 1] * zeroI + p1)
        zfirstI = 0.5 * (perR[0, 1] * zeroI + perI[0, 1] * zeroR + p2)
        zsecondR = 0.5 * (perR[1, 1] * zeroR - perI[1, 1] * zeroI + p3)
        zsecondI = 0.5 * (perR[1, 1] * zeroI + perI[1, 1] * zeroR + p4)
        
        z = jnp_array([zfirstR + 1j * zfirstI, zsecondR + 1j * zsecondI])
        
        # Compute theta function
        f = hyp_theta_fourier(z, riemannM, char, jnp_array([]), minMax)
        af = jnp_abs(f)
        
        # Compute derivatives for Newton step
        a = 0.5 * (
            perR[0, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, char, minMax)
            - perI[0, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, char, minMax)
            + perR[1, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            - perI[1, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
        )
        
        c = 0.5 * (
            perR[0, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, char, minMax)
            + perI[0, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, char, minMax)
            + perR[1, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            + perI[1, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
        )
        
        b = -c
        d = a
        det = a * d - b * c
        
        f_real = jnp_real(f)
        f_imag = jnp_im(f)
        
        # Newton update
        zeroR_new = zeroR - (d * f_real - b * f_imag) / det
        zeroI_new = zeroI - (-c * f_real + a * f_imag) / det
        
        # Update counter
        count += 1
        
        # Check convergence
        converged = af <= eps
        
        return (zeroR_new, zeroI_new, count), (zeroR_new + 1j * zeroI_new, f, af, converged)
    
    # Run fixed number of Newton iterations
    carry = (zeroR, zeroI, 0)
    _, (all_zeros, all_fs, all_afs, all_converged) = lax.scan(
        newton_step, carry, xs=jnp_arange(30)
    )
    
    # Find first converged iteration
    converged_mask = all_converged
    converged_idx = argmax(converged_mask)
    
    # Check if any iteration converged
    any_converged = jnp_any(converged_mask)
    
    def success_case():
        # Use the converged values
        final_zero = all_zeros[converged_idx]
        final_zeroR = jnp_real(final_zero)
        final_zeroI = jnp_im(final_zero)
        
        # Final z for converged iteration
        zfirstR_final = 0.5 * (perR[0, 1] * final_zeroR - perI[0, 1] * final_zeroI + p1)
        zfirstI_final = 0.5 * (perR[0, 1] * final_zeroI + perI[0, 1] * final_zeroR + p2)
        zsecondR_final = 0.5 * (perR[1, 1] * final_zeroR - perI[1, 1] * final_zeroI + p3)
        zsecondI_final = 0.5 * (perR[1, 1] * final_zeroI + perI[1, 1] * final_zeroR + p4)
        
        z_final = jnp_array([zfirstR_final + 1j * zfirstI_final, 
                            zsecondR_final + 1j * zsecondI_final])
        
        # Compute solution
        s1 = hyp_theta_fourier(z_final, riemannM, char, jnp_array([1]), minMax)
        s2 = hyp_theta_fourier(z_final, riemannM, char, jnp_array([2]), minMax)
        
        # Compute - (s1 * per[0,:] + s2 * per[1,:]) / (s1 * per[0,:] + s2 * per[1,:])
        numerator = s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0]
        denominator = s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1]
        sol = -numerator / denominator
        
        return sol, final_zero, jnp_array([affineParameter, final_zero])
    
    def failure_case():
        # Return all iterations and last function value
        return all_zeros, all_fs[-1], jnp_array([], dtype=jnp_complex128)
    
    # Conditional return based on convergence
    result = lax.cond(
        any_converged,
        success_case,
        failure_case
    )
    
    if not any_converged:
        # Failure case
        zeros_list = [complex(z) for z in all_zeros]
        print("In solution: Iteration process stopped after 30 iterations.")
        print("hyp_theta(1/2 * omega1inv * (phi, initNewton)^t) = ", complex(all_fs[-1]))
        return [zeros_list, complex(all_fs[-1])]
    else:
        # Success case
        sol, final_zero, pair = result
        return [complex(sol), complex(final_zero), 
                [complex(final_zero), complex(affineParameter)]]

@jit
def solution_second(affineParameter: complex, initNewton: complex, eps: float, minMax: int):
    """
    Computes the unphysical component of the theta divisor.
    """
    # Convert to JAX arrays
    affineParameter = jnp_complex128(affineParameter)
    initNewton = jnp_complex128(initNewton)
    eps = jnp_float64(eps)
    
    # Access global variables
    global periods_inverse, riemannM
    
    # Convert periods_inverse to JAX array if needed
    if not isinstance(periods_inverse, jnp_ndarray):
        periods_inverse = jnp_array(periods_inverse, dtype=jnp_complex128)
    
    # Characteristic vector
    char = jnp_array([[0.5, 0.5], [0.0, 0.5]], dtype=jnp_complex128)
    
    # Initialize
    zero = initNewton
    aff = affineParameter
    
    # Extract real and imaginary parts
    zeroR = jnp_real(zero)
    zeroI = jnp_im(zero)
    affR = jnp_real(aff)
    affI = jnp_im(aff)
    
    # Extract period matrix components
    perR = jnp_real(periods_inverse)
    perI = jnp_im(periods_inverse)
    
    # Precompute p values (different indices from solution_first)
    p1 = perR[0, 1] * affR - perI[0, 1] * affI
    p2 = perR[0, 1] * affI + perI[0, 1] * affR
    p3 = perR[1, 1] * affR - perI[1, 1] * affI
    p4 = perR[1, 1] * affI + perI[1, 1] * affR
    
    # Define function for one Newton iteration
    def newton_step(carry, _):
        zeroR, zeroI, count = carry
        
        # Compute z (different indices from solution_first)
        zfirstR = 0.5 * (perR[0, 0] * zeroR - perI[0, 0] * zeroI + p1)
        zfirstI = 0.5 * (perR[0, 0] * zeroI + perI[0, 0] * zeroR + p2)
        zsecondR = 0.5 * (perR[1, 0] * zeroR - perI[1, 0] * zeroI + p3)
        zsecondI = 0.5 * (perR[1, 0] * zeroI + perI[1, 0] * zeroR + p4)
        
        z = jnp_array([zfirstR + 1j * zfirstI, zsecondR + 1j * zsecondI])
        
        # Compute theta function
        f = hyp_theta_fourier(z, riemannM, char, jnp_array([]), minMax)
        af = jnp_abs(f)
        
        # Compute derivatives for Newton step (different indices from solution_first)
        a = 0.5 * (
            perR[0, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, char, minMax)
            - perI[0, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, char, minMax)
            + perR[1, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            - perI[1, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
        )
        
        c = 0.5 * (
            perR[0, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, char, minMax)
            + perI[0, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 0, riemannM, char, minMax)
            + perR[1, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            + perI[1, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
        )
        
        b = -c
        d = a
        det = a * d - b * c
        
        f_real = jnp_real(f)
        f_imag = jnp_im(f)
        
        # Newton update
        zeroR_new = zeroR - (d * f_real - b * f_imag) / det
        zeroI_new = zeroI - (-c * f_real + a * f_imag) / det
        
        # Update counter
        count += 1
        
        # Check convergence
        converged = af <= eps
        
        return (zeroR_new, zeroI_new, count), (zeroR_new + 1j * zeroI_new, f, af, converged)
    
    # Run fixed number of Newton iterations
    carry = (zeroR, zeroI, 0)
    _, (all_zeros, all_fs, all_afs, all_converged) = lax.scan(
        newton_step, carry, xs=jnp_arange(30)
    )
    
    # Find first converged iteration
    converged_mask = all_converged
    converged_idx = argmax(converged_mask)
    
    # Check if any iteration converged
    any_converged = jnp_any(converged_mask)
    
    def success_case():
        # Use the converged values
        final_zero = all_zeros[converged_idx]
        final_zeroR = jnp_real(final_zero)
        final_zeroI = jnp_im(final_zero)
        
        # Final z for converged iteration
        zfirstR_final = 0.5 * (perR[0, 0] * final_zeroR - perI[0, 0] * final_zeroI + p1)
        zfirstI_final = 0.5 * (perR[0, 0] * final_zeroI + perI[0, 0] * final_zeroR + p2)
        zsecondR_final = 0.5 * (perR[1, 0] * final_zeroR - perI[1, 0] * final_zeroI + p3)
        zsecondI_final = 0.5 * (perR[1, 0] * final_zeroI + perI[1, 0] * final_zeroR + p4)
        
        z_final = jnp_array([zfirstR_final + 1j * zfirstI_final, 
                            zsecondR_final + 1j * zsecondI_final])
        
        # Compute solution
        s1 = hyp_theta_fourier(z_final, riemannM, char, jnp_array([1]), minMax)
        s2 = hyp_theta_fourier(z_final, riemannM, char, jnp_array([2]), minMax)
        
        # Compute - (s1 * per[0,:] + s2 * per[1,:]) / (s1 * per[0,:] + s2 * per[1,:])
        numerator = s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0]
        denominator = s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1]
        sol = -numerator / denominator
        
        # Different return order from solution_first: [x[-1], affineParameter]
        return sol, final_zero, jnp_array([final_zero, affineParameter])
    
    def failure_case():
        # Return all iterations and last function value
        return all_zeros, all_fs[-1], jnp_array([], dtype=jnp_complex128)
    
    # Conditional return based on convergence
    result = lax.cond(
        any_converged,
        success_case,
        failure_case
    )
    
    # Convert to Python types for compatibility
    if not any_converged:
        # Failure case
        zeros_list = [complex(z) for z in all_zeros]
        print("In solution: Iteration process stopped after 30 iterations.")
        print("hyp_theta(1/2 * omega1inv * (phi, initNewton)^t) = ", complex(all_fs[-1]))
        return [zeros_list, complex(all_fs[-1])]
    else:
        # Success case
        sol, final_zero, pair = result
        return [complex(sol), complex(final_zero), 
                [complex(final_zero), complex(affineParameter)]]
