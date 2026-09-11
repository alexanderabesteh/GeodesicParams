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

import numpy as _np
from jax import jit, lax, vmap
from jax.numpy import (
    abs as jnp_abs,
    any as jnp_any,
    arange as jnp_arange,
    argmax,
    array as jnp_array,
    bool_ as jnp_bool,
    complex128 as jnp_complex128,
    dot,
    exp as jnp_exp,
    float64 as jnp_float64,
    imag as jnp_im,
    log as jnp_log,
    logical_and as jnp_logical_and,
    max as jnp_max,
    ndarray as jnp_ndarray,
    pi as jnp_pi,
    real as jnp_real,
    round as jnp_round,
    sort as jnp_sort,
    sqrt as jnp_sqrt,
    stack,
    sum as jnp_sum,
    where as jnp_where,
    zeros as jnp_zeros,
    zeros_like as jnp_zeros_like,
)
from mpmath import exp, fabs, im, matrix, pi, re, sign
from numpy import array, load as npload, ndarray as np_ndarray, save, zeros as np_zeros
from sympy import (
    Poly,
    Symbol,
    apart,
    asin as spasin,
    collect,
    degree,
    lambdify,
    oo,
    sin as spsin,
    solve,
    sqrt as spsqrt,
    sympify,
)
from torch import sqrt as t_sqrt
from torchquad import GaussLegendre, set_up_backend

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
from ..riemannsurfaces.riemann_funcs.hyperelp_funcs import (
    hyp_theta_fourier,
    hyp_theta_IR,
    hyp_theta_RR,
)
from ..utilities import inlist, separate_zeros

# Global period matrices
periods_inverse = 0
riemannM = 0
set_up_backend("torch", data_type = "float64", torch_enable_cuda = True)


def _torch_to_complex(t):
    """
    Convert a torchquad integration result (a torch.Tensor) into a plain
    Python complex, so it interoperates with the surrounding mpmath
    arithmetic (mpmath.matrix construction, mpmath scalar ops) used
    alongside it in this module.
    """

    return complex(t.detach().cpu().item())

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
        res_func = lambda s : gl.integrate(integrand_subs, dim=1, N=101, integration_domain=[[inits[0], s]], backend = "torch")
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

        f_parfrac = apart(integrand.evalf(digits), full=True).doit()
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

        # Precompute the purely symbolic setup eagerly -- this doesn't touch
        # disk and doesn't depend on orbitdata existing yet.
        poly_first_coeffs = []
        poly_second_coeffs = []

        for i in range(len(poly_fracs)):
            deg = degree(poly_fracs[i].as_poly(p_sym), p_sym)
            if deg > 1:
                raise ValueError(
                    "Equations of motion of hyperelliptic type and second kind are not supported"
                )
            elif deg == 0:
                poly_first_coeffs.append(complex(poly_fracs[i]))
            elif deg == 1:
                poly_second_coeffs.append(complex(poly_fracs[i].coeff(p_sym, 1)))

        poly_first_jax = jnp_array(poly_first_coeffs, dtype=jnp_complex128) if poly_first_coeffs else jnp_array([], dtype=jnp_complex128)
        poly_second_jax = jnp_array(poly_second_coeffs, dtype=jnp_complex128) if poly_second_coeffs else jnp_array([], dtype=jnp_complex128)

        has_rat_fracs = len(rat_fracs_final) > 0
        if has_rat_fracs:
            periodMatrix_loaded, invert_data, eps = npload(f"{datafile}.npy", allow_pickle=True)
            minMax_loaded = invert_data[2]
            eta, r1, r2 = compute_secondkind_periods(
                zeros, eps, periodMatrix_loaded, datafile, digits, minMax_loaded
            )
            rat_coeff_values = [complex(rf.as_numer_denom()[0]) for rf in rat_fracs_final]
            rat_coeffs_jax = jnp_array(rat_coeff_values, dtype=jnp_complex128)
            rat_integration_funcs = [
                integrate_hyperelliptic_third(zeros, r1, r2, eta, i, datafile, digits)
                for i in rat_fracs_final
            ]

        def res_hyp(s):
            # integrate_hyperelliptic_first reads back the per-mino orbitdata
            # cache (<datafile>_orbitdata.npy), which is only populated once
            # sol_r(mino) has actually been called with this same <s> --
            # so, unlike the purely symbolic setup above, these calls must
            # stay inside this lazily-evaluated closure rather than being
            # hoisted out to run at integrate_eom's own call time (when that
            # cache doesn't exist yet). Each call returns a plain list of
            # already-computed values, one per orbitdata row -- not a
            # function to call again with <s>.
            poly_first_int = [
                integrate_hyperelliptic_first(1, datafile) for _ in poly_first_coeffs
            ]
            poly_second_int = [
                integrate_hyperelliptic_first(2, datafile) for _ in poly_second_coeffs
            ]

            s_array = jnp_array(s, dtype=jnp_complex128)

            # Polynomial first-kind contributions
            if len(poly_first_int) > 0:
                first_integrals = stack([jnp_array(lst, dtype=jnp_complex128) for lst in poly_first_int], axis=0)
                first_contrib = jnp_sum(poly_first_jax[:, None] * first_integrals, axis=0)
            else:
                first_contrib = jnp_zeros_like(s_array, dtype=jnp_complex128)

            # Polynomial second-kind contributions
            if len(poly_second_int) > 0:
                second_integrals = stack([jnp_array(lst, dtype=jnp_complex128) for lst in poly_second_int], axis=0)
                second_contrib = jnp_sum(poly_second_jax[:, None] * second_integrals, axis=0)
            else:
                second_contrib = jnp_zeros_like(s_array, dtype=jnp_complex128)

            # Rational fraction contributions
            if has_rat_fracs:
                integrations = stack([jnp_array(f(s), dtype=jnp_complex128) for f in rat_integration_funcs], axis=0)
                rat_res = jnp_sum(rat_coeffs_jax[:, None] * integrations, axis=0)
            else:
                rat_res = jnp_zeros_like(s_array, dtype=jnp_complex128)

            final_results = first_contrib + second_contrib + rat_res

            if len(s_array) == 1:
                return complex(final_results[0])
            else:
                return [complex(x) for x in final_results]

        return lambda s: res_hyp(s)
    elif deg_p == 3:
        periods, g2, g3, int_init, inits = npload(f"{datafile}.npy", allow_pickle=True)

        # Precompute all SymPy operations outside the solution function
        elp_poly_first_coeffs = []
        elp_poly_second_coeffs = []

        for i in range(len(poly_fracs)):
            deg = degree(poly_fracs[i].as_poly(p_sym), p_sym)
            if deg == 0:
                elp_poly_first_coeffs.append(complex(poly_fracs[i]))
            elif deg == 1:
                elp_poly_second_coeffs.append(complex(poly_fracs[i].coeff(p_sym, 1)))

        elp_poly_first_jax = jnp_array(elp_poly_first_coeffs, dtype=jnp_complex128) if elp_poly_first_coeffs else jnp_array([], dtype=jnp_complex128)
        elp_poly_second_jax = jnp_array(elp_poly_second_coeffs, dtype=jnp_complex128) if elp_poly_second_coeffs else jnp_array([], dtype=jnp_complex128)

        periods_array = jnp_array(periods, dtype=jnp_complex128)
        inits_array = jnp_array(inits, dtype=jnp_complex128)
        omega1 = periods_array[0]
        omega3 = periods_array[1]
        s0 = inits_array[0]

        elp_has_rat_fracs = len(rat_fracs_final) > 0
        if elp_has_rat_fracs:
            elp_integration_funcs = [integrate_elliptic_third(i, datafile) for i in rat_fracs_final]

        def res_elp(s):
            s_array = jnp_array(s, dtype=jnp_complex128)

            if elp_has_rat_fracs:
                integrations_stack = stack([jnp_array(func(s), dtype=jnp_complex128) for func in elp_integration_funcs], axis=0)
                rat_res = jnp_sum(integrations_stack, axis=0)
            else:
                rat_res = jnp_zeros_like(s_array, dtype=jnp_complex128)

            if len(elp_poly_first_jax) > 0:
                bounds = s_array - s0
                first_contrib = jnp_sum(elp_poly_first_jax) * bounds
            else:
                first_contrib = jnp_zeros_like(s_array, dtype=jnp_complex128)

            if len(elp_poly_second_jax) > 0:
                zeta_args = s_array - s0
                zeta_vals = vmap(lambda x: weierstrass_zeta(x, omega1, omega3))(zeta_args)
                second_contrib = jnp_sum(elp_poly_second_jax) * zeta_vals
            else:
                second_contrib = jnp_zeros_like(s_array, dtype=jnp_complex128)

            final_results = first_contrib + second_contrib + rat_res

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
        wp_val = weierstrass_P(
            initial_values[0] - int_initial, periodMatrix[0], periodMatrix[1], 1
        )
        if sign(complex(wp_val).real) != int_sign:
            int_initial *= -1

    # Save to datafile
    if datafile != None:
        integrate_initial = int_initial
        initials = initial_values
        data = _np.array([periodMatrix, g2, g3, integrate_initial, initials], dtype=object)
        save(datafile, data)
        print(f"In invert_elliptic_first: periodMatrix saved to, {datafile}.npy")

    @jit
    def vectorized_computation(s_array):
        s_jax = jnp_array(s_array, dtype=jnp_complex128).ravel()

        args = jnp_sqrt(constant) * s_jax - initial_values[0] - int_initial
        weierstrass_results = weierstrass_P(args, periodMatrix[0], periodMatrix[1])
        real_results = jnp_real(weierstrass_results)

        return vmap(substitution)(real_results)

    def sol_func(s):
        s_arr = jnp_array(s, dtype=jnp_float64)
        is_scalar = s_arr.ndim == 0
        result = vectorized_computation(s_arr)
        if is_scalar:
            return float(result[0])
        return result

    return sol_func


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
                zeros, initial_values[1], realNS[k], digits, periodMatrix
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
                zeros, initial_values[1], realNS[k], digits, periodMatrix
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
        # invert_data is a ragged structure (two pairs plus a bare int <max>),
        # not a rectangular array -- store it as a plain nested list (like
        # periodMatrix.tolist() below) rather than forcing it into a
        # jnp_array, which requires a homogeneous shape.
        save(datafile, _np.array([periodMatrix.tolist(), invert_data, eps], dtype=object))
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
    extended_orbitdata = npload(datafile + "_orbitdata.npy", allow_pickle=True)
    invert_data = npload(datafile + ".npy", allow_pickle=True)

    # Extract the component value from invert_data
    component_val = invert_data[1][1][component - 1]

    # Each row of extended_orbitdata is [subs_coord, coordinates, x, divisor,
    # affine_list] -- 4 scalars and one pair (divisor). Only divisor is used
    # here, so pull just that field into its own rectangular array rather
    # than trying to convert the whole heterogeneous (scalar, scalar,
    # scalar, pair, scalar) row structure with one jnp_array(...) call,
    # which fails since the row itself isn't uniformly-shaped.
    divisor_jax = jnp_array(
        [[complex(v) for v in row[3]] for row in extended_orbitdata],
        dtype=jnp_complex128,
    )
    component_val_jax = jnp_complex128(component_val)

    # JIT-compiled computation part
    @jit
    def compute_result(divisor, comp_val):
        divisor_comps = divisor[:, component - 1]
        result = divisor_comps + comp_val
        return result

    # Compute result
    result_jax = compute_result(divisor_jax, component_val_jax)

    # Convert back to Python list if needed
    return [complex(x) for x in result_jax]


def _reduce_theta_arg(z, riemannM):
    """
    Reduce a genus-2 theta-function argument z (a 2-element complex array)
    to z - shift, where shift = p + riemannM @ q for integer 2-vectors p, q
    chosen so the result has small real and imaginary parts, and return
    that <shift>.

    Why this is needed: the theta Fourier series (hyp_theta_fourier/RR/IR)
    sums terms that grow like exp(+-2*pi*Im(z_i)), so for |Im(z)| in the
    thousands (which is the norm, not a corner case, once the period
    matrix has entries as large as they do for a near-degenerate root
    configuration -- see project_genus2_ima2per2_bug.md) the series
    overflows/underflows to inf/nan before it can be summed. Reducing z by
    a lattice vector first keeps every term representable.

    Why it's valid for root-finding (Newton's method): the exact
    quasi-periodicity identity for theta[g,h] is
        theta[g,h](z + p + tau@q; tau)
            = exp(2*pi*i*g.p) * exp(-pi*i*q.(tau@q) - 2*pi*i*q.(z+h))
              * theta[g,h](z; tau)
    for integer p, q -- derivable directly from the Fourier-series
    definition by re-indexing the summation n -> n - q. The prefactor is
    never zero, so theta(z)=0 iff theta(z-shift)=0: the two have exactly
    the same roots. And since <shift> here is held fixed for an entire
    Newton run (computed once, from the run's starting z, not
    recomputed per-iteration), d/dx theta(z(x) - shift) = theta'(z(x) -
    shift) * dz/dx -- the same derivative Newton's method would use on the
    unreduced z, just evaluated at the shifted point. So running the whole
    iteration on z - shift converges to exactly the same root as it would
    (if it could) on z itself, with no exponential-prefactor bookkeeping
    needed anywhere in the Newton loop.

    Parameters
    ----------
    z : jnp.ndarray
        A 2-element complex array, the unreduced theta argument.
    riemannM : jnp.ndarray
        The 2x2 complex Riemann matrix tau (Im(tau) positive definite).

    Returns
    -------
    jnp.ndarray
        A 2-element complex array: the shift to subtract from z (and from
        every z encountered later in the same Newton run) before evaluating
        the theta Fourier series or its derivatives.

    """

    imTau = jnp_im(riemannM)
    imZ = jnp_im(z)

    det = imTau[0, 0] * imTau[1, 1] - imTau[0, 1] * imTau[1, 0]
    q0 = (imTau[1, 1] * imZ[0] - imTau[0, 1] * imZ[1]) / det
    q1 = (-imTau[1, 0] * imZ[0] + imTau[0, 0] * imZ[1]) / det
    q = jnp_round(jnp_array([q0, q1])).astype(jnp_complex128)

    tau_q = dot(riemannM, q)
    p = jnp_round(jnp_real(z - tau_q)).astype(jnp_complex128)

    return p + tau_q


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
    # hyp_theta_fourier is JAX-jitted and cannot trace an mpmath matrix;
    # convert once here (matching the same guard used elsewhere in this
    # module, e.g. solution_first/solution_second). mpmath matrices don't
    # iterate into nested rows the way jnp_array expects, so build the
    # nested list explicitly rather than passing the matrix straight through
    # (which silently flattens it to 1-D).
    if not isinstance(riemannM, jnp_ndarray):
        riemannM = jnp_array(
            [[complex(riemannM[i, j]) for j in range(2)] for i in range(2)],
            dtype=jnp_complex128,
        )

    # The manual Fourier summation below (the fallback used to determine the
    # bound <max>) is plain Python/mpmath arithmetic, so it needs riemannM's
    # entries as plain complex scalars rather than the JAX array passed to
    # hyp_theta_fourier.
    riemannM_py = [[complex(riemannM[i, j]) for j in range(2)] for i in range(2)]

    g = [1 / 2, 1 / 2]
    h = [0, 1 / 2]
    char = [g, h]

    if physical_comp == 1:
        z_mat = (
            1
            / 2
            * periods_inverse
            * matrix([initial_values[0] - modified_init, initNewton])
        )
    else:
        z_mat = (
            1
            / 2
            * periods_inverse
            * matrix([initNewton, initial_values[0] - modified_init])
        )

    # hyp_theta_fourier is JAX-jitted and expects z as a plain list of two
    # complex numbers (per its docstring), not an mpmath matrix.
    z = [complex(z_mat[0]), complex(z_mat[1])]

    # Reduce z into the fundamental domain before summing the Fourier
    # series -- see _reduce_theta_arg's docstring for why this is both
    # necessary (unreduced z routinely has |Im(z)| in the thousands for a
    # near-degenerate root configuration, overflowing the series) and valid
    # (the reduced and unreduced z are theta-quasi-periodic, so a small |f|
    # at one genuinely means the same near a root at the other -- this
    # `max`-search doesn't even need the exponential quasi-periodicity
    # prefactor, since it's only ever comparing |f| to a threshold).
    imTau = [[riemannM_py[i][j].imag for j in range(2)] for i in range(2)]
    imZ = [z[0].imag, z[1].imag]
    det = imTau[0][0] * imTau[1][1] - imTau[0][1] * imTau[1][0]
    q0 = round((imTau[1][1] * imZ[0] - imTau[0][1] * imZ[1]) / det)
    q1 = round((-imTau[1][0] * imZ[0] + imTau[0][0] * imZ[1]) / det)
    tau_q0 = riemannM_py[0][0] * q0 + riemannM_py[0][1] * q1
    tau_q1 = riemannM_py[1][0] * q0 + riemannM_py[1][1] * q1
    p0 = round((z[0] - tau_q0).real)
    p1 = round((z[1] - tau_q1).real)
    z = [z[0] - p0 - tau_q0, z[1] - p1 - tau_q1]

    max = 5
    f = complex(hyp_theta_fourier(z, riemannM, char, minMax=max))

    # Determine summation bound
    while (fabs(re(f)) > eps / 10 or fabs(im(f)) > eps / 10) and max < 30:
        # Evaluate theta function
        for m1 in range(-max - 1, max + 2):
            m = [m1, -max - 1]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM_py[i][j] * (m[j] + g[j])
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            f += exp(1j * pi * char_sum)

            m = [m1, max + 1]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM_py[i][j] * (m[j] + g[j])
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            f += exp(1j * pi * char_sum)

        for m2 in range(-max, max + 1):
            m = [-max - 1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM_py[i][j] * (m[j] + g[j])
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])
            f += exp(1j * pi * char_sum)

            m = [max + 1, m2]
            char_sum = 0

            for i in range(2):
                tau_sum = 0
                for j in range(2):
                    tau_sum += riemannM_py[i][j] * (m[j] + g[j])
                char_sum += (m[i] + g[i]) * (tau_sum + 2 * z[i] + 2 * h[i])

            f += exp(1j * pi * char_sum)
        # print("f = ", f)
        max += 1

    # Evaluate the derivatives of the sigma function
    s1 = complex(hyp_theta_fourier(z, riemannM, char, (1,), max))
    s2 = complex(hyp_theta_fourier(z, riemannM, char, (2,), max))
    u = -(
        (s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0])
        / (s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1])
    )

    # Check if u(initial_values[0]) is within the correct accuracy of initial_values[1] for the initial Newton method value
    if (initial_values[1] == oo and fabs(u) < 1 / eps * (10 ** (-4))) or (
        initial_values[1] < oo and fabs(u - initial_values[1]) > eps * 10**4
    ):
        max += 1
        s1 = complex(hyp_theta_fourier(z, riemannM, char, (1, 0), max))
        s2 = complex(hyp_theta_fourier(z, riemannM, char, (0, 1), max))
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

def _solve_step_with_refinement(prev_affine, target_affine, modified_init, x_prev, eps, minMax, physical_comp, success_tol, max_substeps=64):
    """
    Advance Newton's continuation on the theta divisor from (prev_affine,
    x_prev) to target_affine, subdividing into successively finer sub-steps
    whenever a direct jump doesn't converge to a valid (near-real) root.

    Why this is needed: a single-shot Newton call from orbitdata can fail
    simply because the mino step is too large relative to how fast the
    physical coordinate is moving right there -- e.g. near a turning point,
    or for a component (like nu/theta) whose natural period is much
    shorter than the mino sampling interval used for the other component.
    This is the standard remedy in any predictor-corrector continuation
    method: retry with a shorter step (a better initial guess for Newton),
    not give up outright. It does not touch the convergence/correctness
    criterion itself (`jnp_abs(jnp_im(coord[0])) < success_tol`) -- a step that still
    fails at the finest refinement tried is reported as failed, exactly as
    before.

    Parameters
    ----------
    prev_affine, target_affine : complex
        The raw (pre-modified_init) affine parameter values at the last
        converged point and the point being advanced to.
    modified_init : complex
        Passed through to `solution` unchanged (see `orbitdata`).
    x_prev : complex
        The Newton seed at prev_affine (i.e. the previously converged
        theta-divisor coordinate).
    eps, minMax, physical_comp
        Passed through to `solution` unchanged.
    success_tol : float
        The acceptance tolerance on Im(coord[0]) (see `orbitdata`'s
        docstring note on why this is not simply `100 * eps`).
    max_substeps : int
        The largest number of equal sub-steps to try before giving up.
        Doubled at each refinement level (1, 2, 4, 8, ...).

    Returns
    -------
    list
        Whatever `solution(...)` returned for the finest attempt made: a
        3-element list on success, or its failure-shape return otherwise.

    """

    n = 1
    coord = None
    while n <= max_substeps:
        step = (target_affine - prev_affine) / n
        x_seed = x_prev
        ok = True

        for k in range(1, n + 1):
            affine_k = prev_affine + step * k
            coord = solution(affine_k - modified_init, x_seed, eps, minMax, physical_comp)

            # NOTE: this must be jnp_abs(Im(coord[0])), not the signed value
            # -- a genuine, longstanding bug (present in the pre-JAX
            # codebase too, `im(coord[0]) < 100 * eps`, so it's not a JAX
            # port regression) that accepted ANY negative imaginary residual
            # regardless of magnitude, since a negative number is always
            # "< success_tol" for a positive tolerance. Confirmed directly:
            # for this session's cosmo!=0 r-motion, Im(coord[0]) swings from
            # ~0 down to -0.0136 and back before finally crossing to
            # positive -- every one of those large-magnitude-but-negative
            # steps was being silently accepted as "converged", producing
            # wildly wrong physical r values (observed: -20 -> -267 -> +206
            # before settling), and the walk only "failed" once the residual
            # happened to cross zero into positive territory.
            if len(coord) == 3 and jnp_abs(jnp_im(coord[0])) < success_tol:
                x_seed = coord[1]
            else:
                ok = False
                break

        if ok:
            return coord

        n *= 2

    return coord


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
    Computes orbit data using Newton iteration on the theta divisor.
    """
    global periods_inverse, riemannM

    # Convert to JAX types
    initNewton = jnp_complex128(initNewton)
    modified_init = jnp_complex128(modified_init)
    eps = jnp_float64(eps)
    affine_array = jnp_array(affine_list, dtype=jnp_complex128)

    init_coord = jnp_complex128(initial_values[1])
    init_gamma = jnp_complex128(initial_values[0])

    # The acceptance tolerance for a Newton solution used to be a flat
    # 100 * eps, where <eps> is inherited from the period matrix's
    # Legendre-relation residual -- fine when that residual was large
    # (the ~342 defect from the precision bug this session started with),
    # but once the period matrix is accurate (eps ~1e-12), 100 * eps becomes
    # ~1e-10 for the *derived* physical coordinate (r or nu), which is
    # amplified by division in `solution`'s numerator/denominator and
    # routinely lands its imaginary residual around 1e-6 to 1e-5 even for a
    # thoroughly-converged Newton root -- not because Newton failed, but
    # because that bound was never an achievable target for this quantity.
    # Floor it at a fixed, physically-reasonable tolerance (1e-4, still far
    # tighter than anything visible in a plotted orbit) so an accurate
    # period matrix doesn't make convergence acceptance *stricter* than a
    # sloppy one did.
    success_tol = jnp_where(eps * 100 > 1e-4, eps * 100, 1e-4)

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
    stop = False
    for i in range(max_len - pos0 - 1):
        if stop:
            break
        current_idx = pos0 + i + 1
        if current_idx >= max_len:
            break

        # Get last valid x
        prev_idx = int(jnp_max(jnp_arange(max_len) * valid_mask))
        x_prev = x_arr[prev_idx]

        coord = _solve_step_with_refinement(
            affine_array[prev_idx], affine_array[current_idx], modified_init,
            x_prev, eps, minMax, physical_comp, success_tol,
        )

        is_success = len(coord) == 3 and jnp_abs(jnp_im(coord[0])) < success_tol

        if is_success:
            x_arr = x_arr.at[current_idx].set(coord[1])
            coords_arr = coords_arr.at[current_idx].set(jnp_real(coord[0]))
            subs_arr = subs_arr.at[current_idx].set(substitution(jnp_real(coord[0])))
            valid_mask = valid_mask.at[current_idx].set(True)
        else:
            stop = True

    # Process backward
    stop = False
    for i in range(pos0):
        if stop:
            break
        current_idx = pos0 - i - 1
        if current_idx < 0:
            break

        first_idx = int(argmax(valid_mask))
        x_prev = x_arr[first_idx]

        coord = _solve_step_with_refinement(
            affine_array[first_idx], affine_array[current_idx], modified_init,
            x_prev, eps, minMax, physical_comp, success_tol,
        )

        is_success = len(coord) == 3 and jnp_abs(jnp_im(coord[0])) < success_tol

        if is_success:
            x_arr = x_arr.at[current_idx].set(coord[1])
            coords_arr = coords_arr.at[current_idx].set(jnp_real(coord[0]))
            subs_arr = subs_arr.at[current_idx].set(substitution(jnp_real(coord[0])))
            valid_mask = valid_mask.at[current_idx].set(True)
        else:
            stop = True

    # Extract results
    valid_indices = jnp_arange(max_len)[valid_mask]
    valid_indices_sorted = jnp_sort(valid_indices)
    subs_coord_result = [complex(subs_arr[i]) for i in valid_indices_sorted]

    # Handle datafile saving
    if datafile is not None:
        subs_coord_np = array([complex(x) for x in subs_arr[valid_mask]])
        coords_np = array([complex(x) for x in coords_arr[valid_mask]])
        x_np = array([complex(x) for x in x_arr[valid_mask]])
        affine_np = array([complex(x) for x in affine_array[valid_mask]])

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
    sig_minus = weierstrass_sigma(x - y, omega1, omega3)
    sig_plus = weierstrass_sigma(x + y, omega1, omega3)
    value = complex(jnp_log(sig_minus / sig_plus))

    if value.imag != 0:
        gl = GaussLegendre()
        value = gl.integrate(lambda s: weierstrass_zeta(s - y, omega1, omega3) - weierstrass_zeta(s + y, omega1, omega3),
                                  dim=1, N=101, integration_domain=[[0, complex(x).real]], backend = "jax")
        value = complex(value)

    return value

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

    # Keep mpmath types for mpmath-based calls
    y_mp = complex(y) if not isinstance(y, (int, float, complex)) else y
    o1_mp = complex(omega1) if not isinstance(omega1, (int, float, complex)) else omega1
    o3_mp = complex(omega3) if not isinstance(omega3, (int, float, complex)) else omega3

    # Convert to JAX for array computations
    y_jax = jnp_complex128(y_mp)
    o1_jax = jnp_complex128(o1_mp)
    o3_jax = jnp_complex128(o3_mp)

    # Compute eta (uses mpmath internally)
    eta_mp = periods_secondkind(o1_mp, o3_mp)[0]
    eta = jnp_complex128(complex(eta_mp))

    # Branch factor (uses mpmath internally via sigma_ln_numerical)
    c = jnp_complex128(complex(
        sigma_ln_numerical(o1_mp, y_mp, o1_mp, o3_mp) - sigma_ln_numerical(
            1e-10, y_mp, o1_mp, o3_mp
        )
    ))

    # Compute branch (using JAX-compatible rounding)
    branch_component = -o1_jax / jnp_pi * (jnp_im(c) / o1_jax + jnp_im(2 * eta * y_jax / o1_jax))
    branch = jnp_round(branch_component)  # JAX equivalent of nint
    switch = branch * jnp_pi * 1j

    # Define function for single x value
    def compute_single(i):
        sigmatilde1 = weierstrass_sigma(i - y_jax, o1_jax, o3_jax) * jnp_exp(
            -eta * (i - y_jax) ** 2 / (2 * o1_jax)
        )
        sigmatilde2 = weierstrass_sigma(i + y_jax, o1_jax, o3_jax) * jnp_exp(
            -eta * (i + y_jax) ** 2 / (2 * o1_jax)
        )

        value = (
            jnp_log((sigmatilde1 / sigmatilde2) * jnp_exp(switch * (i / o1_jax - 1)))
            - switch * (i / o1_jax - 1)
            - 2 * eta * i * y_jax / o1_jax
        )
        return jnp_where(jnp_abs(value) < 1e-15, 0.0 + 0.0j, value)

    # Vectorize over all x values
    is_scalar = x_array.ndim == 0
    if is_scalar:
        x_array = x_array.reshape(1)
    result = vmap(compute_single)(x_array)
    if is_scalar:
        return result[0]
    return result
   
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
    coeff = complex(integrand.as_numer_denom()[0])

    # Locate the value v1 in the fundamental period parallelogram such that:
    # weierstrass_P(v1) = pole
    inv_integrand = 1 / integrand
    pole = solve(inv_integrand, Poly(inv_integrand).gen)
    v1 = inverse_weierstrass_P(pole[0], periods[0], periods[1])

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

    def result(s):
        """Wrapper that handles both single values and lists."""
        is_batch = isinstance(s, (list, tuple, np_ndarray, jnp_ndarray))
        s_array = jnp_array(s, dtype=jnp_complex128)
        val = jnp_array(compute_single(s_array))
        if is_batch:
            return [complex(x) for x in val.ravel()]
        else:
            return complex(val.ravel()[0]) if val.ndim > 0 else complex(val)
    
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
        _np.array([second_kind_periods.tolist(), eps], dtype=object),
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
        int_dz = myint_genus2(zeros, pole, realNS[-1], 1, digits)
        int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, realNS[-1], 1, digits)
        int_3 = 2 * myint_genus2_second(
            zeros, [0, 0, 0, 1], pole, realNS[-1], 1, digits
        )
        inf1 = eval_period(len(realNS) - 1, oo, realNS, zeros, periodMatrix, 0)
        inf2 = eval_period(len(realNS) - 1, oo, realNS, zeros, periodMatrix, 1)
    # in the remaining cases there is at least one real zero > pole!
    elif k == 3:
        if im(zeros[3]) == 0:
            int_dz = myint_genus2(zeros, pole, zeros[3], 1, digits)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[3], 1, digits)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[3], 1, digits
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
            int_dz = myint_genus2(zeros, pole, zeros[1], 1, digits)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[1], 1, digits)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[1], 1, digits
            )
            inf1 = eval_period(
                inlist(zeros[1], realNS), oo, realNS, zeros, periodMatrix, 0
            )
            inf2 = eval_period(
                inlist(zeros[1], realNS), oo, realNS, zeros, periodMatrix, 1
            )

        elif im(zeros[2]) == 0 and im(zeros[3]) == 0:
            int_dz = myint_genus2(zeros, pole, zeros[2], 1, digits)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[2], 1, digits)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[2], 1, digits
            )
            inf1 = eval_period(
                inlist(zeros[2], realNS), oo, realNS, zeros, periodMatrix, 0
            )
            inf2 = eval_period(
                inlist(zeros[2], realNS), oo, realNS, zeros, periodMatrix, 1
            )
        elif pole == re(zeros[1]):
            int_dz = myint_genus2(zeros, pole, realNS[0], 1, digits)
            int_2 = 2 * myint_genus2_second(
                zeros, [0, 0, 1], pole, realNS[0], 1, digits
            )
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, realNS[0], 1, digits
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
                    1j * _torch_to_complex(gl.integrate(lambda x: 1 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[1])]], backend = "torch")),
                    1j * _torch_to_complex(gl.integrate(lambda x: x / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[1])]], backend = "torch")),
                ]
            )
            int_2 = 2 * 1j * _torch_to_complex(gl.integrate(lambda x: x**2 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[1])]], backend = "torch"))
            int_3 = 2 * 1j * _torch_to_complex(gl.integrate(lambda x: x**3 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[1])]], backend = "torch"))
            int_dz += int_genus2_complex(
                zeros, re(zeros[1]), im(zeros[1]), 0, 1, digits
            )

            int_2 = int_2 + 2 * int_genus2_complex_second(
                zeros, [0, 0, 1], re(zeros[0]), fabs(im(zeros[0])), 0, 1, digits
            )

            int_3 += 2 * int_genus2_complex_second(
                zeros, [0, 0, 0, 1], re(zeros[0]), fabs(im(zeros[0])), 0, 1, digits
            )

            inf1 = periodMatrix[0, 2] - periodMatrix[0, 0]
            inf2 = periodMatrix[1, 2] - periodMatrix[1, 0]

    elif k == 1:
        if im(zeros[0]) == 0:
            int_dz = myint_genus2(zeros, pole, zeros[0], 1, digits)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[0], 1, digits)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[0], 1, digits
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
            int_dz = myint_genus2(zeros, pole, zeros[0], 1, digits)
            int_2 = 2 * myint_genus2_second(zeros, [0, 0, 1], pole, zeros[0], 1, digits)
            int_3 = 2 * myint_genus2_second(
                zeros, [0, 0, 0, 1], pole, zeros[0], 1, digits
            )
            inf1 = eval_period(0, oo, realNS, zeros, periodMatrix, 0)
            inf2 = eval_period(0, oo, realNS, zeros, periodMatrix, 1)
        elif im(zeros[0]) != 0 and im(zeros[1]) != 0:
            r = 0
            for i in range(1, 7):
                r += re(coeffsP[i - 1]) * x ** (i - 1)
            r = lambdify(x, r, "torch")
            gl = GaussLegendre()

            int_dz = (
                matrix(
                    [
                        1j * _torch_to_complex(gl.integrate(lambda x: 1 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[0])]], backend = "torch")),
                        1j * _torch_to_complex(gl.integrate(lambda x: x / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[0])]], backend = "torch")),
                    ]
                )
                + int_genus2_complex(zeros, re(zeros[0]), fabs(im(zeros[0])), 0, 1, digits)
            )
            int_2 = (
                2 * 1j * _torch_to_complex(gl.integrate(lambda x: x**2 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[0])]], backend = "torch"))
                + 2 * int_genus2_complex_second(zeros, [0, 0, 1], re(zeros[0]), fabs(im(zeros[0])), 0, 1, digits)
            )

            int_3 = (
                2 * 1j * _torch_to_complex(gl.integrate(lambda x: x**3 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, re(zeros[0])]], backend = "torch"))
                + 2 * int_genus2_complex_second(zeros, [0, 0, 0, 1], re(zeros[0]), fabs(im(zeros[0])), 0, 1, digits)
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
                        1j * _torch_to_complex(gl.integrate(lambda x: 1 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, zeros[0]]], backend = "torch")),
                        1j * _torch_to_complex(gl.integrate(lambda x: x / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, zeros[0]]], backend = "torch")),
                    ]
                )
                + myint_genus2(zeros, zeros[0], re(zeros[1]), 1, digits)
                + int_genus2_complex(zeros, re(zeros[1]), fabs(im(zeros[1])), 1, 1, digits)
            )

            int_2 = (
                2 * 1j * _torch_to_complex(gl.integrate(lambda x: x**2 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, zeros[0]]], backend = "torch"))
                + 2 * myint_genus2_second(zeros, [0, 0, 1], zeros[0], re(zeros[1]), 1, digits)
                + 2 * int_genus2_complex_second(zeros, [0, 0, 1], re(zeros[1]), fabs(im(zeros[1])), 1, 1, digits)
            )

            int_3 = (
                2 * 1j * _torch_to_complex(gl.integrate(lambda x: x**3 / t_sqrt(-r(x)), dim=1, N=101, integration_domain=[[pole, zeros[0]]], backend = "torch"))
                + 2 * myint_genus2_second(zeros, [0, 0, 0, 1], zeros[0], re(zeros[1]), 1, digits)
                + 2 * int_genus2_complex_second(zeros, [0, 0, 0, 1], re(zeros[1]), fabs(im(zeros[1])), 1, 1 , digits)
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
        extended_orbitdata = npload(datafile + "_orbitdata.npy", allow_pickle=True)
    
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
        # kappa is a 2x2 mpmath matrix; it doesn't iterate into nested rows
        # the way jnp_array expects, so build the nested list explicitly
        # rather than passing the matrix straight through (which silently
        # flattens it to 1-D).
        kappa_jax = jnp_array(
            [[complex(kappa[i, j]) for j in range(2)] for i in range(2)],
            dtype=jnp_complex128,
        )
        s0_jax = jnp_array(s0, dtype=jnp_complex128)
        int_dr1_jax = jnp_complex128(int_dr1)
        int_dr2_jax = jnp_complex128(int_dr2)
    
        def compute_results():
            xi_terms = divisor_jax + 2 * xi_inf_jax
            yi_terms = divisor_jax + 2 * yi_inf_jax
        
            # sigma() is written for a single 2-vector z (dot(periods_inverse,
            # z) with periods_inverse 2x2); xi_terms/yi_terms are a whole
            # (N, 2) batch of points at once, so vmap it over the batch
            # dimension rather than calling it directly on the batch.
            vars_vec = vmap(sigma)(xi_terms) / vmap(sigma)(yi_terms)
        
            # Vectorized log_vars computation
            diff_vec = xi_inf_jax - yi_inf_jax
            # Batched version of the same bilinear form v @ kappa @ w used
            # for the single row divisor_jax[pos0] in "const" below: apply
            # kappa to every row of divisor_jax via a real matrix multiply
            # (divisor_jax * kappa_jax was an elementwise product against a
            # (2, 2) array, not the intended row @ kappa @ diff_vec).
            log_vars_vec = 2 * jnp_sum((divisor_jax @ kappa_jax) * diff_vec, axis=1)
        
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
                results = results.at[current_idx].set(result_with_branch)
            
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
                results = results.at[current_idx].set(result_with_branch)
            
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
        
            # Final scaling (vectorized). pole/zeros are sympy objects
            # (zeros holds the polynomial's roots as sympy numbers), so this
            # product is a sympy expression -- convert it to a plain complex
            # number before handing it to jnp_real/jnp_sqrt, which (like
            # every other JAX-jitted function in this module) can't trace a
            # raw sympy object.
            product = complex(
                (pole - zeros[0]) * (pole - zeros[1]) * (pole - zeros[2]) *
                (pole - zeros[3]) * (pole - zeros[4])
            )
            scaling_factor = jnp_sqrt(jnp_real(jnp_complex128(product)))
        
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

def solution_first(affineParameter: complex, initNewton: complex, eps: float, minMax: int) -> tuple:
    """
    Computes the unphysical component of the theta divisor.
    """
    global periods_inverse, riemannM

    # Convert to JAX arrays
    affineParameter = jnp_complex128(affineParameter)
    initNewton = jnp_complex128(initNewton)
    eps = jnp_float64(eps)

    if not isinstance(periods_inverse, jnp_ndarray):
        # mpmath matrices don't iterate into nested rows the way jnp_array
        # expects, so build the nested list explicitly rather than passing
        # the matrix straight through (which silently flattens it to 1-D).
        periods_inverse = jnp_array(
            [[complex(periods_inverse[i, j]) for j in range(2)] for i in range(2)],
            dtype=jnp_complex128,
        )

    # Characteristic vector
    char = jnp_array([[0.5, 0.5], [0.0, 0.5]], dtype=jnp_complex128)

    # Initialize Newton iteration
    zero = initNewton
    aff = affineParameter

    zeroR = jnp_real(zero)
    zeroI = jnp_im(zero)
    affR = jnp_real(aff)
    affI = jnp_im(aff)

    perR = jnp_real(periods_inverse)
    perI = jnp_im(periods_inverse)

    p1 = perR[0, 0] * affR - perI[0, 0] * affI
    p2 = perR[0, 0] * affI + perI[0, 0] * affR
    p3 = perR[1, 0] * affR - perI[1, 0] * affI
    p4 = perR[1, 0] * affI + perI[1, 0] * affR

    # The theta argument z built from (zeroR, zeroI) below routinely has
    # |Im(z)| in the thousands for a near-degenerate root configuration
    # (large period-matrix entries), which overflows the Fourier series in
    # hyp_theta_fourier/RR/IR. Reduce it into the fundamental domain by a
    # fixed lattice shift computed once from this run's starting point, and
    # subtract that same shift at every iteration below -- see
    # _reduce_theta_arg's docstring for why this is exact for root-finding.
    z0first = 0.5 * (perR[0, 1] * zeroR - perI[0, 1] * zeroI + p1) + 1j * 0.5 * (perR[0, 1] * zeroI + perI[0, 1] * zeroR + p2)
    z0second = 0.5 * (perR[1, 1] * zeroR - perI[1, 1] * zeroI + p3) + 1j * 0.5 * (perR[1, 1] * zeroI + perI[1, 1] * zeroR + p4)
    theta_shift = _reduce_theta_arg(jnp_array([z0first, z0second]), riemannM)

    # Newton iteration step
    def newton_step(carry, _):
        zeroR, zeroI, count = carry

        zfirst = 0.5 * (perR[0, 1] * zeroR - perI[0, 1] * zeroI + p1) + 1j * 0.5 * (perR[0, 1] * zeroI + perI[0, 1] * zeroR + p2) - theta_shift[0]
        zsecond = 0.5 * (perR[1, 1] * zeroR - perI[1, 1] * zeroI + p3) + 1j * 0.5 * (perR[1, 1] * zeroI + perI[1, 1] * zeroR + p4) - theta_shift[1]
        zfirstR, zfirstI = jnp_real(zfirst), jnp_im(zfirst)
        zsecondR, zsecondI = jnp_real(zsecond), jnp_im(zsecond)

        z = jnp_array([zfirst, zsecond])

        f = hyp_theta_fourier(z, riemannM, char, (), minMax)
        af = jnp_abs(f)

        a = 0.5 * (
            perR[0, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            - perI[0, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            + perR[1, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 2, riemannM, char, minMax)
            - perI[1, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 2, riemannM, char, minMax)
        )

        c = 0.5 * (
            perR[0, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            + perI[0, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            + perR[1, 1] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 2, riemannM, char, minMax)
            + perI[1, 1] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 2, riemannM, char, minMax)
        )

        b = -c
        d = a
        det = a * d - b * c

        f_real = jnp_real(f)
        f_imag = jnp_im(f)

        # zeroR/zeroI are the real and imaginary parts of a Newton iterate,
        # real by definition -- but a/b/c/d are built from hyp_theta_RR/IR,
        # which (like the rest of this module) are complex128-typed even
        # when their mathematical value is real, so zeroR_new/zeroI_new
        # otherwise pick up a vestigial complex dtype that lax.scan's carry
        # (seeded as float64 below) rejects.
        zeroR_new = jnp_real(zeroR - (d * f_real - b * f_imag) / det)
        zeroI_new = jnp_real(zeroI - (-c * f_real + a * f_imag) / det)

        count += 1
        converged = af <= eps

        return (zeroR_new, zeroI_new, count), (zeroR_new + 1j * zeroI_new, f, af, converged)

    carry = (zeroR, zeroI, 0)
    _, (all_zeros, all_fs, all_afs, all_converged) = lax.scan(
        newton_step, carry, xs=jnp_arange(30)
    )

    converged_mask = all_converged
    converged_idx = argmax(converged_mask)
    any_converged = bool(jnp_any(converged_mask))

    if not any_converged:
        print("In solution: Iteration process stopped after 30 iterations.")
        print("hyp_theta(1/2 * omega1inv * (phi, initNewton)^t) = ", complex(all_fs[-1]))
        # Match the shape of the converged-case return below (a 3-element
        # list, whose 3rd element is itself a 2-element list) rather than
        # the wildly different 2-element [zeros_list, complex] shape this
        # used to return: orbitdata() stacks every affine_list sample's
        # result into one array, and a differently-shaped failure result
        # makes that array ragged, which crashes far downstream (in
        # integrate_hyperelliptic_first) instead of here where the actual
        # convergence failure happened.
        nan = complex(float("nan"), float("nan"))
        return [nan, nan, [nan, complex(affineParameter)]]

    final_zero = all_zeros[converged_idx]
    final_zeroR = jnp_real(final_zero)
    final_zeroI = jnp_im(final_zero)

    zfirst_final = 0.5 * (perR[0, 1] * final_zeroR - perI[0, 1] * final_zeroI + p1) + 1j * 0.5 * (perR[0, 1] * final_zeroI + perI[0, 1] * final_zeroR + p2) - theta_shift[0]
    zsecond_final = 0.5 * (perR[1, 1] * final_zeroR - perI[1, 1] * final_zeroI + p3) + 1j * 0.5 * (perR[1, 1] * final_zeroI + perI[1, 1] * final_zeroR + p4) - theta_shift[1]

    z_final = jnp_array([zfirst_final, zsecond_final])

    s1 = hyp_theta_fourier(z_final, riemannM, char, (1,), minMax)
    s2 = hyp_theta_fourier(z_final, riemannM, char, (2,), minMax)

    numerator = s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0]
    denominator = s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1]
    sol = -numerator / denominator

    return [complex(sol), complex(final_zero),
            [complex(final_zero), complex(affineParameter)]]

def solution_second(affineParameter: complex, initNewton: complex, eps: float, minMax: int):
    """
    Computes the unphysical component of the theta divisor.
    """
    global periods_inverse, riemannM

    # Convert to JAX arrays
    affineParameter = jnp_complex128(affineParameter)
    initNewton = jnp_complex128(initNewton)
    eps = jnp_float64(eps)

    if not isinstance(periods_inverse, jnp_ndarray):
        periods_inverse = jnp_array(
            [[complex(periods_inverse[i, j]) for j in range(2)] for i in range(2)],
            dtype=jnp_complex128,
        )

    # Characteristic vector
    char = jnp_array([[0.5, 0.5], [0.0, 0.5]], dtype=jnp_complex128)

    zero = initNewton
    aff = affineParameter

    zeroR = jnp_real(zero)
    zeroI = jnp_im(zero)
    affR = jnp_real(aff)
    affI = jnp_im(aff)

    perR = jnp_real(periods_inverse)
    perI = jnp_im(periods_inverse)
    
    p1 = perR[0, 1] * affR - perI[0, 1] * affI
    p2 = perR[0, 1] * affI + perI[0, 1] * affR
    p3 = perR[1, 1] * affR - perI[1, 1] * affI
    p4 = perR[1, 1] * affI + perI[1, 1] * affR

    z0first = 0.5 * (perR[0, 0] * zeroR - perI[0, 0] * zeroI + p1) + 1j * 0.5 * (perR[0, 0] * zeroI + perI[0, 0] * zeroR + p2)
    z0second = 0.5 * (perR[1, 0] * zeroR - perI[1, 0] * zeroI + p3) + 1j * 0.5 * (perR[1, 0] * zeroI + perI[1, 0] * zeroR + p4)
    theta_shift = _reduce_theta_arg(jnp_array([z0first, z0second]), riemannM)

    # Newton iteration step
    def newton_step(carry, _):
        zeroR, zeroI, count = carry

        zfirst = 0.5 * (perR[0, 0] * zeroR - perI[0, 0] * zeroI + p1) + 1j * 0.5 * (perR[0, 0] * zeroI + perI[0, 0] * zeroR + p2) - theta_shift[0]
        zsecond = 0.5 * (perR[1, 0] * zeroR - perI[1, 0] * zeroI + p3) + 1j * 0.5 * (perR[1, 0] * zeroI + perI[1, 0] * zeroR + p4) - theta_shift[1]
        zfirstR, zfirstI = jnp_real(zfirst), jnp_im(zfirst)
        zsecondR, zsecondI = jnp_real(zsecond), jnp_im(zsecond)

        z = jnp_array([zfirst, zsecond])

        f = hyp_theta_fourier(z, riemannM, char, (), minMax)
        af = jnp_abs(f)

        a = 0.5 * (
            perR[0, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            - perI[0, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            + perR[1, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 2, riemannM, char, minMax)
            - perI[1, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 2, riemannM, char, minMax)
        )

        c = 0.5 * (
            perR[0, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            + perI[0, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 1, riemannM, char, minMax)
            + perR[1, 0] * hyp_theta_IR(zfirstR, zfirstI, zsecondR, zsecondI, 2, riemannM, char, minMax)
            + perI[1, 0] * hyp_theta_RR(zfirstR, zfirstI, zsecondR, zsecondI, 2, riemannM, char, minMax)
        )

        b = -c
        d = a
        det = a * d - b * c

        f_real = jnp_real(f)
        f_imag = jnp_im(f)

        zeroR_new = jnp_real(zeroR - (d * f_real - b * f_imag) / det)
        zeroI_new = jnp_real(zeroI - (-c * f_real + a * f_imag) / det)

        count += 1
        converged = af <= eps

        return (zeroR_new, zeroI_new, count), (zeroR_new + 1j * zeroI_new, f, af, converged)

    carry = (zeroR, zeroI, 0)
    _, (all_zeros, all_fs, all_afs, all_converged) = lax.scan(
        newton_step, carry, xs=jnp_arange(30)
    )

    converged_mask = all_converged
    converged_idx = argmax(converged_mask)
    any_converged = bool(jnp_any(converged_mask))

    if not any_converged:
        print("In solution: Iteration process stopped after 30 iterations.")
        print("hyp_theta(1/2 * omega1inv * (phi, initNewton)^t) = ", complex(all_fs[-1]))
        nan = complex(float("nan"), float("nan"))
        return [nan, nan, [nan, complex(affineParameter)]]

    final_zero = all_zeros[converged_idx]
    final_zeroR = jnp_real(final_zero)
    final_zeroI = jnp_im(final_zero)

    zfirst_final = 0.5 * (perR[0, 0] * final_zeroR - perI[0, 0] * final_zeroI + p1) + 1j * 0.5 * (perR[0, 0] * final_zeroI + perI[0, 0] * final_zeroR + p2) - theta_shift[0]
    zsecond_final = 0.5 * (perR[1, 0] * final_zeroR - perI[1, 0] * final_zeroI + p3) + 1j * 0.5 * (perR[1, 0] * final_zeroI + perI[1, 0] * final_zeroR + p4) - theta_shift[1]

    z_final = jnp_array([zfirst_final, zsecond_final])

    s1 = hyp_theta_fourier(z_final, riemannM, char, (1,), minMax)
    s2 = hyp_theta_fourier(z_final, riemannM, char, (2,), minMax)

    numerator = s1 * periods_inverse[0, 0] + s2 * periods_inverse[1, 0]
    denominator = s1 * periods_inverse[0, 1] + s2 * periods_inverse[1, 1]
    sol = -numerator / denominator

    return [complex(sol), complex(final_zero),
            [complex(final_zero), complex(affineParameter)]]
