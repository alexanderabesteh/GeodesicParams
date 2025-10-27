#!/usr/bin/env python3
"""
Provides mini helper procedures utilized throughout various modules.

Helper functions include sorting procedures for the roots of polynomials,
clearing directories, checking if a number is in a list, and numerically
evaluating symbolic numbers in a list.

References
----------
[] E. Hackmann, Geodesic equations in black hole space-times with cosmological constant (2010).
[] A. Meurer, C. P. Smith, M. Paprocki, O. ˇCert´ık, S. B. Kirpichev, M. Rocklin, A. Kumar, S. Ivanov, J. K.
    Moore, S. Singh, T. Rathnayake, S. Vig, B. E. Granger, R. P. Muller, F. Bonazzi, H. Gupta, S. Vats, F. Johans-
    son, F. Pedregosa, M. J. Curry, A. R. Terrel, v. Rouˇcka, A. Saboo, I. Fernando, S. Kulal, R. Cimrman, and
    A. Scopatz, Sympy: symbolic computing in python, PeerJ Computer Science 3, e103 (2017).

"""

from os import makedirs, path
from shutil import rmtree

import jax.numpy as jnp
from jax import config, jit, lax
from sympy import Expr

config.update("jax_enable_x64", True)


@jit
def inlist(element, lst):
    """
    Given a complex number <element>, find what index the element is
    located at in <lst>. Return -1 if <element> is not in <lst>.

    Parameters
    ----------
    element : complex
        A complex number.
    lst : list
        A list of complex or real numbers.

    Returns
    -------
    result : integer
        The index at which <element> is located in <lst>.
    """
    lst_arr = jnp.asarray(lst, dtype=jnp.complex128)
    matches = lst_arr == element

    # find index where match occurs; returns 0 if none, will correct later
    idx = jnp.argmax(matches)

    # If no match, argmax returns 0, so check if element is really present
    result = lax.cond(
        jnp.any(matches),
        lambda _: idx,
        lambda _: -1,
        operand=None,
    )
    return result


@jit
def extract_multiple_elems(lst):
    """
    Remove duplicate elements from <lst> and store them in a separate
    list.

    Parameters
    ----------
    lst : list
        A list of complex or real numbers.

    Returns
    -------
    clean_list : list
        The original list, but with duplicate elements removed.
    mult_elems : list
        A list of the duplicate elements removed from <lst>.
    """
    arr = jnp.asarray(lst, dtype=jnp.complex128)

    # Sort by real part
    sorted_arr = arr[jnp.argsort(jnp.real(arr))]

    # Identify duplicates (adjacent equal elements)
    duplicates_mask = jnp.concatenate(
        [jnp.array([False]), sorted_arr[1:] == sorted_arr[:-1]]
    )

    mult_elems = sorted_arr[duplicates_mask]

    # Keep only unique elements (first occurrence)
    clean_list = jnp.array(
        [x for i, x in enumerate(sorted_arr) if not duplicates_mask[i]]
    )

    return clean_list, mult_elems


@jit
def find_next(expression, lst):
    """
    Find a number in <lst> that is closest to the value of <expression>.

    Parameters
    ----------
    expression : real
        A real number.
    lst : list
        A list of real numbers.

    Returns
    -------
    real
        The number in <lst> closest to <expression>.
    """
    if isinstance(expression, (list, jnp.ndarray)):
        expression = expression[0]

    arr = jnp.asarray(lst, dtype=jnp.float64)
    diffs = jnp.abs(arr - expression)
    idx = jnp.argmin(diffs)
    return arr[idx]


@jit
def separate_zeros(zeros):
    """
    Seperate the zeros of a polynomial into its real roots
    and complex roots.

    Parameters
    ----------
    zeros : list
        A list of complex and real numbers representing the roots
        of the polynomial.

    Returns
    -------
     realNS : list
        A list of real numbers ordered from least to greatest.
     complexNS : list
        A list of complex numbers ordered from least to greatest.
    """
    zeros_arr = jnp.asarray(zeros, dtype=jnp.complex128)

    # Boolean mask for real numbers (imaginary part == 0)
    is_real = jnp.isclose(jnp.imag(zeros_arr), 0.0)

    realNS = jnp.sort(jnp.real(zeros_arr[is_real]))
    complexNS = zeros_arr[~is_real]

    return realNS, complexNS


@jit
def eval_roots(lst):
    """
    Evaluate the symbolic roots of a polynomial numerically.

    Parameters
    ----------
    lst : list
        A list of complex or real numbers representing the roots of a
        polynomial symbolically.

    Returns
    -------
    lst: list
        The original list of numbers evaluated numerically instead of symbolically.

    """

    def _eval(x):
        if isinstance(x, Expr):
            return float(x.evalf())
        return float(x)

    return [_eval(x) for x in lst]


def clear_directory(dir_path):
    """
    Remove all files in a directory.

    Parameters
    ----------
    dir : string
        A string representing the path to the directory to be cleared.

    Returns
    -------
    None

    """

    if path.exists(dir_path):
        rmtree(dir_path)
    makedirs(dir_path, exist_ok=True)
