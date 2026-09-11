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


from pathlib import Path
from shutil import Error, rmtree

from jax import config
from jax.numpy import arange, argmax, array, float64, isclose, logical_and, nan, where
from sympy import im, re

config.update("jax_enable_x64", True)


def inlist(element: complex | str, lst: list[complex | str]) -> int:
    """
    Given an element, find what index the element is located at in <lst>.
    Return -1 if <element> is not in <lst>. Supports complex numbers
    (with approximate matching) and strings (with exact matching).

    Parameters
    ----------
    element : complex or str
        A complex number or string.
    lst : list
        A list of complex numbers, real numbers, or strings.

    Returns
    -------
    result : integer
        The index at which <element> is located in <lst>.
    """
    if isinstance(element, str):
        for i, item in enumerate(lst):
            if item == element:
                return i
        return -1

    elem = complex(element)
    element_array = array([elem.real, elem.imag], dtype = float64)

    # Entries that aren't a single number (e.g. a nested list/tuple) can never
    # equal <element>; mark them with NaN so they fall out of isclose() below
    # instead of raising, matching plain Python's "==" behavior on a type
    # mismatch (always False, never an exception).
    pairs = []
    for z in lst:
        try:
            c = complex(z)
            pairs.append([c.real, c.imag])
        except TypeError:
            pairs.append([nan, nan])

    lst_array = array(pairs, dtype = float64)

    real_matches = isclose(lst_array[:, 0], element_array[0], rtol=1e-10, atol=1e-12)
    imag_matches = isclose(lst_array[:, 1], element_array[1], rtol=1e-10, atol=1e-12)
    matches = logical_and(real_matches, imag_matches)

    indices = arange(len(lst))
    match_indices = indices * matches

    max_index = argmax(match_indices)

    result_index = where(
        matches[max_index] > 0,
        max_index,
        -1
    )

    return int(result_index)

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

    sorted_list = sorted(lst, key=lambda x: re(x))
    mult_elems = []
    clean_list = sorted_list
    i = 0

    while i < len(sorted_list):
        if i == len(sorted_list) - 1:
            break
        else:
            if sorted_list[i] == sorted_list[i + 1]:
                mult_elems.append(sorted_list[i])
                clean_list.pop(i)
            else:
                i += 1
    return clean_list, mult_elems


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

    if isinstance(expression, list):
        expression = expression[0]

    d = abs(expression - lst[0])
    j = 0

    for i in range(1, len(lst)):
        if abs(expression - lst[i]) < d:
            d = abs(expression - lst[i])
            j = i
    return lst[j]


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

    realNS = []
    complexNS = []

    for i in range(len(zeros)):
        if im(zeros[i]) == 0:
            realNS.append(zeros[i])
        else:
            complexNS.append(zeros[i])
    realNS.sort()

    return realNS, complexNS


def eval_roots(lst, digits=15):
    """
    Evaluate the symbolic roots of a polynomial numerically.

    Parameters
    ----------
    lst : list
        A list of complex or real numbers representing the roots of a
        polynomial symbolically.
    digits : int, optional
        The number of significant digits to evaluate each root to. Defaults
        to 15 (sympy's own `.evalf()` default), which is not enough to
        resolve roots that end up tightly clustered after a genus-2
        substitution (e.g. a small but nonzero cosmological constant):
        expanding a branch-cut polynomial from such near-coincident roots
        is a Wilkinson-style ill-conditioned computation that can lose 10+
        significant digits to cancellation, so callers working with such
        configurations should pass the ambient `digits`/`mp.prec` precision
        instead of relying on this default.

    Returns
    -------
    lst: list
        The original list of numbers evaluated numerically instead of symbolically.

    """

    eval_lst = [lst[i].evalf(digits) for i in range(len(lst))]
    return eval_lst

def clear_directory(dir_path):
    """
    Remove all files and subdirectories in a directory.

    Parameters
    ----------
    dir_path : str or Path
        Path to the directory to be cleared.

    Returns
    -------
    None
    
    Raises
    ------
    OSError
        If directory operations fail.
    """
    dir_path = Path(dir_path)
    
    try:
        if dir_path.exists():
            for item in dir_path.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    rmtree(item)
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    except (OSError, PermissionError, Error) as e:
        raise OSError(f"Failed to clear directory {dir_path}: {e}")
