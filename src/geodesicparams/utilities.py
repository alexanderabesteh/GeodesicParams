#!/usr/bin/env python3
"""
Provides mini helper procedures utilized throughout various modules.

Helper functions include sorting procedures for the roots of polynomials,
clearing directories, checking if a number is in a list, and numerically
evaluating symbolic numbers in a list.
"""

from os import path, unlink, walk
from shutil import rmtree

from sympy import re, im

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

    for i in range(len(lst)):
        if element == lst[i]:
            result = i
            break
        else:
            result = -1
    return result

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

    sorted_list = sorted(lst, key = lambda x : re(x))
    mult_elems = []
    clean_list = sorted_list
    i = 0

    while i < len(sorted_list):
        if i == len(sorted_list) - 1:
            break
        else:
            if sorted_list[i] == sorted_list[i+1]:
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

    if type(expression) == list:
        expression = expression[0]

    d = abs(expression - lst[0])
    j = 0

    for i in range(1, len(lst)):
        if abs(expression-lst[i]) < d:
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

    eval_lst = [lst[i].evalf() for i in range(len(lst))]
    return eval_lst

def clear_directory(dir):
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

    for root, dirs, files in walk(dir):
        for f in files:
            unlink(path.join(root, f))
        for d in dirs:
            rmtree(path.join(root, d))
