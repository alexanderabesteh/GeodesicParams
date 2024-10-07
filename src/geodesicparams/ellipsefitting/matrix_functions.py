#!/usr/bin/env python3
"""
Functions for performing element-wise operations on matrices.

These operations include element-wise exponentiation, multiplication, 
and division.

"""

from mpmath import matrix

def element_pow(mat1, mat2):
    """
    Compute the element-wise exponentiation of the matrix <mat1> to the matrix <mat2>
    powers.

    Specifically, the result[i, j] = mat1[i, j] ** mat2[i, j]. <mat2> can be an integer; 
    in this case, the elements of mat1 will each be raised to the <mat2> power.

    Parameters
    ----------
    mat1 : matrix
        An mpmath matrix with each element serving as the bases to exponentiated by
        <mat2>.
    mat2 : matrix, integer
        An mpmath matrix or an integer. If a matrix is used, the elements of <mat1>
        are raised to the power of each element in <mat2>.

    Returns
    -------
    result : matrix
        An mpmath matrix containing the result of the element-wise exponentiation of 
        <mat1> and <mat2>.

    """

    result = matrix(mat1.rows, mat1.cols)

    # If <mat2> is not an integer
    if type(mat2) != type(mat1):
        matTemp = result
        for i in range(mat1.rows):
            for j in range(mat1.cols):
                matTemp[i, j] = mat2
    else:
        matTemp = mat2

    for i in range(mat1.rows):
        for j in range(mat1.cols):
            result[i, j] = mat1[i, j] ** matTemp[i, j]

    return result

def element_mul(mat1, mat2):
    """
    Compute the element-wise multiplication of the matrix <mat1> and the matrix <mat2>.

    Specifically, the result[i, j] = mat1[i, j] * mat2[i, j].

    Parameters
    ----------
    mat1 : matrix
        An mpmath matrix.
    mat2 : matrix
        An mpmath matrix.

    Returns
    -------
    result : matrix
        An mpmath matrix containing the result of the element-wise multiplication of 
        <mat1> and <mat2>.

    """

    result = matrix(mat1.rows, mat1.cols)

    for i in range(mat1.rows):
        for j in range(mat1.cols):
            result[i, j] = mat1[i, j] * mat2[i, j]

    return result

def element_div(mat1, mat2):
    """
    Compute the element-wise division of the matrix <mat1> by the matrix <mat2>.

    Specifically, the result[i, j] = mat1[i, j] / mat2[i, j].

    Parameters
    ----------
    mat1 : matrix
        An mpmath matrix with each element serving as the dividend.
    mat2 : matrix
        An mpmath matrix with each element serving as the divisor.

    Returns
    -------
    result : matrix
        An mpmath matrix containing the result of the element-wise division of 
        <mat1> by <mat2>.

    """

    result = matrix(mat1.rows, mat1.cols)

    for i in range(mat1.rows):
        for j in range(mat1.cols):
            result[i, j] = mat1[i, j] / mat2[i, j]

    return result
