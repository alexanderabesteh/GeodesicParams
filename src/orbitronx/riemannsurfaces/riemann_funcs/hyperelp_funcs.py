#!/usr/bin/env python3
"""
A collection of procedures for computing hyperelliptic functions for genus a genus 2 Riemann
surface.

In particular, the functions implemented are the hyperelliptic theta function (with various
algorithms), and the derivatives of the theta function and Kleinian sigma functions.

References
----------
[] E. Hackmann, Geodesic equations in black hole space-times with cosmological constant (2010).
[] T. mpmath development team, mpmath: a Python library for arbitrary-precision floating-point arithmetic
    (version 1.3.0) (2023), http://mpmath.org/.
[] H. Labrande, Explicit computation of the Abel-Jacobi map and its inverse, Theses,
    Universit´e de Lorraine (2016).

"""

from functools import partial

from jax import config, jit, vmap
from jax.numpy import (
    arange,
    array,
    asarray,
    broadcast_arrays,
    complex128,
    cos,
    einsum,
    exp,
    float64,
    imag,
    int32,
    meshgrid,
    pi,
    real,
    sin,
    stack,
    sum,
)

config.update("jax_enable_x64", True)


def derivative_factor(derivatives):
    """
    Computes the 2*pi*1j factor in front of the Fourier series definition of the theta function
    after taking partial derivatives.

    Parameters
    ----------
    derivatives : list
        A list containing the integers representing the derivatives with respect to z1 or z2.
        A 0 means no derivative is computed, while 1 and 2 represent the derivatives with respect
        to z1 and z2 respectively.

    Returns
    -------
    complex
        The derivative factor in front of the Fourier series.
    """
    if derivatives is None:
        return 1 + 0j
    count = len([d for d in derivatives if d != 0])
    return (2 * pi * 1j) ** count


@partial(jit, static_argnames=('derivatives', 'minMax'))
def hyp_theta_fourier(z, riemannM, char, derivatives=(), minMax=5):
    """
    Computes the hyperelliptic theta function on a genus 2 Riemann surface using the Fourier
    series definition of the function. Optional derivatives can be computed.

    Parameters
    ----------
    z : list
        A list containing two complex numbers.
    riemannM : matrix
        The Riemann matrix of the Riemann surface.
    char : list
        A list containing two lists of length 2. These lists represent the g and h
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    derivatives : list, optional
        A list containing integers, either 0, 1, or 2. These represent the partial derivatives of the
        theta function with respect to the first and second components of the vector z, z1 and z2
        respectively. For example, [1, 2 ,1] computes the partial derivative of the theta function
        with respect to z1, z2, and finally z1. An integer of 0 means no derivative is computed.
    minMax : natural, optional
        A natural number from 5 <= minMax <= 30 (the summation bound).

    Returns
    -------
    result : complex
        The value of the hyperelliptic theta function evaluated at <z> with Riemann matrix
        <riemannM>.

    """
    riemannM = asarray(riemannM, dtype=complex128)
    g, h = char
    g = asarray(g, dtype=complex128)
    h = asarray(h, dtype=complex128)

    za = asarray(z, dtype=complex128)
    if za.ndim == 1:
        batch_mode = False
        z_batch = za[None, :]
    else:
        batch_mode = True
        if za.shape[-1] != 2:
            raise ValueError("z must have last dimension of size 2")
        leading_shape = za.shape[:-1]
        z_batch = za.reshape((-1, 2))

    if not (5 <= minMax <= 30):
        raise ValueError("minMax must satisfy 5 <= minMax <= 30")
    k = arange(-minMax, minMax + 1, dtype=int32)
    M1, M2 = meshgrid(k, k, indexing="ij")
    M1 = M1.astype(complex128)
    M2 = M2.astype(complex128)

    m1g = M1 + g[0]
    m2g = M2 + g[1]

    tau_sum_0 = riemannM[0, 0] * m1g + riemannM[0, 1] * m2g
    tau_sum_1 = riemannM[1, 0] * m1g + riemannM[1, 1] * m2g

    derivatives = () if derivatives is None else tuple(derivatives)
    count1 = len([d for d in derivatives if d == 1])
    count2 = len([d for d in derivatives if d == 2])

    derivs_grid = (m1g ** array(count1, dtype=int32)) * (
        m2g ** array(count2, dtype=int32)
    )

    def _single_theta(z2):
        z0 = z2[0]
        z1 = z2[1]
        term0 = m1g * (tau_sum_0 + 2.0 * z0 + 2.0 * h[0])
        term1 = m2g * (tau_sum_1 + 2.0 * z1 + 2.0 * h[1])
        char_sum = term0 + term1

        ex = exp(1j * pi * char_sum)

        summand = ex * derivs_grid
        res = sum(summand)

        deriv_factor = derivative_factor(derivatives)
        return deriv_factor * res

    batched_single = vmap(_single_theta, in_axes=0, out_axes=0)
    results = batched_single(z_batch)

    if not batch_mode:
        return results[0]
    else:
        results = results.reshape(leading_shape)
        return results


@partial(jit, static_argnames=('l', 'minMax'))
def hyp_theta_RR(xR, xI, wR, wI, l, riemannM, char, minMax=5):
    """
    Computes the partial derivative of the real part of the hyperelliptic theta function
    with respect <xR> or <wR>, depending on <l>. The vector z of the theta function is split
    into two complex variables <x> and <w> (i.e. z = [xR + 1j * xI, wR + 1j * xI]).

    Parameters
    ----------
    xR : real
        The real part of the first component of z.
    xI : real
        The imaginary part of the first component of z.
    wR : real
        The real part of the second component of z.
    wI : real
        The imaginary part of the second component of z.
    l : int
        An integer, either 1 or 2 representing the partial derivative with respect to
        <xR> for <l> = 1 and <wR> for <l> = 2.
    char : list
        A list containing two lists of length 2. These lists represent the g and h
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    minMax : natural, optional
        A natural number from 5 <= minMax <= 30 (the summation bound).

    Returns
    -------
    result : complex
        The partial deriative of the real part of the theta function with respect to <xR> or
        <wR>.

    """
    if not (5 <= minMax <= 30):
        raise ValueError("minMax must satisfy 5 <= minMax <= 30")

    xR_a = asarray(xR, dtype=float64)
    xI_a = asarray(xI, dtype=float64)
    wR_a = asarray(wR, dtype=float64)
    wI_a = asarray(wI, dtype=float64)
    try:
        xR_b, xI_b, wR_b, wI_b = broadcast_arrays(xR_a, xI_a, wR_a, wI_a)
    except Exception as e:
        raise ValueError(f"hyp_theta_RR: xR,xI,wR,wI must be broadcastable: {e}")

    batch_shape = xR_b.shape
    flat_xR = xR_b.ravel()
    flat_xI = xI_b.ravel()
    flat_wR = wR_b.ravel()
    flat_wI = wI_b.ravel()

    R = asarray(riemannM, dtype=complex128)
    Rr = real(R)
    Ri = imag(R)

    g, h = char
    g = asarray(g, dtype=complex128)
    h = asarray(h, dtype=complex128)

    k = arange(-minMax, minMax + 1, dtype=int32)
    M1, M2 = meshgrid(k, k, indexing="ij")
    M1 = M1.astype(complex128)
    M2 = M2.astype(complex128)

    m1g = M1 + g[0]
    m2g = M2 + g[1]

    tau_sum_r_0 = Rr[0, 0] * m1g + Rr[0, 1] * m2g
    tau_sum_r_1 = Rr[1, 0] * m1g + Rr[1, 1] * m2g
    tau_sum_i_0 = -(Ri[0, 0] * m1g + Ri[0, 1] * m2g)
    tau_sum_i_1 = -(Ri[1, 0] * m1g + Ri[1, 1] * m2g)

    if not (l == 1 or l == 2):
        raise ValueError("l must be 1 or 2")
    l_idx = l - 1
    m_selected = m1g if l_idx == 0 else m2g

    def _single(xR0, xI0, wR0, wI0):
        varR0 = array([xR0, wR0], dtype=float64)
        varI0 = array([xI0, wI0], dtype=float64)

        char_sumExp = m1g * (tau_sum_i_0 - 2.0 * varI0[0]) + m2g * (
            tau_sum_i_1 - 2.0 * varI0[1]
        )
        char_sumSin = m1g * (tau_sum_r_0 + 2.0 * varR0[0] + 2.0 * h[0]) + m2g * (
            tau_sum_r_1 + 2.0 * varR0[1] + 2.0 * h[1]
        )

        ex = exp(pi * char_sumExp)
        s = sin(pi * char_sumSin)
        multiplier = 2.0 * pi * m_selected

        summand = -ex * s * multiplier
        total = sum(summand)
        return total

    batched = vmap(_single, in_axes=(0, 0, 0, 0), out_axes=0)
    flat_res = batched(flat_xR, flat_xI, flat_wR, flat_wI)

    res = flat_res.reshape(batch_shape)
    return res


@partial(jit, static_argnames=('l', 'minMax'))
def hyp_theta_IR(xR, xI, wR, wI, l, riemannM, char, minMax=5):
    """
    Computes the partial derivative of the imaginary part of the hyperelliptic theta function
    with respect <xR> or <wR>, depending on <l>. The vector z of the theta function is split
    into two complex variables <x> and <w> (i.e. z = [xR + 1j * xI, wR + 1j * xI]).

    Parameters
    ----------
    xR : real
        The real part of the first component of z.
    xI : real
        The imaginary part of the first component of z.
    wR : real
        The real part of the second component of z.
    wI : real
        The imaginary part of the second component of z.
    l : int
        An integer, either 1 or 2 representing the partial derivative with respect to
        <xR> for <l> = 1 and <wR> for <l> = 2.
    char : list
        A list containing two lists of length 2. These lists represent the g and h
        characteristics of the theta function (the elements of these lists are either 0 or 1/2).
    minMax : natural, optional
        A natural number from 5 <= minMax <= 30 (the summation bound).

    Returns
    -------
    result : complex
        The partial deriative of the real part of the theta function with respect to <xR> or
        <wR>.

    """

    if not (5 <= minMax <= 30):
        raise ValueError("minMax must satisfy 5 <= minMax <= 30")

    xR_a = asarray(xR, dtype=float64)
    xI_a = asarray(xI, dtype=float64)
    wR_a = asarray(wR, dtype=float64)
    wI_a = asarray(wI, dtype=float64)
    try:
        xR_b, xI_b, wR_b, wI_b = broadcast_arrays(xR_a, xI_a, wR_a, wI_a)
    except Exception as e:
        raise ValueError(f"hyp_theta_IR: xR,xI,wR,wI must be broadcastable: {e}")

    batch_shape = xR_b.shape
    flat_xR = xR_b.ravel()
    flat_xI = xI_b.ravel()
    flat_wR = wR_b.ravel()
    flat_wI = wI_b.ravel()

    R = asarray(riemannM, dtype=complex128)
    Rr = real(R)
    Ri = imag(R)

    g, h = char
    g = asarray(g, dtype=complex128)
    h = asarray(h, dtype=complex128)

    k = arange(-minMax, minMax + 1, dtype=int32)
    M1, M2 = meshgrid(k, k, indexing="ij")
    M1 = M1.astype(complex128)
    M2 = M2.astype(complex128)

    m1g = M1 + g[0]
    m2g = M2 + g[1]

    tau_sum_r_0 = Rr[0, 0] * m1g + Rr[0, 1] * m2g
    tau_sum_r_1 = Rr[1, 0] * m1g + Rr[1, 1] * m2g
    tau_sum_i_0 = -(Ri[0, 0] * m1g + Ri[0, 1] * m2g)
    tau_sum_i_1 = -(Ri[1, 0] * m1g + Ri[1, 1] * m2g)

    if not (l == 1 or l == 2):
        raise ValueError("l must be 1 or 2")
    l_idx = l - 1
    m_selected = m1g if l_idx == 0 else m2g

    def _single(xR0, xI0, wR0, wI0):
        varR0 = array([xR0, wR0], dtype=float64)
        varI0 = array([xI0, wI0], dtype=float64)

        char_sumExp = m1g * (tau_sum_i_0 - 2.0 * varI0[0]) + m2g * (
            tau_sum_i_1 - 2.0 * varI0[1]
        )
        char_sumCos = m1g * (tau_sum_r_0 + 2.0 * varR0[0] + 2.0 * h[0]) + m2g * (
            tau_sum_r_1 + 2.0 * varR0[1] + 2.0 * h[1]
        )

        ex = exp(pi * char_sumExp)
        c = cos(pi * char_sumCos)

        multiplier = 2.0 * pi * m_selected

        summand = ex * c * multiplier
        total = sum(summand)
        return total

    batched = vmap(_single, in_axes=(0, 0, 0, 0), out_axes=0)
    flat_res = batched(flat_xR, flat_xI, flat_wR, flat_wI)

    res = flat_res.reshape(batch_shape)
    return res


@partial(jit, static_argnames=('minMax',))
def sigma1(z, riemannM, minMax=5):

    z = asarray(z, dtype=complex128)
    riemannM = asarray(riemannM, dtype=complex128)
    g = array([0.5, 0.5])
    h = array([0.0, 0.5])

    m_range = arange(-minMax, minMax + 1)
    m1, m2 = meshgrid(m_range, m_range, indexing="ij")

    m = stack([m1, m2], axis=-1).reshape(-1, 2)
    mg = m + g

    tau_sum = einsum("ij,...j->...i", riemannM, mg)
    char_sum = sum(mg * (tau_sum + 2 * z + 2 * h), axis=-1)

    term = exp(1j * pi * char_sum) * (2 * pi * 1j * (m[:, 0] + g[0]))
    result = sum(term, axis=0)

    return result


@partial(jit, static_argnames=('minMax',))
def sigma2(z, riemannM, minMax=5):

    z = asarray(z, dtype=complex128)
    riemannM = asarray(riemannM, dtype=complex128)
    g = array([0.5, 0.5])
    h = array([0.0, 0.5])

    m_range = arange(-minMax, minMax + 1)
    m1, m2 = meshgrid(m_range, m_range, indexing="ij")

    m = stack([m1, m2], axis=-1).reshape(-1, 2)
    mg = m + g

    tau_sum = einsum("ij,...j->...i", riemannM, mg)  # (..., 2)

    char_sum = sum(mg * (tau_sum + 2 * z + 2 * h), axis=-1)

    term = exp(1j * pi * char_sum) * (2 * pi * 1j * (m[:, 1] + g[1]))
    result = sum(term, axis=0)

    return result
