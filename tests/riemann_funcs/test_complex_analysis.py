"""
Tests for orbitronx.riemannsurfaces.riemann_funcs.complex_analysis

Covers: jacobi_theta, qfrom, chop, carlson_first, agm
"""
import cmath
import math

import jax.numpy as jnp
import mpmath
import pytest
from orbitronx.riemannsurfaces.riemann_funcs.complex_analysis import (
    agm,
    carlson_first,
    chop,
    jacobi_theta,
    qfrom,
)

RTOL = 1e-8
ATOL = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# qfrom
# ─────────────────────────────────────────────────────────────────────────────

class TestQfrom:
    def test_pure_imaginary_1j(self):
        """qfrom(1j) = exp(-π)."""
        q = complex(qfrom(1j))
        assert abs(q - cmath.exp(-math.pi)) < ATOL

    def test_pure_imaginary_2j(self):
        """qfrom(2j) = exp(-2π)."""
        q = complex(qfrom(2j))
        assert abs(q - cmath.exp(-2 * math.pi)) < ATOL

    def test_modulus_less_than_one(self):
        """|q| < 1 for all τ with Im(τ) > 0."""
        for tau in [0.5j, 1j, 2j, 0.3 + 1j, -0.7 + 2j]:
            assert abs(complex(qfrom(tau))) < 1.0

    def test_definition(self):
        """qfrom(τ) = exp(iπτ) by definition."""
        tau = 0.4 + 1.2j
        q = complex(qfrom(tau))
        expected = cmath.exp(1j * math.pi * tau)
        assert abs(q - expected) < ATOL


# ─────────────────────────────────────────────────────────────────────────────
# chop
# ─────────────────────────────────────────────────────────────────────────────

class TestChop:
    def test_real_below_tol_becomes_zero(self):
        assert complex(chop(5e-16 + 0j)) == 0j

    def test_imag_below_tol_becomes_zero(self):
        assert complex(chop(0 + 5e-16j)) == 0j

    def test_above_tol_unchanged(self):
        val = 1e-14 + 0j
        assert abs(complex(chop(val)) - val) < 1e-30

    def test_custom_tol_chops(self):
        assert complex(chop(1e-11 + 0j, tol=1e-10)) == 0j

    def test_custom_tol_does_not_chop(self):
        val = 1e-9 + 0j
        assert abs(complex(chop(val, tol=1e-10)) - val) < 1e-30

    def test_normal_value_unchanged(self):
        val = 3.14159 + 2.71828j
        assert abs(complex(chop(val)) - val) < ATOL


# ─────────────────────────────────────────────────────────────────────────────
# jacobi_theta
# ─────────────────────────────────────────────────────────────────────────────

class TestJacobiTheta:
    # Use a nome well within the convergence radius
    q = complex(qfrom(1.2j))   # exp(-1.2π) ≈ 0.023
    z = 0.4 + 0.3j

    # ── θ₁ ──────────────────────────────────────────────────────────────────

    def test_theta1_zero_at_origin(self):
        """θ₁(0, q) = 0 (θ₁ is an odd function)."""
        val = complex(jacobi_theta(1, 0.0 + 0j, self.q))
        assert abs(val) < ATOL

    def test_theta1_odd_symmetry(self):
        """θ₁(-z, q) = -θ₁(z, q)."""
        pos = complex(jacobi_theta(1, self.z, self.q))
        neg = complex(jacobi_theta(1, -self.z, self.q))
        assert abs(neg + pos) < ATOL

    def test_theta1_matches_mpmath(self):
        val = complex(jacobi_theta(1, self.z, self.q))
        ref = complex(mpmath.jtheta(1, self.z, self.q))
        assert abs(val - ref) / (abs(ref) + 1e-30) < RTOL

    # ── θ₂ ──────────────────────────────────────────────────────────────────

    def test_theta2_even_symmetry(self):
        """θ₂(-z, q) = θ₂(z, q)."""
        pos = complex(jacobi_theta(2, self.z, self.q))
        neg = complex(jacobi_theta(2, -self.z, self.q))
        assert abs(neg - pos) < ATOL

    def test_theta2_matches_mpmath(self):
        val = complex(jacobi_theta(2, self.z, self.q))
        ref = complex(mpmath.jtheta(2, self.z, self.q))
        assert abs(val - ref) / (abs(ref) + 1e-30) < RTOL

    # ── θ₃ ──────────────────────────────────────────────────────────────────

    def test_theta3_even_symmetry(self):
        """θ₃(-z, q) = θ₃(z, q)."""
        pos = complex(jacobi_theta(3, self.z, self.q))
        neg = complex(jacobi_theta(3, -self.z, self.q))
        assert abs(neg - pos) < ATOL

    def test_theta3_matches_mpmath(self):
        val = complex(jacobi_theta(3, self.z, self.q))
        ref = complex(mpmath.jtheta(3, self.z, self.q))
        assert abs(val - ref) / (abs(ref) + 1e-30) < RTOL

    # ── θ₄ ──────────────────────────────────────────────────────────────────

    def test_theta4_even_symmetry(self):
        """θ₄(-z, q) = θ₄(z, q)."""
        pos = complex(jacobi_theta(4, self.z, self.q))
        neg = complex(jacobi_theta(4, -self.z, self.q))
        assert abs(neg - pos) < ATOL

    def test_theta4_matches_mpmath(self):
        val = complex(jacobi_theta(4, self.z, self.q))
        ref = complex(mpmath.jtheta(4, self.z, self.q))
        assert abs(val - ref) / (abs(ref) + 1e-30) < RTOL

    # ── validation / edge cases ──────────────────────────────────────────────

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            jacobi_theta(5, 0.0 + 0j, self.q)

    def test_theta1_derivative1_finite_difference(self):
        """d/dz θ₁(z,q)|_{z=self.z} via central finite difference."""
        h = 1e-5
        deriv = complex(jacobi_theta(1, self.z, self.q, derivative=1))
        fd = (
            complex(jacobi_theta(1, self.z + h, self.q))
            - complex(jacobi_theta(1, self.z - h, self.q))
        ) / (2 * h)
        assert abs(deriv - fd) / (abs(fd) + 1e-30) < 1e-5

    def test_batched_z_consistent_with_scalar(self):
        """jacobi_theta(3, [z1, z2], q) equals two scalar calls."""
        z1, z2 = 0.3 + 0.1j, -0.5 + 0.2j
        batch = jnp.array([z1, z2])
        result_batch = jacobi_theta(3, batch, self.q)
        r1 = complex(jacobi_theta(3, z1, self.q))
        r2 = complex(jacobi_theta(3, z2, self.q))
        assert abs(complex(result_batch[0]) - r1) < ATOL
        assert abs(complex(result_batch[1]) - r2) < ATOL


# ─────────────────────────────────────────────────────────────────────────────
# carlson_first  R_F(x, y, z)
# ─────────────────────────────────────────────────────────────────────────────

class TestCarlsonFirst:
    def test_equal_args_rf_111_equals_1(self):
        """R_F(1, 1, 1) = 1/√1 = 1."""
        val = complex(carlson_first(1.0, 1.0, 1.0))
        assert abs(val - 1.0) < ATOL

    def test_equal_args_rf_aaa_equals_1_over_sqrt_a(self):
        """R_F(4, 4, 4) = 1/√4 = 0.5."""
        val = complex(carlson_first(4.0, 4.0, 4.0))
        assert abs(val - 0.5) < RTOL

    def test_homogeneity(self):
        """R_F(λx, λy, λz) = λ^{-1/2} R_F(x, y, z)."""
        x, y, z, lam = 1.0, 2.0, 3.0, 4.0
        base = complex(carlson_first(x, y, z))
        scaled = complex(carlson_first(lam * x, lam * y, lam * z))
        assert abs(scaled - base / math.sqrt(lam)) / (abs(base) + 1e-30) < RTOL

    def test_known_value_rf_0_1_2(self):
        """R_F(0, 1, 2) matches mpmath.elliprf(0, 1, 2) ≈ 1.311.

        The implementation substitutes 0 with tol=1e-12, so error is ~O(√tol) ≈ 1e-6.
        """
        val = complex(carlson_first(0.0, 1.0, 2.0))
        ref = complex(mpmath.elliprf(0, 1, 2))
        assert abs(val - ref) / abs(ref) < 1e-5

    def test_symmetry(self):
        """R_F is symmetric in all three arguments."""
        x, y, z = 1.0, 3.0, 5.0
        base = complex(carlson_first(x, y, z))
        assert abs(complex(carlson_first(y, x, z)) - base) < ATOL
        assert abs(complex(carlson_first(z, y, x)) - base) < ATOL
        assert abs(complex(carlson_first(x, z, y)) - base) < ATOL

    def test_two_zeros_gives_inf(self):
        """R_F(0, 0, 1) → ∞."""
        val = complex(carlson_first(0.0, 0.0, 1.0))
        assert math.isinf(abs(val))

    def test_complex_inputs_match_mpmath(self):
        """R_F for complex arguments matches mpmath.elliprf."""
        x, y, z = 1.0 + 0.5j, 2.0 - 0.3j, 3.0 + 0.1j
        val = complex(carlson_first(x, y, z))
        ref = complex(mpmath.elliprf(x, y, z))
        assert abs(val - ref) / abs(ref) < RTOL


# ─────────────────────────────────────────────────────────────────────────────
# agm
# ─────────────────────────────────────────────────────────────────────────────

class TestAgm:
    def test_equal_arguments_identity(self):
        """agm(a, a) = a for real a."""
        for a in [1.0, 2.5, 0.3]:
            result = complex(agm(a, a))
            assert abs(result - a) < ATOL

    def test_unit_inputs(self):
        """agm(1, 1) = 1."""
        assert abs(complex(agm(1.0, 1.0)) - 1.0) < ATOL

    def test_known_value_1_sqrt2_matches_mpmath(self):
        """agm(1, √2) matches mpmath.agm(1, √2)."""
        a, b = 1.0, math.sqrt(2)
        val = complex(agm(a, b))
        ref = complex(mpmath.agm(a, b))
        assert abs(val - ref) / abs(ref) < RTOL

    def test_homogeneity(self):
        """agm(λa, λb) = λ · agm(a, b)."""
        a, b, lam = 1.0, 2.0, 3.0
        base = complex(agm(a, b))
        scaled = complex(agm(lam * a, lam * b))
        assert abs(scaled - lam * base) / abs(lam * base) < RTOL

    def test_symmetry(self):
        """agm(a, b) = agm(b, a)."""
        a, b = 1.5, 2.7
        assert abs(complex(agm(a, b)) - complex(agm(b, a))) < ATOL

    def test_default_b_is_one(self):
        """agm(a) with no b argument equals agm(a, 1.0)."""
        a = 3.0
        with_default = complex(agm(a))
        ref = complex(mpmath.agm(a, 1.0))
        assert abs(with_default - ref) / abs(ref) < RTOL
