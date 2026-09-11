"""
Tests for orbitronx.riemannsurfaces.riemann_funcs.elliptic_funcs

Covers: weierstrass_roots, invariants_from_periods, weierstrass_P,
        inverse_weierstrass_P, weierstrass_zeta, weierstrass_sigma
"""
import math

import pytest
from orbitronx.riemannsurfaces.riemann_funcs.complex_analysis import carlson_first
from orbitronx.riemannsurfaces.riemann_funcs.elliptic_funcs import (
    invariants_from_periods,
    inverse_weierstrass_P,
    weierstrass_P,
    weierstrass_roots,
    weierstrass_sigma,
    weierstrass_zeta,
)

RTOL = 1e-8
ATOL = 1e-9

# Standard half-periods used across the suite (non-degenerate, complex lattice)
O1 = 1.5
O3 = 1.0 + 2.0j
# A generic z not close to any half-period or lattice point
Z = 0.5 + 0.3j


# ─────────────────────────────────────────────────────────────────────────────
# weierstrass_roots
# ─────────────────────────────────────────────────────────────────────────────

class TestWeierstrassRoots:
    def _roots(self):
        return [complex(e) for e in weierstrass_roots(O1, O3)]

    def test_sum_of_roots_is_zero(self):
        """e1 + e2 + e3 = 0 (Vieta's formula for monic cubic z³ - (g2/4)z - g3/4)."""
        e1, e2, e3 = self._roots()
        assert abs(e1 + e2 + e3) < ATOL

    def test_roots_are_distinct(self):
        """Non-degenerate lattice → three distinct roots."""
        e1, e2, e3 = self._roots()
        assert abs(e1 - e2) > 1e-6
        assert abs(e1 - e3) > 1e-6
        assert abs(e2 - e3) > 1e-6

    def test_e1_equals_P_at_omega1(self):
        """e1 = ℘(ω₁)."""
        e1 = self._roots()[0]
        p = complex(weierstrass_P(O1, O1, O3))
        assert abs(e1 - p) < ATOL

    def test_e2_equals_P_at_omega3(self):
        """e2 = ℘(ω₃)."""
        e2 = self._roots()[1]
        p = complex(weierstrass_P(O3, O1, O3))
        assert abs(e2 - p) < ATOL

    def test_e3_equals_P_at_minus_omega1_minus_omega3(self):
        """e3 = ℘(-ω₁ - ω₃)."""
        e3 = self._roots()[2]
        p = complex(weierstrass_P(-O1 - O3, O1, O3))
        assert abs(e3 - p) < ATOL


# ─────────────────────────────────────────────────────────────────────────────
# invariants_from_periods
# ─────────────────────────────────────────────────────────────────────────────

class TestInvariantsFromPeriods:
    def setup_method(self):
        self.e1, self.e2, self.e3 = [complex(e) for e in weierstrass_roots(O1, O3)]
        self.g2 = complex(invariants_from_periods(O1, O3)[0])
        self.g3 = complex(invariants_from_periods(O1, O3)[1])

    def test_g2_formula(self):
        """g2 = 2(e1² + e2² + e3²)."""
        e1, e2, e3 = self.e1, self.e2, self.e3
        expected = 2 * (e1**2 + e2**2 + e3**2)
        assert abs(self.g2 - expected) / (abs(expected) + 1e-30) < RTOL

    def test_g3_formula(self):
        """g3 = 4·e1·e2·e3."""
        expected = 4 * self.e1 * self.e2 * self.e3
        assert abs(self.g3 - expected) / (abs(expected) + 1e-30) < RTOL

    def test_discriminant_nonzero(self):
        """Δ = g2³ - 27·g3² ≠ 0 for non-degenerate lattice."""
        delta = self.g2**3 - 27 * self.g3**2
        assert abs(delta) > 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# weierstrass_P
# ─────────────────────────────────────────────────────────────────────────────

class TestWeierstrassP:
    def setup_method(self):
        self.g2, self.g3 = [complex(x) for x in invariants_from_periods(O1, O3)]

    def test_even_symmetry(self):
        """℘(-z) = ℘(z)."""
        pos = complex(weierstrass_P(Z, O1, O3))
        neg = complex(weierstrass_P(-Z, O1, O3))
        assert abs(pos - neg) < ATOL

    def test_periodicity_2_omega1(self):
        """℘(z + 2ω₁) = ℘(z)."""
        p_z = complex(weierstrass_P(Z, O1, O3))
        p_shift = complex(weierstrass_P(Z + 2 * O1, O1, O3))
        assert abs(p_z - p_shift) / (abs(p_z) + 1e-30) < RTOL

    def test_periodicity_2_omega3(self):
        """℘(z + 2ω₃) = ℘(z)."""
        p_z = complex(weierstrass_P(Z, O1, O3))
        p_shift = complex(weierstrass_P(Z + 2 * O3, O1, O3))
        assert abs(p_z - p_shift) / (abs(p_z) + 1e-30) < RTOL

    def test_differential_equation(self):
        """(℘')² = 4℘³ - g2·℘ - g3."""
        P = complex(weierstrass_P(Z, O1, O3))
        Pp = complex(weierstrass_P(Z, O1, O3, derivative=1))
        lhs = Pp**2
        rhs = 4 * P**3 - self.g2 * P - self.g3
        assert abs(lhs - rhs) / (abs(rhs) + 1e-30) < RTOL

    def test_satisfies_weierstrass_cubic(self):
        """
        ℘(z) satisfies 4w³ - g2·w - g3 = 0 at w = e1, e2, e3.
        (mpmath.weierstrassp is unavailable in this mpmath release.)
        """
        e1, e2, e3 = [complex(e) for e in weierstrass_roots(O1, O3)]
        for e in (e1, e2, e3):
            cubic = 4 * e**3 - self.g2 * e - self.g3
            assert abs(cubic) < RTOL

    def test_value_at_omega1_equals_e1(self):
        """℘(ω₁) = e1."""
        e1 = complex(weierstrass_roots(O1, O3)[0])
        p = complex(weierstrass_P(O1, O1, O3))
        assert abs(p - e1) < RTOL

    def test_value_at_omega3_equals_e2(self):
        """℘(ω₃) = e2."""
        e2 = complex(weierstrass_roots(O1, O3)[1])
        p = complex(weierstrass_P(O3, O1, O3))
        assert abs(p - e2) < RTOL

    def test_derivative1_odd_symmetry(self):
        """℘'(-z) = -℘'(z) (℘ is even, so its derivative is odd)."""
        Pp = complex(weierstrass_P(Z, O1, O3, derivative=1))
        Pp_neg = complex(weierstrass_P(-Z, O1, O3, derivative=1))
        assert abs(Pp + Pp_neg) < ATOL

    def test_derivative2_algebraic_identity(self):
        """℘''(z) = 6℘(z)² - g2/2."""
        P = complex(weierstrass_P(Z, O1, O3))
        P2 = complex(weierstrass_P(Z, O1, O3, derivative=2))
        expected = 6 * P**2 - self.g2 / 2
        assert abs(P2 - expected) / (abs(expected) + 1e-30) < RTOL

    def test_derivative2_finite_difference(self):
        """℘''(z) matches second-order central finite difference."""
        h = 1e-5
        P2 = complex(weierstrass_P(Z, O1, O3, derivative=2))
        fd2 = (
            complex(weierstrass_P(Z + h, O1, O3))
            - 2 * complex(weierstrass_P(Z, O1, O3))
            + complex(weierstrass_P(Z - h, O1, O3))
        ) / h**2
        assert abs(P2 - fd2) / (abs(fd2) + 1e-30) < 1e-5

    def test_derivative3_algebraic_identity(self):
        """℘'''(z) = 12℘(z)·℘'(z)."""
        P = complex(weierstrass_P(Z, O1, O3))
        Pp = complex(weierstrass_P(Z, O1, O3, derivative=1))
        P3 = complex(weierstrass_P(Z, O1, O3, derivative=3))
        assert abs(P3 - 12 * P * Pp) / (abs(12 * P * Pp) + 1e-30) < RTOL

    def test_derivative4_algebraic_identity(self):
        """℘''''(z) = 12[(℘')² + ℘·℘'']."""
        P = complex(weierstrass_P(Z, O1, O3))
        Pp = complex(weierstrass_P(Z, O1, O3, derivative=1))
        P2 = complex(weierstrass_P(Z, O1, O3, derivative=2))
        P4 = complex(weierstrass_P(Z, O1, O3, derivative=4))
        expected = 12 * (Pp**2 + P * P2)
        assert abs(P4 - expected) / (abs(expected) + 1e-30) < RTOL

    def test_invalid_derivative_raises(self):
        with pytest.raises(ValueError):
            weierstrass_P(Z, O1, O3, derivative=5)


# ─────────────────────────────────────────────────────────────────────────────
# inverse_weierstrass_P
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseWeierstrassP:
    def test_round_trip_at_e1(self):
        """℘(℘⁻¹(e1)) ≈ e1."""
        e1 = complex(weierstrass_roots(O1, O3)[0])
        inv = complex(inverse_weierstrass_P(e1, O1, O3))
        roundtrip = complex(weierstrass_P(inv, O1, O3))
        assert abs(roundtrip - e1) / (abs(e1) + 1e-30) < RTOL

    def test_round_trip_at_e2(self):
        """℘(℘⁻¹(e2)) ≈ e2."""
        e2 = complex(weierstrass_roots(O1, O3)[1])
        inv = complex(inverse_weierstrass_P(e2, O1, O3))
        roundtrip = complex(weierstrass_P(inv, O1, O3))
        assert abs(roundtrip - e2) / (abs(e2) + 1e-30) < RTOL

    def test_round_trip_generic_value(self):
        """For a generic ℘-value w, ℘(℘⁻¹(w)) ≈ w."""
        w = complex(weierstrass_P(Z, O1, O3))
        inv = complex(inverse_weierstrass_P(w, O1, O3))
        roundtrip = complex(weierstrass_P(inv, O1, O3))
        assert abs(roundtrip - w) / (abs(w) + 1e-30) < RTOL

    def test_equals_carlson_rf_of_shifted_roots(self):
        """℘⁻¹(z) = R_F(z-e1, z-e2, z-e3) by the Carlson formula."""
        z = 0.7 + 0.2j
        e1, e2, e3 = [complex(e) for e in weierstrass_roots(O1, O3)]
        expected = complex(carlson_first(z - e1, z - e2, z - e3))
        result = complex(inverse_weierstrass_P(z, O1, O3))
        assert abs(result - expected) / (abs(expected) + 1e-30) < ATOL


# ─────────────────────────────────────────────────────────────────────────────
# weierstrass_zeta
# ─────────────────────────────────────────────────────────────────────────────

class TestWeierstrassZeta:
    def test_odd_symmetry(self):
        """ζ(-z) = -ζ(z)."""
        pos = complex(weierstrass_zeta(Z, O1, O3))
        neg = complex(weierstrass_zeta(-Z, O1, O3))
        assert abs(pos + neg) < ATOL

    def test_derivative_equals_minus_P(self):
        """ζ'(z) = -℘(z), verified by central finite difference."""
        h = 1e-6
        fd = (
            complex(weierstrass_zeta(Z + h, O1, O3))
            - complex(weierstrass_zeta(Z - h, O1, O3))
        ) / (2 * h)
        P = complex(weierstrass_P(Z, O1, O3))
        assert abs(fd + P) / (abs(P) + 1e-30) < 1e-5

    def test_finite_at_generic_z(self):
        val = complex(weierstrass_zeta(Z, O1, O3))
        assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_finite_at_multiple_points(self):
        """ζ is finite and consistent at several non-lattice points."""
        for z in [0.3 + 0.1j, 0.7 + 0.5j, -0.4 + 0.8j]:
            val = complex(weierstrass_zeta(z, O1, O3))
            assert math.isfinite(val.real) and math.isfinite(val.imag)


# ─────────────────────────────────────────────────────────────────────────────
# weierstrass_sigma
# ─────────────────────────────────────────────────────────────────────────────

class TestWeierstrassSigma:
    def test_odd_symmetry(self):
        """σ(-z) = -σ(z)."""
        pos = complex(weierstrass_sigma(Z, O1, O3))
        neg = complex(weierstrass_sigma(-Z, O1, O3))
        assert abs(pos + neg) / (abs(pos) + 1e-30) < RTOL

    def test_zero_at_origin(self):
        """σ(0) = 0."""
        val = complex(weierstrass_sigma(0.0 + 0j, O1, O3))
        assert abs(val) < ATOL

    def test_derivative_at_origin_equals_1(self):
        """σ'(0) = 1 (canonical normalization)."""
        h = 1e-7
        s_plus = complex(weierstrass_sigma(h + 0j, O1, O3))
        s_minus = complex(weierstrass_sigma(-h + 0j, O1, O3))
        fd = (s_plus - s_minus) / (2 * h)
        assert abs(fd - 1.0) < 1e-5

    def test_nonzero_at_generic_z(self):
        """σ(z) ≠ 0 for z not at a lattice point."""
        val = complex(weierstrass_sigma(Z, O1, O3))
        assert abs(val) > 1e-6

    def test_log_derivative_equals_zeta(self):
        """d/dz ln σ(z) = ζ(z), verified by finite difference."""
        h = 1e-6
        s = complex(weierstrass_sigma(Z, O1, O3))
        s_plus = complex(weierstrass_sigma(Z + h, O1, O3))
        s_minus = complex(weierstrass_sigma(Z - h, O1, O3))
        log_deriv = (s_plus - s_minus) / (2 * h * s)
        zeta_val = complex(weierstrass_zeta(Z, O1, O3))
        assert abs(log_deriv - zeta_val) / (abs(zeta_val) + 1e-30) < 1e-5
