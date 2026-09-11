"""
Tests for orbitronx.riemannsurfaces.integrations.integrate_hyperelliptic

Covers: int_genus2_real_exp, myint_genus2, int_genus2_complex_exp,
        int_genus2_complex, int_genus2_first

These tests exercise the mpmath-based quadrature in this module and
cross-check numerical agreement against independent mpmath.quad reference
computations of the same differentials.
"""
import cmath
import math

import mpmath
import torch
from mpmath import matrix
from orbitronx.riemannsurfaces.integrations.integrate_hyperelliptic import (
    int_genus2_complex,
    int_genus2_complex_exp,
    int_genus2_first,
    int_genus2_real_exp,
    myint_genus2,
)

RTOL = 1e-6
ATOL = 1e-8

# The number of digits used across these tests -- passed as the <digits>
# argument every genus-2 integration function requires.
DIGITS = 20

# A quintic with 5 distinct real roots.
ZEROS_REAL = [-3.0, -1.0, 0.5, 2.0, 4.0]

# A quintic with one complex-conjugate pair and 3 real roots.
ZEROS_COMPLEX = [1.0 - 2.0j, 1.0 + 2.0j, -3.0, -1.0, 4.0]

# A quintic with 5 tightly-clustered real roots, spacing on the order of
# 1e-3/1e-6 -- the same order of magnitude as the roots produced by a small
# nonzero cosmological constant (e.g. cosmo = 1/3e-5) in the genus-2 radial
# equation of motion. Expanding (x - root_i) into monomial form and
# evaluating at fixed double precision loses ~11 of ~15 significant digits
# to cancellation here and previously went to NaN; see
# TestClusteredRootsRegression below.
ZEROS_CLUSTERED = [
    -0.0015005544945112700,
    -0.0014965856780957800 - 0.0000092578075318400j,
    -0.0014965856780957800 + 0.0000092578075318400j,
    -0.0014958266155973400,
    -0.0007453589264721270,
]


def _poly(zeros, t):
    """Evaluate prod(t - z) at a real mpmath point t for a list of zeros."""
    r = mpmath.mpf(1)
    for z in zeros:
        r *= t - z
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Bug A regression: torchquad precision must actually be float64/complex128,
# not silently downgraded to float32 by an invalid "complex64" data_type.
# (int_genus2_first's Case 3 branch still uses torchquad/GPU quadrature.)
# ─────────────────────────────────────────────────────────────────────────────

class TestTorchPrecisionRegression:
    def test_default_dtype_is_float64(self):
        """set_up_backend must actually configure float64 (not silently fall
        back to float32, which happens if an invalid data_type is passed)."""
        assert torch.get_default_dtype() == torch.float64


# ─────────────────────────────────────────────────────────────────────────────
# Bug C regression: integrating between two zeros adjacent in the full zero
# ordering (the exact pattern used throughout periods_genus2_first.py).
# ─────────────────────────────────────────────────────────────────────────────

class TestMyintGenus2AdjacentZeros:
    def test_no_exception_and_finite(self):
        result = myint_genus2(ZEROS_REAL, ZEROS_REAL[0], ZEROS_REAL[1], 1, DIGITS)
        for entry in result:
            val = complex(entry)
            assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_nontrivial_result(self):
        """The integral should not degenerate to zero."""
        result = myint_genus2(ZEROS_REAL, ZEROS_REAL[0], ZEROS_REAL[1], 1, DIGITS)
        assert abs(complex(result[0])) > 1e-6
        assert abs(complex(result[1])) > 1e-6

    def test_all_adjacent_pairs_succeed(self):
        """Every adjacent pair of real zeros should integrate without error."""
        for i in range(len(ZEROS_REAL) - 1):
            result = myint_genus2(
                ZEROS_REAL, ZEROS_REAL[i], ZEROS_REAL[i + 1], 1, DIGITS
            )
            for entry in result:
                val = complex(entry)
                assert math.isfinite(val.real) and math.isfinite(val.imag)


# ─────────────────────────────────────────────────────────────────────────────
# Bug B/D/F regression: base case (one bound is a zero, the other is not).
# Cross-checked against an independent mpmath.quad reference.
# ─────────────────────────────────────────────────────────────────────────────

class TestIntGenus2RealExpBaseCase:
    def test_finite(self):
        val = complex(int_genus2_real_exp(ZEROS_REAL, -2.0, -1.0, 0, 1, DIGITS))
        assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_matches_mpmath_exponent0(self):
        val = complex(int_genus2_real_exp(ZEROS_REAL, -2.0, -1.0, 0, 1, DIGITS))
        ref = mpmath.quad(
            lambda t: 1 / mpmath.sqrt(abs(_poly(ZEROS_REAL, t))), [-2.0, -1.0]
        )
        assert abs(abs(val) - float(ref)) / float(ref) < 1e-5

    def test_matches_mpmath_exponent1(self):
        val = complex(int_genus2_real_exp(ZEROS_REAL, -2.0, -1.0, 1, 1, DIGITS))
        ref = mpmath.quad(
            lambda t: abs(t) / mpmath.sqrt(abs(_poly(ZEROS_REAL, t))), [-2.0, -1.0]
        )
        assert abs(abs(val) - float(ref)) / float(ref) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Bug E regression: int_genus2_first Case 3 (neither bound is a real zero,
# integrand may be negative throughout the interval and needs a complex sqrt).
# ─────────────────────────────────────────────────────────────────────────────

class TestIntGenus2FirstCase3:
    # A blank period matrix is fine here: this interval never reaches the
    # ub == oo sub-branch, so period_matrix is unused.
    PERIOD_MATRIX = matrix(2, 4)

    def test_case3_finite(self):
        """lb, ub strictly inside a gap, touching no zero."""
        result = int_genus2_first(ZEROS_REAL, -0.75, 0.25, DIGITS, self.PERIOD_MATRIX)
        for entry in result:
            val = complex(entry)
            assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_case3_matches_mpmath(self):
        lb, ub = -0.75, 0.25
        result = int_genus2_first(ZEROS_REAL, lb, ub, DIGITS, self.PERIOD_MATRIX)

        ref1 = mpmath.quad(lambda t: 1 / mpmath.sqrt(_poly(ZEROS_REAL, t)), [lb, ub])
        ref2 = mpmath.quad(lambda t: t / mpmath.sqrt(_poly(ZEROS_REAL, t)), [lb, ub])

        val1 = complex(result[0])
        val2 = complex(result[1])
        assert abs(val1 - complex(ref1)) / abs(complex(ref1)) < RTOL
        assert abs(val2 - complex(ref2)) / abs(complex(ref2)) < RTOL

    def test_case1_adjacent_real_zeros_matches_mpmath(self):
        """lb, ub are themselves adjacent real zeros (Case 1, exercises the
        Bug C fix through the full int_genus2_first entry point)."""
        lb, ub = -1.0, 0.5
        result = int_genus2_first(ZEROS_REAL, lb, ub, DIGITS, self.PERIOD_MATRIX)

        ref1 = mpmath.quad(lambda t: 1 / mpmath.sqrt(_poly(ZEROS_REAL, t)), [lb, ub])
        ref2 = mpmath.quad(lambda t: t / mpmath.sqrt(_poly(ZEROS_REAL, t)), [lb, ub])

        val1 = complex(result[0])
        val2 = complex(result[1])
        assert abs(val1 - complex(ref1)) / abs(complex(ref1)) < 1e-5
        assert abs(val2 - complex(ref2)) / abs(complex(ref2)) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Complex-root path: int_genus2_complex_exp / int_genus2_complex
# ─────────────────────────────────────────────────────────────────────────────

class TestIntGenus2Complex:
    REAL_PART = 1.0
    IMA_PART = 2.0
    POSITION = 0

    def test_complex_exp_finite(self):
        a = int_genus2_complex_exp(
            ZEROS_COMPLEX, self.REAL_PART, self.IMA_PART, self.POSITION, 0, -1, DIGITS
        )
        b = int_genus2_complex_exp(
            ZEROS_COMPLEX, self.REAL_PART, self.IMA_PART, self.POSITION, 1, -1, DIGITS
        )
        assert cmath.isfinite(a)
        assert cmath.isfinite(b)

    def test_complex_finite(self):
        result = int_genus2_complex(
            ZEROS_COMPLEX, self.REAL_PART, self.IMA_PART, self.POSITION, -1, DIGITS
        )
        for entry in result:
            val = complex(entry)
            assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_complex_consistent_with_complex_exp(self):
        """int_genus2_complex combines the exponent-0/1 int_genus2_complex_exp
        results as documented: [a, realPart * a + 1j * b]."""
        a = int_genus2_complex_exp(
            ZEROS_COMPLEX, self.REAL_PART, self.IMA_PART, self.POSITION, 0, -1, DIGITS
        )
        b = int_genus2_complex_exp(
            ZEROS_COMPLEX, self.REAL_PART, self.IMA_PART, self.POSITION, 1, -1, DIGITS
        )
        result = int_genus2_complex(
            ZEROS_COMPLEX, self.REAL_PART, self.IMA_PART, self.POSITION, -1, DIGITS
        )
        assert abs(complex(result[0]) - a) < ATOL
        assert abs(complex(result[1]) - (self.REAL_PART * a + 1j * b)) < ATOL


# ─────────────────────────────────────────────────────────────────────────────
# Regression: tightly-clustered/near-degenerate roots (the cosmo != 0 crash).
#
# Expanding (x - root_i) into monomial coefficients and evaluating sqrt() of
# the result at fixed double precision loses almost all significant digits
# here and previously raised "TypeError: cannot create mpf from nan" via
# int_genus2_real_exp/int_genus2_complex_exp. mpmath's arbitrary-precision
# quad (restored below, matching the pre-GPU-migration implementation)
# resolves the same cancellation cleanly instead.
# ─────────────────────────────────────────────────────────────────────────────

class TestClusteredRootsRegression:
    def test_real_exp_no_longer_nan(self):
        """This exact (lb, ub, exponent) triple previously produced NaN."""
        lb = ZEROS_CLUSTERED[3]
        ub = ZEROS_CLUSTERED[4]
        for exponent in (0, 1):
            val = complex(
                int_genus2_real_exp(ZEROS_CLUSTERED, lb, ub, exponent, 1, DIGITS)
            )
            assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_real_exp_matches_mpmath(self):
        lb = ZEROS_CLUSTERED[3]
        ub = ZEROS_CLUSTERED[4]
        val = complex(int_genus2_real_exp(ZEROS_CLUSTERED, lb, ub, 0, 1, DIGITS))
        ref = mpmath.quad(
            lambda t: 1 / mpmath.sqrt(abs(_poly(ZEROS_CLUSTERED, t))), [lb, ub]
        )
        assert abs(abs(val) - float(ref)) / float(ref) < 1e-4

    def test_myint_genus2_finite(self):
        """The full myint_genus2 entry point (as called by
        periods_genus2_first.py) must also stay finite for clustered roots."""
        result = myint_genus2(
            ZEROS_CLUSTERED, ZEROS_CLUSTERED[3], ZEROS_CLUSTERED[4], 1, DIGITS
        )
        for entry in result:
            val = complex(entry)
            assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_complex_exp_finite(self):
        """int_genus2_complex_exp (the analogous complex-root path) must also
        stay finite for the clustered-root configuration."""
        rea = ZEROS_CLUSTERED[1].real
        ima = abs(ZEROS_CLUSTERED[1].imag)
        for exponent in (0, 1):
            val = int_genus2_complex_exp(
                ZEROS_CLUSTERED, rea, ima, 1, exponent, -1, DIGITS
            )
            assert cmath.isfinite(val)
