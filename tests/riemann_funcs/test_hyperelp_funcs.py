"""
Tests for orbitronx.riemannsurfaces.riemann_funcs.hyperelp_funcs

Covers: derivative_factor, hyp_theta_fourier, hyp_theta_RR, hyp_theta_IR,
        sigma1, sigma2
"""
import math

import jax.numpy as jnp
import pytest
from orbitronx.riemannsurfaces.riemann_funcs.hyperelp_funcs import (
    derivative_factor,
    hyp_theta_fourier,
    hyp_theta_IR,
    hyp_theta_RR,
    sigma1,
    sigma2,
)

# Looser tolerance: 2D Fourier sums have more truncation error than 1D series
RTOL = 1e-6
ATOL = 1e-8

# Genus-2 Riemann matrix with positive-definite imaginary part
RM = [[2j, 0.5j], [0.5j, 3j]]

# Characteristics hardcoded in sigma1 / sigma2
CHAR_HALF = [[0.5, 0.5], [0.0, 0.5]]

# Zero characteristic
CHAR_ZERO = [[0.0, 0.0], [0.0, 0.0]]

# A generic point in ℂ²
Z = [0.3 + 0.1j, 0.2 + 0.05j]


# ─────────────────────────────────────────────────────────────────────────────
# derivative_factor
# ─────────────────────────────────────────────────────────────────────────────

class TestDerivativeFactor:
    def test_empty_tuple_gives_one(self):
        """No derivatives → (2πi)⁰ = 1."""
        assert derivative_factor(()) == 1 + 0j

    def test_none_gives_one(self):
        """None → (2πi)⁰ = 1."""
        assert derivative_factor(None) == 1 + 0j

    def test_single_derivative_gives_2pi_i(self):
        """One non-zero entry → (2πi)¹."""
        val = derivative_factor((1,))
        expected = 2 * math.pi * 1j
        assert abs(val - expected) < ATOL

    def test_two_derivatives_gives_minus_4pi_sq(self):
        """Two non-zero entries → (2πi)² = -4π²."""
        val = derivative_factor((1, 2))
        expected = (2 * math.pi * 1j) ** 2
        assert abs(val - expected) < ATOL

    def test_three_derivatives(self):
        """Three non-zero entries → (2πi)³ = -8π²i."""
        val = derivative_factor((1, 2, 1))
        expected = (2 * math.pi * 1j) ** 3
        assert abs(val - expected) < ATOL

    def test_zero_entries_not_counted(self):
        """Entries equal to 0 do not increase the derivative count."""
        val_with_zero = derivative_factor((1, 0, 2))
        val_without = derivative_factor((1, 2))
        assert abs(val_with_zero - val_without) < ATOL


# ─────────────────────────────────────────────────────────────────────────────
# hyp_theta_fourier
# ─────────────────────────────────────────────────────────────────────────────

class TestHypThetaFourier:
    def test_minmax_below_5_raises(self):
        with pytest.raises(ValueError):
            hyp_theta_fourier(Z, RM, CHAR_ZERO, minMax=4)

    def test_minmax_above_30_raises(self):
        with pytest.raises(ValueError):
            hyp_theta_fourier(Z, RM, CHAR_ZERO, minMax=31)

    def test_scalar_z_returns_scalar(self):
        """1D input z of length 2 should return a 0-d JAX array."""
        result = hyp_theta_fourier(Z, RM, CHAR_ZERO)
        assert jnp.asarray(result).ndim == 0

    def test_batched_z_consistent_with_scalar(self):
        """Batched evaluation matches two independent scalar calls."""
        z1 = [0.3 + 0.1j, 0.2 + 0.05j]
        z2 = [0.1 + 0.2j, 0.4 + 0.1j]
        batch = jnp.array([z1, z2])
        result_batch = hyp_theta_fourier(batch, RM, CHAR_ZERO)
        r1 = complex(hyp_theta_fourier(z1, RM, CHAR_ZERO))
        r2 = complex(hyp_theta_fourier(z2, RM, CHAR_ZERO))
        assert abs(complex(result_batch[0]) - r1) < ATOL
        assert abs(complex(result_batch[1]) - r2) < ATOL

    def test_no_derivative_finite(self):
        """Basic evaluation produces a finite complex number."""
        val = complex(hyp_theta_fourier(Z, RM, CHAR_ZERO))
        assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_derivative1_finite(self):
        val = complex(hyp_theta_fourier(Z, RM, CHAR_HALF, derivatives=(1,)))
        assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_derivative2_finite(self):
        val = complex(hyp_theta_fourier(Z, RM, CHAR_HALF, derivatives=(2,)))
        assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_derivative1_matches_finite_difference_z1(self):
        """∂θ/∂z₁ matches central finite difference in the z₁ direction."""
        h = 1e-5
        z_plus  = [Z[0] + h, Z[1]]
        z_minus = [Z[0] - h, Z[1]]
        fd = (
            complex(hyp_theta_fourier(z_plus,  RM, CHAR_HALF))
            - complex(hyp_theta_fourier(z_minus, RM, CHAR_HALF))
        ) / (2 * h)
        deriv = complex(hyp_theta_fourier(Z, RM, CHAR_HALF, derivatives=(1,)))
        assert abs(deriv - fd) / (abs(fd) + 1e-30) < 1e-4

    def test_derivative2_matches_finite_difference_z2(self):
        """∂θ/∂z₂ matches central finite difference in the z₂ direction."""
        h = 1e-5
        z_plus  = [Z[0], Z[1] + h]
        z_minus = [Z[0], Z[1] - h]
        fd = (
            complex(hyp_theta_fourier(z_plus,  RM, CHAR_HALF))
            - complex(hyp_theta_fourier(z_minus, RM, CHAR_HALF))
        ) / (2 * h)
        deriv = complex(hyp_theta_fourier(Z, RM, CHAR_HALF, derivatives=(2,)))
        assert abs(deriv - fd) / (abs(fd) + 1e-30) < 1e-4

    def test_sigma1_equals_fourier_with_deriv1(self):
        """sigma1(z, R) == hyp_theta_fourier(z, R, CHAR_HALF, derivatives=(1,))."""
        ref = complex(hyp_theta_fourier(Z, RM, CHAR_HALF, derivatives=(1,)))
        val = complex(sigma1(Z, RM))
        assert abs(val - ref) / (abs(ref) + 1e-30) < RTOL

    def test_sigma2_equals_fourier_with_deriv2(self):
        """sigma2(z, R) == hyp_theta_fourier(z, R, CHAR_HALF, derivatives=(2,))."""
        ref = complex(hyp_theta_fourier(Z, RM, CHAR_HALF, derivatives=(2,)))
        val = complex(sigma2(Z, RM))
        assert abs(val - ref) / (abs(ref) + 1e-30) < RTOL

    def test_minmax_convergence(self):
        """Result at minMax=20 is very close to minMax=10 (exponential convergence)."""
        val10 = complex(hyp_theta_fourier(Z, RM, CHAR_ZERO, minMax=10))
        val20 = complex(hyp_theta_fourier(Z, RM, CHAR_ZERO, minMax=20))
        assert abs(val20 - val10) / (abs(val20) + 1e-30) < 1e-8


# ─────────────────────────────────────────────────────────────────────────────
# hyp_theta_RR and hyp_theta_IR
# ─────────────────────────────────────────────────────────────────────────────

class TestHypThetaRRandIR:
    """
    hyp_theta_RR: ∂Re[θ]/∂xR (l=1) or ∂Re[θ]/∂wR (l=2).
    hyp_theta_IR: ∂Im[θ]/∂xR (l=1) or ∂Im[θ]/∂wR (l=2).

    Cross-validated against finite differences of hyp_theta_fourier.
    """
    xR = float(Z[0].real)
    xI = float(Z[0].imag)
    wR = float(Z[1].real)
    wI = float(Z[1].imag)

    def _theta_re(self, xR, wR):
        z = [xR + 1j * self.xI, wR + 1j * self.wI]
        return complex(hyp_theta_fourier(z, RM, CHAR_HALF)).real

    def _theta_im(self, xR, wR):
        z = [xR + 1j * self.xI, wR + 1j * self.wI]
        return complex(hyp_theta_fourier(z, RM, CHAR_HALF)).imag

    def test_minmax_out_of_range_raises(self):
        with pytest.raises(ValueError):
            hyp_theta_RR(self.xR, self.xI, self.wR, self.wI, 1, RM, CHAR_HALF, minMax=4)

    def test_invalid_l_raises(self):
        with pytest.raises(ValueError):
            hyp_theta_RR(self.xR, self.xI, self.wR, self.wI, 3, RM, CHAR_HALF)

    def test_RR_output_is_finite(self):
        val = float(jnp.asarray(hyp_theta_RR(
            self.xR, self.xI, self.wR, self.wI, 1, RM, CHAR_HALF
        )).real)
        assert math.isfinite(val)

    def test_IR_output_is_finite(self):
        val = float(jnp.asarray(hyp_theta_IR(
            self.xR, self.xI, self.wR, self.wI, 1, RM, CHAR_HALF
        )).real)
        assert math.isfinite(val)

    def test_RR_batched_shape(self):
        """Batched scalar arrays produce output with the expected shape."""
        xR_arr = jnp.array([self.xR, self.xR + 0.1])
        xI_arr = jnp.array([self.xI, self.xI])
        wR_arr = jnp.array([self.wR, self.wR])
        wI_arr = jnp.array([self.wI, self.wI])
        result = hyp_theta_RR(xR_arr, xI_arr, wR_arr, wI_arr, 1, RM, CHAR_HALF)
        assert jnp.asarray(result).shape == (2,)

    def test_RR_l1_differs_from_l2(self):
        """∂Re[θ]/∂xR and ∂Re[θ]/∂wR are in general different."""
        v1 = float(jnp.asarray(hyp_theta_RR(
            self.xR, self.xI, self.wR, self.wI, 1, RM, CHAR_HALF
        )).real)
        v2 = float(jnp.asarray(hyp_theta_RR(
            self.xR, self.xI, self.wR, self.wI, 2, RM, CHAR_HALF
        )).real)
        assert abs(v1 - v2) > 1e-10

    def test_IR_l1_differs_from_l2(self):
        """∂Im[θ]/∂xR and ∂Im[θ]/∂wR are in general different."""
        v1 = float(jnp.asarray(hyp_theta_IR(
            self.xR, self.xI, self.wR, self.wI, 1, RM, CHAR_HALF
        )).real)
        v2 = float(jnp.asarray(hyp_theta_IR(
            self.xR, self.xI, self.wR, self.wI, 2, RM, CHAR_HALF
        )).real)
        assert abs(v1 - v2) > 1e-10

    def test_RR_l1_matches_fd(self):
        """∂Re[θ]/∂xR (l=1): compare to central FD."""
        h = 1e-5
        fd = (self._theta_re(self.xR + h, self.wR) - self._theta_re(self.xR - h, self.wR)) / (2 * h)
        val = float(jnp.asarray(hyp_theta_RR(
            self.xR, self.xI, self.wR, self.wI, 1, RM, CHAR_HALF
        )).real)
        assert abs(val - fd) / (abs(fd) + 1e-30) < 1e-4

    def test_RR_l2_matches_fd(self):
        """∂Re[θ]/∂wR (l=2): compare to central FD."""
        h = 1e-5
        fd = (self._theta_re(self.xR, self.wR + h) - self._theta_re(self.xR, self.wR - h)) / (2 * h)
        val = float(jnp.asarray(hyp_theta_RR(
            self.xR, self.xI, self.wR, self.wI, 2, RM, CHAR_HALF
        )).real)
        assert abs(val - fd) / (abs(fd) + 1e-30) < 1e-4

    def test_IR_l1_matches_fd(self):
        """∂Im[θ]/∂xR (l=1): compare to central FD."""
        h = 1e-5
        fd = (self._theta_im(self.xR + h, self.wR) - self._theta_im(self.xR - h, self.wR)) / (2 * h)
        val = float(jnp.asarray(hyp_theta_IR(
            self.xR, self.xI, self.wR, self.wI, 1, RM, CHAR_HALF
        )).real)
        assert abs(val - fd) / (abs(fd) + 1e-30) < 1e-4

    def test_IR_l2_matches_fd(self):
        """∂Im[θ]/∂wR (l=2): compare to central FD."""
        h = 1e-5
        fd = (self._theta_im(self.xR, self.wR + h) - self._theta_im(self.xR, self.wR - h)) / (2 * h)
        val = float(jnp.asarray(hyp_theta_IR(
            self.xR, self.xI, self.wR, self.wI, 2, RM, CHAR_HALF
        )).real)
        assert abs(val - fd) / (abs(fd) + 1e-30) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# sigma1 and sigma2
# ─────────────────────────────────────────────────────────────────────────────

class TestSigmaFunctions:
    def test_sigma1_finite(self):
        val = complex(sigma1(Z, RM))
        assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_sigma2_finite(self):
        val = complex(sigma2(Z, RM))
        assert math.isfinite(val.real) and math.isfinite(val.imag)

    def test_sigma1_not_equal_sigma2(self):
        """sigma1 and sigma2 compute different quantities (m₁ vs m₂ weighted)."""
        v1 = complex(sigma1(Z, RM))
        v2 = complex(sigma2(Z, RM))
        assert abs(v1 - v2) > 1e-10

    def test_sigma1_minmax_convergence(self):
        """sigma1 result barely changes between minMax=5 and minMax=10."""
        v5  = complex(sigma1(Z, RM, minMax=5))
        v10 = complex(sigma1(Z, RM, minMax=10))
        assert abs(v10 - v5) / (abs(v10) + 1e-30) < 1e-6

    def test_sigma2_minmax_convergence(self):
        """sigma2 result barely changes between minMax=5 and minMax=10."""
        v5  = complex(sigma2(Z, RM, minMax=5))
        v10 = complex(sigma2(Z, RM, minMax=10))
        assert abs(v10 - v5) / (abs(v10) + 1e-30) < 1e-6

    def test_sigma1_consistent_with_fourier_deriv1(self):
        """sigma1(z, R) == hyp_theta_fourier(z, R, CHAR_HALF, derivatives=(1,))."""
        ref = complex(hyp_theta_fourier(Z, RM, CHAR_HALF, derivatives=(1,)))
        val = complex(sigma1(Z, RM))
        assert abs(val - ref) / (abs(ref) + 1e-30) < RTOL

    def test_sigma2_consistent_with_fourier_deriv2(self):
        """sigma2(z, R) == hyp_theta_fourier(z, R, CHAR_HALF, derivatives=(2,))."""
        ref = complex(hyp_theta_fourier(Z, RM, CHAR_HALF, derivatives=(2,)))
        val = complex(sigma2(Z, RM))
        assert abs(val - ref) / (abs(ref) + 1e-30) < RTOL
