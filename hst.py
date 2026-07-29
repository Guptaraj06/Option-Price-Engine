from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad


@dataclass
class HestonParams:
    kappa: float  # mean reversion speed
    theta: float  # long-run variance
    xi: float  # vol-of-vol
    rho: float  # spot-vol correlation
    v0: float  # initial variance


class HestonModel:
    def char_fn(
        self, u: complex, S0: float, K: float, T: float, r: float, p: HestonParams
    ) -> complex:
        """Heston characteristic function φ(u) — Albrecher et al. formulation."""
        iu = 1j * u
        d = np.sqrt((p.rho * p.xi * iu - p.kappa) ** 2 + p.xi**2 * (iu + u**2))
        g = (p.kappa - p.rho * p.xi * iu - d) / (p.kappa - p.rho * p.xi * iu + d)

        # Stable formulation (avoids branch cuts)
        exp_dT = np.exp(-d * T)
        C = (p.kappa * p.theta / p.xi**2) * (
            (p.kappa - p.rho * p.xi * iu - d) * T
            - 2 * np.log((1 - g * exp_dT) / (1 - g))
        )
        D = (
            (p.kappa - p.rho * p.xi * iu - d)
            / p.xi**2
            * (1 - exp_dT)
            / (1 - g * exp_dT)
        )

        return np.exp(C + D * p.v0 + iu * np.log(S0 * np.exp(r * T)))

    def price_call(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        p: HestonParams,
        alpha: float = 1.5,
    ) -> float:
        """Carr-Madan FFT pricing. alpha: dampening factor (1 < alpha < 2 typical)."""
        k = np.log(K)

        def integrand(u):
            psi = (
                np.exp(-r * T)
                * self.char_fn(u - (alpha + 1) * 1j, S0, K, T, r, p)
                / (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u)
            )
            return np.real(np.exp(-1j * u * k) * psi)

        integral, _ = quad(integrand, 0, 100, limit=200)
        return np.exp(-alpha * k) / np.pi * integral

    def simulate_milstein(
        self,
        S0: float,
        T: float,
        r: float,
        p: HestonParams,
        n_paths: int = 50_000,
        n_steps: int = 252,
    ) -> tuple:
        """
        Milstein scheme for Heston SDE — more accurate than Euler for CIR.
        Full truncation scheme to handle v_t ≤ 0 (discretization artifact).
        """
        dt = T / n_steps
        sqdt = np.sqrt(dt)
        rng = np.random.default_rng(42)

        S = np.full(n_paths, S0, dtype=float)
        v = np.full(n_paths, p.v0, dtype=float)
        S_paths = np.zeros((n_steps + 1, n_paths))
        S_paths[0] = S

        for i in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = p.rho * Z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)

            v_pos = np.maximum(v, 0)  # full truncation scheme
            sv = np.sqrt(v_pos)

            # Milstein correction for CIR (improves convergence vs Euler)
            dv = (
                p.kappa * (p.theta - v_pos) * dt
                + p.xi * sv * sqdt * Z2
                + 0.25 * p.xi**2 * dt * (Z2**2 - 1)
            )

            # Log-Euler for stock price (avoids negative prices)
            dlogS = (r - 0.5 * v_pos) * dt + sv * sqdt * Z1
            S = S * np.exp(dlogS)
            v = v + dv

            S_paths[i + 1] = S

        return S_paths, v  # full path history + final variance

    def price_from_paths(
        self,
        S_paths: np.ndarray,
        K: float,
        r: float,
        T: float,
        option_type: str = "call",
    ) -> float:
        ST = S_paths[-1]
        if option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        return float(np.exp(-r * T) * payoffs.mean())
