import numpy as np
from scipy.stats.qmc import Sobol


class MonteCarloPricer:
    def __init__(self, n_paths=100_000, n_steps=252, seed=42):
        self.N = n_paths
        self.steps = n_steps
        self.rng = np.random.default_rng(seed)

    # ── EUROPEAN OPTIONS ──────────────────────────────────────
    def price_european(self, opt: Option, method: str = "antithetic") -> dict:
        """Vanilla European call/put via MC."""
        T, S0, K = opt.T, opt.S, opt.K
        r, q, sig = opt.r, opt.q, opt.sigma

        if method == "antithetic":
            Z = self.rng.standard_normal(self.N // 2)
            ST_pos = S0 * np.exp((r - q - 0.5 * sig**2) * T + sig * np.sqrt(T) * Z)
            ST_neg = S0 * np.exp((r - q - 0.5 * sig**2) * T - sig * np.sqrt(T) * Z)
            ST = np.concatenate([ST_pos, ST_neg])

        elif method == "sobol":
            # Quasi-Monte Carlo: Sobol sequence → Normal via inverse CDF
            sampler = Sobol(d=1, scramble=True, seed=42)
            u = sampler.random(self.N).flatten()
            Z = norm.ppf(np.clip(u, 1e-10, 1 - 1e-10))
            ST = S0 * np.exp((r - q - 0.5 * sig**2) * T + sig * np.sqrt(T) * Z)

        else:  # standard
            Z = self.rng.standard_normal(self.N)
            ST = S0 * np.exp((r - q - 0.5 * sig**2) * T + sig * np.sqrt(T) * Z)

        if opt.option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        disc_payoffs = np.exp(-r * T) * payoffs
        price = disc_payoffs.mean()
        se = disc_payoffs.std() / np.sqrt(self.N)

        return {
            "price": price,
            "se": se,
            "ci_95": (price - 1.96 * se, price + 1.96 * se),
        }

    # ── PATH-DEPENDENT: ASIAN OPTION ───────────────────────────
    def price_asian(self, opt: Option, avg_type: str = "arithmetic") -> dict:
        """
        Asian call/put with control variate.
        Geometric average Asian has closed-form → perfect control variate.
        """
        r, q, sig = opt.r, opt.q, opt.sigma
        S0, K, T = opt.S, opt.K, opt.T
        dt = T / self.steps

        # Simulate full paths
        Z = self.rng.standard_normal((self.steps, self.N))
        # Antithetic on full paths
        Z = np.concatenate([Z, -Z], axis=1)
        N_total = self.N * 2

        log_S = np.log(S0) + np.cumsum(
            (r - q - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z, axis=0
        )
        S_paths = np.exp(log_S)  # shape: (steps, N_total)

        # Arithmetic and geometric averages
        A_avg = S_paths.mean(axis=0)
        G_avg = np.exp(log_S.mean(axis=0))

        if opt.option_type == "call":
            payoff_A = np.maximum(A_avg - K, 0)
            payoff_G = np.maximum(G_avg - K, 0)
        else:
            payoff_A = np.maximum(K - A_avg, 0)
            payoff_G = np.maximum(K - G_avg, 0)

        # Exact geometric Asian price (Kemna-Vorst closed form)
        sig_g = sig * np.sqrt((2 * self.steps + 1) / (6 * (self.steps + 1)))
        r_g = 0.5 * (r - q - sig**2 / 2 + sig_g**2)
        opt_g = Option(S0, K, T, r, sig_g, q, opt.option_type)
        G_exact = BlackScholes().price(
            Option(S0, K, T, r_g + sig_g**2 / 2, sig_g, 0, opt.option_type)
        )

        # Control variate adjustment
        disc = np.exp(-r * T)
        G_mc = disc * payoff_G.mean()
        beta_opt = np.cov(payoff_A, payoff_G)[0, 1] / np.var(payoff_G)
        price_cv = disc * payoff_A.mean() - beta_opt * (G_mc - disc * G_exact / disc)

        se_raw = (disc * payoff_A).std() / np.sqrt(N_total)
        resid = payoff_A - beta_opt * payoff_G
        se_cv = (disc * resid).std() / np.sqrt(N_total)

        return {
            "price_raw": disc * payoff_A.mean(),
            "price_cv": price_cv,
            "se_raw": se_raw,
            "se_cv": se_cv,
            "var_reduction": 1 - (se_cv / se_raw) ** 2,
        }

    # ── BARRIER OPTION ─────────────────────────────────────────
    def price_barrier(
        self, opt: Option, barrier: float, barrier_type: str = "down-and-out"
    ) -> float:
        """Down-and-out / up-and-in barrier options — path-dependent."""
        r, q, sig = opt.r, opt.q, opt.sigma
        S0, K, T = opt.S, opt.K, opt.T
        dt = T / self.steps

        Z = self.rng.standard_normal((self.steps, self.N))
        log_S = np.log(S0) + np.cumsum(
            (r - q - 0.5 * sig**2) * dt + sig * np.sqrt(dt) * Z, axis=0
        )
        S_paths = np.exp(log_S)

        ST = S_paths[-1]
        payoff = np.maximum(ST - K, 0)

        if barrier_type == "down-and-out":
            knocked_out = S_paths.min(axis=0) <= barrier
            payoff[knocked_out] = 0
        elif barrier_type == "down-and-in":
            knocked_in = S_paths.min(axis=0) <= barrier
            payoff[~knocked_in] = 0
        elif barrier_type == "up-and-out":
            knocked_out = S_paths.max(axis=0) >= barrier
            payoff[knocked_out] = 0

        return float(np.exp(-r * T) * payoff.mean())
