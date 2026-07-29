import numpy as np
from scipy.linalg import solve_banded


class FiniteDIfferencePricer:
    def __init__(self, N_S: int = 200, N_t: int = 400, S_max_mult: float = 4.0):
        self.N_S = N_S
        self.N_t = N_t
        self.S_mult = S_max_mult

    def _build_cn_matrices(self, S_grid, sigma, r, dt):
        N = len(S_grid)
        dS = S_grid[1] - S_grid[0]

        alpha = 0.25 * dt * (sigma**2 * S_grid**2 / dS**2 - r * S_grid / dS)
        beta = -0.5 * dt * (sigma**2 * S_grid**2 / dS**2 + r)
        gamma = 0.25 * dt * (sigma**2 * S_grid**2 / dS**2 + r * S_grid / dS)

        A_sub = -alpha[1:]
        A_diag = 1 - beta
        A_sup = -gamma[:-1]

        B_sub = alpha[1:]
        B_diag = 1 + beta
        B_sup = gamma[:-1]

        A_band = np.zeros((3, N))
        A_band[0, 1:] = A_sup
        A_band[1, :] = A_diag
        A_band[2, :-1] = A_sub

        return A_band, B_sub, B_diag, B_sup

    def price(self, opt: Option, american: bool = False) -> dict:

        S_max = self.S_mult * opt.K
        S_grid = np.linspace(0, S_max, self.N_S + 1)
        dt = opt.T / self.N_t
        dS = S_grid[1] - S_grid[0]

        if opt.option_type == "call":
            V = np.maximum(S_grid - opt.K, 0)
            intrinsic = V.copy()
        else:
            V = np.maximum(opt.K - S_grid, 0)
            intrinsic = V.copy()

        A_band, B_sub, B_diag, B_sup = self._build_cn_matrices(
            S_grid, opt.sigma, opt.r, dt
        )

        for _ in range(self.N_t):
            rhs = B_diag * V
            rhs[1:] += B_sub * V[:-1]
            rhs[:-1] += B_sup * V[1:]

            if opt.option_type == "call":
                rhs[0] = 0
                rhs[-1] = S_max - opt.K * np.exp(-opt.r * dt)
            else:
                rhs[0] = opt.K * np.exp(-opt.r * dt)
                rhs[-1] = 0

            V_new = solve_banded((1, 1), A_band, rhs)

            if american:
                V_new = np.maximum(V_new, intrinsic)

            V = V_new

            idx = np.searchsorted(S_grid, opt.S)
            idx = np.clip(idx, 1, self.N_S - 1)
            w = (opt.S - S_grid[idx - 1]) / dS
            price = (1 - w) * V[idx - 1] + w * V[idx]

            delta = (V[idx + 1] - V[idx - 1]) / (2 * dS)
            gamma = (V[idx + 1] - 2 * V[idx] + V[idx - 1]) / dS**2

            return {
                "price": price,
                "delta": delta,
                "gamma": gamma,
                "S_grid": S_grid,
                "V_grid": V,
                "american": american,
            }
