import numpy as np
from scipy.optimize import least_squares


class HestonCalibrator:
    def __init__(self, S0: float, r: float, q: float = 0.0):
        self.S0 = S0
        self.r = r
        self.q = q
        self.heston = HestonModel()
        self.bs = BlackScholes()
        self.iv_surf = ImpliedVolSurface()

    def _residuals(
        self,
        params: np.ndarray,
        strikes: np.ndarray,
        expiries: np.ndarray,
        market_ivs: np.ndarray,
    ) -> np.ndarray:
        """
        Objective: minimize sum of squared (model_IV - market_IV)².
        params = [kappa, theta, xi, rho, v0] (all in natural units)
        """
        kappa, theta, xi, rho, v0 = params
        p = HestonParams(kappa, theta, xi, rho, v0)
        resids = []

        for K, T, iv_mkt in zip(strikes, expiries, market_ivs):
            try:
                price_heston = self.heston.price_call(self.S0, K, T, self.r, p)
                opt_inv = Option(self.S0, K, T, self.r, 0.2, self.q, "call")
                iv_model = self.iv_surf.implied_vol(price_heston, opt_inv)
                resids.append(iv_model - iv_mkt)
            except:
                resids.append(1.0)  # large penalty for failed evaluations

        return np.array(resids)

    def calibrate(self, iv_surface_df: pd.DataFrame) -> HestonParams:
        """
        Fit Heston params to observed IV surface via nonlinear least-squares.
        iv_surface_df: columns [strike, expiry, iv]
        """
        K = iv_surface_df["strike"].values
        T = iv_surface_df["expiry"].values
        ivs = iv_surface_df["iv"].values

        # Initial guess and bounds
        x0 = [2.0, 0.04, 0.5, -0.7, 0.04]
        lower = [0.1, 0.001, 0.01, -0.99, 0.001]
        upper = [15.0, 1.0, 2.0, 0.99, 1.0]

        result = least_squares(
            self._residuals,
            x0,
            args=(K, T, ivs),
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",  # robust to outlier IV quotes
            max_nfev=500,
        )

        kappa, theta, xi, rho, v0 = result.x
        fitted = HestonParams(kappa, theta, xi, rho, v0)

        # Verify Feller condition
        feller = 2 * kappa * theta > xi**2
        rmse_iv = np.sqrt((result.fun**2).mean())
        print(f"Calibration RMSE: {rmse_iv*100:.3f} vol points")
        print(f"Feller condition: {'satisfied' if feller else 'VIOLATED'}")
        print(f"Fitted: κ={kappa:.3f} θ={theta:.4f} ξ={xi:.3f} ρ={rho:.3f} v₀={v0:.4f}")

        return fitted
