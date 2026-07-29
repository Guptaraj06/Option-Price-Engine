import pandas as pd


class ImpliedVolSurface:
    def __init__(self):
        self.bs = BlackScholes()

    def implied_vol(
        self, market_price: float, opt: Option, tol: float = 1e-8, max_iter: int = 100
    ) -> float:
        disc = np.exp(-opt.r * opt.T)
        if opt.Option_type == "call":
            lower = max(opt.S * np.exp(-opt.q * opt.T) - opt.K * disc, 0)
        else:
            lower = max(opt.K * disc - opt.S * np.exp(-opt.q * opt.T), 0)

        if market_price <= lower + 1e-10:
            return np.nan

        sigma = market_price / (opt.S * np.sqrt(opt.T / (2 * np.pi)))
        sigma = np.clip(sigma, 1e-4, 5.0)

        for _ in range(max_iter):
            opt_trial = Option(
                opt.S, opt.K, opt.T, opt.r, sigma, opt.q, opt.option_type
            )
            price_trial = self.bs.price(opt_trial)
            greeks = self.bs.greeks(opt_trial)
            vega = greeks["vega"] * 100

            if abs(vega) < 1e-12:
                break

            diff = price_trial - market_price
            sigma = sigma - diff / vega
            sigma = np.clip(sigma, 1e-4, 5.0)

            if abs(diff) < tol:
                break

        return float(sigma)

    def build_surface(
        self, option_chain: pd.DataFrame, S0: float, r: float, q: float = 0.0
    ) -> pd.DataFrame:
        ivs = []
        for _, row in option_chain.iterrows():
            opt = Option(
                S=S0,
                K=row["strike"],
                T=row["expiry"],
                r=r,
                sigma=0.2,
                q=q,
                option_type=row["option_type"],
            )
            iv = self.implied_vol(row["mid_price"], opt)
            ivs.append(
                {
                    "strike": row["strike"],
                    "expiry": row["expiry"],
                    "moneyness_k": np.log(
                        row["strike"] / S0 * np.exp(r * row["expiry"])
                    ),
                    "iv": iv,
                    "total_var": iv**2 * row["expiry"] if not np.isnan(iv) else np.nan,
                    "option_type": row["option_type"],
                }
            )
            return pd.DataFrame(ivs).dropna(subset=["iv"])

    def svi_fit(self, k: np.ndarray, tot_var: np.ndarray) -> dict:

        from scipy.optimize import curve_fit

        def svi(k, a, b, rho, m, sigma):
            return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))

        p0 = [tot_var.mean(), 0.1, -0.7, 0.0, 0.1]
        bounds = ([0, 0, -1, -1, 0], [1, 1, 1, 1, 1])

        popt, _ = curve_fit(
            svi, k, tot_var, p0=p0, bounds=bounds, maxfev=10000, method="trf"
        )
        a, b, rho, m, sigma = popt

        k_test = np.linspace(k.min(), k.max(), 500)
        w = svi(k_test, *popt)
        d2w = np.gradient(np.gradient(w, k_test), k_test)
        butterfly_free = bool((d2w >= -1e-8).all())

        return {
            "params": {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma},
            "butterfly_free": butterfly_free,
            "svi_fn": lambda kk: svi(kk, *popt),
        }
