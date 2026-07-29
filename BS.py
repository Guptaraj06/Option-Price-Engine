from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class Option:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    option_type: str = "call"


class BlackScholes:
    def _d1d2(self, opt: Option) -> tuple:
        d1 = (
            np.log(opt.S / opt.K) + (opt.r - opt.q + 0.5 * opt.sigma**2) * opt.T
        ) / (opt.sigma * np.sqrt(opt.T))
        d2 = d1 - opt.sigma * np.sqrt(opt.T)
        return d1, d2

    def price(self, opt: Option) -> float:
        if opt.T <= 0:
            if opt.option_type == "call":
                return max(opt.S - opt.K, 0)
            return max(opt.K - opt.S, 0)

        d1, d2 = self._d1d2(opt)
        disc = np.exp(-opt.r * opt.T)
        fwd = opt.S * np.exp(-opt.q * opt.T)

        if opt.option_type == "call":
            return fwd * norm.cdf(d1) - opt.K * disc * norm.cdf(d2)
        else:
            return opt.K * disc * norm.cdf(-d2) - fwd * norm.cdf(-d1)

    def put_call_parity_check(self, call: Option, put: Option) -> bool:
        lhs = self.price(call) - self.price(put)
        rhs = call.S * np.exp(-call.q * call.T) - call.K * np.exp(-call.r * call.T)
        return abs(lhs - rhs) < 1e-10

    def greeks(self, opt: Option) -> dict:
        d1, d2 = self._d1d2(opt)
        sqrtT = np.sqrt(opt.T)
        disc = np.exp(-opt.r * opt.T)
        fwd = opt.S * np.exp(-opt.q * opt.T)
        phi_d1 = norm.pdf(d1)

        sign = 1 if opt.option_type == "call" else -1
        Nd1 = norm.cdf(sign * d1)
        Nd2 = norm.cdf(sign * d2)

        delta = sign * np.exp(-opt.q * opt.T) * norm.cdf(sign * d1)
        gamma = (np.exp(-opt.q * opt.T) * phi_d1) / (opt.S * opt.sigma * sqrtT)
        vega = fwd * phi_d1 * sqrtT / 100
        theta = (
            -(fwd * phi_d1 * opt.sigma) / (2 * sqrtT)
            - (sign * opt.r * opt.K * disc * Nd2)
            + (sign * opt.q * fwd * Nd1)
        ) / 365
        rho = sign * opt.K * opt.T * disc * Nd2 / 100

        vanna = -np.exp(-opt.q * opt.T) * phi_d1 * d2 / opt.sigma
        volga = fwd * phi_d1 * sqrtT * d1 * d2 / opt.sigma
        charm = (
            -np.exp(-opt.q * opt.T)
            * phi_d1
            * (2 * (opt.r - opt.q) * sqrtT - d2 * opt.sigma)
            / (2 * opt.T * opt.sigma * sqrtT)
        ) / 365

        return {
            "price": self.price(opt),
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "vega": round(vega, 6),
            "theta": round(theta, 6),
            "rho": round(rho, 6),
            "vanna": round(vanna, 6),
            "volga": round(volga, 6),
            "charm": round(charm, 6),
        }
