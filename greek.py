class GreeksBumpAndReprice:
    def __init__(self, pricer, dS=0.01, dsigma=0.0001, dr=0.0001, dT=1 / 365):
        self.pricer = pricer
        self.dS = dS
        self.dsigma = dsigma
        self.dr = dr
        self.dT = dT

    def delta(self, opt: Option) -> float:
        up = self.pricer(Option(**{**vars(opt), "S": opt.S + self.dS}))
        down = self.pricer(Option(**{**vars(opt), "S": opt.S - self.dS}))
        return (up - down) / (2 * self.dS)

    def gamma(self, opt: Option) -> float:
        mid = self.pricer(opt)
        up = self.pricer(Option(**{**vars(opt), "S": opt.S + self.dS}))
        down = self.pricer(Option(**{**vars(opt), "S": opt.S - self.dS}))
        return (up - 2 * mid + down) / self.dS**2

    def vega(self, opt: Option) -> float:
        up = self.pricer(Option(**{**vars(opt), "sigma": opt.sigma + self.dsigma}))
        down = self.pricer(Option(**{**vars(opt), "sigma": opt.sigma - self.dsigma}))
        return (up - down) / (2 * self.dsigma)

    def theta(self, opt: Option) -> float:
        if opt.T <= self.dT:
            return 0.0
        fwd = self.pricer(Option(**{**vars(opt), "T": opt.T - self.dT}))
        return (fwd - self.pricer(opt)) / self.dT

    def all_greeks(self, opt: Option) -> dict:
        return {
            "price": self.pricer(opt),
            "delta": self.delta(opt),
            "gamma": self.gamma(opt),
            "vega": self.vega(opt),
            "theta": self.theta(opt),
        }
