import numpy as np
import pandas as pd
import yfinance as yf

if __name__ == "__main__":
    # 1. Load underlying and options data
    ticker = "SPY"
    S0 = yf.download(ticker, period="5d", auto_adjust=True)["Close"].iloc[-1]
    r = 0.053  # approximate risk-free rate
    q = 0.013  # SPY dividend yield

    # Download option chain for nearest 3 expiries
    spy = yf.Ticker(ticker)
    chain_data = []
    for exp in spy.options[:4]:
        chain = spy.option_chain(exp)
        T = (pd.Timestamp(exp) - pd.Timestamp.today()).days / 365
        for df, otype in [(chain.calls, "call"), (chain.puts, "put")]:
            df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
            df["mid_price"] = (df["bid"] + df["ask"]) / 2
            df["expiry"] = T
            df["option_type"] = otype
            chain_data.append(df[["strike", "expiry", "mid_price", "option_type"]])
    chain_df = pd.concat(chain_data).reset_index(drop=True)

    # 2. Build implied vol surface
    iv_engine = ImpliedVolSurface()
    surface = iv_engine.build_surface(chain_df, S0, r, q)
    print(surface.groupby("expiry")[["iv"]].describe())

    # 3. Price same option across all methods and compare
    test_opt = Option(S=S0, K=S0, T=0.25, r=r, sigma=0.18, q=q, option_type="call")

    bs_price = BlackScholes().price(test_opt)
    bs_greeks = BlackScholes().greeks(test_opt)

    mc = MonteCarloPricer(n_paths=200_000)
    mc_result = mc.price_european(test_opt, method="antithetic")

    fd = FiniteDifferencePricer(N_S=300, N_t=500)
    fd_result = fd.price(test_opt, american=False)

    print(f"\nPrice comparison (ATM Call, T=0.25yr, σ=18%):")
    print(f"  Black-Scholes: {bs_price:.4f}")
    print(f"  Monte Carlo:   {mc_result['price']:.4f} ± {mc_result['se']:.4f}")
    print(f"  Finite Diff:   {fd_result['price']:.4f}")

    # 4. Calibrate Heston to market surface
    # Filter to ATM region for robust calibration
    surface_fit = surface[
        (surface["moneyness_k"].abs() < 0.3) & (surface["iv"].between(0.05, 1.5))
    ]
    calib = HestonCalibrator(S0, r, q)
    heston_params = calib.calibrate(surface_fit)

    # 5. Price exotic with MC — Asian call
    asian_result = mc.price_asian(test_opt)
    print(f"\nAsian call: {asian_result['price_cv']:.4f}")
    print(f"Variance reduction from CV: {asian_result['var_reduction']*100:.1f}%")

    # 6. American put via FD (early exercise premium)
    put_euro = fd.price(Option(S0, S0, 0.25, r, 0.18, q, "put"), american=False)
    put_amer = fd.price(Option(S0, S0, 0.25, r, 0.18, q, "put"), american=True)
    ee_premium = put_amer["price"] - put_euro["price"]
    print(f"\nEarly exercise premium: {ee_premium:.4f}")
