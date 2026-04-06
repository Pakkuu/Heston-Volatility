#!/usr/bin/env python
"""
Heston Model Calibration to Option Prices

YouTube Tutorial (Published: Mar 25, 2022): https://youtu.be/Jy4_AVEyO0w

Heston's Stochastic Volatility Model implementation with calibration
to market option prices using characteristic functions approach.

References:
- Heston Girsanov's Formula: https://quant.stackexchange.com/questions/61927
- Heston PDE: https://uwspace.uwaterloo.ca/bitstream/handle/10012/7541/Ye_Ziqun.pdf
- Heston Characteristic Eq: https://www.maths.univ-evry.fr/pages_perso/crepey/Finance/051111_mikh%20heston.pdf
- Heston Implementation: https://hal.sorbonne-universite.fr/hal-02273889/document
- Heston Calibration: https://calebmigosi.medium.com/build-the-heston-model-from-scratch-in-python-part-ii
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import NonlinearConstraint
from datetime import datetime as dt, date, timedelta, timezone

from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols


# =============================================================================
# Part 1: Heston Characteristic Function
# =============================================================================

def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, tau, r):
    """
    Compute the Heston model characteristic function.
    
    Parameters:
    -----------
    phi : float or array
        Characteristic function argument
    S0 : float
        Initial asset price
    v0 : float
        Initial variance
    kappa : float
        Mean reversion rate of variance process
    theta : float
        Long-term mean variance
    sigma : float
        Volatility of volatility
    rho : float
        Correlation between variance and stock process
    tau : float
        Time to maturity
    r : float
        Risk-free interest rate
        
    Returns:
    --------
    complex
        Value of the characteristic function
    """
    # Constants
    a = kappa * theta
    b = kappa
    rspi = rho * sigma * phi * 1j

    # Define d parameter (re(d) >= 0)
    d = np.sqrt((rho * sigma * phi * 1j - b)**2 + (phi * 1j + phi**2) * sigma**2)

    # Define g parameter (stable formulation uses -d)
    g = (b - rspi - d) / (b - rspi + d)

    # Calculate characteristic function components (Numerically Stable Version)
    # Using the e^(-d*tau) formulation to avoid overflow and branch cut issues
    exp_minus_d_tau = np.exp(-d * tau)
    
    # D term (coefficient of v0)
    D = ((b - rspi - d) / sigma**2) * ((1 - exp_minus_d_tau) / (1 - g * exp_minus_d_tau))
    
    # C term (constant/linear in tau)
    # Using the Albrecher et al. (2007) stable formulation for the logarithm
    C = (r * phi * 1j * tau) + (a / sigma**2) * (
        (b - rspi - d) * tau - 2 * np.log((1 - g * exp_minus_d_tau) / (1 - g))
    )
    
    result = S0**(phi * 1j) * np.exp(C + D * v0)
    
    # Numerical guard: replace NaN/inf with 0 to prevent optimizer corruption
    if np.isscalar(result):
        if not np.isfinite(result):
            return 0.0 + 0.0j
    else:
        result = np.where(np.isfinite(result), result, 0.0 + 0.0j)
    
    return result


# =============================================================================
# Part 2: Carr-Madan FFT Option Pricing (replaces rectangular integration)
# =============================================================================

def heston_price_fft(S0, K, v0, kappa, theta, sigma, rho, tau, r,
                     N=12, alpha=1.5, eta=0.25):
    """
    Price European calls via Carr-Madan (1999) IFFT method.
    
    Prices ALL strikes simultaneously using np.fft.ifft, replacing the old
    rectangular integration loop that ran 10,000 iterations per option.
    
    Parameters:
    -----------
    S0 : float
        Initial asset price
    K : float or array
        Strike price(s) to price
    v0, kappa, theta, sigma, rho : float
        Heston model parameters
    tau : float
        Time to maturity (single maturity per call)
    r : float
        Risk-free rate
    N : int
        Power of 2 for FFT grid size (2^N points). Default 12 -> 4096.
    alpha : float
        Carr-Madan damping factor. Default 1.5 for calls.
    eta : float
        Spacing in frequency domain. Default 0.25.
        
    Returns:
    --------
    np.ndarray
        European call option price(s)
    """
    n_points = 2**N
    
    # Frequency-domain grid: v_j = j * eta
    v = np.arange(n_points) * eta
    
    # Log-strike grid spacing and bounds
    lda = 2 * np.pi / (n_points * eta)       # log-strike spacing (lambda)
    b = n_points * lda / 2                    # half-width of log-strike grid
    ku = -b + lda * np.arange(n_points) + np.log(S0)  # centered on log(S0)
    
    # Carr-Madan modified characteristic function psi(v)
    # psi(v) = exp(-r*tau) * phi(v - (alpha+1)*i) / (alpha^2 + alpha - v^2 + i*(2*alpha+1)*v)
    cf_values = heston_charfunc(v - (alpha + 1) * 1j,
                                S0, v0, kappa, theta, sigma, rho, tau, r)
    denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
    psi = np.exp(-r * tau) * cf_values / denom
    
    # Simpson's rule weights: [1, 4, 2, 4, 2, ..., 4] / 3
    sw = 3 + (-1) ** np.arange(1, n_points + 1)
    sw[0] = 1
    sw = sw / 3
    
    # Build IFFT input with phase shift
    # Using identity: Re[FFT(x)] = N * Re[IFFT(conj(x))]
    # where x = exp(i*v*b) * psi * eta * sw  (standard Carr-Madan FFT input)
    x = np.exp(1j * v * b) * psi * eta * sw
    
    # Inverse FFT: frequency space -> strike space
    payoff = np.real(np.fft.ifft(np.conj(x))) * n_points / np.pi
    
    # Apply damping factor to recover call prices on the grid
    call_prices_grid = np.exp(-alpha * ku) * payoff
    
    # Interpolate from grid to requested strikes
    K_arr = np.atleast_1d(np.asarray(K, dtype=float))
    log_K = np.log(K_arr)
    prices = np.interp(log_K, ku, call_prices_grid)
    
    return prices if prices.shape[0] > 1 else prices[0]


def heston_price_quad(S0, K, v0, kappa, theta, sigma, rho, tau, r, alpha=1.5):
    """
    Price a single European call using scipy.integrate.quad (Carr-Madan integrand).
    
    Useful for sanity-checking individual prices against the FFT output.
    
    Parameters:
    -----------
    S0, K, v0, kappa, theta, sigma, rho, tau, r : float
        Model and option parameters
    alpha : float
        Carr-Madan damping factor (default 1.5)
        
    Returns:
    --------
    float
        European call option price
    """
    def _integrand(v):
        cf = heston_charfunc(v - (alpha + 1) * 1j,
                             S0, v0, kappa, theta, sigma, rho, tau, r)
        denom = alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v
        psi = np.exp(-r * tau) * cf / denom
        return np.real(np.exp(-1j * v * np.log(K)) * psi)
    
    integral_val, _ = quad(_integrand, 0, 100)
    
    return np.exp(-alpha * np.log(K)) * integral_val / np.pi


# =============================================================================
# Part 4: Yield Curve Calibration (Live FRED Data + Nelson-Siegel-Svensson)
# =============================================================================

# FRED series IDs mapped to their maturities in years
_FRED_SERIES = {
    'DGS1MO': 1/12,
    'DGS3MO': 3/12,
    'DGS6MO': 6/12,
    'DGS1':   1,
    'DGS2':   2,
    'DGS5':   5,
    'DGS10':  10,
    'DGS30':  30,
}


def calibrate_yield_curve():
    """
    Calibrate a yield curve using the Nelson-Siegel-Svensson model
    with live US Treasury Par Yield Curve rates from FRED.
    
    Fetches the most recent daily rates for 8 standard tenors from
    FRED's public CSV endpoints (no API key required), then fits
    an NSS curve via ordinary least squares.
    
    Falls back to hardcoded rates if FRED is unreachable.
    
    Returns:
    --------
    NelsonSiegelSvenssonCurve
        Fitted yield curve object that can be called with maturity to get rate
    """
    yield_maturities = []
    yields_decimal = []
    
    try:
        for series_id, maturity in _FRED_SERIES.items():
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            df = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')
            # Take the most recent non-NaN value
            latest = df.dropna().iloc[-1, 0]
            yield_maturities.append(maturity)
            yields_decimal.append(float(latest) / 100.0)  # percentage -> decimal
        
        print(f"  Fetched live Treasury yields for {len(yields_decimal)} tenors from FRED")
    except Exception as e:
        print(f"  WARNING: FRED fetch failed ({e}), using fallback yields")
        yield_maturities = [1/12, 3/12, 6/12, 1, 2, 5, 10, 30]
        yields_decimal = [0.0372, 0.0365, 0.0358, 0.0347, 0.0347, 0.0374, 0.0419, 0.0486]
    
    yield_maturities = np.array(yield_maturities)
    yields_decimal = np.array(yields_decimal)
    
    # Calibrate NSS model using ordinary least squares
    curve_fit, status = calibrate_nss_ols(yield_maturities, yields_decimal)
    
    return curve_fit


# =============================================================================
# Part 5: Market Data Fetching (Massive API — Live SPX Options)
# =============================================================================

def fetch_and_process_market_data():
    """
    Fetch real SPX call options from the Massive API snapshot endpoint
    and return processed data ready for Heston calibration.
    
    Reads the MASSIVE_API_KEY environment variable automatically.
    Paginates through all results for I:SPX calls expiring between
    today and one year from today, then applies quality filters.
    
    Returns:
    --------
    tuple
        (S0, DataFrame) where S0 is the underlying spot price and
        DataFrame has columns: maturity, strike, price, weight
    """
    from massive import RESTClient
    
    client = RESTClient()  # reads MASSIVE_API_KEY from env
    
    today = date.today()
    one_year = today + timedelta(days=365)
    
    # Paginate through all call options on I:SPX
    contracts = []
    for o in client.list_snapshot_options_chain(
        "I:SPX",
        params={
            "contract_type": "call",
            "expiration_date.gte": today.strftime("%Y-%m-%d"),
            "expiration_date.lte": one_year.strftime("%Y-%m-%d"),
        },
    ):
        contracts.append(o)
    
    print(f"  Raw contracts fetched: {len(contracts)}")
    
    if not contracts:
        raise RuntimeError("No option contracts returned from Massive API")
    
    # Extract spot price from underlying asset
    S0 = contracts[0].underlying_asset.price
    
    rows = []
    for c in contracts:
        bid    = c.last_quote.bid
        ask    = c.last_quote.ask
        strike = c.details.strike_price
        exp    = c.details.expiration_date
        
        mid    = (bid + ask) / 2.0
        spread = ask - bid
        tau    = (dt.strptime(exp, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                  - dt.now(timezone.utc)).days / 365.25
        
        # Quality filters
        if spread <= 0 or bid <= 0:
            continue
        if tau < 0.04 or tau > 2.0:
            continue
        if strike < S0 * 0.70 or strike > S0 * 1.15:
            continue
        
        rows.append({
            "maturity": tau,
            "strike":   strike,
            "price":    mid,
            "weight":   1.0 / spread,
        })
    
    df = pd.DataFrame(rows)
    return S0, df


# =============================================================================
# Part 6: Calibration Optimization
# =============================================================================

# Default parameter configuration for calibration
DEFAULT_PARAMS = {
    "v0": {"x0": 0.1, "lbub": [1e-3, 0.1]},
    "kappa": {"x0": 3, "lbub": [1e-3, 5]},
    "theta": {"x0": 0.05, "lbub": [1e-3, 0.1]},
    "sigma": {"x0": 0.3, "lbub": [1e-2, 1]},
    "rho": {"x0": -0.8, "lbub": [-1, 0]},
}


def create_objective_function(S0, K, tau, r, P, weights=None):
    """
    Create the squared error objective function for calibration.
    
    Parameters:
    -----------
    S0 : float
        Spot price
    K : array
        Strike prices
    tau : array
        Times to maturity
    r : array
        Risk-free rates
    P : array
        Market option prices
    weights : array, optional
        Per-option weights (e.g. 1/bid-ask spread). None = equal weighting.
        
    Returns:
    --------
    callable
        Objective function that takes parameter vector x and returns squared error
    """
    def SqErr(x):
        v0, kappa, theta, sigma, rho = x
        
        # FFT prices all strikes for a single maturity at once.
        # Group by unique (tau, r) pairs and run one FFT per maturity.
        heston_prices = np.empty_like(P)
        for t_val in np.unique(tau):
            mask = tau == t_val
            r_val = r[mask][0]  # same maturity -> same rate
            K_subset = K[mask]
            prices = heston_price_fft(S0, K_subset, v0, kappa, theta,
                                      sigma, rho, t_val, r_val)
            heston_prices[mask] = np.atleast_1d(prices)
        
        # Weighted mean squared error
        if weights is not None:
            err = np.sum(weights * (P - heston_prices)**2) / np.sum(weights)
        else:
            err = np.sum((P - heston_prices)**2 / len(P))
        
        return err
    
    return SqErr


def calibrate_heston(S0, K, tau, r, P, weights=None, params=None, verbose=True):
    """
    Calibrate Heston model parameters to market option prices.
    
    Uses two-phase optimization:
      1. differential_evolution (global search) to find a good basin
      2. SLSQP (local refinement) to fine-tune from the DE result
    
    Parameters:
    -----------
    S0 : float
        Spot price
    K : array
        Strike prices
    tau : array
        Times to maturity  
    r : array
        Risk-free rates
    P : array
        Market option prices
    params : dict, optional
        Parameter configuration with initial values and bounds
    weights : array, optional
        Per-option weights (e.g. 1/bid-ask spread). None = equal weighting.
    verbose : bool
        Whether to print optimization progress
        
    Returns:
    --------
    dict
        Calibrated parameters and optimization result
    """
    if params is None:
        params = DEFAULT_PARAMS
    
    # Extract initial values and bounds
    x0 = [param["x0"] for key, param in params.items()]
    bnds = [param["lbub"] for key, param in params.items()]
    
    # Create objective function
    SqErr = create_objective_function(S0, K, tau, r, P, weights=weights)
    
    # Feller condition: 2*kappa*theta > sigma^2
    # Guarantees variance stays positive; prevents CF blowups
    feller_constraint_slsqp = [{'type': 'ineq', 'fun': lambda x: 2 * x[1] * x[2] - x[3]**2}]
    feller_constraint_de = NonlinearConstraint(lambda x: 2 * x[1] * x[2] - x[3]**2, 0, np.inf)
    
    if verbose:
        print("Starting calibration...")
    
    # Phase 1: Global search via differential evolution
    if verbose:
        print("  Phase 1: Differential evolution (global search)...")
    
    de_result = differential_evolution(
        SqErr, bounds=bnds,
        seed=42, maxiter=100, tol=1e-5,
        polish=False,  # we polish with SLSQP below
        constraints=feller_constraint_de
    )
    
    if verbose:
        print(f"    DE converged: {de_result.success}, error: {de_result.fun:.6f}")
        print(f"  Phase 2: SLSQP (local refinement)...")
    
    # Phase 2: Local refinement from DE solution
    result = minimize(SqErr, de_result.x, tol=1e-3, method='SLSQP',
                     options={'maxiter': int(1e4)}, bounds=bnds,
                     constraints=feller_constraint_slsqp)
    
    # Extract calibrated parameters
    v0, kappa, theta, sigma, rho = result.x
    
    calibrated = {
        'v0': v0,
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma,
        'rho': rho,
        'optimization_result': result
    }
    
    if verbose:
        print(f"\nCalibration {'succeeded' if result.success else 'failed'}!")
        print(f"  Final error: {result.fun:.6f}")
        print(f"  Iterations: {result.nit}")
        print(f"\nCalibrated parameters:")
        print(f"  v0     = {v0:.6f}  (initial variance)")
        print(f"  kappa  = {kappa:.6f}  (mean reversion rate)")
        print(f"  theta  = {theta:.6f}  (long-term variance)")
        print(f"  sigma  = {sigma:.6f}  (vol of vol)")
        print(f"  rho    = {rho:.6f}  (correlation)")
    
    return calibrated


# =============================================================================
# Part 7: Visualization
# =============================================================================

def plot_calibration_results(tau_arr, K_arr, market_prices, heston_prices, 
                              output_file='calibration_results.html'):
    """
    Create 3D visualization comparing market prices vs calibrated Heston prices.
    
    Parameters:
    -----------
    tau_arr : array
        Times to maturity
    K_arr : array
        Strike prices
    market_prices : array
        Market option prices
    heston_prices : array
        Calibrated Heston model prices
    output_file : str
        Path to save HTML file (None to skip saving)
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The 3D figure object
    """
    import plotly.graph_objects as go
    
    # Create 3D mesh for market prices
    fig = go.Figure()
    
    # Add market prices as mesh surface
    fig.add_trace(go.Mesh3d(
        x=tau_arr,
        y=K_arr,
        z=market_prices,
        color='mediumblue',
        opacity=0.55,
        name='Market Prices'
    ))
    
    # Add Heston prices as scatter markers
    fig.add_trace(go.Scatter3d(
        x=tau_arr,
        y=K_arr,
        z=heston_prices,
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Heston Prices'
    ))
    
    # Update layout
    fig.update_layout(
        title_text='Market Prices (Mesh) vs Calibrated Heston Prices (Markers)',
        scene=dict(
            xaxis_title='TIME (Years)',
            yaxis_title='STRIKES (Pts)',
            zaxis_title='OPTION PRICE (Pts)'
        ),
        height=800,
        width=800
    )
    
    # Save to HTML file
    if output_file:
        fig.write_html(output_file)
        print(f"Visualization saved to: {output_file}")
    
    return fig


def plot_error_surface(tau_arr, K_arr, errors, output_file='pricing_errors.html'):
    """
    Create 3D visualization of pricing errors.
    
    Parameters:
    -----------
    tau_arr : array
        Times to maturity
    K_arr : array
        Strike prices
    errors : array
        Pricing errors (market - heston)
    output_file : str
        Path to save HTML file
        
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Color by error magnitude
    fig.add_trace(go.Scatter3d(
        x=tau_arr,
        y=K_arr,
        z=errors,
        mode='markers',
        marker=dict(
            size=8,
            color=errors,
            colorscale='RdBu',
            colorbar=dict(title='Error'),
            cmin=-max(abs(errors)),
            cmax=max(abs(errors))
        ),
        name='Pricing Errors'
    ))
    
    fig.update_layout(
        title_text='Heston Model Pricing Errors',
        scene=dict(
            xaxis_title='TIME (Years)',
            yaxis_title='STRIKES (Pts)',
            zaxis_title='ERROR (Pts)'
        ),
        height=700,
        width=800
    )
    
    if output_file:
        fig.write_html(output_file)
        print(f"Error visualization saved to: {output_file}")
    
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Heston Model Calibration Pipeline (Live SPX Data)")
    print("=" * 60)
    
    # 1. Fetch & process market data from Massive API
    print("\n" + "-" * 60)
    print("1. Fetching SPX Options from Massive API")
    print("-" * 60)
    
    S0, df = fetch_and_process_market_data()
    print(f"  Spot (S0): {S0:.2f}")
    print(f"  Contracts after filtering: {len(df)}")
    
    K_arr   = df["strike"].values
    tau_arr = df["maturity"].values
    P_arr   = df["price"].values
    w_arr   = df["weight"].values
    
    # 2. Calibrate yield curve from live Treasury data
    print("\n" + "-" * 60)
    print("2. Calibrating Yield Curve (Live FRED Data)")
    print("-" * 60)
    
    curve = calibrate_yield_curve()
    r_arr = np.array([curve(t) for t in tau_arr])
    print(f"  1-year rate: {curve(1.0)*100:.4f}%")
    
    # 3. Calibrate Heston model
    print("\n" + "-" * 60)
    print("3. Calibrating Heston Model")
    print("-" * 60 + "\n")
    
    calibrated = calibrate_heston(S0, K_arr, tau_arr, r_arr, P_arr, weights=w_arr)
    
    # 4. Compute Calibrated Prices and Statistics
    print("\n" + "-" * 60)
    print("4. Computing Calibrated Prices")
    print("-" * 60)
    
    cal_prices = np.empty_like(P_arr)
    for t_val in np.unique(tau_arr):
        mask = tau_arr == t_val
        r_val = r_arr[mask][0]
        cal_prices[mask] = np.atleast_1d(heston_price_fft(
            S0, K_arr[mask],
            calibrated['v0'], calibrated['kappa'], calibrated['theta'],
            calibrated['sigma'], calibrated['rho'],
            t_val, r_val
        ))
    
    # Summary statistics
    errors = P_arr - cal_prices
    print(f"\nPricing error statistics:")
    print(f"  Mean error:     {np.mean(errors):>10.4f}")
    print(f"  Mean abs error: {np.mean(np.abs(errors)):>10.4f}")
    print(f"  Max abs error:  {np.max(np.abs(errors)):>10.4f}")
    print(f"  RMSE:           {np.sqrt(np.mean(errors**2)):>10.4f}")
    
    # 5. Create Visualizations
    print("\n" + "-" * 60)
    print("5. Creating Visualizations")
    print("-" * 60 + "\n")
    
    fig1 = plot_calibration_results(tau_arr, K_arr, P_arr, cal_prices)
    fig2 = plot_error_surface(tau_arr, K_arr, errors)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - calibration_results.html (3D price comparison)")
    print("  - pricing_errors.html (error surface)")
    print("\nOpen the HTML files in a browser to view interactive 3D plots.")
