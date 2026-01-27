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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize
from datetime import datetime as dt

from eod import EodHistoricalData
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols


# =============================================================================
# Part 1: Heston Characteristic Function
# =============================================================================

def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
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
    lambd : float
        Variance risk premium
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
    b = kappa + lambd
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
    
    return S0**(phi * 1j) * np.exp(C + D * v0)


# =============================================================================
# Part 2: Integrand Function
# =============================================================================

def integrand(phi, S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """
    Compute the integrand for Heston option pricing.
    
    Parameters:
    -----------
    phi : float
        Integration variable
    S0 : float
        Initial asset price
    K : float
        Strike price
    v0, kappa, theta, sigma, rho, lambd, tau, r : float
        Heston model parameters (see heston_charfunc)
        
    Returns:
    --------
    complex
        Value of the integrand at phi
    """
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = np.exp(r * tau) * heston_charfunc(phi - 1j, *args) - K * heston_charfunc(phi, *args)
    denominator = 1j * phi * K**(1j * phi)
    return numerator / denominator


# =============================================================================
# Part 3: Heston Option Pricing Functions
# =============================================================================

def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """
    Calculate European call option price using Heston model with rectangular integration.
    
    This method uses rectangular (midpoint) integration which is faster and works
    with arrays, making it suitable for calibration.
    
    Parameters:
    -----------
    S0 : float
        Initial asset price
    K : float or array
        Strike price(s)
    v0 : float
        Initial variance
    kappa : float
        Mean reversion rate
    theta : float
        Long-term mean variance
    sigma : float
        Volatility of volatility
    rho : float
        Correlation
    lambd : float
        Variance risk premium
    tau : float or array
        Time to maturity
    r : float or array
        Risk-free rate
        
    Returns:
    --------
    float or array
        European call option price(s)
    """
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    P, umax, N = 0, 100, 10000
    dphi = umax / N  # dphi is width

    for i in range(1, N):
        # Rectangular integration using midpoint
        phi = dphi * (2 * i + 1) / 2  # midpoint to calculate height
        numerator = np.exp(r * tau) * heston_charfunc(phi - 1j, *args) - K * heston_charfunc(phi, *args)
        denominator = 1j * phi * K**(1j * phi)

        P += dphi * numerator / denominator

    return np.real((S0 - K * np.exp(-r * tau)) / 2 + P / np.pi)


def heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """
    Calculate European call option price using Heston model with scipy quad integration.
    
    This method uses scipy's adaptive quadrature for higher accuracy on single values.
    
    Parameters:
    -----------
    S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r : float
        Model and option parameters (see heston_price_rec for details)
        
    Returns:
    --------
    float
        European call option price
    """
    args = (S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)

    real_integral, err = np.real(quad(integrand, 0, 100, args=args))

    return (S0 - K * np.exp(-r * tau)) / 2 + real_integral / np.pi


# =============================================================================
# Part 4: Yield Curve Calibration (Nelson-Siegel-Svensson)
# =============================================================================

def calibrate_yield_curve(yield_maturities=None, yields=None):
    """
    Calibrate a yield curve using the Nelson-Siegel-Svensson model.
    
    Uses US Daily Treasury Par Yield Curve Rates as default data.
    Reference: https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics
    
    Parameters:
    -----------
    yield_maturities : array, optional
        Maturities in years for the yield curve points
    yields : array, optional
        Corresponding yield rates (as decimals, not percentages)
        
    Returns:
    --------
    NelsonSiegelSvenssonCurve
        Fitted yield curve object that can be called with maturity to get rate
    """
    if yield_maturities is None:
        # Default US Treasury maturities (in years)
        yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
    
    if yields is None:
        # Sample Treasury yields (converted from percentage to decimal)
        yields = np.array([0.15, 0.27, 0.50, 0.93, 1.52, 2.13, 2.32, 2.34, 2.37, 2.32, 2.65, 2.52]) / 100
    
    # Calibrate NSS model using ordinary least squares
    curve_fit, status = calibrate_nss_ols(yield_maturities, yields)
    
    return curve_fit


# =============================================================================
# Part 5: Market Data Fetching (EOD Historical Data API)
# =============================================================================

def fetch_market_data(api_key=None, symbol='GSPC.INDX'):
    """
    Fetch option market data from EOD Historical Data API.
    
    Parameters:
    -----------
    api_key : str, optional
        EOD API key. If None, will try to get from EOD_API environment variable.
    symbol : str
        Symbol to fetch options for (default: S&P500 Index)
        
    Returns:
    --------
    tuple
        (S0, market_prices) where S0 is spot price and market_prices is dict of option data
    """
    if api_key is None:
        api_key = os.environ.get('EOD_API')
    
    if api_key is None:
        raise ValueError("EOD API key not found. Set EOD_API environment variable or pass api_key.")
    
    # Create the client instance
    client = EodHistoricalData(api_key)
    
    # Fetch option data
    resp = client.get_stock_options(symbol)
    
    S0 = resp['lastTradePrice']
    market_prices = {}
    
    for i in resp['data']:
        market_prices[i['expirationDate']] = {
            'strike': [name['strike'] for name in i['options']['CALL']],
            'price': [(name['bid'] + name['ask']) / 2 for name in i['options']['CALL']]
        }
    
    return S0, market_prices


def process_market_data(S0, market_prices, curve_fit):
    """
    Process raw market data into format suitable for calibration.
    
    Parameters:
    -----------
    S0 : float
        Spot price
    market_prices : dict
        Dictionary of option prices by expiration date
    curve_fit : NelsonSiegelSvenssonCurve
        Fitted yield curve for rate interpolation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: maturity, strike, price, rate
    """
    # Find common strikes across all maturities
    all_strikes = [v['strike'] for i, v in market_prices.items()]
    common_strikes = set.intersection(*map(set, all_strikes))
    common_strikes = sorted(common_strikes)
    
    print(f"Number of common strikes: {len(common_strikes)}")
    
    # Build price matrix
    prices = []
    maturities = []
    
    for date, v in market_prices.items():
        maturities.append((dt.strptime(date, '%Y-%m-%d') - dt.today()).days / 365.25)
        price = [v['price'][i] for i, x in enumerate(v['strike']) if x in common_strikes]
        prices.append(price)
    
    # Create volatility surface DataFrame
    price_arr = np.array(prices, dtype=object)
    volSurface = pd.DataFrame(price_arr, index=maturities, columns=common_strikes)
    
    # Filter to reasonable range (maturity: 0.04-1 years, strikes: 3000-5000)
    volSurface = volSurface.iloc[
        (volSurface.index > 0.04) & (volSurface.index < 1),
        (volSurface.columns > 3000) & (volSurface.columns < 5000)
    ]
    
    # Convert to long format for calibration
    volSurfaceLong = volSurface.melt(ignore_index=False).reset_index()
    volSurfaceLong.columns = ['maturity', 'strike', 'price']
    
    # Calculate risk-free rate for each maturity
    volSurfaceLong['rate'] = volSurfaceLong['maturity'].apply(curve_fit)
    
    return volSurfaceLong


def generate_test_market_data():
    """
    Generate synthetic test market data for testing without API access.
    
    Returns:
    --------
    tuple
        (S0, r, K, tau, P) arrays for calibration testing
    """
    S0 = 4500.0  # Approximate S&P 500 level
    
    # Generate synthetic option data
    strikes = np.array([4000, 4200, 4400, 4500, 4600, 4800, 5000])
    maturities = np.array([0.1, 0.25, 0.5, 0.75])
    
    # Use Heston model to generate "market" prices with known parameters
    v0_true = 0.04
    kappa_true = 2.0
    theta_true = 0.04
    sigma_true = 0.3
    rho_true = -0.7
    lambd_true = 0.1
    
    # Build arrays
    K_list, tau_list, r_list, P_list = [], [], [], []
    
    curve = calibrate_yield_curve()
    
    for tau in maturities:
        r = curve(tau)
        for K in strikes:
            price = heston_price_rec(S0, K, v0_true, kappa_true, theta_true, 
                                     sigma_true, rho_true, lambd_true, tau, r)
            K_list.append(K)
            tau_list.append(tau)
            r_list.append(r)
            P_list.append(price)
    
    return (S0, 
            np.array(r_list), 
            np.array(K_list), 
            np.array(tau_list), 
            np.array(P_list))


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
    "lambd": {"x0": 0.03, "lbub": [-1, 1]},
}


def create_objective_function(S0, K, tau, r, P):
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
        
    Returns:
    --------
    callable
        Objective function that takes parameter vector x and returns squared error
    """
    def SqErr(x):
        v0, kappa, theta, sigma, rho, lambd = x
        
        # Calculate Heston prices for all options
        heston_prices = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
        
        # Mean squared error
        err = np.sum((P - heston_prices)**2 / len(P))
        
        # Optional penalty term (currently zero - no prior guesses)
        pen = 0
        
        return err + pen
    
    return SqErr


def calibrate_heston(S0, K, tau, r, P, params=None, verbose=True):
    """
    Calibrate Heston model parameters to market option prices.
    
    Uses SLSQP optimization to minimize squared pricing error.
    
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
    SqErr = create_objective_function(S0, K, tau, r, P)
    
    if verbose:
        print("Starting calibration...")
        print(f"  Initial params: v0={x0[0]}, kappa={x0[1]}, theta={x0[2]}, "
              f"sigma={x0[3]}, rho={x0[4]}, lambd={x0[5]}")
    
    # Run optimization
    result = minimize(SqErr, x0, tol=1e-3, method='SLSQP', 
                     options={'maxiter': int(1e4)}, bounds=bnds)
    
    # Extract calibrated parameters
    v0, kappa, theta, sigma, rho, lambd = result.x
    
    calibrated = {
        'v0': v0,
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma,
        'rho': rho,
        'lambd': lambd,
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
        print(f"  lambd  = {lambd:.6f}  (variance risk premium)")
    
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
    print("Heston Model Calibration Pipeline (Synthetic Data Mode)")
    print("=" * 60)
    
    # 1. Yield Curve Calibration
    print("\n" + "-" * 60)
    print("1. Calibrating Yield Curve")
    print("-" * 60)
    curve = calibrate_yield_curve()
    print(f"1-year rate: {curve(1.0)*100:.4f}%")
    
    # 2. Market Data Acquisition
    print("\n" + "-" * 60)
    print("2. Generating Synthetic Market Data")
    print("-" * 60)
    print("\nTrue parameters: v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7, lambd=0.1")
    
    S0, r_arr, K_arr, tau_arr, P_arr = generate_test_market_data()
    print(f"Data generated for {len(P_arr)} option prices.")
    
    # 3. Calibrate Heston model
    print("\n" + "-" * 60)
    print("3. Calibrating Heston Model")
    print("-" * 60 + "\n")
    
    calibrated = calibrate_heston(S0, K_arr, tau_arr, r_arr, P_arr)
    
    # 4. Compute Calibrated Prices and Statistics
    print("\n" + "-" * 60)
    print("4. Computing Calibrated Prices")
    print("-" * 60)
    
    cal_prices = heston_price_rec(
        S0, K_arr, 
        calibrated['v0'], calibrated['kappa'], calibrated['theta'],
        calibrated['sigma'], calibrated['rho'], calibrated['lambd'],
        tau_arr, r_arr
    )
    
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
