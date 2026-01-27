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

    # Common terms w.r.t phi
    rspi = rho * sigma * phi * 1j

    # Define d parameter given phi and b
    d = np.sqrt((rho * sigma * phi * 1j - b)**2 + (phi * 1j + phi**2) * sigma**2)

    # Define g parameter given phi, b and d
    g = (b - rspi + d) / (b - rspi - d)

    # Calculate characteristic function by components
    exp1 = np.exp(r * phi * 1j * tau)
    term2 = S0**(phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g))**(-2 * a / sigma**2)
    exp2 = np.exp(
        a * tau * (b - rspi + d) / sigma**2 
        + v0 * (b - rspi + d) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau))) / sigma**2
    )

    return exp1 * term2 * exp2


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


if __name__ == "__main__":
    print("=" * 60)
    print("Step 4 Complete: Market Data Fetching")
    print("=" * 60)
    
    # Test Heston pricing (quick check)
    S0_test = 100.0
    K_test = 100.0
    v0, kappa, theta = 0.1, 1.5768, 0.0398
    sigma, lambd, rho = 0.3, 0.575, -0.5711
    tau, r = 1.0, 0.03
    
    price = heston_price_rec(S0_test, K_test, v0, kappa, theta, sigma, rho, lambd, tau, r)
    print(f"\nHeston price check: {price:.6f}")
    
    # Test yield curve
    print("\n" + "-" * 60)
    print("Yield Curve")
    print("-" * 60)
    curve = calibrate_yield_curve()
    print(f"1-year rate: {curve(1.0)*100:.4f}%")
    
    # Test synthetic market data generation
    print("\n" + "-" * 60)
    print("Synthetic Market Data Generation")
    print("-" * 60)
    
    S0, r_arr, K_arr, tau_arr, P_arr = generate_test_market_data()
    print(f"\nSpot price (S0): {S0}")
    print(f"Number of options: {len(P_arr)}")
    print(f"Strike range: {K_arr.min()} - {K_arr.max()}")
    print(f"Maturity range: {tau_arr.min():.2f} - {tau_arr.max():.2f} years")
    print(f"Price range: {P_arr.min():.2f} - {P_arr.max():.2f}")
    
    # Show sample of data
    print("\nSample prices (first 5):")
    for i in range(5):
        print(f"  K={K_arr[i]:.0f}, tau={tau_arr[i]:.2f}, r={r_arr[i]*100:.2f}%, P={P_arr[i]:.2f}")
