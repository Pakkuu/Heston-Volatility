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


if __name__ == "__main__":
    # Test parameters
    S0 = 100.0      # initial asset price
    K = 100.0       # strike price
    v0 = 0.1        # initial variance
    kappa = 1.5768  # rate of mean reversion of variance process
    theta = 0.0398  # long-term mean variance
    sigma = 0.3     # volatility of volatility
    lambd = 0.575   # risk premium of variance
    rho = -0.5711   # correlation between variance and stock process
    tau = 1.0       # time to maturity
    r = 0.03        # risk free rate
    
    print("=" * 60)
    print("Step 2 Complete: Heston Option Pricing Functions")
    print("=" * 60)
    
    # Test characteristic function
    phi = 1.0
    char_result = heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    print(f"\nCharacteristic function (phi=1): {char_result}")
    
    # Test pricing with rectangular integration
    price_rec = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
    print(f"\nHeston price (rectangular): {price_rec:.6f}")
    
    # Test pricing with scipy quad
    price_quad = heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
    print(f"Heston price (scipy quad):  {price_quad:.6f}")
    
    print(f"\nDifference: {abs(price_rec - price_quad):.8f}")
