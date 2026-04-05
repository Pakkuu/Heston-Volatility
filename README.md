# Heston Volatility Model Calibration Suite

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A pure-Python implementation of the **Heston Stochastic Volatility Model**, designed for European option pricing and automated model calibration to market data.

## Overview

This project provides a comprehensive pipeline for modeling stochastic volatility in financial markets. It implements the semi-analytical solution for European call options using characteristic functions and Fourier transform-based numerical integration. The suite includes tools for market data acquisition (synthetic & live), yield curve modeling, and robust parameter optimization.

## Key Features

- **Heston Model Pricing**: Semi-analytical European option pricing via the Heston characteristic function.
- **Yield Curve Calibration**: Automated fitting of the **Nelson-Siegel-Svensson (NSS)** model to market interest rates.
- **Parameter Optimization**: Robust calibration of the Heston parameter set ($\kappa, \theta, \sigma, \rho, v_0, \lambda$) using **SLSQP optimization** to minimize mean squared pricing error.
- **Interactive Visualizations**: Generates dynamic 3D Plotly visualizations of the volatility surface and pricing error residuals.
- **Modular Design**: Clean, pure-Python architecture with full support for dependency management via `uv`.

## 🛠️ Technology Stack

- **Quantitative Python**: `numpy`, `scipy`, `pandas`
- **Visualization**: `plotly`, `matplotlib`
- **Yield Modeling**: `nelson-siegel-svensson`
- **Dependency Management**: `uv`

## 🚀 Getting Started

### Prerequisites

Ensure you have [uv](https://github.com/astral-sh/uv) installed to manage the project environment seamlessly.

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/Heston-Volatility.git
cd Heston-Volatility
uv sync
```

### Usage

To run the full calibration pipeline (Synthetic Market Data Mode):

```bash
uv run python main.py
```

### Visualizing Results

The pipeline generates two interactive HTML files for in-depth analysis:
- `calibration_results.html`: 3D comparison between market data mesh and Heston model markers.
- `pricing_errors.html`: 3D error surface showing price residuals across strikes and maturities.

## 📊 Methodology

The implementation follows the characteristic function approach described by Heston (1993). The objective function for calibration is the Mean Squared Error (MSE) between market-observed prices and model-generated prices:

$$\min_{\Theta} \sum_{i,j} (C_{Market}(K_i, \tau_j) - C_{Heston}(K_i, \tau_j, \Theta))^2$$

Where $\Theta = (v_0, \kappa, \theta, \sigma, \rho, \lambda)$.
