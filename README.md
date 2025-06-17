# ESMA 2025 BSE Project: Outlier Detection Toolkit

## Overview

This repository contains a robust outlier detection toolkit built during the 2025 BSE master's project in collaboration with the European Securities and Markets Authority (ESMA). The toolkit is designed to support the identification of anomalous values in financial time series datasets submitted by EU member states, using scalable methods implemented in PySpark, pandas, and advanced machine learning frameworks.

The toolkit supports three major detection strategies:
- **Statistical Thresholds** (Median ± 3/4 Standard Deviations)
- **Forecast-Based Anomaly Detection** (ARIMA models, Recursive Forecasting)
- **Machine Learning Methods** (Random Forest residuals, Autoencoders)

All methods are wrapped to be reusable on grouped, quarterly/monthly time series data in Spark or Pandas, and are visualized with official-style ESMA plots using Plotly.

---

## Features

### ✅ Statistical Threshold Flagging (PySpark)
- Median ± 3σ / 4σ outlier detection per group
- Optional log-transform and group filtering
- Implemented for distributed Spark DataFrames with optional `.groupBy()`

### ⚖️ Random Forest Regressor
- Predicts value using other numeric features
- Flags based on high residual errors
- Can be used per group or globally

### 🔢 ARIMA-Based Forecasting
- Automatically fits seasonal ARIMA on each key
- Flags points with forecast residuals > 3 standard deviations
- Generates interactive ESMA-style Plotly charts with forecast lines and flagged outliers

### 🧠 Sliding-Window Outliers
- Rolling mean + rolling standard deviation or MAD
- Robust to seasonal drift
- Interactive visualizations included

### 🧬 Autoencoder Detection
- Sliding-window neural network for reconstruction error analysis
- Flags high-error segments
- Plots MSE and visual timelines of detected outliers

---

## Getting Started

### Requirements

- Python 3.10+
- PySpark
- pandas, numpy, matplotlib, seaborn
- `pmdarima`, `scikit-learn`, `tensorflow`
- `plotly`, `esmaplotly` (custom wrapper)

### Running the Spark Modules

```bash
python run_thresholds.py
python run_randomforest.py
