# ESMA 2025 BSE Project: Outlier Detection Toolkit

## Overview

This repository contains a robust outlier detection toolkit built during the 2025 BSE master's project in collaboration with the European Securities and Markets Authority (ESMA). The toolkit is designed to support the identification of anomalous values in financial time series datasets submitted by EU member states, using scalable methods implemented in PySpark, pandas, and advanced machine learning frameworks.

## 🚀 Quick Start

Basic usage with the main spot function for different detection methods including percentile-based detection, random forest detection, and ARIMA-based detection.

## Detection Methods

### Statistical Methods
- **Thresholds**: Median ± 3σ/4σ detection per group
- **Percentile**: Top/bottom percentile flagging
- **Sliding Window**: Rolling mean ± threshold detection

### Machine Learning Methods
- **Random Forest**: Residual-based outlier detection
- **HBOS**: Histogram-based outlier scoring
- **Autoencoder**: Neural network reconstruction error

### Time Series Methods
- **ARIMA**: Seasonal forecasting with residual analysis
- **Sliding Window**: Robust trend-based detection

## 📊 Visualization

Built-in ESMA-style plots with Plotly for ARIMA results and global outlier visualization.

##Installation

Required packages include pyspark, pandas, numpy, matplotlib, pmdarima, scikit-learn, tensorflow, and plotly.

## Main Function: spot()

The spot() function is your main entry point supporting all detection modes.

### Parameters
- spark_df: Input Spark DataFrame
- mode: Detection method (percentile, thresholds, random_forest_regressor, hbos, arima, sliding_window, autoencoder)
- numbercol: Column to analyze (default: OBS_VALUE)
- groupbycols: Grouping columns for analysis
- return_mode: all or outliers

### Example Usage

Statistical threshold detection, percentile-based detection, and machine learning detection examples available.

## Key Features

- **Scalable**: Built for PySpark with fallback to pandas
- **Flexible**: Multiple detection algorithms
- **Visual**: ESMA-style interactive charts
- **Robust**: Handles missing data and edge cases
- **Grouped**: Per-group analysis for financial datasets

## Available Functions

### Core Detection
- spot() - Main unified detection function
- add_outlier_thresholds() - Statistical threshold detection
- flag_outliers_by_percentile() - Percentile-based flagging

### Machine Learning
- random_forest_outliers() - RF-based detection
- add_outlier_hbos() - Histogram-based outlier scoring
- fit_autoencoder_and_flag_outliers() - Neural network detection

### Time Series
- fit_arima_and_flag_outliers() - ARIMA-based detection
- flag_outliers_sliding_window() - Rolling window detection

### Visualization
- plot_arima_esma() - ARIMA results visualization
- plot_global_series_with_outliers() - Global outlier plots
- plot_sliding_window_esma_plotly() - Sliding window plots

## Return Values

All functions return DataFrames with original data columns, boolean outlier flags, and additional method-specific columns for scores and predictions.

## Performance Notes

- **Spark ML**: Attempted first for scalability
- **Pandas Fallback**: Used when Spark ML fails
- **Grouped Processing**: Efficient per-group analysis
- **Memory Optimized**: Handles large datasets
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
