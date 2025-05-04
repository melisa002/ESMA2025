# ESMA2025

# name_of_pckg**

This package is a Python module designed to **flag ESMA outliers** from different databases, supporting **various granularities** (quarterly, monthly, yearly) and **different anomaly detection methods** (statistical and machine learning).

It works flexibly with **Spark DataFrames** (for big data) and **Pandas DataFrames** (for smaller datasets) and helps you handle **non-standard time formats** and **multi-level groupings** common in ESMA datasets.

---

## ✨ **Features**

- **Flexible time parsing:**
    - Supports `YYYY-QX`, `YYYY-MM`, `YYYY` formats.
- **Statistical outlier detection:**
    - Detects outliers based on **median ± 3σ/4σ** thresholds.
- **Random Forest outlier detection:**
    - Train a **RandomForest model** (Spark or Pandas) and flag outliers based on residuals.
- **Lag feature engineering:**
    - Add lagged features & time indices per group.
- **Visualization tools:**
    - Interactive outlier plotting using **Plotly**.

---

## 🔧 **Installation**

Dependencies:

```bash
pip install requirements.txt
