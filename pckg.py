"""
ESMA Outliers - Complete Outlier Detection Package
==================================================

A comprehensive package containing all outlier detection methods from your original code.
All functions preserved exactly as written.

Usage:
    from esma_outliers import spot

    result = spot(
        spark_df=your_data,
        mode='percentile',  # or any other mode
        numbercol='OBS_VALUE',
        groupbycols=['SEC_TYPE_CFI', 'ISSUER_COU']
    )
"""

import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import warnings
import os
import scipy.stats as stats
from pyspark.sql.functions import udf, col
from pyspark.sql.types import TimestampType
from pyspark.sql.window import Window
from skforecast.recursive import ForecasterRecursive
from sklearn.tree import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import pyspark.sql.functions as f
from functools import reduce
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from pyod.models.hbos import HBOS
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest

try:
    import esmaplotly as epy
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"


# =============================================================================
# 1. THRESHOLDS METHODS
# =============================================================================

## This is the very basic outlier thersholds that we had in the beginning.
## add_outlier_thresholds_percentile leves out the extreme data points when computing stddev.

def add_outlier_thresholds(
    data,
    numbercol,
    groupbycols=None,
    showstats=False,
    use_logs=False,
    outlier_mode= 4  # options: '3sd', '4sd', 'both'
):
    """
    Flags outliers in the specified numeric column based on group-level
    median ± 3 or 4 standard deviations.

    Parameters:
    - data: Spark DataFrame
    - numbercol: str, column to evaluate
    - groupbycols: list of str, columns to group by
    - showstats: bool, print number of outliers
    - use_logs: bool, apply log to the column before comparison
    - outlier_mode: str, one of ['3sd', '4sd', 'both']
    """
    if groupbycols is None:
        groupbycols = []

    col3sd = f"{numbercol}_3sd"
    col4sd = f"{numbercol}_4sd"

    # Step 1: compute median + stddev per group
    stats_df = (
        data.groupBy(groupbycols)
        .agg(
            f.expr(f'percentile_approx({numbercol}, 0.5)').alias('median'),
            f.stddev(numbercol).alias('stddev')
        )
    )

    # Step 2: join stats back to original data
    if groupbycols:
        join_condition = reduce(
            lambda x, y: x & y,
            [f.col(f"data.{col}") == f.col(f"stats.{col}") for col in groupbycols]
        )
        data = data.alias("data").join(stats_df.alias("stats"), on=join_condition, how="left")

    else:
        data = (
            data.withColumn("dummykey", f.lit(1))
            .join(stats_df.withColumn("dummykey", f.lit(1)), on="dummykey", how="left")
            .drop("dummykey")
        )


    data = data.drop(*[f"stats.{col}" for col in groupbycols])
    # Step 3: prepare value for thresholding
    val = f.log(f.abs(f.col(numbercol))) if use_logs else f.col(numbercol)

    # Step 4: flag outliers
    if outlier_mode in (3, 'both'):
        data = data.withColumn(
            col3sd, (val > f.col("median") + 3 * f.col("stddev")) | (val < f.col("median") - 3 * f.col("stddev"))
        )
    if outlier_mode in (4, 'both'):
        data = data.withColumn(
            col4sd, (val > f.col("median") + 4 * f.col("stddev")) | (val < f.col("median") - 4 * f.col("stddev"))
        )

    # Step 5: assign `is_outlier`
    if outlier_mode == 3:
        data = data.withColumn("is_outlier", f.col(col3sd))
    elif outlier_mode == 4:
        data = data.withColumn("is_outlier", f.col(col4sd))
    elif outlier_mode == 'both':
        # don't overwrite is_outlier, just return both _3sd and _4sd flags
        pass
    else:
        raise ValueError("Invalid outlier_mode. Choose from '3sd', '4sd', or 'both'.")

    # Step 6: optional summary
    if showstats:
        if outlier_mode in (3, 'both'):
            print(f"[3SD OUTLIERS] {numbercol}: {data.filter(f.col(col3sd)).count()}")
        if outlier_mode in (4, 'both'):
            print(f"[4SD OUTLIERS] {numbercol}: {data.filter(f.col(col4sd)).count()}")

    # Step 7: clean up
    return data.drop('median', 'stddev')

## This is the new version of add_outlier_thresholds that removes the extreme data points
## before computing the stddev.

def add_outlier_thresholds_percentile(
    data,
    numbercol,
    groupbycols=None,
    showstats=False,
    use_logs=False,
    outlier_mode=4,  # options: '3sd', '4sd', 'both'
    pre_filter_percentile=None  # e.g., 0.99 to drop top 1% values before computing std
):
    """
    Flags outliers in the specified numeric column based on group-level
    median ± 3 or 4 standard deviations.

    Parameters:
    - data: Spark DataFrame
    - numbercol: str, column to evaluate
    - groupbycols: list of str, columns to group by
    - showstats: bool, print number of outliers
    - use_logs: bool, apply log to the column before comparison
    - outlier_mode: str, one of ['3sd', '4sd', 'both']
    - pre_filter_percentile: float between 0 and 1 to drop top values before computing std
    """
    if groupbycols is None:
        groupbycols = []

    col3sd = f"{numbercol}_3sd"
    col4sd = f"{numbercol}_4sd"
    outlier_col = f"{numbercol}_is_outlier"  # 🆕 dynamic column name

    # Step 0: Optional pre-filtering by percentile
    if pre_filter_percentile is not None:
        perc_expr = f"percentile_approx({numbercol}, {pre_filter_percentile})"
        perc_df = data.groupBy(groupbycols).agg(f.expr(perc_expr).alias("cutoff"))

        if groupbycols:
            join_condition = reduce(
                lambda x, y: x & y,
                [f.col(f"data.{col}") == f.col(f"cut.{col}") for col in groupbycols]
            )
            data = (
                data.alias("data")
                .join(perc_df.alias("cut"), on=join_condition, how="left")
                .filter(f.col(numbercol) <= f.col("cut.cutoff"))
                .drop("cut.cutoff")
            )
        else:
            cutoff = data.selectExpr(perc_expr).collect()[0][0]
            data = data.filter(f.col(numbercol) <= cutoff)

    # Step 1: compute median + stddev per group
    stats_df = (
        data.groupBy(groupbycols)
        .agg(
            f.expr(f'percentile_approx({numbercol}, 0.5)').alias('median'),
            f.stddev(numbercol).alias('stddev')
        )
    )

    # Step 2: join stats back to original data
    if groupbycols:
        join_condition = reduce(
            lambda x, y: x & y,
            [f.col(f"data.{col}") == f.col(f"stats.{col}") for col in groupbycols]
        )
        data = data.alias("data").join(stats_df.alias("stats"), on=join_condition, how="left")
    else:
        data = (
            data.withColumn("dummykey", f.lit(1))
            .join(stats_df.withColumn("dummykey", f.lit(1)), on="dummykey", how="left")
            .drop("dummykey")
        )

    data = data.drop(*[f"stats.{col}" for col in groupbycols])

    # Step 3: prepare value for thresholding
    val = f.log(f.abs(f.col(numbercol))) if use_logs else f.col(numbercol)

    # Step 4: flag outliers
    if outlier_mode in (3, 'both'):
        data = data.withColumn(
            col3sd, (val > f.col("median") + 3 * f.col("stddev")) | (val < f.col("median") - 3 * f.col("stddev"))
        )
    if outlier_mode in (4, 'both'):
        data = data.withColumn(
            col4sd, (val > f.col("median") + 4 * f.col("stddev")) | (val < f.col("median") - 4 * f.col("stddev"))
        )

    # Step 5: assign dynamic outlier flag
    if outlier_mode == 3:
        data = data.withColumn(outlier_col, f.col(col3sd))
    elif outlier_mode == 4:
        data = data.withColumn(outlier_col, f.col(col4sd))
    elif outlier_mode == 'both':
        # keep both flags but do not create final flag
        pass
    else:
        raise ValueError("Invalid outlier_mode. Choose from '3sd', '4sd', or 'both'.")

    # Step 6: optional summary
    if showstats:
        if outlier_mode in (3, 'both'):
            print(f"[3SD OUTLIERS] {numbercol}: {data.filter(f.col(col3sd)).count()}")
        if outlier_mode in (4, 'both'):
            print(f"[4SD OUTLIERS] {numbercol}: {data.filter(f.col(col4sd)).count()}")

    # Step 7: clean up
    return data.drop('median', 'stddev')

## The function that allows flaggin by percentile.

def flag_outliers_by_percentile(
    data,
    numbercol,
    groupbycols=None,
    percentile=0.99,
    output_col=None
):
    """
    Flags values in `numbercol` above the specified percentile.

    Parameters:
    - data: Spark DataFrame
    - numbercol: numeric column to evaluate (e.g. 'OBS_VALUE')
    - groupbycols: list of columns to group by (e.g. ['key'])
    - percentile: float in (0,1), e.g. 0.99 = top 1%
    - output_col: optional str, name of output column. Defaults to numbercol+'_is_outlier'
    """
    if groupbycols is None:
        groupbycols = []

    if output_col is None:
        output_col = numbercol + "_is_outlier"

    # 1. Compute the threshold per group
    stats_df = (
        data.groupBy(groupbycols)
        .agg(
            f.expr(f'percentile_approx({numbercol}, {percentile})').alias('threshold')
        )
    )

    # 2. Join back to original data
    if groupbycols:
        join_condition = reduce(
            lambda x, y: x & y,
            [f.col(f"data.{col}") == f.col(f"stats.{col}") for col in groupbycols]
        )
        data = data.alias("data").join(stats_df.alias("stats"), on=join_condition, how="left")
    else:
        data = (
            data.withColumn("dummykey", f.lit(1))
            .join(stats_df.withColumn("dummykey", f.lit(1)), on="dummykey", how="left")
            .drop("dummykey")
        )

    data = data.withColumn(
        output_col,
        f.when(f.col(numbercol) > f.col("threshold"), f.lit(True)).otherwise(f.lit(False))
    )

    return data.drop("threshold")


# =============================================================================
# 2. TIMESERIES METHODS
# =============================================================================

## this function is needed for the arima. Maybe I can indent in ARIMA class.
def _infer_freq_and_m(index: pd.DatetimeIndex):
    """
    Infer the frequency string and seasonal period m from a datetime index.
    """
    freq = pd.infer_freq(index)
    if freq is None:
        # fallback: check day differences
        diffs = np.diff(index.values.astype('datetime64[D]')).astype(int)
        if np.all(diffs == diffs[0]):
            freq = 'D' if diffs[0] == 1 else 'M' if 28 <= diffs[0] <= 31 else None
        if freq is None:
            raise ValueError(f"Could not infer a consistent freq from index: {index[:5]}")
    # map to seasonal period
    if freq.startswith('Q'):
        m = 4
    elif freq.startswith('M'):
        m = 12
    elif freq.startswith('W'):
        m = 52
    elif freq.startswith('D'):
        m = 7
    else:
        raise ValueError(f"Unsupported frequency: {freq}")
    return freq, m


def fit_arima_and_flag_outliers(
        df: pd.DataFrame,
        key: str,
        key_col: str = 'key',
        date_col: str = 'period',
        value_col: str = 'value'):
    """
    Filter df by key, fit a seasonal ARIMA, and flag outliers beyond 3σ.

    Returns:
      model      - the fitted pmdarima model
      results_df - DataFrame with ['value','pred','resid','outlier'] indexed by date
    """
    sub = df[df[key_col] == key].copy()
    if sub.empty:
        raise ValueError(f"No data found for key={key}")

    # parse and set index
    sub[date_col] = pd.to_datetime(sub[date_col])
    sub = sub.set_index(date_col).sort_index()

    # infer freq & seasonal period, restrict to actual data span
    freq, m = _infer_freq_and_m(sub.index)
    sub = sub.asfreq(freq)

    # fit seasonal ARIMA (auto selects d and D for stationarity)
    model = pm.auto_arima(
        sub[value_col],
        seasonal=True,
        m=m,
        error_action='ignore',
        suppress_warnings=True
    )

    # in-sample predictions (skip warm-up)
    preds = pd.Series(
        model.predict_in_sample(start=m),
        index=sub.index[m:]
    )
    actual = sub[value_col].iloc[m:]
    resid = actual - preds

    # flag points >3σ
    sigma = resid.std()
    outliers = resid.abs() > 3 * sigma

    results_df = pd.DataFrame({
        'value': actual,
        'pred': preds,
        'resid': resid,
        'outlier': outliers
    })
    return model, results_df


def flag_outliers_sliding_window(
        df: pd.DataFrame,
        key: str,
        key_col: str = 'KEY',
        date_col: str = 'TIME_PERIOD',
        value_col: str = 'OBS_VALUE',
        window: int = 12,
        threshold: float = 4.0,
        robust: bool = False):
    """
    Rolling-window outlier detection.
    If robust=False uses rolling std, else rolling MAD.
    Returns DataFrame with value, rolling_mean, rolling_scale, resid, outlier.
    """
    sub = df[df[key_col] == key].copy()
    if sub.empty:
        raise ValueError(f"No data for key={key}")
    sub[date_col] = pd.to_datetime(sub[date_col])
    sub = sub.set_index(date_col).sort_index()
    series = sub[value_col].astype(float).asfreq(pd.infer_freq(sub.index))

    # rolling mean
    roll_mean = series.rolling(window=window, min_periods=1, center=True).mean()
    if robust:
        # rolling MAD
        roll_scale = series.rolling(window=window, min_periods=1, center=True)
        roll_scale = roll_scale.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=False)
    else:
        roll_scale = series.rolling(window=window, min_periods=1, center=True).std()

    resid = series - roll_mean
    outliers = resid.abs() > threshold * roll_scale

    results_df = pd.DataFrame({
        'value': series,
        'rolling_mean': roll_mean,
        'rolling_scale': roll_scale,
        'resid': resid,
        'outlier': outliers
    })
    return results_df


def fit_autoencoder_and_flag_outliers(
        df: pd.DataFrame,
        key: str,
        key_col: str = 'KEY',
        date_col: str = 'TIME_PERIOD',
        value_col: str = 'OBS_VALUE',
        window_size: int = 5,
        latent_dim: int = 2,
        epochs: int = 50,
        batch_size: int = 16,
        threshold: float = 2.0,
        threshold_method: str = 'std'):
    """
    Train a simple autoencoder on sliding windows and flag outliers by reconstruction error.

    Parameters:
      df               - DataFrame with key/date/value columns
      key              - filter value in key_col
      window_size      - number of time steps per window
      latent_dim       - dimensionality of encoded representation
      epochs           - training epochs
      batch_size       - training batch size
      threshold        - multiplier (for 'std') or quantile (for 'quantile')
      threshold_method - 'std' uses mean+threshold*std; 'quantile' uses (1-threshold) quantile

    Returns:
      model      - trained keras autoencoder
      results_df - DataFrame indexed by date (window end) with columns:
                   ['value', 'recon_error', 'outlier']
    """

    ### I need the try for the spark.ml here ...
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam

    # filter and prepare
    sub = df[df[key_col] == key].copy()
    if sub.empty:
        raise ValueError(f"No data for key={key}")
    sub[date_col] = pd.to_datetime(sub[date_col])
    sub = sub.set_index(date_col).sort_index()
    series = sub[value_col].astype(float).asfreq(pd.infer_freq(sub.index))
    values = series.values.reshape(-1, 1)

    # scale to [0,1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    # build sliding windows
    X = []
    for i in range(len(scaled) - window_size + 1):
        X.append(scaled[i:i + window_size].flatten())
    X = np.array(X)

    # train/test split: use all for training
    X_train = X

    # autoencoder architecture
    input_dim = window_size
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(latent_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoenc = Model(inputs=input_layer, outputs=decoded)
    autoenc.compile(optimizer=Adam(), loss='mse')

    # train
    autoenc.fit(X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=0)

    # reconstruction
    X_pred = autoenc.predict(X)
    # compute mse per window
    mse = np.mean(np.power(X - X_pred, 2), axis=1)

    # determine threshold
    if threshold_method == 'std':
        mean_mse = mse.mean()
        std_mse = mse.std()
        err_thresh = mean_mse + threshold * std_mse
    elif threshold_method == 'quantile':
        err_thresh = np.quantile(mse, 1 - threshold)
    else:
        raise ValueError("threshold_method must be 'std' or 'quantile'")

    # map errors & outliers to dates (window end)
    dates = series.index[window_size - 1:]
    recon_error = pd.Series(mse, index=dates)
    outliers = recon_error > err_thresh

    results_df = pd.DataFrame({
        'value': series[window_size - 1:],
        'recon_error': recon_error,
        'outlier': outliers
    })
    return autoenc, results_df


# =============================================================================
# 3. MACHINE LEARNING METHODS
# =============================================================================

def add_outlier_hbos(
    data,
    numbercol,
    groupbycols=None,
    showstats=False,
    n_bins=10,
    alpha=0.1,
    tol=0.5,
    contamination=0.1,
    output_col=None,
    min_group_size=20
):
    """
    Histogram-Based Outlier Score (HBOS) detection using PyOD for Spark DataFrames.

    Uses the pyod.models.hbos.HBOS implementation for efficient outlier detection.

    Parameters:
    -----------
    data : pyspark.sql.DataFrame
        Input Spark DataFrame
    numbercol : str
        Numeric column to analyze for outliers
    groupbycols : list of str, optional
        Columns to group by (default: None)
    showstats : bool, optional
        Whether to print outlier statistics (default: False)
    n_bins : int or str, optional
        Number of bins for histogram. Use "auto" for automatic selection (default: 10)
    alpha : float, optional
        Regularization parameter to prevent overflow (default: 0.1)
    tol : float, optional
        Tolerance for points falling outside bins (default: 0.5)
    contamination : float, optional
        Expected proportion of outliers (default: 0.1)
    output_col : str, optional
        Name for outlier flag column (default: f"{numbercol}_hbos_outlier")
    min_group_size : int, optional
        Minimum group size for HBOS analysis. Groups smaller than this will have no outliers flagged (default: 20)

    Returns:
    --------
    pyspark.sql.DataFrame
        DataFrame with added HBOS outlier scores and binary outlier flags
    """
    if groupbycols is None:
        groupbycols = []

    if output_col is None:
        output_col = f"{numbercol}_hbos_outlier"

    score_col = f"{numbercol}_hbos_score"

    # Convert to Pandas for PyOD HBOS computation
    df_pd = data.toPandas()

    def compute_hbos_group(group_df):
        """Compute HBOS scores for a single group using PyOD"""
        group_size = len(group_df)

        # Check minimum group size
        if group_size < min_group_size:
            if showstats:
                print(f"Skipping group with {group_size} samples (min required: {min_group_size})")
            group_df[score_col] = 0.0
            group_df[output_col] = False
            return group_df

        # Prepare data for PyOD (needs 2D array)
        values = group_df[numbercol].values.reshape(-1, 1)

        # Remove NaN values
        valid_mask = ~np.isnan(values.flatten())
        valid_count = valid_mask.sum()

        if valid_count < min_group_size:
            if showstats:
                print(f"Skipping group with {valid_count} valid samples (min required: {min_group_size})")
            group_df[score_col] = 0.0
            group_df[output_col] = False
            return group_df

        # Adjust n_bins based on group size
        if isinstance(n_bins, str) and n_bins.lower() == "auto":
            effective_bins = max(5, min(int(np.sqrt(valid_count)), valid_count // 4))
        else:
            effective_bins = min(n_bins, valid_count // 4)

        # Initialize and fit HBOS model
        hbos = HBOS(
            n_bins=effective_bins,
            alpha=alpha,
            tol=tol,
            contamination=contamination
        )

        try:
            # Fit the model and get outlier labels and scores
            outlier_labels = hbos.fit_predict(values)
            outlier_scores = hbos.decision_scores_

            # Store results
            group_df[score_col] = outlier_scores
            group_df[output_col] = outlier_labels == 1  # PyOD uses 1 for outliers

            if showstats:
                outlier_count = (outlier_labels == 1).sum()
                print(f"Group size: {group_size}, Outliers: {outlier_count} ({outlier_count/group_size*100:.1f}%), Bins used: {effective_bins}")

        except Exception as e:
            print(f"HBOS failed for group: {e}")
            group_df[score_col] = 0.0
            group_df[output_col] = False

        return group_df

    # Apply HBOS computation per group
    if groupbycols:
        df_result = (
            df_pd.groupby(groupbycols, group_keys=False)
            .apply(compute_hbos_group)
            .reset_index(drop=True)
        )
    else:
        df_result = compute_hbos_group(df_pd)

    # Show statistics if requested
    if showstats:
        total_outliers = df_result[output_col].sum()
        total_samples = len(df_result)
        print(f"[HBOS OUTLIERS] {numbercol}: {total_outliers}/{total_samples} "
              f"({total_outliers/total_samples*100:.2f}%)")

        if groupbycols:
            group_stats = df_result.groupby(groupbycols)[output_col].agg(['sum', 'count'])
            group_stats['percentage'] = (group_stats['sum'] / group_stats['count'] * 100).round(2)
            print("Per-group outlier statistics:")
            print(group_stats)

    # Convert back to Spark DataFrame
    return data.sql_ctx.createDataFrame(df_result)


def flag_outliers_hbos(
    data,
    numbercol,
    groupbycols=None,
    n_bins=10,
    contamination=0.05,
    output_col=None
):
    """
    Simplified HBOS outlier detection that follows your existing pattern.

    This function matches the style of your flag_outliers_by_percentile function.

    Parameters:
    -----------
    data : pyspark.sql.DataFrame
        Input Spark DataFrame
    numbercol : str
        Numeric column to analyze
    groupbycols : list of str, optional
        Columns to group by
    n_bins : int or str, optional
        Number of histogram bins (default: 10). Use "auto" for automatic selection
    contamination : float, optional
        Expected proportion of outliers (default: 0.05)
    output_col : str, optional
        Name for output column (default: f"{numbercol}_hbos_outlier")

    Returns:
    --------
    pyspark.sql.DataFrame
        DataFrame with HBOS outlier flag column added
    """
    if output_col is None:
        output_col = f"{numbercol}_hbos_outlier"

    result_df = add_outlier_hbos(
        data=data,
        numbercol=numbercol,
        groupbycols=groupbycols,
        n_bins=n_bins,
        contamination=contamination,
        output_col=output_col,
        showstats=False
    )

    # Return only original columns plus the outlier flag (like your percentile function)
    score_col = f"{numbercol}_hbos_score"
    return result_df.drop(score_col)


# =============================================================================
# 4. VISUALIZATION METHODS
# =============================================================================

def plot_global_series_with_outliers(df, date_col='TIME_PERIOD',
                                     value_col='OBS_VALUE',
                                     outlier_col='OBS_VALUE_is_outlier'):
    if PLOTLY_AVAILABLE:
        eplot = epy.esmaplotly()
        fig = go.Figure()

        # Plot the full series
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[value_col],
            mode='lines',
            name='OBS_VALUE',
            line=dict(width=1),
            showlegend=True
        ))

        # Plot global outliers as red dots
        outliers = df[df[outlier_col]]
        fig.add_trace(go.Scatter(
            x=outliers[date_col],
            y=outliers[value_col],
            mode='markers',
            name='Outliers (Top 1%)',
            marker=dict(color='red', size=6),
            showlegend=True
        ))

        fig = eplot.update_chart_trv(
            fig,
            annotation_text="Red dots represent global top 1% percentile outliers."
        )
        fig.update_layout(title='Global Outlier Detection (Percentile-Based)')

        return fig
    else:
        # Fallback to matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(df[date_col], df[value_col], 'b-', alpha=0.7, label='Normal')
        outliers = df[df[outlier_col]]
        if len(outliers) > 0:
            plt.scatter(outliers[date_col], outliers[value_col], c='red', s=20, alpha=0.8, label='Outliers')
        plt.title('Global Outlier Detection (Percentile-Based)')
        plt.xlabel('Time Period')
        plt.ylabel('Value')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_arima_esma(
    results_df,
    date_col='TIME_PERIOD',
    obs_col='value',
    pred_col='pred',
    outlier_col='outlier',
    title =' '
):
    """
    ESMA‐style interactive chart:
      • solid blue line = actuals (excluding outliers)
      • dashed blue line = ARIMA forecast
      • orange dots = outliers
    plus light-grey horizontal grid lines
    """
    if PLOTLY_AVAILABLE:
        df = results_df.copy()
        if date_col not in df.columns:
            df.index.name = date_col
            df = df.reset_index()

        df[outlier_col] = df[outlier_col].astype(bool)
        normal = df[~df[outlier_col]]
        out    = df[df[outlier_col]]

        eplot = epy.esmaplotly()
        fig = go.Figure()

        # 1) actuals line (ESMA blue)
        fig.add_trace(go.Scatter(
            x=normal[date_col],
            y=normal[obs_col],
            mode='lines',
            name='Actual',
            line=dict(color='#007EFF', width=2),
            legendgroup='act'
        ))

        # 2) forecast line (dashed ESMA blue)
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[pred_col],
            mode='lines',
            name='Forecast',
            line=dict(color='#7BD200', dash='dash', width=2),
            legendgroup='fc'
        ))

        # 3) outlier markers (ESMA orange)
        fig.add_trace(go.Scatter(
            x=out[date_col],
            y=out[obs_col],
            mode='markers',
            name='Outlier',
            marker=dict(color='#DB5700', size=8),
            legendgroup='pts'
        ))

        # apply ESMA layout & annotation
        fig = eplot.update_chart_trv(
            fig,
            annotation_text="Orange dots represent the flagged outliers; dashed line is the ARIMA forecast. <br> Note: The monthly ARIMA needs 12 periods to learn the dynamics of the timeseries."
        )
        fig.update_layout(title=title)

        # --- NEW: turn on horizontal grid lines on the y-axis ---
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',  # light grey
            gridwidth=1
        )

        # export high-res if you like
        fig.write_image("chart_highres.png", width=1600, height=900, scale=2)
        fig.write_image("chart1.svg")

        fig.show()
    else:
        # Fallback to matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(results_df.index, results_df['value'], label='Actual')
        plt.plot(results_df.index, results_df['pred'], label='Predicted')
        out = results_df[results_df['outlier']]
        plt.scatter(out.index, out['value'], label='Outliers', color='red')
        plt.legend()
        plt.title(title)
        plt.show()


def plot_sliding_window_esma_plotly(
        flagged: pd.DataFrame,
        date_col: str = 'TIME_PERIOD',
        value_col: str = 'value',
        rolling_mean_col: str = 'rolling_mean',
        outlier_col: str = 'outlier'):
    if PLOTLY_AVAILABLE:
        # 1) Bring index into a column
        df = flagged.reset_index()
        df[date_col] = pd.to_datetime(df[date_col])

        # 2) Single x-axis, y-series, mask
        x = df[date_col]
        y = df[value_col]
        rm = df[rolling_mean_col]
        mask = df[outlier_col].astype(bool)

        eplot = epy.esmaplotly()
        fig = go.Figure()

        # A) Plot ALL actuals (blue solid) – include outliers in the line
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Value',
            line=dict(color='#007EFF', width=4),
            legendgroup='val'
        ))

        # B) Rolling mean (green dashed)
        fig.add_trace(go.Scatter(
            x=x,
            y=rm,
            mode='lines',
            name='Rolling Mean',
            line=dict(color='#7BD200', dash='dash', width=4),
            legendgroup='mean'
        ))

        # C) Outlier markers (orange open circles)
        fig.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            mode='markers',
            name='Outliers',
            marker=dict(color='#DB5700', size=14, symbol='circle'),
            legendgroup='out'
        ))

        # Horizontal gridlines on y-axis
        fig.update_yaxes(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.3)',
            gridwidth=1
        )
        fig.update_xaxes(showgrid=False)

        # Transparent background & layout
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(size=14),
            legend=dict(font=dict(size=14)),
            margin=dict(l=60, r=40, t=60),  # extra bottom for annotation
            width=1000, height=600
        )

        # ESMA footer annotation — manually added so it exports
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0, y=-0.15,
            showarrow=False,
            text="Note: rolling-window outliers flagged at ±4σ around the dashed trend.",
            font=dict(size=16),
            align="left"
        )

        # Save & show
        fig.write_image('sliding_window_esma.png', width=1000, height=600, scale=2)
        fig.show()
        return fig
    else:
        # Fallback to matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(flagged.index, flagged['value'], label='Value')
        plt.plot(flagged.index, flagged['rolling_mean'], label='Rolling Mean')
        out = flagged[flagged['outlier']]
        plt.scatter(out.index, out['value'], facecolors='none', edgecolors='red', label='Outliers')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Sliding-Window Outliers')
        plt.show()


# =============================================================================
# 5. MAIN SPOT FUNCTION
# =============================================================================

def spot(
    spark_df,
    mode='thresholds',
    numbercol='OBS_VALUE',
    groupbycols=None,
    showstats=False,
    use_logs=False,
    min_filter=None,
    min_date=None,
    feature_cols=None,
    datecol='parsed_date',
    return_mode = 'all',
    nr_sd = 4,
    # Additional parameters for different modes
    percentile=0.95,
    contamination=0.05,
    n_bins=10,
    # ARIMA parameters
    key=None,
    key_col='KEY',
    date_col='TIME_PERIOD',
    value_col='OBS_VALUE',
    # Sliding window parameters
    window=12,
    threshold=4.0,
    robust=False,
    # Autoencoder parameters
    window_size=5,
    latent_dim=2,
    epochs=50,
    batch_size=16,
    autoencoder_threshold=2.0,
    threshold_method='std'
):
    """
    Unified outlier detection function supporting multiple methods.

    Parameters:
    -----------
    spark_df : pyspark.sql.DataFrame
        Input Spark DataFrame
    mode : str
        Detection method: 'thresholds', 'percentile', 'hbos', 'random_forest',
        'arima', 'sliding_window', 'autoencoder', 'isolation_forest'
    numbercol : str
        Numeric column to analyze (default: 'OBS_VALUE')
    groupbycols : list, optional
        Columns to group by
    showstats : bool
        Print statistics
    return_mode : str
        'all' or 'outliers'

    Mode-specific parameters:
    - percentile: percentile threshold
    - contamination: expected outlier rate for ML methods
    - n_bins: bins for HBOS
    - key: specific key for time series methods
    - window: window size for sliding window
    - And many others...

    Returns:
    --------
    pyspark.sql.DataFrame or pandas.DataFrame
        Results with outlier flags
    """
    if groupbycols is None:
        groupbycols = []

    # Apply filters first
    if min_date:
        spark_df = spark_df.filter(f.col(datecol) >= f.lit(min_date))

    if min_filter:
        spark_df = spark_df.filter(f.col(numbercol) >= min_filter)

    # Statistical methods
    if mode == 'percentile':
        df_with_outliers = flag_outliers_by_percentile(
            data=spark_df,
            numbercol=numbercol,
            groupbycols=groupbycols,
            percentile=percentile
        )

        if return_mode == 'all':
            return df_with_outliers
        elif return_mode == 'outliers':
            return df_with_outliers.filter(f.col(f'{numbercol}_is_outlier') == True)

    elif mode == 'thresholds':
        df_thresh = spark_df
        if min_filter is not None:
            df_thresh = df_thresh.filter(f.col(numbercol) >= min_filter)

        outliers_only = add_outlier_thresholds(
            data=df_thresh,
            numbercol=numbercol,
            groupbycols=groupbycols,
            showstats=showstats,
            use_logs=use_logs,
            outlier_mode=nr_sd
        )

        if return_mode == 'all':
            return outliers_only
        elif return_mode == 'outliers':
            return outliers_only.filter(f.col('is_outlier') == True)

    # Machine Learning methods
    elif mode == 'hbos':
        df_with_outliers = add_outlier_hbos(
            data=spark_df,
            numbercol=numbercol,
            groupbycols=groupbycols,
            showstats=showstats,
            n_bins=n_bins,
            contamination=contamination
        )

        outlier_col = f"{numbercol}_hbos_outlier"
        df_with_outliers = df_with_outliers.withColumnRenamed(outlier_col, "is_outlier")

        if return_mode == 'all':
            return df_with_outliers
        elif return_mode == 'outliers':
            return df_with_outliers.filter(f.col('is_outlier') == True)

    elif mode == 'random_forest_regressor':
        df_pd = spark_df.toPandas()

        if feature_cols is None:
            numeric = df_pd.select_dtypes(include=['float','int']).columns.tolist()
            feature_cols = [c for c in numeric if c != numbercol]

        def process_group(grp):
            if grp.shape[0] < 10:
                grp['is_outlier'] = False
                return grp

            X = grp[feature_cols]
            y = grp[numbercol]
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)

            grp['pred_rfr']      = model.predict(X)
            grp['residual_rfr']  = grp[numbercol] - grp['pred_rfr']
            sd = grp['residual_rfr'].std()
            thresh = int(nr_sd) * sd

            grp['is_outlier'] = grp['residual_rfr'].abs() > thresh
            return grp

        if groupbycols:
            df_out = (
                df_pd
                .groupby(groupbycols, group_keys=False)
                .apply(process_group)
                .reset_index(drop=True)
            )
        else:
            df_out = process_group(df_pd)

        if showstats:
            print(df_out['is_outlier'].value_counts())

        if min_filter is not None:
            df_out.loc[df_out[numbercol] < min_filter, 'is_outlier'] = False

        if return_mode == 'all':
            return spark_df.sql_ctx.createDataFrame(df_out)
        elif return_mode == 'outliers':
            return spark_df.sql_ctx.createDataFrame(df_out[df_out['is_outlier']== True])

    elif mode == 'isolation_forest':
        df_pd = spark_df.toPandas()

        def compute_isolation_forest(group_df):
            if len(group_df) < 30:
                group_df['is_outlier'] = False
                return group_df

            values = group_df[numbercol].values.reshape(-1, 1)

            try:
                iso = IsolationForest(contamination=contamination, random_state=42)
                outliers = iso.fit_predict(values) == -1
                group_df['is_outlier'] = outliers
            except Exception:
                group_df['is_outlier'] = False

            return group_df

        if groupbycols:
            df_result = (
                df_pd.groupby(groupbycols, group_keys=False)
                .apply(compute_isolation_forest)
                .reset_index(drop=True)
            )
        else:
            df_result = compute_isolation_forest(df_pd)

        if return_mode == 'all':
            return spark_df.sql_ctx.createDataFrame(df_result)
        elif return_mode == 'outliers':
            return spark_df.sql_ctx.createDataFrame(df_result[df_result['is_outlier'] == True])

    # Time series methods
    elif mode == 'arima':
        df_pd = spark_df.toPandas()

        if key is None:
            raise ValueError("ARIMA mode requires a 'key' parameter")

        model, results_df = fit_arima_and_flag_outliers(
            df=df_pd,
            key=key,
            key_col=key_col,
            date_col=date_col,
            value_col=value_col
        )

        # Add is_outlier column for consistency
        results_df['is_outlier'] = results_df['outlier']

        if return_mode == 'all':
            return results_df
        elif return_mode == 'outliers':
            return results_df[results_df['is_outlier'] == True]

    elif mode == 'sliding_window':
        df_pd = spark_df.toPandas()

        if key is None:
            raise ValueError("Sliding window mode requires a 'key' parameter")

        results_df = flag_outliers_sliding_window(
            df=df_pd,
            key=key,
            key_col=key_col,
            date_col=date_col,
            value_col=value_col,
            window=window,
            threshold=threshold,
            robust=robust
        )

        # Add is_outlier column for consistency
        results_df['is_outlier'] = results_df['outlier']

        if return_mode == 'all':
            return results_df
        elif return_mode == 'outliers':
            return results_df[results_df['is_outlier'] == True]

    elif mode == 'autoencoder':
        df_pd = spark_df.toPandas()

        if key is None:
            raise ValueError("Autoencoder mode requires a 'key' parameter")

        model, results_df = fit_autoencoder_and_flag_outliers(
            df=df_pd,
            key=key,
            key_col=key_col,
            date_col=date_col,
            value_col=value_col,
            window_size=window_size,
            latent_dim=latent_dim,
            epochs=epochs,
            batch_size=batch_size,
            threshold=autoencoder_threshold,
            threshold_method=threshold_method
        )

        # Add is_outlier column for consistency
        results_df['is_outlier'] = results_df['outlier']

        if return_mode == 'all':
            return results_df
        elif return_mode == 'outliers':
            return results_df[results_df['is_outlier'] == True]

    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# 6. PACKAGE EXPORTS
# =============================================================================

__all__ = [
    'spot',
    'add_outlier_thresholds',
    'add_outlier_thresholds_percentile',
    'flag_outliers_by_percentile',
    'fit_arima_and_flag_outliers',
    'flag_outliers_sliding_window',
    'fit_autoencoder_and_flag_outliers',
    'add_outlier_hbos',
    'flag_outliers_hbos',
    'plot_global_series_with_outliers',
    'plot_arima_esma',
    'plot_sliding_window_esma_plotly'
]
