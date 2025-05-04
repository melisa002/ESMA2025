import pandas as pd
import pyspark.sql.functions as f
from pyspark.sql.types import TimestampType, DoubleType
from pyspark.sql.window import Window
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor as SparkRF
from pyspark.ml import Pipeline
from sklearn.ensemble import RandomForestRegressor as SklearnRF
from functools import reduce

# =====================
# 1️⃣ UDF: Flexible Date Parser
# =====================
def get_parse_date_udf():
    @f.udf(returnType=TimestampType())
    def parse_date_flex(date_str):
        try:
            if '-Q' in date_str:
                year_str, q_str = date_str.split('-Q')
                year = int(year_str)
                quarter = int(q_str)
                month = (quarter - 1) * 3 + 1
                return pd.Timestamp(year=year, month=month, day=1)
            elif '-' in date_str:
                year_str, month_str = date_str.split('-')
                year = int(year_str)
                month = int(month_str)
                return pd.Timestamp(year=year, month=month, day=1)
            elif len(date_str) == 4 and date_str.isdigit():
                year = int(date_str)
                return pd.Timestamp(year=year, month=1, day=1)
            else:
                return None
        except:
            return None
    return parse_date_flex


# =====================
# 2️⃣ Threshold Outliers
# =====================
def add_outlier_thresholds(
    data: DataFrame,
    numbercol: str,
    groupbycols=None,
    showstats=False,
    use_logs=False
):
    if groupbycols is None:
        groupbycols = []

    col3sd = f"{numbercol}_3sd"
    col4sd = f"{numbercol}_4sd"

    # Stats per group
    stats_df = (
        data.groupBy(groupbycols)
        .agg(
            f.expr(f'percentile_approx({numbercol}, 0.5)').alias('median'),
            f.stddev(numbercol).alias('stddev')
        )
    )

    # Join back
    if groupbycols:
        join_condition = reduce(
            lambda x, y: x & y,
            [f.col(f"data.{col}") == f.col(f"stats.{col}") for col in groupbycols]
        )
        data = (
            data.alias("data")
            .join(stats_df.alias("stats"), on=join_condition, how="left")
        )
    else:
        data = (
            data.withColumn('dummykey', f.lit(1))
            .join(stats_df.withColumn('dummykey', f.lit(1)), on='dummykey', how='left')
            .drop('dummykey')
        )

    # Flag outliers
    if use_logs:
        data = data.withColumn(
            col3sd,
            f.when(f.log(f.abs(f.col(numbercol)) - f.col('median')) > 3 * f.col('stddev'), True).otherwise(False)
        ).withColumn(
            col4sd,
            f.when(f.log(f.abs(f.col(numbercol)) - f.col('median')) > 4 * f.col('stddev'), True).otherwise(False)
        )
    else:
        data = data.withColumn(
            col3sd,
            (f.abs(f.col(numbercol) - f.col('median')) > 3 * f.col('stddev'))
        ).withColumn(
            col4sd,
            (f.abs(f.col(numbercol) - f.col('median')) > 4 * f.col('stddev'))
        )

    if showstats:
        count_3sd = data.filter(f.col(col3sd)).count()
        count_4sd = data.filter(f.col(col4sd)).count()
        print(f"[3SD OUTLIERS] {numbercol}: {count_3sd}")
        print(f"[4SD OUTLIERS] {numbercol}: {count_4sd}")

    return data


# =====================
# 3️⃣ Random Forest Outliers (Pandas)
# =====================
def rf_outliers_pandas(
    df_pd,
    numbercol,
    feature_cols,
    groupbycols=None,
    showstats=False
):
    if groupbycols is None:
        groupbycols = []

    def process_group(grp):
        if grp.shape[0] < 10:
            grp['rfr_outlier'] = False
            return grp
        X = grp[feature_cols]
        y = grp[numbercol]
        model = SklearnRF(n_estimators=50, random_state=42)
        model.fit(X, y)
        grp['pred_rfr'] = model.predict(X)
        grp['residual_rfr'] = grp[numbercol] - grp['pred_rfr']
        sd = grp['residual_rfr'].std()
        grp['rfr_outlier'] = grp['residual_rfr'].abs() > 4 * sd
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
        print(df_out['rfr_outlier'].value_counts())

    return df_out


# =====================
# 4️⃣ Lag Feature Prep (Spark)
# =====================
def add_lag_features(
    df: DataFrame,
    groupbycols,
    date_col,
    target_col,
    lag=1,
    time_index=True
):
    w = Window.partitionBy(*groupbycols).orderBy(date_col)
    df = df.withColumn(f'lag{lag}_{target_col}', f.lag(target_col, lag).over(w))

    if time_index:
        df = df.withColumn(
            'time_index',
            f.year(date_col) * 10 + ((f.quarter(date_col) - 1))
        )
    return df


# =====================
# 5️⃣ RandomForest Spark Model
# =====================
def rf_outliers_spark(
    df: DataFrame,
    feature_cols,
    numbercol='OBS_VALUE'
):
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    rf = SparkRF(
        featuresCol='features',
        labelCol=numbercol,
        predictionCol='prediction',
        numTrees=100
    )
    pipeline = Pipeline(stages=[assembler, rf])
    model = pipeline.fit(df)
    preds = model.transform(df)
    preds = preds.withColumn('residual', f.col(numbercol) - f.col('prediction'))

    # Get per-group stddev of residuals if grouping exists
    return preds


# =====================
# 🔧 Utility: Plotting (Matplotlib)
# =====================
def plot_outliers_matplotlib(
    df_pd,
    date_col,
    target_col,
    outlier_col,
    title="Time Series Outliers",
    xlabel="Date",
    ylabel="Value"
):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(df_pd[date_col], df_pd[target_col], marker='o', linestyle='', alpha=0.2, label='All data')
    outliers = df_pd[df_pd[outlier_col]]
    plt.scatter(outliers[date_col], outliers[target_col], color='red', label='Outliers')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
