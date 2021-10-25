import pandas as pd


def get_bounds_limits(df, alpha=1.5):
    """
    Determine bounds on variables

    Args:
        df: pd.DataFrame that we need to analyse
        alpha: float, window size of iqr

    Returns:
        dataframe bounds

    """
    bounds = []
    for col in df.columns:
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        if iqr == 0:
            iqr = 1             
        q1 = df[col].quantile(0.25) - alpha * iqr
        q3 = df[col].quantile(0.75) + alpha * iqr
        # print(col, "Q1 - 1.5*IQR:", q1)
        # print(col, "Q3 + 1.5*IQR:", q3)
        bounds.append([col, max(df[col].min(), q1), min(q3, df[col].max())])

    bounds = pd.DataFrame(
        bounds, columns=["variable", "lower_lim", "upper_lim"])
    print(bounds)
    return bounds