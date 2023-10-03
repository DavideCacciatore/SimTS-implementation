from typing import Literal

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _get_time_features(dt: pd.DatetimeIndex) -> np.ndarray:
    """
    Get time-related features from a datetime index.

    Args:
        dt (pd.DatetimeIndex): Datetime index.

    Returns:
        np.ndarray: Array containing minute, hour, day of the week, day of the month,
                    day of the year, month, and week of the year.
    """
    return np.stack(
        [
            dt.minute.to_numpy(),
            dt.hour.to_numpy(),
            dt.dayofweek.to_numpy(),
            dt.day.to_numpy(),
            dt.dayofyear.to_numpy(),
            dt.month.to_numpy(),
            dt.weekofyear.to_numpy(),
        ],
        axis=1,
    ).astype(np.float64)


def load_forecast_csv(
    name: Literal["ETTh1", "ETTh2", "ETTm1", "ETTm2"], univar: bool = True
) -> tuple:
    """
    Load a CSV dataset for time series forecasting.

    Args:
        name (str): Name of the dataset ("ETTh1", "ETTh2", "ETTm1", or "ETTm2).
        univar (bool): Whether to use univariate data (default True).

    Returns:
        tuple: A tuple containing data, train_slice, valid_slice, test_slice, scaler,
               pred_lens, and n_covariate_cols.
    """
    # Read the CSV file into a pandas DataFrame, parsing dates and using "date" as the index.
    data = pd.read_csv(f"datasets/{name}.csv", index_col="date", parse_dates=True)

    # Extract time-related features from the datetime index.
    dt_embed = _get_time_features(data.index)
    n_covariate_cols = dt_embed.shape[-1]

    # If univar is True, keep only the "OT" column in the DataFrame.
    if univar:
        data = data[["OT"]]

    # Convert the DataFrame to a numpy array.
    data = data.to_numpy()

    # Define train, validation, and test slices based on the dataset name.
    if name == "ETTh1" or name == "ETTh2":
        train_slice = slice(None, 12 * 30 * 24)
        valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
        test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
    elif name == "ETTm1" or name == "ETTm2":
        train_slice = slice(None, 12 * 30 * 24 * 4)
        valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        test_slice = slice(15 * 30 * 24 * 4, 20 * 30 * 24 * 4)

    # Create a StandardScaler and fit it to the training data
    scaler = StandardScaler().fit(data[train_slice])
    data = scaler.transform(data)
    # Add an extra dimension to the data array.
    data = np.expand_dims(data, 0)

    # If there are covariate columns, scale and concatenate them with the data.
    if n_covariate_cols > 0:
        dt_scaler = StandardScaler().fit(dt_embed[train_slice])
        dt_embed = np.expand_dims(dt_scaler.transform(dt_embed), 0)
        data = np.concatenate(
            [np.repeat(dt_embed, data.shape[0], axis=0), data], axis=-1
        )

    # Define prediction lengths based on the dataset name.
    if name in ("ETTh1", "ETTh2"):
        pred_lens = [24, 48, 168, 336, 720]
    elif name in ("ETTm1", "ETTm2"):
        pred_lens = [24, 48, 96, 288, 672]

    return (
        data,
        train_slice,
        valid_slice,
        test_slice,
        scaler,
        pred_lens,
        n_covariate_cols,
    )
