import os
import pickle
from typing import Literal

import numpy as np


def pkl_save(name: str, var):
    """
    Save a Python variable to a pickle file.

    Args:
        name (str): The name of the pickle file to sate the variable to.
        var: The Python variable to be saved.

    Returns:
        None
    """
    with open(name, "wb") as f:
        pickle.dump(var, f)


def save_checkpoint_callback(
    save_every: int = 1, unit: Literal["epoch", "iter"] = "epoch"
) -> callable:
    """
    Create a callback for saving model checkpoints during training.

    Args:
        save_every (int): Save checkpoints every 'save_every' units.
        unit (str): Specify whether to save based on 'epoch' or 'iter'.

    Returns:
        callable: A callback function that can be used during model training.
    """
    assert unit in ("epoch", "iter")

    run_dir = os.getcwd()  # Get the current working directory.

    def callback(model):
        """
        Callback function to save model checkpoints.

        Args:
            model: The model being trained.
            loss: The current loss value.
        """
        n = model.n_epochs if unit == "epoch" else model.n_iters
        if n % save_every == 0:
            checkpoint_path = os.path.join(run_dir, f"model_{n}.pkl")
            model.save(checkpoint_path)  # Save the model checkpoint.

    return callback


def pad_nan_to_target(
    array: np.ndarray, target_length: int, axis: int = 0, both_side: bool = False
) -> np.ndarray:
    """
    Pad a NumPy array with NaN values to reach a specified target length along a specified axis.

    Args:
        array (np.ndarray): The input array to be padded with NaN values.
        target_length (int): The desired length of the array along the specified axis.
        axis (int, optional): The axis along which the padding should be applied (defaults to 0).
        both_side (bool, optional): If True, padding is applied equally on both sides of the array to reach target length.
            If False, padding is only applied to the end of the array (default is False).

    Returns:
        np.ndarray: The padded array with NaN values.

    Raises:
        AssertionError: If the data type of the input array is not a float.

    Example:
        input_array = np.array([1.0, 2.0, 3.0])
        padded_array = pad_nan_to_target(input_array, 5)
        # Output: array([1.0, 2.0, 3.0, nan, nan])
    """

    assert array.dtype in [np.float16, np.float32, np.float64]

    pad_size = target_length - array.shape[axis]  # Calculate the padding required.

    if pad_size <= 0:  # If no padding needed, return input array.
        return array

    npad = [(0, 0)] * array.ndim

    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode="constant", constant_values=np.nan)


def create_data_segments(train_data: np.ndarray, T: int, K: int) -> np.ndarray:
    """
    Creates data segments and adjusts the input data for training.

    Args:
        train_data (np.ndarray): Input training data.
        T (int): Maximum allowed sequence length for training.
        K (int): Length of history segmentation.

    Returns:
        np.ndarray: Adjusted and segmented training data.
    """
    # Indices for creating data segments.
    index = [i * K for i in range(train_data.shape[1] // 201)]  
    sections = train_data.shape[1] // T  # Number of sections.
    segment = train_data.shape[1] // sections  # Segment size based on sections.

    # Store data segments.
    segments = []
    for i in range(len(index)):  # Create data segments.
        if index[i] + segment < train_data.shape[1]:
            segments.append(train_data[:, index[i] : index[i] + segment, :])

        else:
            segments.append(train_data[:, -segment:, :])

        # Pad each segment with NaN values to match the length of the first segment.
        segments[i] = pad_nan_to_target(segments[i], segments[0].shape[1], axis=1)

    # Concatenate segmented data along the batch dimension.
    adjusted_data = np.concatenate(segments, axis=0)

    return adjusted_data


def centerize_vary_length_series(x: np.ndarray) -> np.ndarray:
    """
    Center-align varying length time series based on their non-NaN values.

    Args:
        x (np.ndarray): Input 2D array where each row represents a time series of varying length.

    Returns:
        np.ndarray: Center-aligned time series with NaN values filling gaps.

    Example:
        input_array = np.array([[1.0, 2.0, nan, nan],
                                [nan, 3.0, 4.0, nan],
                                [nan, nan, 5.0, 6.0]])
        centered_array = centerize_vary_length_series(input_array)
        # Output: array([[nan, nan, 1.0, 2.0],
                         [3.0, 4.0, nan, nan],
                         [nan, 5.0, 6.0, nan])
    """
    # Find first and last non-NaN values in each row.
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    # Calculate offset to align the time series.
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    # Create row and column indices for indexing.
    rows, column_indices = np.ogrid[: x.shape[0], : x.shape[1]]
    offset[offset < 0] += x.shape[1]  # Adjust negative offsets.
    # Calculate new column indices.
    column_indices = column_indices - offset[:, np.newaxis] 

    return x[rows, column_indices]
