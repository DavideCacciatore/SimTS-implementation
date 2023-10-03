import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

def fit_ridge(
    train_features: np.ndarray,
    train_y: np.ndarray,
    valid_features: np.ndarray,
    valid_y: np.ndarray,
    MAX_SAMPLES: int = 100000,
) -> Ridge:
    """
    Fit a Ridge regression model to the training data and return the trained model.

    Args:
        train_features (np.ndarray): Training features (X) as a NumPy array.
        train_y (np.ndarray): Training target values (y) as a NumPy array.
        valid_features (np.ndarray): Validation features (X) as a NumPy array.
        valid_y (np.ndarray): Validation target values (y) as a NumPy array.
        MAX_SAMPLES (int, optional): Maximum target of samples to use for training and validation. (defaults to 100,000)

    Returns:
        sklearn.linear_model.Ridge: Trained Ridge regression model.
    """
    # If the training set is too large, subsample it to reduce computation time.
    if train_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            train_features,
            train_y,
            train_size=MAX_SAMPLES,
            random_state=0,
        )
        train_features = split[0]
        train_y = split[2]

    # If the validation set is too large, subsample it for efficiency.
    if valid_features.shape[0] > MAX_SAMPLES:
        split = train_test_split(
            valid_features,
            valid_y,
            train_size=MAX_SAMPLES,
            random_state=0,
        )
        valid_features = split[0]
        valid_y = split[2]

    # Define a list of alpha values to tune the Ridge model.
    alphas = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    valid_results = []

    # Iterate through alpha values and train Ridge models to find the best alpha.
    for alpha in alphas:
        lr = Ridge(alpha=alpha).fit(train_features, train_y)
        valid_pred = lr.predict(valid_features)

        # Calculate a custom validation score combining RMSE and MAE.
        score = (
            np.sqrt(((valid_pred - valid_y) ** 2).mean())
            + np.abs(valid_pred - valid_y).mean()
        )
        valid_results.append(score)

    # Select the best alpha value based on the validation results.
    best_alpha = alphas[np.argmin(valid_results)]

    # Train a Ridge model with the best alpha.
    lr = Ridge(alpha=best_alpha)
    lr.fit(train_features, train_y)

    return lr


def generate_pred_samples(
    features: np.ndarray, data: np.ndarray, pred_len: int
) -> tuple:
    """
    Generate prediction samples from input features and data.

    Args:
        features (np.ndarray): Input features.
        data (np.ndarray): Inpu data.
        pred_len (int): Length of the prediction.

    Returns:
        tuple: Tuple containing reshaped features and labels.
    """
    n = data.shape[1]  # Number of data points in the input data.

    # Trim input features to match prediction length.
    features = features[:, :-pred_len]
    print(features.shape)
    # Create labels by stacking shifted versions of the input data to match prediction length.
    labels = np.stack(
        [data[:, i : 1 + n + i - pred_len] for i in range(pred_len)], axis=2
    )[:, 1:]

    # Reshape the features and labels for compatibility with modeling.
    return features.reshape(-1, features.shape[-1]), labels.reshape(
        -1, labels.shape[2] * labels.shape[3]
    )


def cal_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Calculate evaluation metrics (MSE and MAE) between predictions and target values.

    Args:
        pred (np.ndarray): Predicted values.
        target (np.ndarray): Target values.

    Returns:
        dict: Dictionary containing calculated metrics (MSE and MAE).
    """
    return {
        "MSE": ((pred - target) ** 2).mean(),
        "MAE": np.abs(pred - target).mean(),
    }


def eval_forecasting(
    model,
    data: np.ndarray,
    train_slice: slice,
    valid_slice: slice,
    test_slice: slice,
    scaler,
    pred_lens: list,
    n_covariate_cols: int
) -> tuple:
    """
    Evaluate a forecasting model on a dataset.

    Args:
        model: The forecasting model to evaluate.
        data (np.ndarray): Input data.
        train_slice (slice): Slice for the training data.
        valid_slice (slice): Slice for the validation data.
        test_slice (slice): Slice for the test data.
        scaler: The scaler used for data transformation.
        pred_lens (list): List of prediction lengths.
        n_covariate_cols (int): Number of covariate columns in the data.
        padding (int, optional): Padding value for generating prediction samples. (defaults to 200)

    Returns:
        tuple: Tuple containing output log and evaluation results.
    """
    t = time.time()

    # Encode input data for forecasting using the given model.
    all_repr = model.encode(data)
    
    # Calculate the time taken for encoding using the model.
    encoder_infer_time = time.time() - t

    # Slice the encoded representations for training, validation, and testing.
    train_repr = all_repr[:, train_slice]
    valid_repr = all_repr[:, valid_slice]
    test_repr = all_repr[:, test_slice]

    # Extract the relevant data for training, validation, and testing.
    train_data = data[:, train_slice, n_covariate_cols:]
    valid_data = data[:, valid_slice, n_covariate_cols:]
    test_data = data[:, test_slice, n_covariate_cols:]

    result = {}
    lr_train_time = {}
    lr_infer_time = {}
    out_log = {}

    # Iterate over different prediction lengths.
    for pred_len in pred_lens:
        # Generate prediction samples for training, validation, and testing.
        train_features, train_labels = generate_pred_samples(
            train_repr, train_data, pred_len, 
        )
        valid_features, valid_labels = generate_pred_samples(
            valid_repr, valid_data, pred_len
        )
        test_features, test_labels = generate_pred_samples(
            test_repr, test_data, pred_len
        )
        
        # Train a Ridge regression model using the training samples.
        t = time.time()
        lr = fit_ridge(
            train_features, train_labels, valid_features, valid_labels
        )
        lr_train_time[pred_len] = time.time() - t

        # Make predictions using the trained Ridge model.
        t = time.time()
        test_pred = lr.predict(test_features)
        lr_infer_time[pred_len] = time.time() - t

        # Reshape the predictions and ground truth to their original shape.
        original_shape = test_data.shape[0], -1, pred_len, test_data.shape[2]
        test_pred = test_pred.reshape(original_shape)
        test_labels = test_labels.reshape(original_shape)

        # Inverse transform the scaled predictions and ground truth to their original scale.
        if test_data.shape[0] > 1:
            test_pred_inv = scaler.inverse_transform(
                test_pred.swapaxes(0, 3)
            ).swapaxes(0, 3)
            test_labels_inv = scaler.inverse_transform(
                test_labels.swapaxes(0, 3)
            ).swapaxes(0, 3)
        else:
            test_pred_inv = scaler.inverse_transform(test_pred)
            test_labels_inv = scaler.inverse_transform(test_labels)

        # Store prediction result and evaluation metrics.
        out_log[pred_len] = {
            "norm": test_pred,
            "raw": test_pred_inv,
            "norm_gt": test_labels,
            "raw_gt": test_labels_inv,
        }

        result[pred_len] = {
            "norm": cal_metrics(test_pred, test_labels),
            "raw": cal_metrics(test_pred_inv, test_labels_inv),
        }

    # Create a dictionary containing evaluation results and timing information.
    eval_res = {
        "result": result,
        "encoder_infer_time": encoder_infer_time,
        "lr_train_time": lr_train_time,
        "lr_infer_time": lr_infer_time,
    }

    return out_log, eval_res
