from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from encoder import *
from loss import *
from utils import centerize_vary_length_series, create_data_segments


class LinearPred(torch.nn.Module):
    def __init__(
        self, input_dims: int, input_len: int, output_dims: int, output_len: int
    ):
        """
        Initialize a Linear model for predictions.

        Args:
            input_dims (int): Dimensionality of the input data.
            input_len (int): Length of the sequence.
            output_dims (int): Dimensionality of the output data.
            output_len (int): Length of the output sequence.
        """
        super(LinearPred, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = "cpu"

        # Linear dense layers.
        self.linear1 = torch.nn.Linear(input_len, output_len).to(self.device)
        self.linear2 = torch.nn.Linear(input_dims, output_dims).to(self.device)

        # Dropout layer with probability 0.25.
        self.dropout = torch.nn.Dropout(0.25)

        # ReLU activation layer.
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        # Apply the first linear layer and ReLU activation and transpose the tensor to match expected shape.
        x_pred = self.relu(self.linear1(x)).transpose(1, 2)

        # Apply second linear layer.
        x_pred2 = self.linear2(x_pred)

        return x_pred2


class SimTS(torch.nn.Module):
    """
    Initialize a SimTS model.

    Args:
        input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
        K (int): The length of history segmentation.
        kernel_list (list): List of kernel sizes for the encoder.
        output_dims (int): The representation dimension.
        lr (float): The learning rate.
        batch_size (int): The batch size.
        T (int or None): The maximum allowed sequence length for training. Sequences longer than this will be split into smaller sequences.
        after_iter_callback (Callable or None): A callback function called after each iteration.
        after_epoch_callback (Callable or None): A callback function called after each epoch.
    """

    def __init__(
        self,
        input_dims: int,
        K: int,
        kernel_list: list,
        output_dims: int = 320,
        lr: float = 0.001,
        batch_size: int = 8,
        T: int | None = None,
        after_iter_callback: callable | None = None,
        after_epoch_callback: callable | None = None,
    ):
        super(SimTS, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = "cpu"

        # Store model parameters
        self.input_dims = input_dims
        self.lr = lr
        self.batch_size = batch_size
        self.T = T
        self.K = K

        # Initialize encoder network.
        self.net = CausalCNNEncoder(
            in_channels=batch_size,  # input_dims,
            reduced_size=320,
            component_dims=output_dims,
            kernel_list=kernel_list,
        ).to(self.device)

        # Store callback functions.
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback

        # Epoch and iteration counters.
        self.n_epochs = 0
        self.n_iters = 0

        self.timestep = T - K  # Timestep size.

        # Dropout layer and Linear predictor module.
        self.dropout = torch.nn.Dropout(0.9, inplace=False)
        self.predictor = LinearPred(output_dims, 1, output_dims, self.timestep)

        self.loss = cosine_loss  # Apply cosine loss.

    def fit(self, train_data: np.ndarray, n_epochs=None, n_iters=None, verbose=False):
        assert train_data.ndim == 3  # Check if input data has 3D shape.

        # Determine number of training iterations if not provided
        if n_iters is None and n_epochs is None:
            # 300 if the data size is small, otherwise 600
            n_iters = 300 if train_data.size <= 100000 else 600

        # Create data segments and adjust input data for training.
        train_data = create_data_segments(train_data, self.T, self.K)

        # Identify timesteps with all NaN values.
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        # If first or last timestep contains all NaN values, adjust the data by centering it.
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)

        # Remove samples where all values are NaN
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]

        # Create a PyTorch TensorDataset and DataLoader
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.batch_size, len(train_dataset)),
            shuffle=True,
            drop_last=True,
        )

        # SGD optimizer with separate learning rates for different model parts.
        optimizer = torch.optim.SGD(
            [
                {  # Causal Convolution Block parameters
                    "params": list(self.net.parameters())
                },
                {  # Linear predictor parameters.
                    "params": list(self.predictor.parameters())
                },
            ],
            lr=self.lr,
        )

        # Initialize logging variables and early stopping.
        loss_log = []
        early_stop_step = 0
        best_loss = 100.0
        best_net = None

        # Training loop.
        while True:
            # Check if the epochs number is reached.
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break

            cum_loss = 0  # Batch loss accumulator.
            n_epochs_iters = 0  # Iterations within the epoch.

            # Flag to break training when iterations number is reached.
            interrupted = False

            for seq in train_loader:
                # Check if iterations number is reached.
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break

                # Extract a sequence batch.
                if self.T is not None and seq[0].size(1) > self.T:
                    window_offset = np.random.randint(seq[0].size(1) - self.T + 1)
                    x = seq[0][:, window_offset : window_offset + self.T]
                else:
                    x = seq[0]

                # Split sequence into historical and future data.
                x_h = x[:, : self.K, :].clone().to(self.device)
                x_f = x[:, self.K : self.T, :].clone().to(self.device)

                optimizer.zero_grad()

                torch.cuda.empty_cache()

                # Pass through the encoder network.
                z_h, z_h_last, z_f = self.net(x_h, x_f)

                # Encode and forecast future embeddings
                encode_future_embeds = z_f.to(self.device)

                fcst_future_embeds = self.predictor(z_h_last.unsqueeze(-1))

                # Calculate the loss
                loss = self.loss(encode_future_embeds, fcst_future_embeds)

                # Backpropagation and optimization step.
                loss.backward()
                optimizer.step()

                cum_loss += loss.item()
                n_epochs_iters += 1
                self.n_iters += 1

                # Invoke the iteration callback if provided.
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self)

            # Check if training is interrupted.
            if interrupted:
                break

            # Calculate average epoch loss.
            cum_loss /= n_epochs_iters
            loss_log.append(cum_loss)

            # Implement early stopping based on loss.
            if best_loss - cum_loss <= 0.0001:
                early_stop_step += 1
            else:
                early_stop_step = 0

            # Track the best model.
            if best_loss > cum_loss:
                best_loss = cum_loss
                best_net = self.net

            # Training logs.
            if verbose:
                print(f"Epoch #{self.n_epochs}. Loss: {cum_loss}")

            self.n_epochs += 1

            # Invoke the epoch callback if provided.
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self)

        return loss_log, best_net

    def eval_with_pooling(self, x):
        # Pass input data through encoder network, disabling training
        # Take only the last representation.
        _, last, _ = self.net(x.to(self.device, non_blocking=True), training=False)

        last = last.unsqueeze(1)

        return last.cpu()

    def encode(self, data: torch.Tensor | np.ndarray, batch_size=None):
        """
        Encode input data using the trained SimTS model.

        Args:
            data (torch.Tensor or np.ndarray): Input data tensor or array with shape (n_samples, sequence_length, input_dims).
            batch_size (int, optional): Batch size for processing data.

        Returns:
            np.ndarray: Encoded representations of the input data.
        """
        # Check if encoder network is trained.
        assert self.net is not None, "Please train or load a net first."

        # Check input data has correct dimensionality.
        if data.ndim != 3:
            data = data.unsqueeze(0)
        assert data.ndim == 3

        # Convert input data to a NumPy array if it's not.
        if not isinstance(data, np.ndarray):
            data = data.clone().detach().cpu().numpy()

        # If it is not provided, use the model's batch size.
        if batch_size is None:
            batch_size = self.batch_size

        # Save original training mode of the encoder network and set it to evaluation mode.
        orig_training = self.net.training
        self.net.eval()

        # Create a PyTorch dataset and data loader.
        dataset = TensorDataset(torch.from_numpy(data.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size)

        # Perform encoding in batches.
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                out = self.eval_with_pooling(x)

                output.append(out)

            # Concatenate the outputs.
            output = torch.cat(output, dim=0)

        # Restore the original training mode of the network.
        self.net.train(orig_training)

        return output.numpy()

    def save(self, filename: str):
        """ 
        Save the model to a file.
        """
        torch.save(self.net.state_dict(), filename)