import torch
from einops import rearrange, reduce


class CausalConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of the applied non-residual convolutions.
        dilation (int): Dilation parameter of non-residual convolutions.
        final (bool): Disables, if True, the last activation function.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False,
    ):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal.
        self.padding = (kernel_size - 1) * dilation
        padding = self.padding

        # First causal convolution.
        self.conv1 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.dropout1 = torch.nn.Dropout(0.1)

        # Second causal convolution.
        self.conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.dropout2 = torch.nn.Dropout(0.1)

        # Residual connection.
        self.resconn = (
            torch.nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

        # Final activation function.
        self.relu = torch.nn.ReLU() if final else None

    def forward(self, x):
        # First causal convolution layer.
        out_causal = self.conv1(x)
        out_causal = self.dropout1(torch.nn.functional.gelu(out_causal))
        # Second causal convolution layer.
        out_causal = self.conv2(out_causal)
        out_causal = self.dropout2(torch.nn.functional.gelu(out_causal))
        # Residual connection, if exists.
        res = (
            x if self.resconn is None else self.resconn(x)
        )  

        if self.relu is None:
            x = out_causal + res  # Add residual connection to the output
        else:
            x = self.relu(out_causal + res)

        return x


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a Causal CNN: the computed representation is
    the output of a fully-connected layer applied to the output of an adaptive
    max-pooling layer applied on top of the Causal CNN, which reduces the length
    of the time series to a fixed size.
    Takes as input a three dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input.

    Args:
        in_channels (int): Number of input channels.
        reduced_size (int): Fixed length to which the output time series of the Causal CNN is reduced.
        component_dims (int): Number of output channels for each kernel size in the Causal CNN.
        kernel_list (list): A list of kernel sizes used in the Causal CNN.
            The kernel sizes are 2^i for i in [0, 1, .., m]; where m = [log K]+1 (K is the length of the historical data).
    """

    def __init__(
        self,
        in_channels: int,
        reduced_size: int,
        component_dims: int,
        kernel_list: list = [1, 2, 4, 8, 16, 32, 64, 128],
    ):
        super(CausalCNNEncoder, self).__init__()

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = "cpu"

        self.input_fc = CausalConvolutionBlock(in_channels, reduced_size, 1, 1)
        self.repr_dropout = torch.nn.Dropout(0.1)
        self.kernel_list = kernel_list

        # Initialize a list of convolutional layers with varying kernel sizes.
        self.multi_cnn = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(reduced_size, component_dims, k, padding=k - 1)
                for k in kernel_list
            ]
        )

    def forward(self, x_h, x_f=None, training=True):
        # Replace NaN values with zeros in historical data.
        nan_mask_h = ~x_h.isnan().any(axis=-1)
        x_h[~nan_mask_h] = 0

        x_h = x_h.transpose(2, 1)  # Transpose input data to apply convolution block.
        x_h = self.input_fc(x_h)  # Causal Convolution Block to historical data.
        x_h = x_h.transpose(2, 1)  # Transpose back to the original shape.
        x_h = x_h.transpose(2, 1)

        # Training.
        if training:
            # Replace NaN values with zeros in future data.
            nan_mask_f = ~x_f.isnan().any(axis=-1)
            x_f[~nan_mask_f] = 0

            x_f = x_f.transpose(2, 1)  # Transpose input data to apply convolution block.
            x_f = self.input_fc(x_f)  # Causal Convolution Block to future data.
            x_f = x_f.transpose(2, 1)  # Transpose back to the original shape.
            x_f = x_f.transpose(2, 1)

            # Store intermediate results.
            trend_h, trend_h_weights = [], []
            trend_f, trend_f_weights = [], []

            for idx, mod in enumerate(self.multi_cnn):
                # Apply convolutional layers to historical and future data.
                out_h = mod(x_h)
                out_f = mod(x_f)

                if self.kernel_list[idx] != 1:
                    # Truncates the output tensors to ensure they have the same length.
                    out_h = out_h[..., : -(self.kernel_list[idx] - 1)]
                    out_f = out_f[..., : -(self.kernel_list[idx] - 1)]

                # Store processed data and weights of the last time step.
                trend_h.append(out_h.transpose(1, 2))
                trend_h_weights.append(out_h.transpose(1, 2)[:, -1, :].unsqueeze(-1))
                trend_f.append(out_f.transpose(1, 2))
                trend_f_weights.append(out_f.transpose(1, 2)[:, -1, :].unsqueeze(-1))

            # Combine results from different convolutional layers.
            trend_h = reduce(
                rearrange(trend_h, "list b t d -> list b t d"),
                "list b t d -> b t d",
                "mean",
            )
            trend_f = reduce(
                rearrange(trend_f, "list b t d -> list b t d"),
                "list b t d -> b t d",
                "mean",
            )

            trend_h = self.repr_dropout(trend_h)  # Dropout regularization.
            trend_h_last = trend_h[:, -1, :]  # Representation of last time step.

            # Gradients are not computed during backpropagation for future data.
            return trend_h, trend_h_last, trend_f.detach() 

        # Inference.
        else:  # Future data is not used during inference.
            trend_h, trend_h_weights = [], []

            for idx, mod in enumerate(self.multi_cnn):
                out_h = mod(x_h)

                if self.kernel_list[idx] != 1:
                    out_h = out_h[..., : -(self.kernel_list[idx] - 1)]

                trend_h.append(out_h.transpose(1, 2))
                trend_h_weights.append(out_h.transpose(1, 2)[:, -1, :].unsqueeze(-1))

            trend_h = reduce(
                rearrange(trend_h, "list b t d -> list b t d"),
                "list b t d -> b t d",
                "mean",
            )

            trend_h = self.repr_dropout(trend_h)
            trend_h_last = trend_h[:, -1, :]

            return trend_h, trend_h_last, None
