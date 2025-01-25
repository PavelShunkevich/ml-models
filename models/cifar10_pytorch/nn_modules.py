import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A residual block as described in the original ResNet paper.
    It consists of two convolutional layers with batch normalization
    and a skip connection that allows the network to learn residual functions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride of the first convolutional layer. Defaults to 1.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class NetDropout(nn.Module):
    """
    A convolutional neural network with residual blocks and dropout.

    Args:
         n_chans1 (int): Number of channels in the first convolutional layer. Defaults to 64.
    """

    def __init__(self, n_chans1: int = 64):
        super().__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_chans1)
        self.relu = nn.ReLU(inplace=True)
        self.res_block1 = ResidualBlock(n_chans1, n_chans1 * 2, stride=2)
        self.res_block2 = ResidualBlock(n_chans1 * 2, n_chans1 * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(n_chans1 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
             x (torch.Tensor): Input tensor.

        Returns:
             torch.Tensor: Output tensor.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.fc2(out)
        return out