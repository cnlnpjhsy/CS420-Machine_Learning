import torch
import torch.nn as nn
import numpy as np


class baselineCNN(nn.Module):
    def __init__(self, input_shape: tuple, n_labels: int) -> None:
        super().__init__()
        # input shape: (C, H, W). Here should be (1, 28, 28)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 500),
            nn.ReLU(),
            nn.Linear(500, n_labels)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # x: (batch, C, H, W)
        x = x / 255.0       # regularize to [0.0, 1.0]
        x = self.conv(x)
        x = x.view(x.size()[0], -1) # Flatten
        y = self.fc(x)
        return y
