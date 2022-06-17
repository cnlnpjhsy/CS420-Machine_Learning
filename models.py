import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
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


class baselineRNN(nn.Module):
    def __init__(self, input_size: int, n_labels: int, hidden_size=256) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(hidden_size * 2, 512)
        self.linear2 = nn.Linear(512, 256)
        self.clf = nn.Linear(256, n_labels)

    def forward(self, batch_x, batch_x_len):
        packed_input = rnn_utils.pack_padded_sequence(
            batch_x, batch_x_len, batch_first=True, enforce_sorted=False
        )
        packed_output, (hx, cx) = self.lstm(packed_input)   # hx: (2, batch_size, hidden_size)
        hx_L = hx[-2]
        hx_R = hx[-1]
        x = torch.cat((hx_L, hx_R), dim=1)  # x: (batch_size, hidden_size * 2)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y = self.clf(x)

        return y


class SimpleTwoBranch(nn.Module):
    def __init__(self, input_size: int, image_shape: tuple, n_labels: int, hidden_size=256) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.conv = nn.Sequential(
            nn.Conv2d(image_shape[0], 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        conv_out_size = self._get_conv_out(image_shape)
        lstm_out_size = hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + lstm_out_size, 500),
            nn.ReLU(),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Linear(250, n_labels)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, seq_x, img_x, seq_x_len):
        packed_input = rnn_utils.pack_padded_sequence(
            seq_x, seq_x_len, batch_first=True, enforce_sorted=False
        )
        packed_output, (hx, cx) = self.lstm(packed_input)
        hx_L = hx[-2]
        hx_R = hx[-1]
        seq_x = torch.cat((hx_L, hx_R), dim=1)

        img_x = self.conv(img_x / 255.0)
        img_x = img_x.view(img_x.size()[0], -1) # Flatten

        x = torch.cat([seq_x, img_x], dim=1)
        y = self.fc(x)

        return y
