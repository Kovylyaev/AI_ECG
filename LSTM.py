import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn
import numpy as np
import time
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from my_dataset import ECGs_Dataset


class LSTM_ECGs_arithm(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=32,  # 128
                 n_layers=1,  # 2
                 dropout=0.,  # 0.25
                 ):
        super().__init__()

        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=False,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.5)

    def forward(self, records):
        """Inputs:

            records, shape is [batch size, num of ecg canals, record len (5000)],

        Intermediate values:

            reshaped, shape is [record len (5000), batch size, num of ecg canals],

            output, shape is [record len (5000), batch size, hid dim],

            hidden/cell, shape is [n layers, batch size, hid dim]

        Outputs hold forward hidden state in the final layer,
        Hidden and cell are the forward hidden and cell states at the final time-step

        Returns:

            prediction, shape is [batch size, output dim]
        """

        reshaped = (records.swapaxes(0,1)).swapaxes(0,2)
        outputs, (hidden, cell) = self.lstm(reshaped)
        p = np.array(reshaped)
        # for i in p:
        #     for j in i:
        #         if any(np.isnan(k) for k in j):
        #             print("NNNNNNAAAAAANNNNNN in reshaped")
        #     #print()
        predictions = self.fc(outputs[-1])

        predictions = self.sigmoid(predictions)

        return predictions
