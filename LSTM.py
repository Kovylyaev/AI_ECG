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




SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

print(0)
train_dataset = ECGs_Dataset(ecgs="/Users/aleksandr/PycharmProjects/AI_ECG/all_train_ECGs",
                             diags="/Users/aleksandr/PycharmProjects/AI_ECG/all_train_Diags")
test_dataset = ECGs_Dataset(ecgs="/Users/aleksandr/PycharmProjects/AI_ECG/all_test_ECGs",
                             diags="/Users/aleksandr/PycharmProjects/AI_ECG/all_test_Diags")

print(1)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=2)



print(2)
class LSTM_ECGs_arithm(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim=32,  # 128
                 n_layers=1,  # 2
                 dropout=0,  # 0.25
                 ):
        super().__init__()

        self.lstm = nn.LSTM(input_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            bidirectional=False,
                            dropout=dropout if n_layers > 1 else 0)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def forward(self, records):
        """Inputs:

            records, shape is [batch size, num of ecg canals, record len (5000)],

        Intermediate values:

            reshaped, shape is [record len (5000), batch size, num of ecg canals],

            output, shape is [record len (5000), batch size, hid dim],

            hidden/cell, shape is [n layers, batch size, hid dim]

        Outputs hold forward hidden state in the final layer,
        Hidden and cell are the  forward hidden and cell states at the final time-step

        Returns:

            prediction, shape is [record len (5000), batch size, output dim]
        """

        reshaped = self.dropout((records.swapaxes(0,1)).swapaxes(0,2))
        outputs, (hidden, cell) = self.lstm(reshaped)
        predictions = self.fc(self.dropout(outputs))

        return predictions




print(3)
# input_dim = len(TEXT.vocab)
#
# output_dim = len(UD_TAGS.vocab)
#
# pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
#
# model = BiLSTMPOSTagger(input_dim,
#                         output_dim,
#                         pad_idx)


print(42)