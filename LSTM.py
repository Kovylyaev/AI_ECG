import math
import torch.nn as nn


class LSTM_ECGs_arithm(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim=128,  # 128
            n_layers=1,  # 2
            dropout=0.0,  # 0.25
    ):
        super().__init__()
        """Initializes LSTM with passed arguments.
        Args:
            input_dim: dim of input vector,
            output_dim: dim of output vector,
            hidden_dim: dim of hidden layer,
            n_layers: num of input layers,
            dropout: probability of randomly zeroed some of the elements of the input tensor.
        """

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

        for name, param in self.named_parameters():
            if name.find("bias") != -1:   # это должно подойти, т.к. у нас функция активации как раз тангенс
                param.data.fill_(0)       # Улучшенная инициализация весов
            else:
                bound = math.sqrt(6)/math.sqrt(param.shape[0]+param.shape[1])
                param.data.uniform_(-bound, bound)

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

        reshaped = (records.swapaxes(0, 1)).swapaxes(0, 2)
        outputs, (hidden, cell) = self.lstm(reshaped)

        predictions = self.fc(outputs[-1])

        return predictions
