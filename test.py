import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def loader(records_path, diags_path, train=True):
    records = np.load(records_path)
    diags = np.load(diags_path)

    if train:
        added_str = 'train'
    else:
        added_str = 'test'

    for i in range(len(records)):
        record = np.array(records[i][:500])
        diag = np.array(diags[i])

        np.savez(f"{i}_record_and_diag_{added_str}", record, diag)
        print(i)



loader("/Users/aleksandr/PycharmProjects/AI_ECG/all_train_ECGs", "/Users/aleksandr/PycharmProjects/AI_ECG/all_train_Diags")

loader("/Users/aleksandr/PycharmProjects/AI_ECG/all_test_ECGs", "/Users/aleksandr/PycharmProjects/AI_ECG/all_test_Diags", train=False)


import numpy as np
import torch
import torch.utils.data


class ECGs_Dataset(torch.utils.data.Dataset):
    def __init__(self, ecgs : str, length : int):
        """Initializes Dataset with passed files.
        Args:
            ecgs: file of ecgs,
            diags: file of diagnoses.
        """
        self.ecgs = ecgs
        self.length = length


    def __getitem__(self, idx: int):
        """Returns the object by given index.
        Args:
            idx - index of the record.
        Returns:
            record and diagnosis.
        """

        record = []
        diag = []
        # f = open(f"{idx}{self.ecgs}.npz")
        file = np.load(f"{idx}{self.ecgs}.npz")
        record, diag = file['arr_0'], file['arr_1']
        #f.close()
        #print(record.shape, diag.shape)

        diag = torch.tensor(float(1 - diag))

        return record, diag                 # 1 - аритмия, 0 - норма

    def __len__(self):
        """Returns length of files containing in dataset."""

        return self.length




train_dataset = ECGs_Dataset(
    ecgs="_record_and_diag_train",
    length=len(np.load("/Users/aleksandr/PycharmProjects/AI_ECG/all_train_Diags"))
)
test_dataset = ECGs_Dataset(
    ecgs="_record_and_diag_test",
    length=len(np.load("/Users/aleksandr/PycharmProjects/AI_ECG/all_test_Diags"))
)



train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=100, shuffle=True, pin_memory=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=100, shuffle=False, pin_memory=True, num_workers=2
)




import math

import torch.nn as nn
# import numpy as np


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

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=False,
            dropout=dropout,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

        for name, param in self.named_parameters():
            # nn.init.normal_(param.data, mean=0, std=0.1)
            if name.find("bias") != -1:   # это должно подойти, т.к. у нас функция активации как раз тангенс
                param.data.fill_(0)
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




input_dim = 12

output_dim = 1  # arithmia or norm

model = LSTM_ECGs_arithm(input_dim, output_dim)
#model.load_state_dict(torch.load('/content/drive/MyDrive/AI_ECG/tut4-model.pt'))


optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
#optimizer.load_state_dict(torch.load('/content/drive/MyDrive/AI_ECG/tut4-optimizer.pt'))

criterion = nn.BCEWithLogitsLoss()





def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    correct = [int((preds[i] > 0.5 and y[i].item() > 0.5) or (preds[i] < 0.5 and y[i].item() < 0.5)) for i in range(len(y))]
    return sum(correct) / len(correct)


def Errors(preds, y):
    tp, tn, fp, fn = 0, 0, 0, 0
    arithm, norm = 0, 0
    for i in range(len(y)):
        if y[i].item() >= 0.5 and preds[i] > 0.5:
            tp += 1
            arithm += 1
        elif y[i].item() <= 0.5 and preds[i] < 0.5:
            tn += 1
            norm += 1
        elif y[i].item() <= 0.5 and preds[i] > 0.5:
            fp += 1
            norm += 1
        else:
            fn += 1
            arithm += 1

    return tp / arithm if arithm != 0 else 0, tn / norm if norm != 0 else 0#, fp, fn




def train(model, loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_TP, epoch_TN = 0, 0

    model.train()

    num = 0
    for records, diags in loader:

        records = records
        diags = diags

        optimizer.zero_grad()

        # records = [batch size, num of ecg canals, record len (5000)]

        predictions = model(records.float())

        # predictions = [batch size, output dim]
        # diags = [batch size, output_dim]

        # predictions = predictions.view(-1, predictions.shape[-1])
        # tags = tags.view(-1)
        #
        # # predictions = [sent len * batch size, output dim]
        # # diags = [sent len * batch size]
        # predictions = predictions.reshape(-1)

        #predictions = torch.squeeze(predictions)

        loss = criterion(torch.squeeze(predictions), diags)

        acc = categorical_accuracy(predictions, diags)
        TP, TN = Errors(predictions, diags)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_TP += TP
        epoch_TN += TN

        num += 1
        print(f"{num}/{len(loader)}    epoch_loss = {epoch_loss}")

    return epoch_loss / len(loader), epoch_acc / len(loader), (epoch_TP / len(loader), epoch_TN / len(loader))





def evaluate(model, loader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_TP, epoch_TN = 0, 0

    model.eval()

    with torch.no_grad():
        for records, diags in loader:

            records = records
            diags = diags

            predictions = model(records.float())

            # predictions = predictions.reshape(-1)

            loss = criterion(torch.squeeze(predictions), diags)

            acc = categorical_accuracy(predictions, diags)
            TP, TN = Errors(predictions, diags)

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_TP += TP
            epoch_TN += TN

    return epoch_loss / len(loader), epoch_acc / len(loader), (epoch_TP / len(loader), epoch_TN / len(loader))




def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




N_EPOCHS = 1000

best_valid_loss = float("inf")

model = model.float()

Train_Loss, Train_Acc, Train_TP, Train_TN = [], [], [], []
Test_Loss, Test_Acc, Test_TP, Test_TN = [], [], [], []

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc, train_Errors = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc, test_Errors = evaluate(model, test_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if test_loss < best_valid_loss:
        best_valid_loss = test_loss
        torch.save(model.state_dict(), "/content/drive/MyDrive/AI_ECG/FFF-model.pt")
        torch.save(optimizer.state_dict(), "/content/drive/MyDrive/AI_ECG/FFF-optimizer.pt")


    print()
    print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(
        f"\tTrain Loss: {train_loss:.5f} | Train Acc: {train_acc * 100:.2f}%\n"
        f"Train true arithmia: {train_Errors[0]:.9f}\n"
        f"Train true norm: {train_Errors[1]:.9f}\n"
    )
    print(
        f"\t Test. Loss: {test_loss:.5f} |  Test. Acc: {test_acc * 100:.2f}%\n"
        f"Test true arithmia: {test_Errors[0]:.9f}\n"
        f"Test true norm: {test_Errors[1]:.9f}\n"
    )

    Train_Loss.append(train_loss)
    Train_Acc.append(train_acc)
    Train_TP.append(train_Errors[0])
    Train_TN.append(train_Errors[1])

    Test_Loss.append(test_loss)
    Test_Acc.append(test_acc)
    Test_TP.append(test_Errors[0])
    Test_TN.append(test_Errors[1])


    fig, ax = plt.subplots(2,3, figsize=(10,8))
    ax[0, 0].plot(Train_Loss)
    ax[0, 0].set_title('Train Loss')
    ax[1, 0].plot(Test_Loss)
    ax[1, 0].set_title('Test Loss')

    ax[0, 1].plot(Train_TP)
    ax[0, 1].set_title('Train TP')
    ax[1, 1].plot(Test_TP)
    ax[1, 1].set_title('Test TP')

    ax[0, 2].plot(Train_TN)
    ax[0, 2].set_title('Train TN')
    ax[1, 2].plot(Test_TN)
    ax[1, 2].set_title('Test TN:')
    plt.show()
