import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
import torch
import torch.utils.data

from my_dataset import ECGs_Dataset
from LSTM import LSTM_ECGs_arithm

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
print("Random set")

TRAIN = False

if TRAIN:
    train_dataset = ECGs_Dataset(
        ecgs="/Users/aleksandr/PycharmProjects/AI_ECG/all_train_ECGs",
        diags="/Users/aleksandr/PycharmProjects/AI_ECG/all_train_Diags",
    )
test_dataset = ECGs_Dataset(
    ecgs="/Users/aleksandr/PycharmProjects/AI_ECG/all_test_ECGs",
    diags="/Users/aleksandr/PycharmProjects/AI_ECG/all_test_Diags",
)
print("Datasets configured")

if TRAIN:
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=200, shuffle=True, pin_memory=True, num_workers=0
    )
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=200, shuffle=False, pin_memory=True, num_workers=0
)
print("Dataloaders configured")

if TRAIN:
    input_dim = train_dataset.ecgs.shape[1]

    output_dim = 1  # arithmia or norm

    model = LSTM_ECGs_arithm(input_dim, output_dim)
    model.load_state_dict(torch.load('tut4-model.pt'))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(torch.load('tut4-optimizer.pt'))

    criterion = nn.BCEWithLogitsLoss()


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    correct = [int((preds[i] > 0.5 and y[i] > 0.5) or (preds[i] < 0.5 and y[i] < 0.5)) for i in range(len(y))]
    return sum(correct) / len(correct)


def errors(preds, y):
    """
    Returns True Positive and True Negative per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    tp, tn = 0, 0
    arithm, norm = 0, 0
    for i in range(len(y)):
        if y[i] >= 0.5 and preds[i] > 0.5:
            tp += 1
            arithm += 1
        elif y[i] <= 0.5 and preds[i] < 0.5:
            tn += 1
            norm += 1
        elif y[i] <= 0.5 and preds[i] > 0.5:
            norm += 1
        else:
            arithm += 1

    return tp / arithm if arithm != 0 else 0, tn / norm if norm != 0 else 0


def train(model, loader, optimizer, criterion):
    """
    Trains network
    Args:
        model - network
        loader - data container
        optimizer - model optimizer
        criterion - loss function
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_TP, epoch_TN = 0, 0

    model.train()

    num = 0
    for records, diags in loader:
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

        # predictions = torch.squeeze(predictions)

        loss = criterion(torch.squeeze(predictions), diags)

        acc = categorical_accuracy(predictions, diags)
        TP, TN = errors(predictions, diags)

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
    """
    Tests network
    """
    epoch_loss = 0
    epoch_acc = 0
    epoch_TP, epoch_TN = 0, 0

    model.eval()
    num = 0

    with torch.no_grad():
        for records, diags in loader:
            predictions = model(records.float())

            # predictions = predictions.reshape(-1)

            loss = criterion(torch.squeeze(predictions), diags)

            acc = categorical_accuracy(predictions, diags)
            TP, TN = errors(predictions, diags)

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_TP += TP
            epoch_TN += TN

            num += 1
            print(f"{num}/{len(loader)}    test_epoch_loss = {epoch_loss}")

    return epoch_loss / len(loader), epoch_acc / len(loader), (epoch_TP / len(loader), epoch_TN / len(loader))


def epoch_time(start_time, end_time):
    """
    Calculates epoch time (min and sec from sec)
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if TRAIN:
    N_EPOCHS = 333

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
            torch.save(model.state_dict(), "tut2-model.pt")
            torch.save(optimizer.state_dict(), "tut2-optimizer.pt")

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

input_dim = 12

output_dim = 1  # arithmia or norm

model = LSTM_ECGs_arithm(input_dim, output_dim)
model.load_state_dict(
    torch.load('/Users/aleksandr/PycharmProjects/AI_ECG/FF-model.pt', map_location=torch.device('cpu')))

criterion = nn.BCEWithLogitsLoss()

test_loss, test_acc, test_Errors = evaluate(model, test_loader, criterion)

print(
    f"\t Test. Loss: {test_loss:.5f} |  Test. Acc: {test_acc * 100:.2f}%\n"
    f"Test true arithmia: {test_Errors[0]:.9f}\n"
    f"Test true norm: {test_Errors[1]:.9f}\n"
)
