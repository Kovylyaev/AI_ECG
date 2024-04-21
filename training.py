import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn
import numpy as np
import time
import random
import torch
from my_dataset import ECGs_Dataset
from LSTM import LSTM_ECGs_arithm


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

print("Random set")
train_dataset = ECGs_Dataset(
    ecgs="/Users/aleksandr/PycharmProjects/AI_ECG/all_train_ECGs",
    diags="/Users/aleksandr/PycharmProjects/AI_ECG/all_train_Diags",
)
test_dataset = ECGs_Dataset(
    ecgs="/Users/aleksandr/PycharmProjects/AI_ECG/all_test_ECGs",
    diags="/Users/aleksandr/PycharmProjects/AI_ECG/all_test_Diags",
)

print("Datasets configured")
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=0
)

print("Dataloaders configured")
input_dim = train_dataset.ecgs.shape[1]

output_dim = 2  # arithmia or norm

model = LSTM_ECGs_arithm(input_dim, output_dim)


optimizer = optim.Adam(model.parameters(), lr=1e-2)

criterion = nn.CrossEntropyLoss()


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    y = np.array(y)
    correct = [int(preds[i][y[i]] > preds[i][1 - y[i]]) for i in range(len(y))]
    return sum(correct) / len(correct)


def F_score(preds, y):
    P = 0
    R = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y)):
        if y[i] == 1 and preds[i][0] < preds[i][1]:
            tp += 1
        elif y[i] == 0 and preds[i][0] > preds[i][1]:
            tn += 1
        elif y[i] == 0 and preds[i][0] < preds[i][1]:
            fp += 1
        else:
            fn += 1

    if tp + fp == 0:
        P = 0
    else:
        P = tp / (tp + fp)

    if tp + fn == 0:
        R = 0
    else:
        R = tp / (tp + fn)

    # print(f"P = {P}, R = {R}")
    if P == 0 and R == 0:
        F = 0
    else:
        F = 2 * P * R / (P + R)
    return F


def train(model, loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_F = 0

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

        loss = criterion(predictions, diags)

        acc = categorical_accuracy(predictions, diags)
        F = F_score(predictions, diags)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc
        epoch_F += F

        num += 1
        print(num, end=" ")
        print(f"epoch_F = {epoch_F}, loss = {epoch_loss}")

    print(f"F = {epoch_F / len(loader)}")

    return epoch_loss / len(loader), epoch_acc / len(loader), epoch_F / len(loader)


def evaluate(model, loader, criterion):
    epoch_loss = 0
    epoch_acc = 0
    epoch_F = 0

    model.eval()

    with torch.no_grad():
        for records, diags in loader:
            predictions = model(records.float())

            # predictions = predictions.reshape(-1)

            loss = criterion(predictions, diags)

            acc = categorical_accuracy(predictions, diags)
            F = F_score(predictions, diags)

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_F += F

    return epoch_loss / len(loader), epoch_acc / len(loader), epoch_F / len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10

best_valid_loss = float("inf")

model = model.float()

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc, train_F = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc, test_F = evaluate(model, test_loader, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if test_loss < best_valid_loss:
        best_valid_loss = test_loss
        torch.save(model.state_dict(), "tut1-model.pt")

    print()
    print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
    print(
        f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}% | Train F: {train_F * 100:.2f}"
    )
    print(
        f"\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_loss * 100:.2f}% |  Test. F: {test_F * 100:.2f}"
    )
