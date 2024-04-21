import numpy as np
import wfdb
import os
from os.path import exists
from pathlib import Path
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from torchvision.transforms import ToTensor

from my_dataset import ECGs_Dataset
#
# test_dataset = ECGs_Dataset(ecgs="/Users/aleksandr/PycharmProjects/AI_ECG/all_test_ECGs",
#                             diags="/Users/aleksandr/PycharmProjects/AI_ECG/all_test_Diags")
#
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=0)


print(torch.tensor(5)[0])