import numpy as np
import wfdb
import os
from os.path import exists
from pathlib import Path
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable


# paths = Path("/Users/aleksandr/PycharmProjects/AI_ECG/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0").rglob("*.mat")
# sorted_paths = sorted(paths)


b = torch.tensor([[[1,5], [2,6]],
                  [[3,7], [4,8]]])
b = b.swapaxes(0,1)
print(0)