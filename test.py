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


def a(i):
     if i == 0:
          return 0
     raise 13

try:
     print(a(1))
except:
     print()