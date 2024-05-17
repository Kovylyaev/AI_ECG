import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

NUMDOTSFORSINFILTER = 29
def my_correlate(ecg, peak_filter):
    """Returns the cross-correlation of two given 1D-arrays with increased linearity.
    Args:
        ecg - record of the ECG.
        peak_filter - sin filter, that helps to find peaks.
    Returns:
        ecg correlated with peak_filter.
    """
    temp = [*([ecg[0]] * (NUMDOTSFORSINFILTER - 1)), *ecg, *([ecg[-1]] * (NUMDOTSFORSINFILTER - 1))]
    ecg_add = np.correlate(temp, peak_filter, mode="same")
    ecg = ecg_add[(NUMDOTSFORSINFILTER - 1):-(NUMDOTSFORSINFILTER - 1)]

    return ecg

Const1 = [2] * 100
v = np.linspace(-0.5 * np.pi, 1.5 * np.pi, NUMDOTSFORSINFILTER)
peak_filter = np.sin(v)
const1_transformed_by_my_cor = my_correlate(Const1, peak_filter)
const1_transformed_by_np_cor = np.correlate(Const1, peak_filter, mode="same")



print(peak_filter)
plt.figure(figsize=(10, 3))
plt.title("Differences in correlations")
plt.plot(Const1)
plt.plot(const1_transformed_by_np_cor, c="red")
plt.plot(const1_transformed_by_my_cor, c="green")
plt.gca().legend(("Raw signal", "Numpy correlate", "My correlate"))
plt.xlabel("X")
plt.ylabel("Y")
plt.ylim(-4, 10)
# plt.show()

# rr_peaks500, _ = find_peaks(ecg_transformed, height=2500, distance=120)
# rr_peaks1000 = rr_peaks500 * ecg_len_to_time_ratio
# plt.scatter(rr_peaks1000, ecg_transformed[rr_peaks500], color="red")
# plt.xlim(-100, 10100)
# plt.title("ECG signal - 500 Hz")
plt.show()