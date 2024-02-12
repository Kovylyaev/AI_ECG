from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import posixpath
import wfdb
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import find_peaks
from scipy.stats import zscore
from statsmodels.graphics import tsaplots


def timedomain(rr):
    results = {}
    hr = 60000/rr
    # HRV metrics
    results['Среднее HR (beats/min)'] = np.mean(hr)
    results['Среднее HR (Kubios\' style) (beats/min)'] = 60000 / np.mean(rr)
    results['Среднее время между RR (ms)'] = np.mean(rr)
    results['Стандартное отклонение (ms)'] = np.std(rr)
    results['Стандартное отклонение HR (beats/min)'] = np.std(hr)
    return results





record = loadmat('/Users/aleksandr/PycharmProjects/AI_ECG/JS00001.mat')
ecg10 = list(record["val"][10])
ecg_len_to_time_ratio = int(10000.0 / len(ecg10))




v = np.linspace(0.5 * np.pi, 1.5 * np.pi, 15)
peak_filter = np.sin(v)
ecg_transformed = np.correlate(ecg10, peak_filter, mode="same")



plt.figure(figsize=(15, 6))
plt.title('ECG signal - 500 Hz')
plt.plot(range(0, 10000, ecg_len_to_time_ratio), ecg_transformed, alpha=0.8, c='orange')
plt.plot(range(0, 10000, ecg_len_to_time_ratio), ecg10, alpha=1)
plt.gca().legend(('filtered', 'raw signal'))
plt.xlabel('Time (milliseconds)')
#plt.show()




rr_peaks, _ = find_peaks(ecg_transformed, height=1500, distance=120)
plt.scatter(ecg_len_to_time_ratio * rr_peaks, ecg_transformed[rr_peaks], color='red')
plt.xlim(0, 10000)
plt.title("ECG signal - 500 Hz")
plt.show()



rr_peaks *= ecg_len_to_time_ratio
rr_diffs = np.diff(rr_peaks)

hr = len(rr_peaks) * (60 / 10)

rr_E = sum(rr_diffs) / len(rr_diffs) * 2

_rr_diffs = []
for i in rr_diffs:
    _rr_diffs.append((i - rr_E) ** 2)

disperssion = sum(_rr_diffs) / len(rr_diffs)


result = timedomain(rr_diffs)
for key in result.keys():
    print(f"{key} = {result[key]}")
