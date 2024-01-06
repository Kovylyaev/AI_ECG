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


def get_HR_times(signals):
    HR_times = []
    for i in range(1, len(signals) - 1):
        if signals[i][0] >= 0:
            if signals[i - 1][0] <= signals[i][0] >= signals[i + 1][0]:
                HR_times.append(i / 360)

    return HR_times


def binary_search_not_higher(array, elem):
    l = 0
    r = len(array)
    while r - l > 1:
        mid = (r + l) // 2
        if array[mid] <= elem:
            l = mid
        else:
            r = mid
    return l

def binary_search_not_less(array, elem):
    l = -1
    r = len(array)
    while r - l > 1:
        mid = (r + l) // 2
        if array[mid] < elem:
            l = mid
        else:
            r = mid
    return r


def get_HR_per_unit_per_period(HR_times, unit = 60, period_begin = 0, period_end = 99999999):
    period_end = min(period_end, HR_times[-1])
    begin = binary_search_not_less(HR_times, period_begin)
    end = binary_search_not_higher(HR_times, period_end)              # not_higher, чтобы включить границы
    count = end - begin + 1
    return count / (period_end - period_begin) * unit


def timedomain(rr):
    results = {}
    hr = 60000/rr
    # HRV metrics
    results['Mean RR (ms)'] = np.mean(rr)
    results['STD RR/SDNN (ms)'] = np.std(rr)
    results['Mean HR (Kubios\' style) (beats/min)'] = 60000 / np.mean(rr)
    results['Mean HR (beats/min)'] = np.mean(hr)
    results['STD HR (beats/min)'] = np.std(hr)
    results['Min HR (beats/min)'] = np.min(hr)
    results['Max HR (beats/min)'] = np.max(hr)
    return results






record = loadmat('/Users/aleksandr/PycharmProjects/AI_ECG/JS00001.mat')
ecg10 = list(record["val"][10])
for i in range(4999, 1, -1):
    ecg10.insert(i, (ecg10[i - 1] + ecg10[i]) / 2)
# plt.plot(ecg10)
# plt.show()


v = np.linspace(0.5 * np.pi, 1.5 * np.pi, 15)
peak_filter = np.sin(v)
ecg_transformed = np.correlate(ecg10, peak_filter, mode="same")

plt.figure(figsize=(15,6))
plt.title('ECG signal - 500 Hz')
plt.plot(ecg_transformed, alpha = 0.8, c='orange')
plt.plot(ecg10, alpha = 1)
plt.gca().legend(('filtered', 'raw signal'))
plt.xlabel('Time (milliseconds)')
#plt.show()


diff_sig_ecg = np.diff(ecg_transformed)
rr_peaks, _ = find_peaks(ecg_transformed, height=1500, distance=500 * 3 / 5)
plt.plot(ecg_transformed, alpha = 0.8)
plt.scatter(rr_peaks, ecg_transformed[rr_peaks], color='red')
plt.xlim(0,10000)
plt.title("ECG signal - 500 Hz")
plt.show()



rr_ecg = np.diff(rr_peaks)

x_ecg = np.cumsum(rr_ecg)/1000
f_ecg = interp1d(x_ecg, rr_ecg, kind='cubic', fill_value= 'extrapolate')
fs = 4
steps = 1 / fs
# sample using the interpolation function
xx_ecg = np.arange(0, np.max(x_ecg), steps)
rr_interpolated_ecg = f_ecg(xx_ecg)
plt.subplot(211)
plt.title('rr-intervals')
plt.plot(x_ecg, rr_ecg, color='k', markerfacecolor='#A999D1',marker='o')
plt.ylabel('rr-interval (ms)')
plt.subplot(212)

plt.title('rr-intervals (cubic interpolation)')
plt.plot(xx_ecg, rr_interpolated_ecg, color='r')
plt.xlabel('Time (s)')
plt.ylabel('RR-interval (ms)')
plt.show()



rr_ecg[np.abs(zscore(rr_ecg)) > 2] = np.median(rr_ecg)
x_ecg = np.cumsum(rr_ecg)/1000
f_ecg = interp1d(x_ecg, rr_ecg, kind='cubic', fill_value= 'extrapolate')
xx_ecg = np.arange(0, np.max(x_ecg), steps)
clean_rr_interpolated_ecg = f_ecg(xx_ecg)
plt.figure(figsize=(25,5))
plt.title('Error using z-score')
plt.plot(rr_interpolated_ecg)
plt.plot(clean_rr_interpolated_ecg)
plt.xlabel('Time (s)')
plt.ylabel('RR-interval (ms)')
plt.show()

print(timedomain(rr_ecg))


# HR_times = get_HR_times(record)
# print(get_HR_per_unit_per_period(HR_times, 60))
#
#
# wfdb.plot_wfdb(record=record, title='Record')
# display(record.__dict__)
