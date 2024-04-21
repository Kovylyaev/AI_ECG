import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks


def timedomain(rr):
    results = {}
    hr = 60000 / rr
    # HRV metrics
    results["Среднее HR (beats/min)"] = np.mean(hr)
    results["Среднее HR (Kubios' style) (beats/min)"] = 60000 / np.mean(rr)
    results["Среднее время между RR (ms)"] = np.mean(rr)
    results["Стандартное отклонение (ms)"] = np.std(rr)
    results["Стандартное отклонение HR (beats/min)"] = np.std(hr)
    return results


def my_correlate(ecg10, peak_filter):
    temp = [*([ecg10[0]] * 29), *(ecg10), *([ecg10[-1]] * 29)]
    ecg10_add = np.correlate(temp, peak_filter, mode="same")
    ecg10 = ecg10_add[29:-29]

    return ecg10


record = loadmat("/Users/aleksandr/PycharmProjects/AI_ECG/JS23462.mat")
ecg10 = np.array(list(record["val"][10]))
ecg_len_to_time_ratio = int(10000.0 / len(ecg10))


v = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 15)
peak_filter = np.sin(v)
ecg_transformed = my_correlate(ecg10, peak_filter)


plt.figure(figsize=(15, 6))
plt.title("ECG signal - 500 Hz")
plt.plot(range(0, 10000, ecg_len_to_time_ratio), ecg_transformed, alpha=0.8, c="orange")
plt.plot(range(0, 10000, ecg_len_to_time_ratio), ecg10, alpha=1)
plt.gca().legend(("filtered", "raw signal"))
plt.xlabel("Time (milliseconds)")
# plt.show()


rr_peaks500, _ = find_peaks(ecg_transformed, height=2500, distance=120)
rr_peaks1000 = rr_peaks500 * ecg_len_to_time_ratio
plt.scatter(rr_peaks1000, ecg_transformed[rr_peaks500], color="red")
plt.xlim(-100, 10100)
plt.title("ECG signal - 500 Hz")
plt.show()


def HR(rr_peaks):
    rr_diffs = np.diff(rr_peaks)

    rr_E = sum(rr_diffs) / len(rr_diffs)

    _rr_diffs = []
    for i in rr_diffs:
        _rr_diffs.append((i - rr_E) ** 2)

    disperssion = sum(_rr_diffs) / len(rr_diffs)

    result = timedomain(rr_diffs)
    for key in result.keys():
        print(f"{key} = {result[key]}")
    print(
        f"Дисперсия времени между зубцами R (квадрат отклонения от мат ожид) (ms) = {disperssion}"
    )


HR(rr_peaks1000)


# Проверка первого и последнего зубца на полное вхождение в запись

min_dist_to_left_edge = int(
    min(500, rr_peaks500[1] - rr_peaks500[0])
)  # минимальное расстояние до границ записи
min_dist_to_right_edge = int(
    min(500, rr_peaks500[-1] - rr_peaks500[-2])
)  # в миллисекундах
plt.figure(figsize=(15, 3))
plt.plot(range(0, 10000, ecg_len_to_time_ratio), ecg10, alpha=1)

ecg10_cutted = ecg10
left_offset = 0

if 10000 - rr_peaks1000[-1] < min_dist_to_right_edge:
    ecg10_cutted = ecg10_cutted[: rr_peaks500[-1]]

if rr_peaks1000[0] < min_dist_to_left_edge:
    ecg10_cutted = ecg10_cutted[
        rr_peaks500[0]:
    ]  # обрезали по первому и последнему пику,
    left_offset += rr_peaks500[0]  # если они слишком близко к краю

ecg_cutted_transformed = my_correlate(ecg10_cutted, peak_filter)
rr_peaks_new, _ = find_peaks(ecg_cutted_transformed, height=2500, distance=120)
right_edge = rr_peaks_new[-1] + int(min_dist_to_right_edge / 2)
left_edge = rr_peaks_new[0] - int(min_dist_to_left_edge / 2)
ecg10_cutted = ecg10_cutted[:right_edge]  # обрезаем, если они
ecg10_cutted = ecg10_cutted[left_edge:]  # слишком далеко от края записи
left_offset += left_edge
left_offset *= ecg_len_to_time_ratio


plt.title("ECG signal - 500 Hz")
lst = list(range(0, len(ecg10_cutted) * ecg_len_to_time_ratio, ecg_len_to_time_ratio))
for i in range(len(lst)):
    lst[i] += left_offset
plt.plot(lst, ecg10_cutted, alpha=0.8, c="orange")
plt.gca().legend(("raw signal", "cutted"))
plt.xlabel("Time (milliseconds)")
plt.show()
