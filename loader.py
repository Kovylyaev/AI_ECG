import os
import random
from os.path import exists
from pathlib import Path
import numpy as np
import wfdb
from matplotlib import pyplot as plt
from scipy.signal import find_peaks


PADDING = 0
NUMDOTSFORSINFILTER = 30


def my_correlate(ecg, peak_filter):
    temp = [*([ecg[0]] * (NUMDOTSFORSINFILTER - 1)), *ecg, *([ecg[-1]] * (NUMDOTSFORSINFILTER - 1))]
    ecg_add = np.correlate(temp, peak_filter, mode="same")
    ecg = ecg_add[(NUMDOTSFORSINFILTER - 1):-(NUMDOTSFORSINFILTER - 1)]

    return ecg


def cut_n_fill(ecg):
    ecg10 = ecg[10]
    ecg = np.array(ecg)
    ecg_len_to_time_ratio = int(10000.0 / len(ecg10))

    v = np.linspace(-0.5 * np.pi, 1.5 * np.pi, NUMDOTSFORSINFILTER)
    peak_filter = np.sin(v)
    ecg_transformed = my_correlate(ecg10, peak_filter)

    rr_peaks500, _ = find_peaks(ecg_transformed, height=2500, distance=120)
    if len(rr_peaks500) == 0:
        raise 13
    rr_peaks1000 = rr_peaks500 * ecg_len_to_time_ratio

    # Проверка первого и последнего зубца на полное вхождение в запись
    min_dist_to_left_edge = int(
        min(500, rr_peaks500[1] - rr_peaks500[0])
    )  # минимальное расстояние до границ записи
    min_dist_to_right_edge = int(
        min(500, rr_peaks500[-1] - rr_peaks500[-2])
    )  # в миллисекундах

    ecg10_cutted = ecg10
    left_offset = 0

    if 10000 - rr_peaks1000[-1] < min_dist_to_right_edge:
        ecg10_cutted = ecg10_cutted[: rr_peaks500[-1]]
        ecg = ecg[:, : rr_peaks500[-1]]

    if rr_peaks1000[0] < min_dist_to_left_edge:
        ecg10_cutted = ecg10_cutted[
            rr_peaks500[0]:
        ]  # обрезали по первому и последнему пику,
        left_offset += rr_peaks500[0]  # если они слишком близко к краю
        ecg = ecg[:, rr_peaks500[0]:]

    ecg_cutted_transformed = my_correlate(ecg10_cutted, peak_filter)
    rr_peaks_new, _ = find_peaks(ecg_cutted_transformed, height=2500, distance=120)
    if len(rr_peaks_new) == 0:
        raise 13
    right_edge = rr_peaks_new[-1] + int(min_dist_to_right_edge / 2)
    left_edge = rr_peaks_new[0] - int(min_dist_to_left_edge / 2)
    ecg10_cutted = ecg10_cutted[:right_edge]  # обрезаем, если они
    ecg10_cutted = ecg10_cutted[left_edge:]  # слишком далеко от края записи
    ecg = ecg[:, :right_edge]
    ecg = ecg[:, left_edge:]

    n = 5000 - len(ecg10_cutted)
    if left_edge < 0 or right_edge < 0 or n > 2000:
        raise 13
    ecg = np.pad(ecg, ((0, 0), (n, 0)), "constant", constant_values=PADDING)

    plt.figure(figsize=(10, 5))
    plt.plot(range(0, 10000, 2), ecg[10], alpha=0.8, c='orange')
    plt.xlabel('Time (milliseconds)')
    plt.show()

    return ecg


def normalization(ecg):
    for i in range(len(ecg)):
        for j in range(len(ecg[i])):
            if np.isnan(ecg[i][j]):
                ecg[i][j] = ecg[i][j - 1]

        minn = min(ecg[i])
        maxx = max(ecg[i])
        if maxx - minn < 0.00001:
            raise 13
        ecg[i] = (ecg[i] - minn) / (maxx - minn)
    return ecg


ECGs_train = []
ECGs_test = []
Diagnoses_train = []
Diagnoses_test = []

paths = Path(
    "/Users/aleksandr/PycharmProjects/AI_ECG/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
).rglob("*.mat")
paths = sorted(paths)
random.shuffle(paths)


for ind, filename in zip(range(len(paths)), paths):
    s = filename.resolve()
    record = wfdb.rdrecord(f"{s.parent}/{s.stem}")
    patient_ecg = np.matrix.transpose(record.p_signal) * 1000

    try:
        patient_ecg = cut_n_fill(patient_ecg)
        patient_ecg = normalization(patient_ecg)
    except:
        continue  # 13 - чересчур странная ЭКГ (не представляется возможным её нормально обрезать/обработать)

    if ind < 37001:
        ECGs_train.append(patient_ecg)
        Diagnoses_train.append(int(record.comments[2][4:] == "426783006"))
    else:
        ECGs_test.append(patient_ecg)
        Diagnoses_test.append(int(record.comments[2][4:] == "426783006"))

    print(ind, filename.resolve().stem)

train_ECGs_np = np.array(ECGs_train)
train_Diags_np = np.array(Diagnoses_train)
test_ECGs_np = np.array(ECGs_test)
test_Diags_np = np.array(Diagnoses_test)

if exists("all_train_ECGs"):
    os.remove("all_train_ECGs")
if exists("all_train_Diags"):
    os.remove("all_train_Diags")
if exists("all_test_ECGs"):
    os.remove("all_test_ECGs")
if exists("all_test_Diags"):
    os.remove("all_test_Diags")

file_train_ecg = open("all_train_ECGs", "wb")
file_train_diag = open("all_train_Diags", "wb")
file_test_ecg = open("all_test_ECGs", "wb")
file_test_diag = open("all_test_Diags", "wb")
np.save(file_train_ecg, train_ECGs_np)
np.save(file_train_diag, train_Diags_np)
np.save(file_test_ecg, test_ECGs_np)
np.save(file_test_diag, test_Diags_np)
file_train_ecg.close()
file_train_diag.close()
file_test_ecg.close()
file_test_diag.close()
del train_ECGs_np
del train_Diags_np
del test_ECGs_np
del test_Diags_np


print(0)
