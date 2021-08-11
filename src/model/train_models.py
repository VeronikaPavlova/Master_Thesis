"""
Code for the Playing_Around folder, with old label and test data
"""

import os

import numpy

from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

import matplotlib.pyplot as plt
import pickle

import pandas as pd

from utils import loading
import read_rosbag

# constants because of the recording
SR = 48000  # samplerate
PICKLE_PROTOCOL = 2  # for saving the learned KNN model

TEST_SIZE = 0.2

BASE_DIR = "/home/nika/git/Master_Thesis/src/data/experiment_data/21_07_27_8LocSensorFusion_RedFinger"
# SOUND_DIR_NAMES = ["sweep_20ms_default", "sweep_20ms_5_8000", "sweep_20ms_20_500", "sweep_1s", "tone_400_Hz", "tone_600_Hz", "tone_800_Hz"]
# SOUND_DIR_NAMES = ["sweep_20ms_default", "sweep_20ms_20_1000", "sweep_20ms_1000_5000", "sweep_20ms_5000_10000",
#                    "sweep_20ms_10000_30000", "tone_C0_16_35_Hz", "tone_C1_32_70_Hz", "tone_C2_65_42_Hz", "tone_C3_130_81_Hz", "tone_C4_261_63_Hz",
#                    "tone_C5_523_25_Hz", "tone_C6_1046_50_Hz", "tone_C7_2093_Hz", "tone_C8_4186_Hz", "sweep_1s"]
# SOUND_DIR_NAMES = ["click", "impulse", "silence_20ms", "sweep_20ms", "sweep_1s", "white_noise_20ms"]
SOUND_DIR_NAMES = ["sweep_1s"]


# SOUND_DIR_NAMES = ["sweep_1s","sweep_20ms_20_10000_repeat2x", "sweep_20ms_20_10000_reverse", "sweep_20ms_10000_20_reverse", "sweep_20ms_20_10000_tone_700_reverse", "sweep_20ms_20_10000_tone_1000_reverse", "sweep_20ms_default" ]

def main():
    for sound in SOUND_DIR_NAMES:
        PATH = os.path.join(BASE_DIR, sound)
        bagFolder = BASE_DIR + "/rosbag/*.bag"
        os.chdir(PATH)
        print("========== SOUND NAME {} ==========".format(sound))

        # load AAS
        print("========== AAS =========")

        sounds, labels = loading.test_train_aas_data([1,5], normalize=False, noise=False)
        y_train = numpy.array(labels)
        X_train = numpy.array([loading.sound_to_spectrum(sound) for sound in sounds])

        sounds, labels = loading.test_train_aas_data([3], normalize=False, noise=False)
        y_test = numpy.array(labels)
        X_test = numpy.array([loading.sound_to_spectrum(sound) for sound in sounds])

        score = 0
        n = 0
        # skf = StratifiedKFold(shuffle=True, n_splits=5)
        # for train_index, test_index in skf.split(spectra, labels):
            # X_train, X_test = spectra[train_index], spectra[test_index]
            # y_train, y_test = labels[train_index], labels[test_index]
        score += run_exps(X_train, y_train, X_test, y_test, 'AAS_' + sound)
            # n += 1
        # print("Score: {}".format(score / n))

        # # Load Strain Sensor
        print("========== SS ==========")
        sensors, labels = read_rosbag.read_bags(bagFolder)
        sensors, labels = read_rosbag.normalized_sensors(sensors, labels, 0)
        sensors = numpy.array(sensors)
        labels = numpy.array(labels)

        score = 0
        n = 0
        # TODO Hier werden zB 8 values in test und die gleichen restlichen 2 values in train gepackt. Deswegen so eine hohe Genauigkeit und bullshit
        skf = StratifiedKFold(shuffle=True, n_splits=4)
        for train_index, test_index in skf.split(sensors, labels):
            X_train, X_test = sensors[train_index], sensors[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            score += run_exps(X_train, y_train, X_test, y_test, 'SS_' + sound)
            n += 1

        print("Score: {}".format(score / n))


def run_exps(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame,
             sensor_name) -> pd.DataFrame:
    models = [
        # ('LogReg', LogisticRegression()),
        ('rf_classifier', RandomForestClassifier()),
        # ('knn_classifier', KNeighborsClassifier()),
        # ('knn_regressor', KNeighborsRegressor()),
        # ('SVM', SVC()),
        # ('GNB', GaussianNB()),
    ]

    for name, model in models:
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # print(name)
        score = clf.score(X_test, y_test)
        print(score)

        # plot_accuracy(y_test, y_pred)

        # Plot confusion matrix for each model and class
        # try:
        #     disp = plot_confusion_matrix(clf, X_test, y_test,
        #                                  # display_labels=class_names,
        #                                  # cmap=plt.cm.Blues,
        #                                  normalize='true')
        #     plt.show()
        #     cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
        #     # cm_filename = "cm_score_{}.json".format(clf.name)
        #     # store_cm_and_classes(cm, score, clf.classes_, result_dir_sound, filename=cm_filename)
        #     # print("test_score_{}: {:.2f}".format(clf.name, score))
        # except AttributeError:
        #     # Regression models don't have '.classes_' attribute
        #     print("Can't calculate confusion matrix for model '{}'".format(clf.name))

        # Save models
        model_name = sensor_name + "_model_" + name + ".pkl"
        with open(os.path.join(BASE_DIR, model_name), 'wb') as f:
            pickle.dump(model, f, protocol=PICKLE_PROTOCOL)
    return score


def plot_accuracy(y_true, y_pred):
    def loc_to_mm(x):
        return (x - 1) * 3

    fig, ax = pyplot.subplots(1)
    fig.set_size_inches((10, 5))

    y_mm = [loc_to_mm(x) for x in y_true]
    sweep_mm = [loc_to_mm(x) for x in y_pred]
    ax.set_xlabel("True Location [mm]")
    ax.set_ylabel("Measured Location [mm]")
    target = [x for x in set(y_mm)]
    ax.plot(target, target, label="Target", c="C0")
    ax.scatter(y_mm, sweep_mm, label="Active", c="C2", s=150)
    ax.legend(framealpha=1)
    ax.grid(True)
    # fig.tight_layout()

    active_rmse = numpy.sqrt(mean_squared_error(y_mm, sweep_mm))
    active_mae = mean_absolute_error(y_mm, sweep_mm)

    print("mean absolute error: {} , mean square error: {}".format(active_mae, active_rmse))


if __name__ == "__main__":
    main()
