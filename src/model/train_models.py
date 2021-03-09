import os

import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import classification_report

import pandas as pd

import load_aas_data

# constants because of the recording
SR = 48000  # samplerate
PICKLE_PROTOCOL = 2  # for saving the learned KNN model

TEST_SIZE = 0.3

BASE_DIR = "../data/experiment_data/08_03_21_Experiment_5_Labels_Test"
SOUND_DIR_NAMES = ["click", "impulse", "silence_20ms", "sweep_1s", "sweep_20ms", "white_noise_20ms"]


def run_exps(X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    dfs = []

    models = [
        # ('LogReg', LogisticRegression()),
        ('RF', RandomForestClassifier()),
        ('KNN', KNeighborsClassifier()),
        ('SVM', SVC()),
        ('GNB', GaussianNB()),
        ('XGB', XGBClassifier())
    ]
    results = []
    names = []

    scoring = ['accuracy']

    for name, model in models:
        kfold = model_selection.KFold(n_splits=2, shuffle=True, random_state=55555)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring)
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)
        print(classification_report(y_test, y_pred))

        results.append(cv_results)
        names.append(name)

        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        final = pd.concat(dfs, ignore_index=True)

    return final


def main():
    PATH = os.path.join(BASE_DIR, SOUND_DIR_NAMES[3])
    os.chdir(PATH)

    sounds, labels = load_aas_data.load_sounds(normalize=False, noise=False)
    labels = numpy.array(labels)
    spectra = numpy.array([load_aas_data.sound_to_spectrum(sound) for sound in sounds])

    # using random state for reprodudicity
    if TEST_SIZE > 0:
        X_train, X_test, y_train, y_test = train_test_split(spectra, labels, test_size=TEST_SIZE, random_state=4)
    else:
        # X_train, y_train = (spectra, labels)
        return

    run_exps(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
