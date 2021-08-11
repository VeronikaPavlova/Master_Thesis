import csv
import json
import os
import logging
import sys
import time
import pickle

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import loading
import read_rosbag

BASE_DIR = "/home/nika/git/Master_Thesis/src/evaluations"

LOGGING_LEVEL = logging.DEBUG
PICKLE_PROTOCOL = 5


def set_up_logger():
    LOG_PATH = os.path.join(BASE_DIR, "logs")
    # os.makedirs(os.path.join(BASE_DIR, "/logs"), exist_ok=True)
    logging.basicConfig(
        filename=LOG_PATH + "/evaluate_{}.log".format(time.strftime("%Y%m%d")),
        format='%(asctime)s %(message)s',
        level=logging.INFO)
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(LOGGING_LEVEL)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logging.info("\nStarting new evaluation - {}\n".format(time.strftime("%H:%M:%S")))


def choose_evaluations(eval_dir=BASE_DIR, choice=None):
    """
    Lists evaluations found in 'eval_dir'.
    Prompts user to select one by number, unless 'choice' is given already.
    """
    evaluations = list()
    for filename in os.listdir(eval_dir):
        if filename.endswith(".json"):
            full_path = os.path.join(eval_dir, filename)
            with open(full_path) as f:
                evaluation_json = json.load(f)
                evaluation_json["filename"] = filename
                evaluation_json["full_path"] = full_path
                evaluations.append(evaluation_json)
    evaluations.sort(key=lambda x: x["name"])
    if choice is None:
        print("Choose an evaluation:")
        for i, evaluation in enumerate(evaluations):
            print("{}: {} - {}".format(i, evaluation["name"], evaluation['description']))
        choice = input()
    valid_choices = range(len(evaluations))
    if int(choice) in valid_choices:
        return evaluations[int(choice)]
    else:
        logging.warning("{} is not a valid choice. Valid choices are {}".format(choice, valid_choices))
        return None


def load_aas_data(eval_json, runs):
    # simply load all sound and labels and save in two a list
    sounds, labels = loading.test_train_aas_data(runs, normalize=False, noise=False)
    labels = numpy.array(labels)

    aas_feature = eval_json["aas"]["feature"]
    logging.info("Sound Feature: {}".format(aas_feature))

    sound_feauture = []
    # extract the neccessary features from the sound, decide from json which to use
    if aas_feature == "spectrum":
        # create from sound a sprectrum array
        sound_feauture = numpy.array([loading.sound_to_spectrum(sound) for sound in sounds])
    # TODO other features implement in loading
    elif aas_feature == "spectrogram_dB":
        sound_feauture = numpy.array([load_aas_data.compute_spectogram_dB(sound) for sound in sounds])
    elif aas_feature == "mel_spectrogram_dB":
        # TODO Mel Spec here
        print("Mel Spectrogram")
    else:
        # TODO error here
        raise NotImplementedError

    return sound_feauture, labels


def loas_ss_data(strain_path, runs, csv_header, csv_data):

    sensors, labels = loading.read_bags(strain_path, runs, csv_header, csv_data, loading.sound_duration)
    sensors, labels = loading.normalized_sensors(sensors, labels, 0)

    return numpy.array(sensors), numpy.array(labels)


def train_models(model):

    if model == "knn_class_default":
        clf = KNeighborsClassifier()
    elif model == "knn_class_grid_search":
        knn_param_grid = [{
            "classifier__n_neighbors": [1, 2, 3, 5],
            "classifier__p": [1, 2],
        }]
        knn_clf = Pipeline([("classifier", KNeighborsClassifier())])
        clf = GridSearchCV(knn_clf, param_grid=knn_param_grid, n_jobs=6, verbose=2)
        clf.name = "knn_grid-search"
    elif model == "knn_regression_default":
        clf = KNeighborsRegressor()
    elif model == "svc_default":
        clf = Pipeline([
            ("standardscaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", gamma=0.001, C=100.0,
                               probability=True))])
    elif model == "svc_poly":
        clf = Pipeline([
            ("standardscaler", StandardScaler()),
            ("classifier", SVC(kernel="poly", gamma=1, C=1, degree=3,
                               probability=True)),
        ])
    elif model == "svc_grid_search":
        svc_param_grid = [{
            "classifier__C": numpy.logspace(-2, 10, 13),
            "classifier__kernel": ("linear", "rbf"),
            "classifier__gamma": ("scale", "auto"),  # previously: numpy.logspace(-9, 3, 13),
        }]
        svc_clf = Pipeline([("standardscaler", StandardScaler()),
                            ("classifier", SVC())
                            ])
        clf = GridSearchCV(svc_clf, param_grid=svc_param_grid, n_jobs=6)
    elif model == "random_forest_default":
        clf = RandomForestClassifier()
    elif model == "random_forest_grid_search":
        rf_param_grid = [{
            "classifier__n_estimators": [10, 50, 100, 500],
            "classifier__max_features": ['sqrt', 'log2'],
            "classifier__max_depth": [5, 10, None],
        }]
        rf_clf = Pipeline([("classifier", RandomForestClassifier())])
        clf = GridSearchCV(rf_clf, param_grid=rf_param_grid, n_jobs=6, verbose=1)
    elif model == "mlp_default":
        clf = Pipeline([("standardscaler", StandardScaler()),
                        ("classifier", MLPClassifier())
                        ])
    elif model == "mlp_grid_search":
        mlp_param_grid = [{
            "classifier__hidden_layer_sizes": [(100,), (200, 200), (300, 300, 300)],
            "classifier__alpha": [0.001, 0.1, 1],
        }]
        mlp_clf = Pipeline([("standardscaler", StandardScaler()),
                            ("classifier", MLPClassifier())])
        clf = GridSearchCV(mlp_clf, param_grid=mlp_param_grid, n_jobs=6, verbose=1)
    else:
        # TODO: read parameters from json
        raise NotImplementedError

    clf.name = model  # name is used for saving
    return clf


def evaluate(eval_json):
    result_dir = os.path.join(eval_json['data_dir'], eval_json["result_dir"])
    os.makedirs(result_dir, exist_ok=True)

    # save evaluation file to result folder, for reference
    eval_json_path = os.path.join(result_dir, eval_json["filename"])
    with open(eval_json_path, 'w') as f:
        json.dump(eval_json, f, indent=4)
    logging.debug("saved evaluation to {}".format(eval_json_path))

    logging.info("Selected sounds for this experiment: {}".format(eval_json["sounds"]))
    for sound in eval_json["sounds"]:
        sound_path = os.path.join(eval_json["data_dir"], sound)
        os.chdir(sound_path)

        # ============ Load AAS Data ==================
        logging.info("Load the AAS from dir {}".format(sound_path))

        # Train_data_set
        train_runs = eval_json["aas"]["train_runs"]
        sound_feature_train, sound_label_train = load_aas_data(eval_json, train_runs)

        # Test_data_set
        test_runs = eval_json["aas"]["test_runs"]
        sound_feature_test, sound_label_test = load_aas_data(eval_json, test_runs)
        logging.info("Splited AAS in train runs {} and test runs {}".format(train_runs, test_runs))

        # For Time synchronization read the csv file and only use the strain data which is sync to the audio data
        sound_csv_file = os.path.join(eval_json["data_dir"], "audio_meta.csv")
        logging.info("Load the csv File for Time synchronization {}".format(sound_csv_file))

        with open(sound_csv_file, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            audio_csv = []
            for num, row in enumerate(csvreader):
                if num == 0:
                    csv_header = row
                else:
                    audio_csv.append(row)

        # =========== Load SS Data =======================
        strain_path = "../rosbag/*.bag"
        logging.info("Load the SS from dir {}".format(eval_json['data_dir'] + "/rosbag"))

        # Train_ data
        train_runs = eval_json["ss"]["train_runs"]
        strain_feature_train, strain_label_train = loas_ss_data(strain_path, train_runs, csv_header, audio_csv)

        # Test Data
        test_runs = eval_json["ss"]["test_runs"]
        strain_feature_test, strain_label_test = loas_ss_data(strain_path, test_runs)

        for model in eval_json["models"]:
            clf = train_models(model)

            clf.fit(sound_feature_train, sound_label_train)

            # save model
            # result_dir_sound = os.path.join(result_dir, sound)
            # if eval_json["models"][model]["save_to_disk"]:
            #     store_model(clf, result_dir_sound)

            y_pred = clf.predict(sound_feature_test)
            score = clf.score(sound_feature_test, sound_label_test)
            logging.info("test_score_{}: {:.2f}".format(clf.name, score))
            try:
                cm = confusion_matrix(sound_label_test, y_pred, labels=clf.classes_)
                cm_filename = "cm_score_{}.json".format(clf.name)
                # store_cm_and_classes(cm, score, clf.classes_, result_dir_sound, filename=cm_filename)
                logging.info("test_score_{}: {:.2f}".format(clf.name, score))
            except AttributeError:
                # Regression models don't have '.classes_' attribute
                logging.info("Can't calculate confusion matrix for model '{}'".format(clf.name))

            # if eval_json["accuracy"]:
                # need to also store y_test and y_pred for the accuracy plot
                # store_ytrue_and_ypred(sound_labels_test, y_pred, result_dir)
                # logging.info("stored predictions to folder {}".format(result_dir_sound))


def store_ytrue_and_ypred(y_true, y_pred, result_dir, filename="predictions.json"):
    """
    Store the true and predicted results to disk in json format.
    """
    predictions = {"y_true": list(y_true), "y_pred": list(y_pred)}
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, filename), 'w') as f:
        json.dump(predictions, f, indent=4)


# TODO split everything in test and train set depending on the runs
# TODO for each model in the model list train on it and save everything in results


def store_model(clf, save_dir):
    """
    Save fitted sklearn model to disk.
    Uses model name (clf.name) as filename
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = "model_{}.pkl".format(clf.name.replace(" ", "_"))
    with open(os.path.join(save_dir, filename), 'wb') as f:
        pickle.dump(clf, f, protocol=PICKLE_PROTOCOL)
        logging.info("saved sensor model to '{}'".format(filename))


def main():
    set_up_logger()

    if len(sys.argv) > 1:
        evaluation = choose_evaluations(choice=sys.argv[1])
    else:
        evaluation = choose_evaluations()

    if evaluation is None:
        logging.info("No valid evaluation selected. Aborting.")
        return

    logging.info("You chose '{}'".format(evaluation['name']))
    logging.debug(json.dumps(evaluation, indent=4, sort_keys=False))

    evaluate(evaluation)


if __name__ == '__main__':
    main()
