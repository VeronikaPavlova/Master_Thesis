import os
import pickle

BASE_DIR = "/home/nika/git/Master_Thesis/src/data/experiment_data"
EXP_NAME = "22_03_21_Experiment_Sound_Test_5_Contactpoints"

MODEL_NAME = "AAS_sweep_20ms_default_model_knn_classifier.pkl"


def main():
    global DATA_DIR
    DATA_DIR = os.path.join(BASE_DIR, EXP_NAME)
    load_model()

def load_model():
    model_path = os.path.join(DATA_DIR, MODEL_NAME)
    with open(model_path, "rb") as f:
        clf = pickle.load(f)


if __name__ == "__main__":
    main()

