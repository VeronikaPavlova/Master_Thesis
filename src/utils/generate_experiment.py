"""
@author: Vincent Wall, Gabriel Zöller, Marius Hebecker
@copyright 2020 Robotics and Biology Lab, TU Berlin
@licence: BSD Licence
"""
import argparse
import sys
from collections import OrderedDict

import numpy
import json
import random
import itertools
import datetime
import os
from utils import sounds


BASE_DIR = os.path.dirname(sys.argv[0]) + "/data/experiment_data"

CUT_TYPES = (
        'SILENCE_SWEEP',
        'REVIEW_DEMO',
        'DYNAMIC_SENSING_SINGLE',
        'MULTISOUND'
        )

SOUNDS = sounds.SOUND_GENERATIONS

def main():
    parser = argparse.ArgumentParser(description="""Creates a new experiment with randomized labels in the given folder.
                It generates unique labels as the product of (no_of_experiments, no_of_runs, list_of_labels).
                The unique label for one sound is a triple (experiment_no: int, run_no: int, label: str)."""
                                                     )
    parser.add_argument('name',           help="Name of experiment to create.", nargs="?", type=str, default=None)
    parser.add_argument('-e', '--no_of_experiments',
                                          help="How many times is this experiment repeated?",
                                          nargs="?", type=int, default=None)
    parser.add_argument('-r', '--no_of_runs',
                                          help="How many sounds of this type does a single repetition contain?",
                                          nargs="?", type=int, default=None)
    parser.add_argument('-l', '--labels',
                                          help="List of unique labels (e.g. contact locations)",
                                          nargs="*", type=str, default=None)
    parser.add_argument('-n', '--notes',  help="Thorough description of experiment.", nargs="?",
                                          type=str, default=None)
    parser.add_argument('--finger_id',    help="ID of the used actuator/finger, e.g. AAS002", nargs="?",
                                          type=str, default=None)
    parser.add_argument('-s', '--sounds', help="sounds to be tested with", nargs="*", choices=SOUNDS.keys(), type=str,
                                          action='append', default=None)
    parser.add_argument('--shuffle_sounds',
                                          help="randomize order in which sounds are played at each pose?",
                                          action='store_true', default=False)
    parser.add_argument('--inflation',    help="Desired inflation of actuator in [mg].",
                                          type=float, nargs="?", default=None)
    parser.add_argument('-rs', '--seed',  help="Random number generator seed to use for this experiment.",
                                          type=float, nargs="?", default=None)
    # old stuff
    parser.add_argument('-c', '--cutting_type',
                                          help="(DEPRECATED) How should sounds be cut?", choices=CUT_TYPES,
                                          nargs="?", type=str, default='MULTISOUND')
    parser.add_argument('-i', '--ignore', help="(DEPRECATED) Should experiment be ignored when automatically evaluating"
                                               " all experiment?", action='store_true', default=False)

    parser.set_defaults(func=create_experiment)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(**vars(args))
    else:
        parser.print_usage()


def make_timestamp():
    # return datetime.datetime.now(datetime.timezone.utc).isoformat()  #doesn't seem to work with Python2
    return datetime.datetime.now().isoformat()  # TODO: check if this breaks compatibility in learning script


def create_experiment(name=None,
                      no_of_experiments=None,
                      no_of_runs=None,
                      labels=None,
                      folder=None,
                      notes=None,
                      finger_id=None,
                      sounds=None,
                      shuffle_sounds=None,
                      record_forces=None,
                      record_pressure=None,
                      velocity=None,
                      inflation=None,
                      seed=None,
                      poses=None,
                      initials=None,
                      ignore=None,
                      cutting_type=None,
                      **kwargs):
    if name is None:
        name = input("_name_ of the experiment:\n")
        assert len(name) > 0, "No name given."
    if no_of_experiments is None:
        no_of_experiments = int(input("_no_of_experiments_ for this experiment:\n"))
    if no_of_runs is None:
        no_of_runs = int(input("_no_of_runs_ for this experiment:\n"))
    if labels is None:
        labels = input("_labels_ for this experiment (comma or space separated, quotes are stripped ):\n").replace('"', '').replace("'", "").replace(',', '').split()
        for i, l in enumerate(labels):
            print("{}: {}".format(i, l))
        assert len(labels) > 0, "No labels given."
    if notes is None:
        print("_notes_ for this experiment:\n(multi-line, end with empty line)")
        notes = []
        while True:
            notes_line = input()
            if len(notes_line) > 0:
                notes.append(notes_line)
            else:
                break
        notes = "\n".join(notes)
    if finger_id is None:
        finger_id = input("_finger_id_ used in this experiment:\n")
    if sounds is None:
        sound_candidates = input("Which test sounds? Multiple can be separated using spaces\n'{}'\n".format(
            SOUNDS)).split(" ")
    else:
        sound_candidates = sounds[0]  # command line entries are put in a list, somehow
    sounds = [sound for sound in sound_candidates if sound in SOUNDS]
    invalid_sounds = [sound for sound in sound_candidates if sound not in SOUNDS]
    print("The experiment will be carried out with the following sounds:{}".format(sounds))
    print("The following sounds you entered are invalid and have not been added to the sound-list:{}".format(
        invalid_sounds))
    if shuffle_sounds:
        print("Sounds will be shuffled before each recording.")
    else:
        print("Sounds will not be shuffled.")
    if velocity is None:
        velocity = float(input("_velocity_ of the robot movement (0 < vel <= 1):\n"))
        assert 0 < velocity <= 1, "velocity not in range"
    if inflation is None:
        inflation = float(input("_inflation_ of the actuator (infl >= 0):\n"))
        assert inflation >= 0, "negative inflation"
        if inflation > 100:
            print("Warning: large inflation given ({})".format(inflation))

    if seed is None:
        seed = random.random()
        print("No seed given, generated seed: {}".format(seed))

    assert cutting_type in CUT_TYPES, "Unknown cutting type given."

    print(sounds)
    input("chance to cancel")

    all_labels = [x for x in itertools.product(range(no_of_experiments), range(no_of_runs), labels)]
    random.seed(seed)
    random.shuffle(all_labels)
    experiment = OrderedDict([
        ("name", name),
        ("notes", notes),
        ("folder", folder),
        ("sounds", sounds),
        ("finger_id", finger_id),
        ("shuffle_sounds", shuffle_sounds),
        ("record_forces", record_forces),
        ("record_pressure", record_pressure),
        ("velocity", velocity),
        ("inflation", inflation),
        ("timestamp", make_timestamp()),
        ("seed", seed),
        ("poses", poses),
        ("initials", initials),
        ("cutting_type", cutting_type),
        ("ignore", ignore),
        ("no_of_experiments", no_of_experiments),
        ("no_of_runs", no_of_runs),
        ("labels", all_labels),  # make sure labels is the last key to make json more readable
    ])
    json_path = mkpath(BASE_DIR, folder, "experiment.json")
    if os.path.exists(json_path):
        ans = input("Experiment exists, overwrite? [y/N]: ")
        if ans.lower() in ["y", "yes"]:
            print("Overwriting experiment.")
        else:
            print("Not overwriting experiment.")
            print("Generated experiment JSON (for copy/pasting):")
            print(json.dumps(experiment, indent=2))
            return
    with open(json_path, "w") as f:
        json.dump(experiment, f, indent=2, sort_keys=False)


def mkpath(*args):
    """ Takes parts of a path (dir or file), joins them, creates the directory if it doesn't exist and returns the path.

        figure_path = mkpath(PLOT_DIR, "experiment", "figure.svg")
    """
    path = os.path.join(*args)
    if os.path.splitext(path)[1]:  # if path has file extension
        base_path = os.path.split(path)[0]
    else:
        base_path = path
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return path


if __name__ == "__main__":
    main()
