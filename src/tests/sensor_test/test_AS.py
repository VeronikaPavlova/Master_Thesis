import subprocess
import librosa
import numpy

import rospy
import os
import sys
import time

import logging

from matplotlib import pyplot

import scipy.io.wavfile
from jacktools.jacksignal import JackSignal
from matplotlib.widgets import Button
import random

LOG_DIR = "../../../logs"
LOGGING_LEVEL = logging.DEBUG
logger = logging.getLogger("record_experiment")

BASE_DIR = "../.."
MODEL_NAME = "Test"

# TODO: Add just 30 Labels or how does it work for regression?
CLASS_LABELS = ["top", "middle", "base"]  # classes to train
SAMPLES_PER_CLASS = 5
SHUFFLE_RECORDING_ORDER = True

CHANNELS = 4

SOUND_NAME = "sweep_1s"
SR = 48000
# Sounds for use
RECORDING_DELAY_SILENCE = numpy.zeros(int(SR * 0.35), dtype='float32')  # the microphone has about .15 seconds delay in
# recording the sound
SOUNDS = dict({
    "sweep_20ms": numpy.hstack(
        [librosa.core.chirp(20, 20000, SR, duration=.02).astype('float32'), RECORDING_DELAY_SILENCE]),
    "sweep_1s": numpy.hstack(
        [librosa.core.chirp(20 , 20000, SR, duration=1).astype('float32'), RECORDING_DELAY_SILENCE]),
    "white_noise_20ms": numpy.hstack(
        [numpy.random.uniform(low=.999, high=1, size=int(SR / 50)).astype('float32'), RECORDING_DELAY_SILENCE])
})


def main():
    # Initialize AS Node
    # rospy.init_node('AcousticSensing', anonymous=True)

    # needs to start after rospy.init or else it will be overwritten
    setup_logger()

    global DATA_DIR
    DATA_DIR = mkpath(BASE_DIR, MODEL_NAME)

    logger.info("Setting up the experiment /n")
    setup_experiment()
    logger.info("Run experiment /n")
    run_experiment()


def setup_experiment():
    # open JACK Audio control interface
    with open(os.devnull, 'w') as fp:
        subprocess.Popen(("qjackctl",), stdout=fp)

    logger.info("Press 'play' Button in the Jack Window \n"
                "Then press <Enter>. \n")
    input()

    global label_list
    global current_idx

    label_list = CLASS_LABELS * SAMPLES_PER_CLASS
    if SHUFFLE_RECORDING_ORDER:
        random.shuffle(label_list)
    current_idx = 0

    # # subscribe to force/torque data from ROS
    # rospy.Subscriber("netft_data", WrenchStamped, get_ft)
    # # subscribe to pressure data from ROS
    # rospy.Subscriber("pneumaticbox/pressure_0", std_msgs.msg.Float64, get_pressure)
    # logger.debug("ros subscribers done")


def run_experiment():
    # Setup Jack
    logger.info("Setup Jack /n")
    setup_jack(SOUND_NAME)

    setup_matplotlib()


def setup_matplotlib():
    global LINES
    global TITLE
    global b_rec

    fix, ax = pyplot.subplots(1)
    ax.set_ylim(-.001, .001)
    pyplot.subplots_adjust(bottom=.2)
    LINES, = ax.plot(Ains[0])
    ax_back = pyplot.axes([0.59, 0.05, 0.1, 0.075])
    b_back = Button(ax_back, '[B]ack')
    b_back.on_clicked(back)
    ax_rec = pyplot.axes([0.81, 0.05, 0.1, 0.075])
    b_rec = Button(ax_rec, '[R]ecord')
    b_rec.on_clicked(record)
    # cid = fig.canvas.mpl_connect('key_press_event', on_key)
    TITLE = ax.set_title(get_current_title())
    pyplot.show()


def l(i):
    try:
        return label_list[i]
    except IndexError:
        # print("current_idx: {}, i: {}".format(current_idx, i))
        return ""


def get_current_title():
    name = "Model: {}".format(MODEL_NAME.replace("_", " "))
    labels = "previous: {}   current: [{}]   next: {}".format(l(current_idx - 1), l(current_idx), l(current_idx + 1))
    number = "#{}/{}: {}".format(current_idx + 1, len(label_list), l(current_idx))
    if current_idx >= len(label_list):
        number += "DONE!"
    title = "{}\n{}\n{}".format(name, labels, number)
    return title


def back(event):
    global current_idx
    # switch to previous
    current_idx = max(0, current_idx - 1)
    update()


def record(event):
    global current_idx
    if current_idx >= len(label_list):
        print("current_idx: {}  >= len(label_list): {}".format(current_idx, len(label_list)))
        return

    global J
    global Ains
    # touch object and start sound
    # wait for recording
    # store current sound
    # plot current sound
    # switch to next label
    J.process()
    J.wait()
    LINES.set_ydata(Ains[0].reshape(-1))
    store()
    current_idx += 1
    update()


def store():
    sound_file = os.path.join(DATA_DIR, "{}_{}.wav".format(current_idx + 1, l(current_idx)))
    scipy.io.wavfile.write(sound_file, SR, Ains[0])


def update():
    TITLE.set_text(get_current_title())
    pyplot.draw()


def setup_jack(sound):
    """
    set up the jack client with standard settings
    """
    global J
    global Ains
    global Aouts

    J = JackSignal("JS")
    logger.debug("J.get_state(): {}".format(J.get_state()))
    assert J.get_state() >= 0, "Creating JackSignal failed."
    name, sr, period = J.get_jack_info()

    # create inputs and outputs
    for i in range(CHANNELS):
        J.create_output(i, "out_{}".format(i))
        J.create_input(i, "in_{}".format(i))
        J.connect_input(i, "system:capture_{}".format(i + 1))
        J.connect_output(i, "system:playback_{}".format(i + 1))
    J.silence()

    # initialize in- and output channels

    sound = SOUNDS[SOUND_NAME]
    Aouts = [sound] * CHANNELS
    Ains = [numpy.zeros_like(sound, dtype=numpy.float32) for __ in range(CHANNELS)]
    for i in range(CHANNELS):
        J.set_output_data(i, Aouts[i])
        J.set_input_data(i, Ains[i])

    logger.info("Save sound files in " + DATA_DIR)
    # store active sound for reference
    sound_file = os.path.join(DATA_DIR, "{}_{}.wav".format(0, SOUND_NAME))
    scipy.io.wavfile.write(sound_file, SR, sound)


def setup_logger():
    """
    Setting up logger for writing to file and to console.
    ATTENTION! rospy.init_node messes up the root logger, which is why we need to create our own custom logger here!
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    global logger
    logname = os.path.join(LOG_DIR, "log_recording_{}.log".format(time.strftime("%Y%m%d")))
    fh = logging.FileHandler(logname)
    logger.setLevel(LOGGING_LEVEL)
    logger.addHandler(fh)
    # define a Handler which writes messages to the sys.stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(LOGGING_LEVEL)
    # add the handler to the root logger
    logger.addHandler(console)
    logger.info("\nStarting new evaluation - {}\n".format(time.strftime("%H:%M:%S")))


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
