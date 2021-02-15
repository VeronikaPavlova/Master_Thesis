import os
import sys
import time
from datetime import date

import subprocess
import random
import librosa
import numpy
import rospy
from jacktools.jacksignal import JackSignal
import scipy.io.wavfile

import roslaunch
import rospkg

import getch
import logging

LOG_DIR = "../logs"
LOGGING_LEVEL = logging.DEBUG
logger = logging.getLogger("record_experiment")


MODEL_NAME = date.today().strftime("%d_%m_%y") + "_Test"
BASE_DIR = os.path.dirname(sys.argv[0]) + "/experiment_data"

# TODO: Add just 30 Labels or how does it work for regression?
CLASS_LABELS = ["top", "middle", "base"]  # classes to train
SAMPLES_PER_CLASS = 5
SHUFFLE_RECORDING_ORDER = True

CHANNELS = 4

SOUND_NAME = "sweep_1s"
SR = 48000
# Sounds for use
RECORDING_DELAY_SILENCE = numpy.zeros(int(SR * 0.15), dtype='float32')  # the microphone has about .15 seconds delay in
# recording the sound
SOUNDS = dict({
    "sweep_20ms": numpy.hstack(
        [librosa.core.chirp(20, 20000, SR, duration=.02).astype('float32'), RECORDING_DELAY_SILENCE]),
    "sweep_1s": numpy.hstack(
        [librosa.core.chirp(20, 20000, SR, duration=1).astype('float32'), RECORDING_DELAY_SILENCE]),
    "white_noise_20ms": numpy.hstack(
        [numpy.random.uniform(low=.999, high=1, size=int(SR / 50)).astype('float32'), RECORDING_DELAY_SILENCE])
})


def main():
    setup_logger()

    global DATA_DIR
    DATA_DIR = mkpath(BASE_DIR, MODEL_NAME)

    logger.info("Setting up Active Acoustic Sensors /n")
    setup_AAS()


    logger.info("Setting up Strain Sensors /n")
    setup_strain_sensors()
    time.sleep(1)

    start_experiment()

    logger.info("Run experiment /n")


def start_experiment():
    logger.info("Setup Jack /n")
    setup_jack(SOUND_NAME)

    looping = True

    while looping:
        print("Space to start recording!")
        char = getch.getch()

        while char != " ":
            if char == "q":
                print("Quitting Experiment")
                return
        record_strain_rosbag()
        record_acoustic()
        time.sleep(2)

        # Do some stuff

        print("Press Space when you are done!")
        char = getch.getch()
        while char != " ":
            if char == "q":
                print("Quitting Experiment")

                return


def setup_AAS():
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


def setup_strain_sensors():
    # Start Strain Sensor Node from launch file
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    rospack = rospkg.RosPack()
    sensorLaunchfile = rospack.get_path("ros_labjack") + "/launch/twoComp_sensors.launch"

    global sensorLaunch
    sensorLaunch = roslaunch.parent.ROSLaunchParent(uuid, [sensorLaunchfile])

    sensorLaunch.start()


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


def record_strain_rosbag():
    # Get available ROS topics
    ros_topics = [top[0] for top in rospy.get_published_topics()]
    print(ros_topics)

    rosbagFolder = DATA_DIR + "/rosbag"

    package = 'rosbag'
    executable = 'record'

    if not os.path.exists(rosbagFolder):
        os.makedirs(rosbagFolder)

    prefix = "test"
    rosbagName = rosbagFolder + os.sep + prefix
    node = roslaunch.core.Node(package, executable,
                               args="-e '(/sensordata/finger)' -o {}".format(rosbagName))

    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()

    global rosbagProcess
    rosbagProcess = launch.launch(node)
    rosbagProcess.rosbagName = rosbagName

    print(" .... recording started!")


def record_acoustic():
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
    # LINES.set_ydata(Ains[0].reshape(-1))
    store()
    current_idx += 1
    # update()


def l(i):
    try:
        return label_list[i]
    except IndexError:
        # print("current_idx: {}, i: {}".format(current_idx, i))
        return ""


def store():
    sound_file = os.path.join(DATA_DIR, "{}_{}.wav".format(current_idx + 1, l(current_idx)))
    scipy.io.wavfile.write(sound_file, SR, Ains[0])

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
