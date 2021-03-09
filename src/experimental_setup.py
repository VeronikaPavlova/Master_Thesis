import os
import signal
import sys
import time
from collections import OrderedDict
from datetime import date

import subprocess
import random
import librosa
import numpy
import psutil
import rospy
from jacktools.jacksignal import JackSignal
import scipy.io.wavfile

import roslaunch
import rospkg
import rosbag

import rosbag_record

import getch
import logging

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

DATE = date.today().strftime("%d_%m_%y")
MODEL = "Experiment_30_Labels_Test"
MODEL_NAME = DATE + "_" + MODEL
BASE_DIR = os.path.dirname(sys.argv[0]) + "/data/experiment_data"

LOG_DIR = BASE_DIR + "/" + MODEL_NAME + "/logs"
LOGGING_LEVEL = logging.DEBUG
logger = logging.getLogger("record_experiment")

# CLASS_LABELS = ["top", "middle", "base"]  # classes to train
CLASS_LABELS = list(range(0, 31))
SAMPLES_PER_CLASS = 8
SHUFFLE_RECORDING_ORDER = True

CHANNELS = 4

SOUND_NAME = "sweep_1s"
SR = 48000
NOISE_SILENCE = numpy.zeros(4000, dtype='float32')
# Sounds for use
RECORDING_DELAY_SILENCE = numpy.zeros(int(SR * 0.25), dtype='float32')  # the microphone has about .15 seconds delay in
# recording the sound
SOUNDS = dict({
    "sweep_20ms": numpy.hstack(
        [NOISE_SILENCE, librosa.core.chirp(20, 20000, SR, duration=.02).astype('float32'), RECORDING_DELAY_SILENCE]),
    "white_noise_20ms": numpy.hstack(
        [NOISE_SILENCE, numpy.random.uniform(low=.999, high=1, size=int(SR / 50)).astype('float32'),
         RECORDING_DELAY_SILENCE]),
    "silence_20ms": numpy.hstack(
        [NOISE_SILENCE, numpy.zeros((int(SR / 50),), dtype='float32'), RECORDING_DELAY_SILENCE]),
    "impulse": numpy.hstack([NOISE_SILENCE, numpy.array([0, 1, -1, 0]).astype(numpy.float32), RECORDING_DELAY_SILENCE]),
    "click": numpy.hstack(
        [NOISE_SILENCE,
         librosa.core.clicks(times=[0], sr=48000, click_freq=2500.0, click_duration=0.01, length=int(48000 * 0.02)),
         RECORDING_DELAY_SILENCE]),
    "sweep_1s": numpy.hstack(
        [NOISE_SILENCE, librosa.core.chirp(20, 20000, SR, duration=1).astype('float32'), RECORDING_DELAY_SILENCE])

})


def main():
    setup_logger()

    global DATA_DIR
    DATA_DIR = mkpath(BASE_DIR, MODEL_NAME)

    logger.info("=================== Setting up Active Acoustic Sensors =================== \n")
    setup_AAS()
    time.sleep(1)

    logger.info("=================== Setting up Strain Sensors =================== \n")
    setup_strain_sensors()
    time.sleep(1)

    start_experiment()

    logger.info("Run experiment /n")


def start_experiment():
    # logger.info(" =================== Setup Jack =================== \n")
    setup_jack(SOUND_NAME)

    looping = True

    while looping:
        print("=================== Space to start recording! =================== \n")
        # char = getch.getch()
        char = " "
        while char != " ":
            if char == "q":
                print("Quitting Experiment")
                return

        setup_matplotlib()

        # Do some stuff

        # print("Press Space when you are done!")
        # char = getch.getch()
        # while char != " ":
        #     if char == "q":
        #         print("Quitting Experiment")
        #         return

        # TODO plot AS and Strain Sensor


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


def setup_jack(sound_name):
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

    sound = SOUNDS[sound_name]
    Aouts = [sound] * CHANNELS
    Ains = [numpy.zeros_like(sound, dtype=numpy.float32) for __ in range(CHANNELS)]
    for i in range(CHANNELS):
        J.set_output_data(i, Aouts[i])
        J.set_input_data(i, Ains[i])

    logger.info("Save sound files in " + DATA_DIR)
    # store active sound for reference
    sound_file = os.path.join(DATA_DIR, "{}_{}.wav".format(0, sound_name))
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


def record_sensors(event):
    # TODO record for different sounds
    # TODO save similar rosbag and .wav names for training them later

    time.sleep(1)
    global current_idx
    if current_idx >= len(label_list):
        print("current_idx: {}  >= len(label_list): {}".format(current_idx, len(label_list)))
        return

    for sound in SOUNDS:
        logger.info(" =================== Setup Jack =================== \n")
        setup_jack(sound)

        sound_folder = DATA_DIR + "/{}".format(sound)

        if not os.path.exists(sound_folder):
            os.makedirs(sound_folder)

        record_strain_rosbag(sound_folder)
        record_acoustic(sound_folder)
        stop_record_strain_rosbag()

        time.sleep(1)

    current_idx += 1

    # update plot title
    update()


def record_strain_rosbag(sound_folder):
    # Get available ROS topics
    ros_topics = [top[0] for top in rospy.get_published_topics()]
    print(ros_topics)

    rosbagFolder = sound_folder + "/rosbag"

    if not os.path.exists(rosbagFolder):
        os.makedirs(rosbagFolder)

    prefix = "num_{}_label_{}".format(current_idx + 1, l(current_idx))
    rosbagName = rosbagFolder + os.sep + prefix

    global rosbagProcess
    global command
    command = "rosbag record -e '(/sensordata/finger)' -O {}".format(rosbagName)
    print(command)
    rosbagProcess = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)

    print("=================== Save files to {} ===================".format(rosbagFolder))

def stop_record_strain_rosbag():

    # Adapted from http://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
    list_cmd = subprocess.Popen("rosnode list", shell=True,  encoding="utf8", stdout=subprocess.PIPE)
    list_output = list_cmd.stdout.read()
    retcode = list_cmd.wait()
    assert retcode == 0, "List command returned %d" % retcode
    for str in list_output.split("\n"):
        if (str.startswith("/record")):
            os.system("rosnode kill " + str)

def record_acoustic(sound_folder):
    global J
    global Ains
    # touch object and start sound
    # wait for recording
    # store current sound
    # plot current sound
    # switch to next label
    J.process()
    J.wait()
    # LINES_AAS.set_ydata(Ains[0].reshape(-1))
    store(sound_folder)
    # TODO update plot after recording


def setup_matplotlib():
    global LINES_AAS
    global ax
    global TITLE
    global b_rec

    fig, ax = plt.subplots(2)
    ax[0].set_ylim(-.03, .03)
    plt.subplots_adjust(bottom=.2)
    LINES_AAS, = ax[0].plot(Ains[0])
    ax_back = plt.axes([0.59, 0.05, 0.1, 0.075])
    b_back = Button(ax_back, '[B]ack')
    b_back.on_clicked(back)
    ax_rec = plt.axes([0.81, 0.05, 0.1, 0.075])
    b_rec = Button(ax_rec, '[R]ecord')
    b_rec.on_clicked(record_sensors)
    # TODO delete wav and rosbag file on "back"
    # cid = fig.canvas.mpl_connect('key_press_event', on_key)
    TITLE = fig.suptitle(get_current_title())
    plt.show()


def back(event):
    global current_idx
    # switch to previous
    current_idx = max(0, current_idx - 1)
    update()


def get_current_title():
    name = "Model: {}".format(MODEL)
    labels = "previous: {}   current: [{}]   next: {}".format(l(current_idx - 1), l(current_idx), l(current_idx + 1))
    number = "#{}/{}: {}".format(current_idx + 1, len(label_list), l(current_idx))
    if current_idx >= len(label_list):
        number += "DONE!"
    title = "{}\n{}\n{}".format(name, labels, number)
    return title


def extract_sensor_data(bag):
    sensor_data = OrderedDict()
    for topic, msg, t in bag.read_messages(topics=["/sensordata/finger"]):
        tsec = msg.header.stamp.to_sec()
        for i, ch in enumerate(msg.channels):
            chFixed = ch
            if chFixed not in sensor_data:
                # totalMsgs = bag.get_type_and_topic_info()[1][topic].message_count
                sensor_data[chFixed] = []
            sensor_data[chFixed].append([tsec, msg.values[i]])

    for i, ch in enumerate(sensor_data.keys()):
        sensor_data[ch] = numpy.array(sensor_data[ch])

    return sensor_data


def l(i):
    try:
        return label_list[i]
    except IndexError:
        # print("current_idx: {}, i: {}".format(current_idx, i))
        return ""


def update():
    LINES_AAS.set_ydata(Ains[0].reshape(-1))
    TITLE.set_text(get_current_title())
    plt.draw()


def store(folder):
    sound_file = os.path.join(folder, "num_{}_label_{}.wav".format(current_idx + 1, l(current_idx)))
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
