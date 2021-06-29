import os
import sys
import time
from collections import OrderedDict
from datetime import date
from collections import Counter

import subprocess
import random
import numpy
import rospy
from jacktools.jacksignal import JackSignal
import scipy.io.wavfile

import roslaunch
import rospkg
import logging
import csv

import rosbag
import read_rosbag

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from utils import sounds

# Save everything in data/experiment_data/date_ModelName
DATE = date.today().strftime("%y_%m_%d")
MODEL = "Test2"
MODEL_NAME = DATE + "_" + MODEL
BASE_DIR = os.path.dirname(sys.argv[0]) + "/data/experiment_data"

LOG_DIR = BASE_DIR + "/" + MODEL_NAME + "/logs"
LOGGING_LEVEL = logging.DEBUG
logger = logging.getLogger("record_experiment")

# CLASS_LABELS = ["top", "middle", "base"]  # classes to train
CLASS_LABELS = list(range(0, 9))
SAMPLES_PER_CLASS = 5  # How often to repeat each class label
ITERATIONS = 1  # Fot each label run the amount of iteration, so put the finger in
SHUFFLE_RECORDING_ORDER = True  # random class order

CHANNELS = 4

SR = 48000

# all sounds from sound.py
SOUNDS = sounds.SOUND_GENERATIONS
sound_name = next(iter(SOUNDS))  # take first sound


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
    logger.info(" =================== Setup Jack =================== \n")
    setup_jack()

    looping = True

    logger.info(" =================== Setup Audio Meta CSV =================== \n")

    # insert header in the audio csv, where we write later the timestamps
    header = ['num', 'rep', 'it', 'curr label', 'sound name', 'start time of record (ros)']
    with open(DATA_DIR + "/audio_meta.csv", 'w') as csv_audio_file:
        writer = csv.writer(csv_audio_file)

        # write the header
        writer.writerow(header)

    while looping:
        setup_matplotlib()

    # TODO plot AS and Strain Sensor


def setup_AAS():
    """
    Setup the Active Acoustic Sensor with jack audio
    @return:
    """
    # open JACK Audio control interface
    with open(os.devnull, 'w') as fp:
        subprocess.Popen(("qjackctl",), stdout=fp)

    logger.info("Press 'play' Button in the Jack Window \n"
                "Then press <Enter>. \n")
    input()

    rospy.init_node("Acoustic_Sensor", anonymous=True)
    # get the labels and ids to show in plot
    global label_list
    global rep_counter
    global current_idx

    label_list = CLASS_LABELS * SAMPLES_PER_CLASS
    rep_counter = dict(Counter(label_list))

    if SHUFFLE_RECORDING_ORDER:
        random.shuffle(label_list)
    current_idx = 0


def setup_jack():
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

    # logger.info("Save sound files in " + DATA_DIR)
    # store active sound for reference
    # sound_file = os.path.join(DATA_DIR, "{}_{}.wav".format(0, sound_name))
    # scipy.io.wavfile.write(sound_file, SR, sound)


def setup_strain_sensors():
    """
    roslaunch the Strain Sensor
    @return:
    """
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


def setup_matplotlib():
    global LINES_AAS
    global ax
    global TITLE
    global b_rec

    fig, ax = plt.subplots(2)
    ax[0].set_ylim(-.02, .02)
    plt.subplots_adjust(bottom=.2)
    LINES_AAS, = ax[0].plot(Ains[0])


    ax_rec = plt.axes([0.81, 0.05, 0.1, 0.075])
    b_rec = Button(ax_rec, '[R]ecord')
    b_rec.on_clicked(record_sensors)

    ax_back = plt.axes([0.59, 0.05, 0.1, 0.075])
    b_back = Button(ax_back, '[B]ack')
    b_back.on_clicked(back)

    ax_exit = plt.axes([0.1, 0.05, 0.1, 0.075])
    b_exit = Button(ax_exit, 'Exit')
    b_exit.on_clicked(exit)

    # cid = fig.canvas.mpl_connect('key_press_event', on_key)
    TITLE = fig.suptitle(get_current_title())
    plt.show()


def record_sensors(event):
    # update the current label id
    global current_idx
    if current_idx >= len(label_list):
        print("current_idx: {}  >= len(label_list): {}".format(current_idx, len(label_list)))
        return

    # Record Rosbag for one label and all sounds

    logger.info("start record strain rosbag for rep label {} \n".format(label(current_idx), rep_counter[label(current_idx)]))
    record_strain_rosbag()

    for sound in SOUNDS:
        for it in range(1, ITERATIONS + 1):
            logger.info("Setup Jack for iteration {} \n".format(it))
            setup_jack()

            sound_folder = DATA_DIR + "/{}".format(sound)

            if not os.path.exists(sound_folder):
                os.makedirs(sound_folder)

            record_acoustic(sound, sound_folder, it)

            time.sleep(.5)

    logger.info("stop record strain rosbag for label {} \n".format(label(current_idx)))
    stop_record_strain_rosbag()

    rep_counter[label(current_idx)] += (- 1)
    current_idx += 1

    # update plot title
    update()


def record_strain_rosbag():
    # Create a rosbag folder
    rosbagFolder = DATA_DIR + "/rosbag"

    if not os.path.exists(rosbagFolder):
        os.makedirs(rosbagFolder)

    prefix = "rep_{}_it_{}_label_{}".format(rep_counter[label(current_idx)], ITERATIONS, label(current_idx))


    global rosbagName
    global rosbagProcess
    global command

    rosbagName = rosbagFolder + os.sep + prefix

    command = "rosbag record -e '(/sensordata/finger)' -O {}".format(rosbagName)
    print(command)
    rosbagProcess = subprocess.Popen(command, stdin=subprocess.PIPE, shell=True)

    print("=================== Save files to {} ===================".format(rosbagFolder))


def stop_record_strain_rosbag():
    # Adapted from http://answers.ros.org/question/10714/start-and-stop-rosbag-within-a-python-script/
    list_cmd = subprocess.Popen("rosnode list", shell=True, encoding="utf8", stdout=subprocess.PIPE)
    list_output = list_cmd.stdout.read()
    retcode = list_cmd.wait()
    assert retcode == 0, "List command returned %d" % retcode
    for str in list_output.split("\n"):
        if (str.startswith("/record")):
            os.system("rosnode kill " + str)


def record_acoustic(sound, sound_folder, it):
    global J
    global Ains
    # touch object and start sound
    # wait for recording
    # store current sound
    # plot current sound
    # switch to next label
    start_audio_time = rospy.Time.now()
    J.process()
    J.wait()
    # LINES_AAS.set_ydata(Ains[0].reshape(-1))
    store(sound_folder, it)

    audio_to_csv(sound, start_audio_time, it)


def audio_to_csv(sound, start_audio_time, it):
    # write the label, sound and name in a new row to the audio csv
    with open(DATA_DIR + "/audio_meta.csv", 'a') as csv_audio_file:
        writer = csv.writer(csv_audio_file)

        data = [current_idx + 1, rep_counter[label(current_idx)], it, label(current_idx), sound, start_audio_time]
        # write the data
        writer.writerow(data)


def back(event):
    global current_idx

    # switch to previous
    current_idx = max(0, current_idx - 1)
    rep_counter[label(current_idx)] += 1
    update()

    # Remove last row in the audio csv
    f = open(DATA_DIR + "/audio_meta.csv", "r+")
    lines = f.readlines()
    lines.pop()
    f = open(DATA_DIR + "/audio_meta.csv", "w+")
    f.writelines(lines)


def exit(event):
    print("Exit")
    sys.exit()


def get_current_title():
    name = "Model: {}".format(MODEL)
    labels = "previous: {}   current: [{}]   next: {}".format(label(current_idx - 1), label(current_idx),
                                                              label(current_idx + 1))
    number = "#{}/{}: {}".format(current_idx + 1, len(label_list), label(current_idx))
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


def label(i):
    try:
        return label_list[i]
    except IndexError:
        # print("current_idx: {}, i: {}".format(current_idx, i))
        return ""


def update():
    LINES_AAS.set_ydata(Ains[0].reshape(-1))
    TITLE.set_text(get_current_title())

    time.sleep(.5)

    # Update strain Sensor plot and get data from the rosbag
    bag = rosbag.Bag(rosbagName + '.bag')
    sensor_data = read_rosbag.extract_sensor_data(bag)
    ax[1].clear()
    ax[1].plot(numpy.array([i[0] for i in sensor_data]).T)
    ax[1].plot(numpy.array([i[1] for i in sensor_data]).T)
    ax[1].plot(numpy.array([i[2] for i in sensor_data]).T)
    ax[1].plot(numpy.array([i[3] for i in sensor_data]).T)
    plt.draw()


def store(folder, it):
    sound_file = os.path.join(folder, "rep_{}_it_{}_label_{}.wav".format(rep_counter[label(current_idx)], it,
                                                                         label(current_idx)))
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
