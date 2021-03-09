import glob
import os
import sys
import time
from collections import OrderedDict

from hurry.filesize import size as _fsize
import numpy

import matplotlib.pyplot as plt
import rosbag

SOUNDS = ["click", "impulse", "silence_20ms", "sweep_1s", "sweep_20ms", "white_noise_20ms"]


def read_bags(bagFolder):
    bags = []
    # If bag files exist, save the in the variable bags
    for infile in sorted(glob.glob(bagFolder)):  # use sorted() to not mix up the bag file order
        bags.append(infile)
    if not bags:
        print("No bag files in folder /bagfiles!")
        exit()
    else:
        print('Processing the following bags:')
        print(bags)

    for bagFile in bags:
        bag = rosbag.Bag(bagFile)

        rosbagInfo(bag)

        plot_data(bag)


def rosbagInfo(bag):
    print("\n--- rosbag info ---\n")
    print("path: \t{}".format(bag.filename))
    print("version: \t{}".format(bag.version))
    print("duration: \t{:.2f} s".format(bag.get_end_time() - bag.get_start_time()))
    print("start: \t\t{}".format(time.ctime(bag.get_start_time())))
    print("end: \t\t{}".format(time.ctime(bag.get_end_time())))
    print("size: \t\t{}".format(_fsize(bag.size)))
    print("messages: \t{}".format(bag.get_message_count()))
    print("topics:\r"),
    typeAndTopic = bag.get_type_and_topic_info()
    topics = sorted(typeAndTopic[1].keys())
    maxLen = max([len(x) for x in topics])

    for topic in topics:
        print("\t\t{:{}}  {:5d} msgs   : {}".format(
            topic, maxLen,
            typeAndTopic[1][topic].message_count,
            typeAndTopic[1][topic].msg_type
        ))


def plot_data(bag):
    sensor_data = extract_sensor_data(bag)

    fig, ax = plt.subplots(1)
    fig.suptitle("Sensor Data of the bag file {}".format(bag.filename))
    ax.set_ylabel("sensor value [V]")

    # seperate axis for pressure sensor
    ax_pressure = ax.twinx()
    ax_pressure.set_ylabel("pressure value [V]", color='grey')
    ax_pressure.tick_params('y', colors='grey')

    for sensor in sorted(sensor_data.keys()):
        sData = numpy.array(sensor_data[sensor]).T
        ls = sensor_data[sensor]
        col = sensor_data[sensor]
        if 'pressure' in sensor:
            axi = ax_pressure
        else:
            axi = ax
        axi.plot(sData[0], sData[1], lw=2, label=sensor)
        # axi.plot(sData[1], lw=2, ls=ls, color=col, label=sensor)
    ax.legend()
    ax_pressure.legend()
    plt.show()


def extract_sensor_data(bag):
    sensor_data = OrderedDict()
    for topic, msg, t in bag.read_messages(topics=["/sensordata/finger"]):
        tsec = msg.header.stamp.to_sec()
        for i, ch in enumerate(msg.channels):
            chFixed = ch
            if chFixed not in sensor_data:
                # totalMsgs = bag.get_type_and_topic_info()[1][topic].message_count
                # totalMsgs = bag.get_type_and_topic_info()[1][topic].message_count
                sensor_data[chFixed] = []
            sensor_data[chFixed].append([tsec, msg.values[i]])

    for i, ch in enumerate(sensor_data.keys()):
        sensor_data[ch] = numpy.array(sensor_data[ch])

    return sensor_data


if __name__ == "__main__":
    bagFolder = '/home/nika/git/Master_Thesis/src/data/experiment_data/08_03_21_Experiment_5_Labels_Test/' + SOUNDS[0] + '/rosbag/*.bag'

    if len(sys.argv) > 1:
        bagFolder += os.sep + sys.argv[1] + os.sep
        print("Rosbag Folder: {}", format(bagFolder))

    read_bags(bagFolder)
