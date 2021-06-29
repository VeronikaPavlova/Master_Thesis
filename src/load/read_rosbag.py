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


def read_bags(bagFolder, plot=False):
    bags = []
    # If bag files exist, save the in the variable bags
    for infile in sorted(glob.glob(bagFolder)):  # use sorted() to not mix up the bag file order
        bags.append(infile)
    if not bags:
        print("No bag files in folder /bagfiles!")
        exit()
    # else:
    #     print('Processing the following bags:')
    #     print(bags)

    labels = []
    sensors = []
    for bagFile in bags:
        bag = rosbag.Bag(bagFile)
        n, label = get_num_and_label(bagFile)
        if plot:
            rosbagInfo(bag)
            plot_data(bag)

        sensors_one_bag = extract_sensor_data(bag)
        sensors.extend(numpy.array(sensors_one_bag))
        labels.extend([label] * len(sensors_one_bag))

    return sensors, labels


def normalized_sensors(sensors, labels, norm_label):
    ss_norm = []
    for i, label in enumerate(labels):
        if label == norm_label:
            ss_norm.append(sensors[i])

    ss_norm = numpy.array(ss_norm)
    ss_norm_mean = numpy.mean(ss_norm, axis=0)
    sensors_norm = [numpy.subtract(t,ss_norm_mean) for t in sensors]

    print("Normalization Array " + str(ss_norm_mean))

    return sensors_norm, labels



def get_num_and_label(filename):
    try:
        # remove file extension
        name = filename.split("/")
        name = name [-1]
        # remove initial number
        name = name.split("_", 3)
        num = int(name[1])
        # label = "_".join(name[2:])
        label = name[3].split(".")
        label = int(label[0])
        return num, label
    except ValueError:
        # filename with different formatting. ignore.
        return -1, None

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


def plot_data(bag, sensor_data):

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
    # sensor_data = OrderedDict()
    sensor_data = []
    for topic, msg, t in bag.read_messages(topics=["/sensordata/finger"]):
        tsec = msg.header.stamp.to_sec()
        values = []
        for i, ch in enumerate(msg.channels):
            chFixed = ch
            # if chFixed not in sensor_data:
                # totalMsgs = bag.get_type_and_topic_info()[1][topic].message_count
                # totalMsgs = bag.get_type_and_topic_info()[1][topic].message_count
                # sensor_data[chFixed] = []
            # sensor_data[chFixed].append([tsec, msg.values[i]])
            if ch.startswith('sensor'):
                values.append(msg.values[i])
        sensor_data.append(values)

    # for i, ch in enumerate(sensor_data.keys()):
    #     sensor_data[ch] = numpy.array(sensor_data[ch])

    return sensor_data


if __name__ == "__main__":
    bagFolder = '/home/nika/git/Master_Thesis/src/data/experiment_data/21_06_26_Test/rosbag/*.bag'

    if len(sys.argv) > 1:
        bagFolder += os.sep + sys.argv[1] + os.sep
        print("Rosbag Folder: {}", format(bagFolder))

    read_bags(bagFolder)
