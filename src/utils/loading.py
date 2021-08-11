import os
from glob import glob
import librosa
import librosa.display
import noisereduce
import numpy
import pandas

import rosbag

import scipy.signal

SR = 48000  # got the sr from the wav files
FRAME_SIZE = 2048  # or 512, 1024, 2048, 4096 ... - the higher, then higher freq. resolution, but worse time resolution
HOP_SIZE = 1024  # or 256, 512, 1024, 2048 ... - 1/2 of Frame Size (could also be 1/4, 1/8 ..)
WINDOW_FKT = scipy.signal.windows.blackmanharris(FRAME_SIZE)  # TODO: Test different window functions Hann,..

sound_duration = 0

# ============ Load AAS ======================

def test_train_aas_data(runs=None, normalize=False, noise=False):
    """
    @param runs:
    @param normalize: if True the raw sound date will be normalized
    @param noise: if True the raw sound will be denoised

    @return:
    """
    global sound_duration

    sounds = []
    labels = []
    for file in glob('*.wav'):
        rep, it, label = get_num_and_label(file)
        if rep in runs:
            sound = librosa.load(file, sr=SR)[0]
            sound_duration = librosa.get_duration(filename=file, sr=SR)
            if noise:
                noisy_part = sound[0:2100]
                sound = noisereduce.reduce_noise(sound, noisy_part)
            sounds.append(sound)
            labels.append(label)
    if normalize:
        sounds = numpy.array(sounds)
        sounds = list((sounds - sounds.mean()) / sounds.std())
        # sounds_norm = noisereduce.reduce_noise(sounds[0], sounds[0])
        # plot_wave(sounds[0], sounds_norm, " ")

    return sounds, labels


#  =============== Load SS =====================
def read_bags(bagFolder, runs=None, csv_header=None, csv_data=None, sound_duration=None):
    bags = []

    # If bag files exist, save the in the variable bags
    for infile in sorted(glob(bagFolder)):  # use sorted() to not mix up the bag file order
        bags.append(infile)
    if not bags:
        print("No bag files in folder /bagfiles!")
        exit()

    labels = []
    sensors = []
    for bagFile in bags:
        bag = rosbag.Bag(bagFile)

        rep, it, label = get_num_and_label(bagFile)

        # TODO Check if sensor one bag is [] it means the rosbag is empty and no sensor was recorded
        if rep in runs:

            # We want the exact start time of the audio from the rep, it and label. This way we can cut the strain sensor data
            if csv_header is not None and csv_data is not None:
                for data in csv_data:
                    if(data[csv_header.index('rep')] == str(rep) and data[csv_header.index('it')] == str(it) and data[csv_header.index('curr label')] == str(label)):
                        audio_time_start = int(data[csv_header.index('start time of record (ros)')])

            sensors_one_bag = extract_sensor_data(bag, audio_time_start)
            sensors.extend(numpy.array(sensors_one_bag))
            labels.extend([label] * len(sensors_one_bag))

    return sensors, labels


def extract_sensor_data(bag, time):
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


def get_num_and_label(filename):
    try:
        # remove file extension
        name = os.path.splitext(filename)[0]
        # remove initial number
        name = name.split("_")
        rep = int(name[1])
        it = int(name[3])
        # label = "_".join(name[2:])
        label = int(name[5])
        return rep, it, label
    except ValueError:
        # filename with different formatting. ignore.
        return -1, None


# ============================= AAS Feature Extraction ========================
def sound_to_spectrum(sound):
    """Convert sounds to frequency spectra"""
    spectrum = numpy.fft.rfft(sound)
    amplitude_spectrum = numpy.abs(spectrum)
    d = 1.0 / SR
    freqs = numpy.fft.rfftfreq(len(sound), d)
    index = pandas.Index(freqs)
    series = pandas.Series(amplitude_spectrum, index=index)

    return series


# ================== SS Function Stuff ============
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