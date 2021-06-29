import os
import time
import subprocess
import pickle


import librosa
import pandas
import roslaunch
import rospkg
import rospy
from jacktools.jacksignal import JackSignal

import matplotlib.pyplot as plt

from ros_labjack.msg import Measurements

import std_msgs.msg as smsg


# ==================
# USER SETTINGS
# ==================
import numpy

BASE_DIR = "/home/nika/git/Master_Thesis/src/data/experiment_data"
MODEL_NAME = "22_03_21_Experiment_Sound_Test_5_Contactpoints"

ACTIVE_SOUND = "0_sweep_20ms_default.wav"

AAS_SENSORMODEL_FILENAME = "AAS_sweep_20ms_default_model_knn_classifier.pkl"
SS_SENSORMODEL_FILENAME = "SS_sweep_20ms_default_model_knn_classifier.pkl"
CONTINUOUSLY = False  # chose between continuous sensing or manually triggered # percentage of samples left out of training and used for reporting test score
# ==================

CHANNELS = 1
SR = 48000
PICKLE_PROTOCOL = 2

plt.ion()


class LiveAcousticSensor(object):
    def __init__(self):
        # load sound from file (starts with "0_")
        # active_sound_filename = [fn for fn in os.listdir(DATA_DIR) if fn[:2] == "0_"][0]
        active_sound_filename = ACTIVE_SOUND
        self.sound = librosa.load(os.path.join(DATA_DIR, active_sound_filename), sr=SR)[0].reshape(
            -1).astype(numpy.float32)
        self.ss_data = []
        self.setup_strain()
        self.setup_jack()
        self.setup_model()
        self.setup_window()

    def setup_strain(self):

        rospy.init_node("StrainSubsctriber", anonymous=True)
        # Start Strain Sensor Node from launch file
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        rospack = rospkg.RosPack()
        sensorLaunchfile = rospack.get_path("ros_labjack") + "/launch/twoComp_sensors.launch"

        global sensorLaunch
        sensorLaunch = roslaunch.parent.ROSLaunchParent(uuid, [sensorLaunchfile])

        sensorLaunch.start()


    def setup_jack(self):

        # open JACK Audio control interface
        with open(os.devnull, 'w') as fp:
            subprocess.Popen(("qjackctl",), stdout=fp)

        print("Press 'play' Button in the Jack Window \n"
              "Then press <Enter>. \n")
        input()

        self.J = JackSignal("JS")
        assert self.J.get_state() >= 0, "Creating JackSignal failed."
        name, sr, period = self.J.get_jack_info()

        for i in range(CHANNELS):
            self.J.create_output(i, "out_{}".format(i))
            self.J.create_input(i, "in_{}".format(i))
            self.J.connect_input(i, "system:capture_{}".format(i + 1))
            self.J.connect_output(i, "system:playback_{}".format(i + 1))
        self.J.silence()

        self.Aouts = [self.sound] * CHANNELS
        self.Ains = [numpy.zeros_like(self.sound, dtype=numpy.float32) for __ in range(CHANNELS)]
        for i in range(CHANNELS):
            self.J.set_output_data(i, self.Aouts[i])
            self.J.set_input_data(i, self.Ains[i])

    def setup_model(self):
        model_path = os.path.join(DATA_DIR, AAS_SENSORMODEL_FILENAME)
        with open(model_path, "rb") as f:
            self.clf_aas = pickle.load(f)
        print(self.clf_aas.classes_)
        model_path = os.path.join(DATA_DIR, SS_SENSORMODEL_FILENAME)
        with open(model_path, "rb") as f:
            self.clf_ss = pickle.load(f)
        print(self.clf_ss.classes_)

    def setup_window(self):
        f = plt.figure(1)
        f.clear()
        f.suptitle("Acoustic Contact Sensing", size=30)
        ax1 = f.add_subplot(2, 2, 1)
        ax1.set_title("Recorded sound (waveform)", size=20)
        ax1.set_xlabel("Time [samples]")
        ax2 = f.add_subplot(2, 2, 2)
        ax2.set_title("Amplitude spectrum", size=20)
        ax2.set_xlabel("Frequency [Hz]")
        self.wavelines, = ax1.plot(self.Ains[0])
        self.spectrumlines, = ax2.plot(self.sound_to_spectrum(self.Ains[0]))
        ax2.set_ylim([0, 120])

        ax3 = f.add_subplot(2, 2, 3)
        ax3.text(0.0, 0.3, "Sensing AAS:", dict(size=40))
        self.predictiontext_aas = ax3.text(.7, 0.25, "", dict(size=50))
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        # ax3.set_title("Contact location")
        ax3.axis('off')

        ax4 = f.add_subplot(2, 2, 4)
        ax4.text(0.0, 0.3, "Sensing SS:", dict(size=40))
        self.predictiontext_ss = ax4.text(.6, 0.25, "", dict(size=50))
        ax4.set_xticklabels([])
        ax4.set_yticklabels([])
        # ax3.set_title("Contact location")
        ax4.axis('off')

        f.show()
        plt.draw()
        plt.pause(0.00001)

    def sound_to_spectrum(self, sound):
        """Convert sounds to frequency spectra"""
        spectrum = numpy.fft.rfft(sound)
        amplitude_spectrum = numpy.abs(spectrum)
        d = 1.0 / SR
        freqs = numpy.fft.rfftfreq(len(sound), d)
        index = pandas.Index(freqs)
        series = pandas.Series(amplitude_spectrum, index=index)
        return series

    def find_nearest(self, array, value):
        array = numpy.asarray(array)
        idx = (numpy.abs(array - value)).argmin()
        return idx


    def predict(self):
        ss_current_data = numpy.array(self.ss_data)
        start_idx = self.find_nearest(ss_current_data[:,0], self.aas_timestamp_start)
        end_idx = self.find_nearest(ss_current_data[:,0], self.aas_timestamp_end)
        data = numpy.array([numpy.subtract(arr, self.ss_norm_mean) for arr in numpy.array(ss_current_data[start_idx-2:end_idx+2, 1])])
        ss_current_data_mean = numpy.mean(data, axis=0)[numpy.newaxis]

        # print(a)
        # print(a.T)
        prediction_ss = self.clf_ss.predict(ss_current_data_mean)
        for i in range(CHANNELS):
            # self.Ains[i][25000:35000] = self.Ains[i][0000:10000]
            # self.Ains[i][55000:64000] = self.Ains[i][0000:9000]
            spectrum = self.sound_to_spectrum(self.Ains[i])
            prediction = self.clf_aas.predict([spectrum])
        self.wavelines.set_ydata(self.Ains[0].reshape(-1))
        self.spectrumlines.set_ydata(spectrum)

        self.predictiontext_aas.set_text(prediction[0])
        self.predictiontext_ss.set_text(prediction_ss[0])

        plt.draw()
        plt.pause(0.00001)

    def strain_callback(self, data):
        # Each subscriber gets 1 callback, and the callback either
        # stores information and/or computes something and/or publishes
        # It _does not!_ return anything
        # print(data.header.stamp)
        tsec = data.header.stamp.to_sec()
        self.ss_data.append([tsec, numpy.array(data.values[2:])])
        # print(data)

    def run(self):
        rospy.Subscriber('/sensordata/finger', Measurements, self.strain_callback)

        key = input("Position finger to 'no contact' for normalization and then press <Enter>")
        if key == '':
            self.ss_data =[]

            time.sleep(1)

            ss_norm = numpy.array(self.ss_data)[:, 1]
            self.ss_norm_mean = numpy.mean(ss_norm, axis=0)
            print("Norm Vector: " + str(self.ss_norm_mean))

        if CONTINUOUSLY:
            while True:
                self.aas_timestamp_start = rospy.Time.now()
                self.ss_data = []
                self.J.process()
                self.J.wait()
                self.predict()
                plt.pause(1)
        else:
            key = input("Press <Enter> to sense! ('q' to abort)")
            while key == '':
                self.aas_timestamp_start = rospy.Time.now().to_sec()
                self.ss_data = []
                self.J.process()
                self.aas_timestamp_end = rospy.Time.now().to_sec()
                self.J.wait()
                self.predict()
                key = input("Press <Enter> to sense! ('q' to abort)")


def main():
    global DATA_DIR
    DATA_DIR = os.path.join(BASE_DIR, MODEL_NAME)

    # ========== AAS Prediction ==========
    predictor = LiveAcousticSensor()
    predictor.run()


if __name__ == '__main__':
    main()
