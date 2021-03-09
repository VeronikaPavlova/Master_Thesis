import os
from glob import glob

import librosa
import librosa.display
import noisereduce
import pandas

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split

import scipy
import scipy.signal
import numpy

import matplotlib
from matplotlib import pyplot

BASE_DIR = "../data/experiment_data/08_03_21_Experiment_5_Labels_Test"
SOUNDS = ["click", "impulse", "silence_20ms", "sweep_1s", "sweep_20ms", "white_noise_20ms"]

SR = 48000  # got the sr from the wav files
FRAME_SIZE = 4096  # or 512, 1024, 2048, 4096 ... - the higher, then higher freq. resolution, but worse time resolution
HOP_SIZE = 2048  # or 256, 512, 1024, 2048 ... - 1/2 of Frame Size (could also be 1/4, 1/8 ..)
WINDOW_FKT = scipy.signal.windows.blackmanharris(FRAME_SIZE)  # TODO: Test different window functions Hann,..

TEST_SIZE = 0.3

def main():

    PATH = os.path.join(BASE_DIR, SOUNDS[3])
    os.chdir(PATH)

    # compute_spectogram_dB(sounds1[0], labels[0], True)
    # sound_to_spectrogram_dB_librosa(sounds[1], labels[1])
    sounds, labels = load_sounds(normalize=False, noise=False)

    compute_spectogram_dB(sounds[0], labels[0], True)
    spectra = [sound_to_spectrum(sound) for sound in sounds]
    classes = list(set(labels))

    if TEST_SIZE > 0:
        X_train, X_test, y_train, y_test = train_test_split(spectra, labels, test_size=TEST_SIZE)
    else:
        X_train, y_train = (spectra, labels)

    clf = KNeighborsClassifier()  # using default KNN classifier
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    print("Fitted sensor model to data!")
    print("Training score: {:.2f}".format(train_score))

    if TEST_SIZE > 0:
        test_score = clf.score(X_test, y_test)
        print("Test score: {:.2f}".format(test_score))

    # sensor_model_filename = "sensor_model.pkl"
    # save_sensor_model(DATA_DIR, clf, sensor_model_filename)
    # print("\nSaved model to '{}'".format(os.path.join(DATA_DIR, sensor_model_filename)))


def sound_to_spectrogram_dB_librosa(sound, label):
    X = librosa.stft(sound, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, win_length=FRAME_SIZE, window=WINDOW_FKT)
    spectrogram = numpy.abs(X)
    spectrogram_dB = librosa.amplitude_to_db(spectrogram, ref=numpy.max)
    fig4 = librosa.display.specshow(spectrogram_dB, y_axis='linear', x_axis='time', sr=SR, hop_length=HOP_SIZE)
    matplotlib.pyplot.colorbar(mappable=fig4)
    matplotlib.pyplot.title(label + '_dB_librosa')
    matplotlib.pyplot.xlabel('Time (seconds)')
    matplotlib.pyplot.ylabel('Frequency (Hz)')
    matplotlib.pyplot.show()

def compute_spectogram_dB(signal, label, show=False):
    stfp_signal = librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    spec_signal = numpy.abs(stfp_signal) ** 2

    # log Amplitude
    spec_log = librosa.power_to_db(spec_signal)

    freq = numpy.linspace(0, SR, len(spec_log))
    index = pandas.Index(freq)
    series = pandas.Series(spec_log, index=index)

    if show:
        print("Plotting...")
        matplotlib.pyplot.figure(figsize=(25, 100))
        librosa.display.specshow(spec_log, sr=SR, hop_length=HOP_SIZE, x_axis="time", y_axis="log")
        matplotlib.pyplot.colorbar()
        matplotlib.pyplot.show()


def plot_wave(signal1, signal2, title):
    matplotlib.pyplot.figure(figsize=(25, 100))
    matplotlib.pyplot.subplot(2, 1, 1)
    librosa.display.waveplot(signal1)

    matplotlib.pyplot.subplot(2, 1, 2)
    librosa.display.waveplot(signal2)

    matplotlib.pyplot.show()

def sound_to_spectrum(sound):
    """Convert sounds to frequency spectra"""
    spectrum = numpy.fft.rfft(sound)
    amplitude_spectrum = numpy.abs(spectrum)
    d = 1.0/SR
    freqs = numpy.fft.rfftfreq(len(sound), d)
    index = pandas.Index(freqs)
    series = pandas.Series(amplitude_spectrum, index=index)
    return series

def load_sounds(normalize=False, noise=False):
    """
    @param normalize: if True the raw sound date will be normalized
    @param noise: if True the raw sound will be denoised

    @return:
    """
    # filenames = sorted(os.listdir(PATH))

    sounds = []
    labels = []
    for fn in glob('*.wav'):
        n, label = get_num_and_label(fn)
        if n < 0:
            # filename with different formatting. ignore.
            continue
        elif n == 0:
            # zero index contains active sound
            global SOUND_NAME
            SOUND_NAME = label
        else:
            sound, SR = librosa.load(fn)
            if noise:
                noisy_part = sound[0:1500]
                sound = noisereduce.reduce_noise(sound, noisy_part)
            sounds.append(sound)
            labels.append(label)

    if normalize:
        sounds = numpy.array(sounds)
        sounds = list((sounds - sounds.mean()) / sounds.std())
        # sounds_norm = noisereduce.reduce_noise(sounds[0], sounds[0])
        # plot_wave(sounds[0], sounds_norm, " ")

    return sounds, labels


def get_num_and_label(filename):
    try:
        # remove file extension
        name = os.path.splitext(filename)[0]
        # remove initial number
        name = name.split("_", 3)
        num = int(name[1])
        # label = "_".join(name[2:])
        label = int(name[3])
        return num, label
    except ValueError:
        # filename with different formatting. ignore.
        return -1, None


if __name__ == "__main__":
    main()
