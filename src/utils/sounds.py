
import librosa
import numpy


NUMPY_RANDOM_SEED = 123456
SR = 48000

# Sounds for use
RECORDING_DELAY_SILENCE = numpy.zeros(int(SR * 0.25), dtype='float32')  # the microphone has about .15 seconds delay in

SOUND_GENERATIONS = dict({
    # "sweep_20ms_default": numpy.hstack(
    #     [librosa.core.chirp(20, 20000, SR, duration=.02).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "sweep_20ms_20_10000_reverse": numpy.hstack(
    #     [librosa.core.chirp(20, 10000, SR, duration=.01).astype('float32'),
    #      librosa.core.chirp(10000, 20, SR, duration=.01).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "sweep_20ms_10000_20_reverse": numpy.hstack(
    #     [librosa.core.chirp(10000, 20, SR, duration=.01).astype('float32'),
    #      librosa.core.chirp(20, 10000, SR, duration=.01).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "sweep_20ms_20_10000_repeat2x": numpy.hstack(
    #     [librosa.core.chirp(20, 10000, SR, duration=.01).astype('float32'),
    #      librosa.core.chirp(20, 10000, SR, duration=.01).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "sweep_20ms_20_10000_tone_1000_reverse": numpy.hstack(
    #     [librosa.core.chirp(20, 10000, SR, duration=.005).astype('float32'),
    #      librosa.core.tone(1000, SR, duration=.01).astype('float32'),
    #      librosa.core.chirp(10000, 20, SR, duration=.005).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "sweep_20ms_20_10000_tone_700_reverse": numpy.hstack(
    #     [librosa.core.chirp(20, 10000, SR, duration=.005).astype('float32'),
    #      librosa.core.tone(700, SR, duration=.01).astype('float32'),
    #      librosa.core.chirp(10000, 20, SR, duration=.005).astype('float32'), RECORDING_DELAY_SILENCE]),
    "sweep_1s": numpy.hstack(
        [librosa.core.chirp(20, 20000, SR, duration=1).astype('float32'), RECORDING_DELAY_SILENCE]),

    # "sweep_20ms_5_8000": numpy.hstack(
    #     [librosa.core.chirp(5, 8000, SR, duration=.02).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "sweep_20ms_20_500": numpy.hstack(
    #     [librosa.core.chirp(20, 500, SR, duration=.02).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "tone_400_Hz": numpy.hstack(
    #     [librosa.core.tone(400, SR, duration=0.2).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "tone_600_Hz": numpy.hstack(
    #     [librosa.core.tone(600, SR, duration=0.2).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "tone_800_Hz": numpy.hstack(
    #     [librosa.core.tone(800, SR, duration=0.2).astype('float32'), RECORDING_DELAY_SILENCE]),

    #
    # "sweep_20ms": numpy.hstack(
    #     [NOISE_SILENCE, librosa.core.chirp(20, 20000, SR, duration=.02).astype('float32'), RECORDING_DELAY_SILENCE]),
    # "white_noise_20ms": numpy.hstack(
    #     [NOISE_SILENCE, numpy.random.uniform(low=.999, high=1, size=int(SR / 50)).astype('float32'),
    #      RECORDING_DELAY_SILENCE]),
    # "silence_20ms": numpy.hstack(
    #     [NOISE_SILENCE, numpy.zeros((int(SR / 50),), dtype='float32'), RECORDING_DELAY_SILENCE]),
    # "impulse": numpy.hstack([NOISE_SILENCE, numpy.array([0, 1, -1, 0]).astype(numpy.float32), RECORDING_DELAY_SILENCE]),
    # "click": numpy.hstack(
    #     [NOISE_SILENCE,
    #      librosa.core.clicks(times=[0], sr=48000, click_freq=2500.0, click_duration=0.01, length=int(48000 * 0.02)),
    #      RECORDING_DELAY_SILENCE]),
})