import wave

import numpy as np


def read_wav(wav_path):
    """
    Extract Raw Audio from Wav File
    :return:
    """
    spf = wave.open(wav_path, 'r')
    signal = spf.readframes(-1)
    signal = np.fromstring(signal, 'Int16')
    print("numpy signal shape:", signal.shape)

    """
    plt.title('Signal Wave...')
    plt.plot(signal)
    plt.show()
    """
    return signal


def noecho(signal):
    # convolve the original signal with delta returns the original signal
    delta = np.array([1., 0., 0.])  # notice: size of delta may be different as long as the first element must be 1
    noecho = np.convolve(signal, delta)
    print("noecho signal:", noecho.shape)
    export_sound_to_file(noecho, 'noecho.wav')


def echo(signal):
    """
    Add echo to the wav sound
    :param signal:
    :return:
    """
    filt = np.zeros(16000)
    filt[0] = 1
    filt[4000] = 0.6
    filt[8000] = 0.3
    filt[12000] = 0.2
    filt[15999] = 0.1

    echo = np.convolve(signal, filt)
    print("echo signal:", echo.shape)
    export_sound_to_file(echo, 'echo.wav')


def export_sound_to_file(sound, path):
    sound = sound.astype(np.int16)  # make sure you do this, otherwise, you will get VERY LOUD NOISE
    obj = wave.open(path, 'w')
    obj.setnchannels(1)  # mono
    obj.setsampwidth(2)
    obj.setframerate(framerate=16000.0)  # hertz
    obj.writeframesraw(sound)
    obj.close()


signal = read_wav('../convolve/example.wav')
noecho(signal)
echo(signal)
