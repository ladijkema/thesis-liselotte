# Creates noise arrays from Pylos data

from math import *
import numpy as np
import numpy.fft as fft
import sys
from scipy import signal
from km3io import acoustics as rawacoustic
from scipy.io.wavfile import read
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import glob

# Import filter functions
def butter_highpass(cutoff, Fs, order=5):
    nyq = 0.5 * Fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, Fs, order=5):
    b, a = butter_highpass(cutoff, Fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandstop(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    sos = butter(order, [low, high], btype='bandstop', analog=False, output='sos')
    return sos

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandstop(lowcut, highcut, fs, order=order)
    y = signal.sosfiltfilt(sos, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut/nyq
    high = highcut/nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    y = butter_lowpass_filter(data, highcut, fs, order)
    return butter_highpass_filter(y, lowcut, fs, order)

sampling_rate = 144000.

def scramble_me(infile):

    # read infile
    fs, data = read(infile)

    # take only an eighth of the file since it is too long
    length = int(len(data)/8)
    start = np.random.randint(0,len(data) - length, 1)[0]
    end = start + length

    data = data[start:end]

    time = len(data)/fs
    timeaxis = np.arange(0,time,time/len(data))

    # interpolate data to KM3NeT sampling frequency
    fs_resampled = rawacoustic.F_S

    # upsample the data to KM3NeT sampling frequency
    number_resampled = int(round(len(data)*fs_resampled/fs)) ## THIS MAY STRANGE RESULTS, do not rely on it..
    data_res = signal.resample(data, number_resampled)

    # generate 1000 neutrino samples, each time with a neutrino in a different location
    for i in range(1,1001):

        # part of the read file of particular length
        # choosing a length of 2048 datapoints, since that was the best window for sperm whale clicks
        datapoints = 2048
        trace_length = datapoints
        trace_start = np.random.randint(0,len(data_res) - datapoints, 1)[0]
        trace_end = trace_start + trace_length

        # determine a noise realisation, (i.e. scramble the noise to remove dolphin clicks or other sources)
        # based on the FFT of the data that have been read
        # using different, random phase.
        # Used polar coordinates for complex numbers
        y = data_res[int(trace_start) : int(trace_end)]

        Y = np.fft.fft(y)
        m = np.abs(Y)
        phase = 2*pi*np.random.rand(len(m)) - pi
        Z = m*np.cos(phase) + 1j*m*np.sin(phase)
        z = np.fft.ifft(Z)
        z = z.real

        mean = sum(z)/len(z)

        z_zero = [i - mean for i in z]

        # apply bandpass filter
        flow = 800
        fhigh = 20000
        z_zero = butter_bandpass_filter(z_zero, flow, fhigh, fs=fs, order=5)
        
        # save noise array as .txt file
        outfile = 'C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/noisedata/noisearrays/noisearray' + str(i) + '.txt'
        np.savetxt(outfile, z_zero)

        plt.specgram(z_zero, Fs=fs_resampled)
        plt.axis('off')
        plt.savefig('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/noisedata/noisespectrums/noisespectrum' + str(i) + '.png', transparent = True, bbox_inches='tight', pad_inches=0)

        time = len(z_zero)/fs_resampled
        timeaxis = np.arange(0,time,time/len(z_zero))

        # save plot to be able to manually delete bad arrays
        plt.figure()
        plt.plot(timeaxis, z_zero)
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude (a.u.)')
        plt.grid()
        plt.savefig('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/noisedata/noiseplots/noiseplot' + str(i) + '.png')
        plt.close()

        print(i+1,'/1000')

def plotnoise():
    data = np.loadtxt('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/testarrays/pylosnoise2.txt')
    # fs, data = read('domfile866_ruis.wav')
    fs = rawacoustic.F_S

    time = len(data)/fs
    timeaxis = np.arange(0,time,time/len(data))

    plt.specgram(data, Fs=fs)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

    # plt.plot(timeaxis, data)
    # plt.ylabel('Amplitude (a.u.)')
    # plt.xlabel('Time (s)')
    # plt.xlim(3.35,3.40)
    # plt.grid()
    # plt.show()

def main(argv):

    filename = '201650198.180129090002.wav'

    # make clip with noise and neutrino
    scramble_me(filename)

    # plotnoise()

if __name__ == '__main__':
    main(sys.argv)