# Creates noise arrays from Pylos data with a sperm whale click in it

from math import *
import numpy as np
import numpy.fft as fft
import sys
from scipy import signal
from km3io import acoustics as rawacoustic
from scipy.io.wavfile import read
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
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
    for i in glob.iglob('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/spermwhalearrays/*.txt'):

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

        spermwhaleclick = np.loadtxt(i)

        # padding to insert the sperm whale click somewhere
        # (random position) in the data stream
        entry_point = np.random.randint(0,len(z_zero) - len(spermwhaleclick), 1)[0]
        # scaling = np.random.randint(5000,15000)
        # spermwhaleclick *= scaling
        x = np.pad(spermwhaleclick, (entry_point, len(z_zero)-len(spermwhaleclick)-entry_point),
                'constant', constant_values=(0., 0.))
        # add noise abd signal and make dure that it of proper format

        spermwhale_noise_array = (x+z_zero)
        spermwhale_noise_array = np.asarray(spermwhale_noise_array, dtype=np.int16)
        spermwhale_noise_array = spermwhale_noise_array.astype(np.float64)
        
        # save noise array as .txt file
        outfile = 'C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/sw_noise_arrays/sw_noise_array_' + str(i) + '.txt'
        np.savetxt(outfile, z_zero)

        # constructing spectrogram
        plt.figure()
        plt.specgram(spermwhale_noise_array, Fs=fs_resampled)
        plt.axis('off')
        plt.savefig('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/sw_noise_spectrums/sw_noise_spectrum_' + str(i) + '.png', transparent = True, bbox_inches='tight', pad_inches=0)
        plt.close()

        time = len(z_zero)/fs_resampled
        timeaxis = np.arange(0,time,time/len(spermwhale_noise_array))

        # save plot to be able to manually delete bad arrays
        plt.figure()
        plt.plot(timeaxis, spermwhale_noise_array)
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude (a.u.)')
        plt.grid()
        plt.savefig('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/spermwhaleplots/spermwhalearray_' '_' + str(i) + '.png')
        plt.close()

        print(i+1,'/1000')

def main(argv):

    filename = '201650198.180129090002.wav'

    # make clip with noise and neutrino
    scramble_me(filename)

if __name__ == '__main__':
    main(sys.argv)