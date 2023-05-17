# Finds sperm whale clicks in audio file and saves it

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

def findclick(filename, flow, fhigh, filenumber):
    # read data
    fs, wavdata = read(filename)

    fs_resampled = rawacoustic.F_S

    # apply bandpass filter
    data_f = butter_bandpass_filter(wavdata, flow, fhigh, fs=fs, order=5)

    # upsample the data to KM3NeT sampling frequency
    number_resampled = int(round(len(data_f)*fs_resampled/fs))
    data = signal.resample(data_f, number_resampled)

    # make timescale
    time = len(data)/fs_resampled
    timeaxis = np.arange(0,time,time/len(data))

    # compute the noise by averaging measurments within one std dev of mean
    counter = 0
    noise = 0

    # using the unfiltered data to compute std, otherwise strange results
    std = np.std(data_f)
    avg = np.mean(data)
    print('std:', std, 'avg:', avg)

    # if value within one standard dev., use as part of noise calculation
    for i in data:
        if (i < (avg + std) and i > (avg - std)):
            noise = noise + abs(i)
            counter = counter + 1
    noise = noise / counter
    print('noise:', noise, 'counter:', counter)

    # threshold is 4.5 Sound-to-Noise ratio
    threshold = 60 * noise
    print('threshold', threshold)

    ylist = []
    for i in range(len(data)):
        ylist.append(threshold)

    # plt.plot(timeaxis, data)
    # plt.plot(timeaxis, ylist)
    # plt.ylabel('Amplitude (a.u.)', fontsize=12)
    # plt.xlabel('Time (s)', fontsize=12)
    # plt.grid()
    # plt.show()

    counter = 0
    clickStarts = []
    clickStartsSeconds = []
    first = True
    nearby = False
    nearbyCount = 0

    # looks at each instance in array representing .wav file
    for i in data:
        # sees if value of instance above or below certain threshold
        if (i > threshold):
            # if no other click is nearby add to counter
            if (not nearby):
                clickStarts.append(counter)
                clickStartsSeconds.append(float(counter) / fs_resampled)
            # set nearby to true and reset nearby count
            nearby = True;
            nearbyCount = 0

        # if it has been .02 seconds since a click, reset nearby to false and nearbyCounter to 0 
        if (nearbyCount > (fs_resampled * .02)):
            nearby = False 
            nearbyCount = 0

        # if nearby a click then increment the clickCounter
        if (nearby):
            nearbyCount += 1

        # increment general counter
        counter += 1

    print('Start clicks:', clickStartsSeconds)

    # construct spectrogram from of num_datapoints and append to time-series arrays
    # 15% before and 85% after
    num_datapoints = 256
    for j in range(len(clickStartsSeconds)):
        
        if int(clickStartsSeconds[j]*fs_resampled - num_datapoints*0.4) >= 0:
            beginning = int(clickStartsSeconds[j]*fs_resampled - num_datapoints*0.4)
            end = int(clickStartsSeconds[j]*fs_resampled + num_datapoints*0.6)
            clickwindow = data[beginning:end]

        else:
            continue

        # make sure that the window is exaclty num_datapoints long
        if len(clickwindow) != num_datapoints:
            clickwindow = clickwindow[0:num_datapoints]
        print(len(clickwindow))
        
        # # constructing spectrogram
        # plt.figure()
        # plt.specgram(clickwindow, Fs=fs)
        # plt.axis('off')
        # plt.savefig('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/spermwhalespectrumsshort/spermwhalespectrumshort_' + str(filenumber) + '_' + str(j) + '.png', transparent = True, bbox_inches='tight', pad_inches=0)
        # plt.close()
        
        timestamps = np.arange(0,num_datapoints/fs_resampled,1/fs_resampled)

        # appending to arrays
        np.savetxt('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/spermwhalearraysshort/spermwhalearrayshort_' + str(filenumber) + '_' + str(j) + '.txt', clickwindow)

        # save plot to be able to manually delete bad arrays
        plt.figure()
        plt.plot(timestamps, clickwindow)
        plt.xlabel('Time(s)')
        plt.ylabel('Amplitude (a.u.)')
        plt.grid()
        plt.savefig('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/spermwhaleplotsshort/spermwhalearrayshort_' + str(filenumber) + '_' + str(j) + '.png')
        plt.close()
        print("#", (j + 1), " out of ", len(clickStartsSeconds)) #TEST
    print('File', filename, 'with number', filenumber, 'has', len(clickStartsSeconds), 'clicks')

def plotfile(filename, flow, fhigh):
    fs, data = read(filename)

    fs_resampled = rawacoustic.F_S

    # upsample the data to KM3NeT sampling frequency
    number_resampled = int(round(len(data)*fs_resampled/fs)) ## THIS MAY STRANGE RESULTS, do not rely on it..
    data_res = signal.resample(data, number_resampled)

    # make timescale
    time0 = len(data_res)/fs_resampled
    timeaxis0 = np.arange(0,time0,time0/len(data_res))

    # apply bandpass filter
    data_bp = butter_bandpass_filter(data_res, flow, fhigh, fs=fs, order=5)

    data_bp = np.loadtxt('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/testswarrays/spermwhalearray3.txt')
    # make timescale
    time1 = len(data_bp)/fs_resampled
    timeaxis1 = np.arange(0,time1,time1/len(data_bp))

    # plt.specgram(data, Fs=fs)
    # plt.xlabel('Time (s)', fontsize=12)
    # plt.ylabel('Frequency (Hz)', fontsize=12)
    # plt.show()

    plt.plot(data_bp)
    plt.ylabel('Amplitude (a.u.)', fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    plt.grid()
    plt.show()

def main(argv):

    flow = 800
    fhigh = 20000

    filenumber = 1
    for filename in glob.iglob('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/swwavfiles/*.wav'):
        findclick(filename, flow, fhigh, filenumber)
        filenumber += 1

    # filename = 'C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/spermwhaledata/swwavfiles/CAS3_20190826_125825.wav'

    # findclick(filename, flow, fhigh, filenumber)

    plotfile(filename, flow, fhigh)

if __name__ == '__main__':
    main(sys.argv)