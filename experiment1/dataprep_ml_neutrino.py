import numpy as np
from scipy import signal
from scipy.io.wavfile import read
from scipy.signal import butter, lfilter
import math
from km3io import acoustics as rawacoustic
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pathlib
import seaborn as sns
from tensorflow import keras
from keras import layers
from keras import models
from IPython import display
import PIL
import PIL.Image

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

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Load the data sets
data_dir_noise1D = "C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/noisedata/noisearrays/noisearray"
noise_files1D = os.listdir('.')

data_dir_neutrino1D = "C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/neutrinodata/neutrinoarrays/neutrinoarray"
neutrino_files1D = os.listdir('.')

data_dir_noise2D = "C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/noisedata/noisespectrums/noisespectrum"
noise_files = os.listdir('.')

data_dir_neutrino2D = "C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/neutrinodata/neutrinospectrums/neutrinospectrum"
neutrino_files = os.listdir('.')

# Now label the 1D data
# 0 for noise
# 1 for neutrino
noisedata1D = []
for file in glob.iglob('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/noisedata/noisearrays/*.txt'):
    noisearray1D = np.loadtxt(file)
    noisedata1D.append([noisearray1D, 0])

noisedata1D = np.array([np.array(x) for x in noisedata1D])

neutrinodata1D = []
for file in glob.iglob('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/neutrinodata/neutrinoarrays/*.txt'):
    neutrinoarray1D = np.load(file)
    neutrinodata1D.append([neutrinoarray1D, 1])

neutrinodata1D = np.array([np.array(x) for x in neutrinodata1D])

# Now label the 2D data
# 0 for noise
# 1 for neutrino
noisedata2D = []
for file in glob.iglob('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/noisedata/noisespectrums/*.txt'):
    noisearray2D = PIL.Image.Image.load(file)
    noisedata2D.append([noisearray2D, 0])

noisedata2D = np.array([np.array(x) for x in noisedata2D])

neutrinodata2D = []
for file in glob.iglob('C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/neutrinodata/neutrinopectrums/*.txt'):
    neutrinoarray2D = PIL.Image.Image.load(file)
    neutrinodata2D.append([neutrinoarray2D, 1])

neutrinodata2D = np.array([np.array(x) for x in neutrinodata2D])

alldata1D = np.vstack((noisedata2D, neutrinodata2D))
alldata2D = np.vstack((noisedata2D, neutrinodata2D))

# shuffle datsets
indices = np.arange(alldata1D.shape[0])
np.random.shuffle(indices)

shuffled_data1D = alldata1D[indices]
shuffled_data2D = alldata2D[indices]

# Reapportion training and testing sets
training_samples = int(shuffled_data1D.shape[0] * 0.8)

# Separate training and testing arrays for 1D CNN
train1D = shuffled_data1D[0:training_samples]
test1D = shuffled_data1D[training_samples:]
train2D = shuffled_data2D[0:training_samples]
test2D = shuffled_data2D[0:training_samples]

# Save the 1D and 2D training and test arrays
np.save("C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/train1D", train1D)
np.save("C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/test1D", test1D)
np.save("C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/train2D", train2D)
np.save("C:/Users/dijkemala/OneDrive - TNO/Documents/ML/data/test2D", test2D)