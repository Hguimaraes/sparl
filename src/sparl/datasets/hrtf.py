import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter
from scipy.signal import lfilter
from scipy.signal import convolve

class HRTF(object):
    def __init__(self, root):
        self.root = root
    
    def convolve(self, mono, sr, azimuth, elevation, subject):
        # Read the HRIR database
        path = os.path.join(self.root, subject, 'hrir_final.mat')
        hrft = loadmat(path)
        hrir_r = hrft['hrir_r']
        hrir_l = hrft['hrir_l']

        # Butterworth highpass filter
        left = self.butter_highpass_filter(mono, 75, sr)
        right = self.butter_highpass_filter(mono, 75, sr)

        # Convolve and return binaural audio
        left_channel = convolve(left, hrir_l[azimuth, elevation], 'same')
        right_channel = convolve(right, hrir_r[azimuth, elevation], 'same')
        return np.vstack((left_channel,right_channel)).astype(np.int16)
    
    def butter_highpass_filter(self, audio, cutoff, rate, order=5):
        cutoff = cutoff / (rate//2)
        b, a = butter(order, cutoff, btype='high', analog=False)
        return lfilter(b, a, audio)
        