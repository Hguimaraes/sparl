import numpy as np
from scipy import signal

def sine_wave(freq, sr, seconds, n_bits=8):
    t = np.arange(int(sr*seconds))
    samples = np.sin(2*np.pi*t*freq/sr).astype(np.float32)

    # Convert to int and return
    to_int = 2**n_bits-1
    return (samples*to_int).astype(np.int16)

def square_wave(freq, sr, seconds, n_bits=8):
    t = np.arange(int(sr*seconds))
    samples = signal.square(2*np.pi*t*freq/sr)

    # Convert to int and return
    to_int = 2**n_bits-1
    return (samples*to_int).astype(np.int16)

def triangle_wave(freq, sr, seconds, n_bits=8):
    t = np.arange(int(sr*seconds))
    samples = signal.sawtooth(2*np.pi*t*freq/sr)

    # Convert to int and return
    to_int = 2**n_bits-1
    return (samples*to_int).astype(np.int16)