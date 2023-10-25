# lpf_filter.py

import numpy as np
from scipy import signal

def create_low_pass_filter(fps, cutoff_frequency, order=2):
    nyquist = 0.5 * fps
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def apply_low_pass_filter(data, b, a, initial_state=None):
    filtered_data = signal.lfilter(b, a, data, zi=initial_state)
    return filtered_data


