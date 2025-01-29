import numpy as np
from scipy.ndimage import median_filter

def apply_median_filter(pitch_track, kernel_size=3):
    filtered_pitch = median_filter(pitch_track, size=kernel_size)
    return filtered_pitch

def remove_outliers(data):
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return np.where((data > lower_bound) & (data < upper_bound), data, np.nan) 