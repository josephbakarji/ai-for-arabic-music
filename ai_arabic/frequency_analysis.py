import numpy as np
import seaborn as sns
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import librosa
def extract_all_frequencies(file_path):
    y, sr = librosa.load(file_path)

    # Use harmonic-percussive source separation (HPS)
    y_harmonic, _ = librosa.effects.hpss(y)

    # Estimate pitch using piptrack
    pitches, magnitudes = librosa.core.piptrack(y=y_harmonic, sr=sr)

    # Extract the highest pitch at each time step
    frequencies = []
    times = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Ignore pitch values of 0
            frequencies.append(pitch)
            times.append(t * (1/sr) * 512)  # Each frame corresponds to 512 samples

    return np.array(times), np.array(frequencies)

def apply_kde_and_extract_peaks(frequencies, bandwidth=0.3, peak_distance=5, peak_height=0.001):
    # Remove NaNs from the filtered frequencies (since KDE can't handle NaNs)
    clean_frequencies = frequencies[~np.isnan(frequencies)]

    # Apply Kernel Density Estimation (KDE) using seaborn's kdeplot
    plt.figure(figsize=(10, 6))
    kde = sns.kdeplot(clean_frequencies, bw_adjust=bandwidth, color='orange')
    plt.title('KDE of Extracted Melody Frequencies')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Density')

    kde_x = kde.get_lines()[0].get_xdata()
    kde_y = kde.get_lines()[0].get_ydata()

    # Find the peaks in the KDE curve to identify the most prominent frequencies
    peaks, _ = find_peaks(kde_y, distance=peak_distance, height=peak_height)

    # Extract the prominent frequencies from the KDE curve
    prominent_frequencies = kde_x[peaks]

    # Plot the detected peaks
    plt.plot(prominent_frequencies, kde_y[peaks], 'ro', label='Detected Peaks')
    plt.legend()
    plt.show()

    return prominent_frequencies

def calculate_all_intervals_in_cents(frequencies):
    num_freqs = len(frequencies)
    intervals_matrix = np.zeros((num_freqs, num_freqs))  # Initialize a matrix to store all the intervals

    # Calculate the interval in cents between every pair of frequencies
    for i in range(num_freqs):
        for j in range(num_freqs):
            if i != j:
                f1 = frequencies[i]
                f2 = frequencies[j]
                intervals_matrix[i, j] = 1200 * np.log2(f2 / f1)

    return intervals_matrix

def calculate_positive_intervals_in_cents(frequencies):
    positive_intervals = []  # Initialize a list to store only positive intervals

    # Calculate the interval in cents between every pair of frequencies
    for i in range(len(frequencies)):
        for j in range(len(frequencies)):
            if i != j:
                f1 = frequencies[i]
                f2 = frequencies[j]
                interval = 1200 * np.log2(f2 / f1)

                # Add only positive intervals to the list
                if interval > 0:
                    positive_intervals.append(interval)

    return np.array(positive_intervals)  # Return as a flat NumPy array

# Define distinctly different jins templates for testing
jins_templates = {
    "Jins 1": [200, 400, 600],  # Example values
    "Jins 2": [100, 100, 500],
    "Jins 3": [300, 500, 700]
}

def compare_intervals_to_jins(extracted_intervals):
    closest_jins = None
    min_difference = float('inf')

    # Compare extracted intervals to each jins template
    for jins_name, template in jins_templates.items():
        # Calculate the difference between the extracted intervals and the template
        difference = np.sum(np.abs(np.array(extracted_intervals) - np.array(template)))

        # Update if this template is the closest match so far
        if difference < min_difference:
            min_difference = difference
            closest_jins = jins_name

    return closest_jins, min_difference 