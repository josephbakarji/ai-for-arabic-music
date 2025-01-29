import os
import librosa

from ai_arabic import (
    load_and_extract_melody, apply_median_filter, extract_melody_librosa,
    plot_melody, plot_time_series, plot_stft, plot_combined_frequencies,
    apply_kde_and_extract_peaks, extract_all_frequencies,
    calculate_all_intervals_in_cents, calculate_positive_intervals_in_cents,
    compare_intervals_to_jins
)

def main():
    # Define file paths
    filename = 'saba--unknown_artist--Un_Voyage_Avec_Le_Oud_Arabe_2--06_Taqsim_Saba_(Baidaphon).mp3'
    file_path = os.path.join('data/recordings', filename)

    # Extract melody frequencies
    melody_frequencies = extract_melody_librosa(file_path, threshold_ratio=0.3)
    print("Extracted Frequencies:", melody_frequencies)

    # Apply KDE and extract prominent frequencies
    prominent_frequencies = apply_kde_and_extract_peaks(melody_frequencies)
    print("Prominent frequencies detected:", prominent_frequencies)

    # Calculate positive intervals in cents
    positive_intervals_array = calculate_positive_intervals_in_cents(prominent_frequencies)
    print("Strictly positive intervals in cents between frequencies:")
    print(positive_intervals_array)

    # Compare intervals to Jins templates
    closest_jins, difference = compare_intervals_to_jins([250, 450, 550])  # Example intervals
    print(f"The extracted intervals are closest to {closest_jins} with a total difference of {difference} cents.")

    # Plotting (Example)
    # Load audio for plotting
    y, sr = librosa.load(file_path, sr=None)
    plot_time_series(y, sr, duration=100000/sr)  # Adjust duration as needed
    plot_stft(y, sr)

    # Extract all frequencies and plot
    times_all, freqs_all = extract_all_frequencies(file_path)
    times_melody, freqs_melody = librosa.times_like(melody_frequencies, sr=sr, hop_length=512), melody_frequencies
    plot_combined_frequencies(times_all, freqs_all, times_melody, freqs_melody)

if __name__ == "__main__":
    main() 