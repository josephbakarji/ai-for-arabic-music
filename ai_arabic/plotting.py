import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def plot_melody(time, frequencies, title="Filtered Melody (Main Notes Only)", color='purple', label='Filtered frequencies'):
    plt.figure(figsize=(10, 6))
    plt.plot(time, frequencies, color=color, label=label)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.show()

def plot_time_series(y, sr, duration=None):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y[:int(duration * sr)] if duration else y, sr=sr, alpha=0.8)
    plt.title("Time Series Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_stft(y, sr):
    D = librosa.stft(y)  # Compute the STFT of the audio signal
    DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure(figsize=(12, 6))
    librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title("STFT (Spectrogram)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

def plot_combined_frequencies(times_all, freqs_all, times_melody, freqs_melody):
    plt.figure(figsize=(10, 6))

    # Plot all frequencies in light gray
    plt.plot(times_all, freqs_all, color='lightgray', label='All Frequencies')

    # Plot filtered melody in purple
    plt.plot(times_melody, freqs_melody, color='purple', label='Main Melody')

    plt.title('All Frequencies vs. Filtered Melody')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.grid(True)
    plt.show() 