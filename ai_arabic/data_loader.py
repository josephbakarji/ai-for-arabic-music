import librosa
import numpy as np
from scipy.ndimage import median_filter

def load_and_extract_melody(file_path):
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    # Get main melody pitch
    pitch_track = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_track.append(pitch if pitch > 0 else 0)

    return np.array(pitch_track)

def extract_melody_librosa(y, sr, fmin=librosa.note_to_hz('C2'),
                          fmax=librosa.note_to_hz('C7'), hop_length=512,
                          frame_length=2048, threshold_ratio=0.1, 
                          size=5, duration=None):

    # Use pYIN to extract fundamental frequencies (f0)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
        hop_length=hop_length, frame_length=frame_length
    )

    # Calculate dynamic threshold based on median voiced probability
    dynamic_threshold = np.median(voiced_probs) * threshold_ratio

    # Post-processing: apply median filter to smooth out short-term noise and spikes
    f0_smoothed = median_filter(f0, size=size)  # Increased filter size for smoother transitions

    # Filter out the frames where the voiced probability is below the dynamic threshold
    f0_filtered = np.where(voiced_probs > dynamic_threshold, f0_smoothed, np.nan)

    return f0_filtered 