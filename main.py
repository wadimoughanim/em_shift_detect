import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils.signal_generation import SignalGenerator
from utils.signal_management import get_100ms_subcuts
from models.wavelet_analysis import WaveletAnalyzer

# Load the trained model (adjust path if necessary)
model = load_model("models/wavelet_cnn_model.keras")  # Use `.keras` model if saved in that format
print("Model loaded successfully.")

# Initialize and generate a new signal similar to the dataset generation
generator = SignalGenerator(
    fs=20000, duration=10, f_base=50, f_min=45, f_max=55, mean_interval=2, transition_duration=0.03, noise_level=0.05
)
signal, true_frequency, change_points = generator.generate_noisy_signal_with_shifts()
print("Generated signal with noise and frequency shifts.")
print(f"Signal shape: {signal.shape}, True frequency shape: {true_frequency.shape}, Change points: {change_points}")

# Segment the signal into 100 ms parts
fs = generator.fs
segments = get_100ms_subcuts(np.arange(0, 10, 1 / fs), signal, true_frequency, fs)
print(f"Generated 100 ms segments: {len(segments)} segments")

# Initialize the WaveletAnalyzer for processing
analyzer = WaveletAnalyzer(fs=fs, min_freq=45, max_freq=55)

# Process each segment and make predictions
detection_times = []
detection_results = []

for i, ((time_segment, signal_segment), (_, frequency_segment)) in enumerate(segments):
    print(f"\nProcessing segment {i + 1} at {time_segment[0]:.2f}s")
    
    # Extract the wavelet spectrum for the segment
    _, _, power_spectrum = analyzer.extract_time_frequency_spectrum(signal_segment)
    power_spectrum = power_spectrum[:10, :50]  # Ensure consistent shape for model input
    power_spectrum = power_spectrum[..., np.newaxis]  # Add channel dimension

    # Make prediction using the model
    prediction = model.predict(np.array([power_spectrum]), verbose=0)
    predicted_label = int(prediction[0][0] > 0.5)  # Classify based on threshold 0.5

    # Store time and prediction
    detection_times.append(time_segment[0])  # Start of each segment
    detection_results.append(predicted_label)

# Plot results: signal, true frequency, and detected shifts
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Plot the noisy signal
ax1.plot(generator.time, signal, color="orange", label="Noisy Signal")
ax1.set_ylabel("Amplitude")
ax1.set_title("Generated Signal with Noise and Frequency Shifts")
ax1.legend()

# Plot the true frequency with change points marked
ax2.plot(generator.time, true_frequency, color="blue", label="True Frequency")
for cp in change_points:
    ax2.axvline(x=cp, color="red", linestyle="--", label="Change Point" if cp == change_points[0] else "")
ax2.set_ylabel("Frequency (Hz)")
ax2.set_title("True Frequency Over Time with Change Points")
ax2.legend()

# Plot the model's predictions
ax3.step(detection_times, detection_results, where='post', color="green", label="Detected Shift (1 = Shift, 0 = No Shift)")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Detection")
ax3.set_title("Detected Frequency Shifts Over Time")
ax3.legend()

plt.tight_layout()
plt.show()
