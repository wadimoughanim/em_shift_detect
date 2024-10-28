import numpy as np

def get_100ms_subcuts(time, signal, true_frequency, fs):
    """
    Splits the input signal and true frequency into 100 ms sub-cuts.

    Parameters:
        time (ndarray): Array of time values.
        signal (ndarray): Array of signal amplitude values.
        true_frequency (ndarray): Array of true frequency values.
        fs (int): Sampling frequency in Hz.

    Returns:
        List of tuples: Each tuple contains:
            - A pair (time_segment, signal_segment) for 100 ms of signal.
            - A pair (time_segment, frequency_segment) for 100 ms of true frequency.
    """
    segment_duration = 0.1  # 100 ms
    samples_per_segment = int(segment_duration * fs)

    segments = []

    for start in range(0, len(time), samples_per_segment):
        end = start + samples_per_segment
        if end <= len(time):  # Ensure we only take full 100 ms segments
            time_segment = time[start:end]
            signal_segment = signal[start:end]
            frequency_segment = true_frequency[start:end]

            # Append each 100 ms segment as pairs (time, signal) and (time, true frequency)
            segments.append(((time_segment, signal_segment), (time_segment, frequency_segment)))

    return segments
