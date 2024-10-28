import numpy as np
import pywt

class WaveletAnalyzer:
    def __init__(self, fs, min_freq=45, max_freq=55, wavelet='morl'):
        """
        Initializes the WaveletAnalyzer class for a specific frequency range.

        Parameters:
            fs (int): Sampling frequency in Hz.
            min_freq (float): Minimum frequency for analysis in Hz.
            max_freq (float): Maximum frequency for analysis in Hz.
            wavelet (str): Type of wavelet to use for analysis (default is 'morl').
        """
        self.fs = fs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.wavelet = wavelet

    def extract_time_frequency_spectrum(self, signal_segment):
        """
        Extracts the time-frequency power spectrum in the 45–55 Hz range from a 100 ms signal segment.

        Parameters:
            signal_segment (ndarray): Array of signal values for the 100 ms segment.

        Returns:
            Tuple of (times, freqs, power_spectrum): Arrays representing the time values within the segment,
            the frequencies analyzed, and the power values for each time-frequency point.
        """
        freqs = np.linspace(self.min_freq, self.max_freq, num=10)  # Reduced frequency points
        scales = pywt.scale2frequency(self.wavelet, freqs) * self.fs
        cwt_coefficients, _ = pywt.cwt(signal_segment, scales, self.wavelet, sampling_period=1/self.fs)
        power_spectrum = np.abs(cwt_coefficients) ** 2
        times = np.linspace(0, 0.1, num=len(signal_segment))  # 100 ms duration
        return times, freqs, power_spectrum

    def batch_extract_time_frequency_spectrum(self, signal_segments):
        """
        Extracts time-frequency power spectra for a batch of 100 ms segments.
        
        Parameters:
            signal_segments (list of ndarray): List of 100 ms signal segments.
        
        Returns:
            np.ndarray: Array of power spectra for each segment in the batch.
        """
        spectra = [self.extract_time_frequency_spectrum(seg)[2] for seg in signal_segments]
        return np.array(spectra)  # Shape: (num_segments, frequency, time)
