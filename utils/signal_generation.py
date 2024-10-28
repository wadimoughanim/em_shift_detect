# utils/signal_generation.py
import numpy as np

class SignalGenerator:
    def __init__(self, fs=20000, duration=10, f_base=50, f_min=45, f_max=55, mean_interval=2, transition_duration=3, noise_level=0.05):
        self.fs = fs
        self.duration = duration
        self.f_base = f_base
        self.f_min = f_min
        self.f_max = f_max
        self.mean_interval = mean_interval
        self.transition_duration = int(transition_duration * fs)
        self.noise_level = noise_level
        self.time = np.arange(0, self.duration, 1 / self.fs)

    def generate_noisy_signal_with_shifts(self):
        signal = np.sin(2 * np.pi * self.f_base * self.time)
        true_frequency = np.ones(len(self.time)) * self.f_base
        change_points = []

        start = np.random.poisson(self.mean_interval * self.fs)
        while start < len(self.time):
            change_points.append(start / self.fs)
            shift_frequency = np.random.uniform(self.f_min, self.f_max)
            transition_end = min(start + self.transition_duration, len(self.time))

            true_frequency[start:transition_end] = np.linspace(self.f_base, shift_frequency, transition_end - start)
            signal[start:transition_end] = np.sin(2 * np.pi * true_frequency[start:transition_end] * self.time[start:transition_end])

            if transition_end < len(self.time):
                true_frequency[transition_end:] = self.f_base
                signal[transition_end:] = np.sin(2 * np.pi * self.f_base * self.time[transition_end:])

            start = transition_end + np.random.poisson(self.mean_interval * self.fs)

        signal = self.add_all_noises(signal)
        return signal, true_frequency, change_points

    def add_white_noise(self, signal, level=None):
        level = level if level is not None else self.noise_level
        white_noise = level * np.random.normal(0, 1, len(signal))
        return signal + white_noise

    def add_low_frequency_noise(self, signal, freq=60, level=None):
        level = level if level is not None else self.noise_level * 0.5
        low_freq_noise = level * np.sin(2 * np.pi * freq * self.time)
        return signal + low_freq_noise

    def add_high_frequency_noise(self, signal, freq=500, level=None):
        level = level if level is not None else self.noise_level * 0.3
        high_freq_noise = level * np.sin(2 * np.pi * freq * self.time)
        return signal + high_freq_noise

    def add_harmonics(self, signal, harmonics=[100, 150], levels=[0.2, 0.1]):
        for freq, level in zip(harmonics, levels):
            harmonic_noise = self.noise_level * level * np.sin(2 * np.pi * freq * self.time)
            signal += harmonic_noise
        return signal

    def add_impulsive_noise(self, signal, impulse_rate=0.001, impulse_level=None):
        impulse_level = impulse_level if impulse_level is not None else self.noise_level * np.random.uniform(5, 10)
        num_impulses = int(len(signal) * impulse_rate)
        impulse_indices = np.random.randint(0, len(signal), num_impulses)
        signal[impulse_indices] += impulse_level * np.random.choice([-1, 1], num_impulses)
        return signal

    def add_all_noises(self, signal):
        """
        Adds all predefined noises to the signal using default settings.
        
        Parameters:
            signal (ndarray): Input signal to add all noises to.
        
        Returns:
            ndarray: Signal with all noise types added.
        """
        signal = self.add_white_noise(signal)
        signal = self.add_low_frequency_noise(signal)
        signal = self.add_high_frequency_noise(signal)
        signal = self.add_harmonics(signal)
        signal = self.add_impulsive_noise(signal)
        return signal
