import numpy as np
import pandas as pd
import random
from utils.signal_generation import SignalGenerator
from utils.signal_management import get_100ms_subcuts
import os

def generate_dataset_from_long_signal(total_duration=60, fs=20000):
    """
    Generates a dataset of 100 ms signal samples from a long signal with random shifts.
    
    Parameters:
        total_duration (float): Total duration of the long signal in seconds.
        fs (int): Sampling frequency in Hz.
        
    Returns:
        pd.DataFrame: DataFrame with columns ['signal', 'label'] where 
                      'signal' contains 100 ms samples and 'label' is 0 or 1.
    """
    # Randomly configure signal generation parameters
    f_base = 50
    f_min = random.uniform(45, 49.9)  # Range for possible frequency shifts
    f_max = random.uniform(50.1, 55)
    noise_level = random.choice([0.01, 0.05, 0.1])

    # Generate the long signal
    generator = SignalGenerator(
        fs=fs,
        duration=total_duration,
        f_base=f_base,
        f_min=f_min,
        f_max=f_max,
        mean_interval=2,
        transition_duration=0.03,
        noise_level=noise_level
    )
    long_signal, true_frequency, change_points = generator.generate_noisy_signal_with_shifts()

    # Split the long signal into 100 ms segments
    segments = get_100ms_subcuts(np.arange(0, total_duration, 1 / fs), long_signal, true_frequency, fs)

    # Prepare the dataset
    dataset = []
    for (time_segment, signal_segment), (_, frequency_segment) in segments:
        # Label as 1 if there’s any deviation from 50 Hz in the segment
        label = 1 if np.any(frequency_segment != 50) else 0
        dataset.append({'signal': signal_segment, 'label': label})

    return pd.DataFrame(dataset)

# Main block to generate and save the dataset
if __name__ == "__main__":
    # Create the dataset directory if it doesn't exist
    os.makedirs("dataset", exist_ok=True)
    
    # Generate dataset from a long signal
    df = generate_dataset_from_long_signal(total_duration=60, fs=20000)
    
    # Save dataset to CSV
    df.to_csv("dataset/dataset_100ms.csv", index=False)
    print("Dataset saved to dataset/dataset_100ms.csv")
