import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

def load_and_visualize_dataset(filepath="dataset/dataset_100ms.csv", num_samples_to_plot=5):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Basic statistics
    print("Dataset Statistics:")
    print(df['label'].value_counts())
    print("\nTotal samples:", len(df))
    print("Samples without frequency shift (label=0):", df['label'].value_counts()[0])
    print("Samples with frequency shift (label=1):", df['label'].value_counts()[1])
    
    # Plot distribution of labels
    plt.figure(figsize=(6, 4))
    df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Label Distribution")
    plt.xlabel("Label (0 = No Shift, 1 = Shift)")
    plt.ylabel("Number of Samples")
    plt.xticks([0, 1], ['No Shift', 'Shift'], rotation=0)
    plt.show()

    # Visualize a few random samples from the dataset
    sample_indices = random.sample(range(len(df)), num_samples_to_plot)
    plt.figure(figsize=(12, 8))
    
    for i, idx in enumerate(sample_indices, 1):
        sample = np.fromstring(df.iloc[idx]['signal'].strip("[]"), sep=", ")
        label = df.iloc[idx]['label']
        plt.subplot(num_samples_to_plot, 1, i)
        plt.plot(sample, color="blue" if label == 0 else "red")
        plt.title(f"Sample {idx} - Label: {'No Shift' if label == 0 else 'Shift'}")
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.show()

# Main block to load and visualize the dataset
if __name__ == "__main__":
    load_and_visualize_dataset()
