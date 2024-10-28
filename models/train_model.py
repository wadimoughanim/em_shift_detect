import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from models.wavelet_analysis import WaveletAnalyzer
from models.nn import create_cnn_model
import os

def prepare_data_for_training(filepath="dataset/dataset_100ms.csv", fs=20000, num_samples=100):
    df = pd.read_csv(filepath).sample(n=num_samples, random_state=42)
    signals = [np.fromstring(sig.strip("[]"), sep=",") for sig in df['signal']]
    labels = df['label'].values

    analyzer = WaveletAnalyzer(fs=fs)
    wavelet_spectra = analyzer.batch_extract_time_frequency_spectrum(signals)
    wavelet_spectra = wavelet_spectra[:, :10, :10]  # Crop or reshape if necessary
    X = wavelet_spectra[..., np.newaxis]
    y = labels

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model():
    X_train, X_test, y_train, y_test = prepare_data_for_training(num_samples=200)
    input_shape = X_train.shape[1:]
    model = create_cnn_model(input_shape)
    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=20,
                        batch_size=16,
                        callbacks=[early_stopping])

    # Save the model using the default Keras format by specifying .keras extension
# Save the model architecture to JSON
    model_json = model.to_json()
    with open("models/wavelet_cnn_model.json", "w") as json_file:
        json_file.write(model_json)

    # Save the weights
    model.save_weights("models/wavelet_cnn_model.weights.h5")


if __name__ == "__main__":
    train_model()
