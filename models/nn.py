from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),  # Explicit Input layer for load_model compatibility
        Conv2D(32, (3, 1), activation='relu'),  # Use (3, 1) kernel size
        MaxPooling2D((2, 1)),  # Pool only along the time dimension
        Conv2D(64, (3, 1), activation='relu'),
        MaxPooling2D((2, 1)),  # Pool only along the time dimension
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model
