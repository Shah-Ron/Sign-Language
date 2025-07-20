import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import os

# Load preprocessed dataset
X = np.load("X.npy")             # shape: (samples, 30, features)
y = np.load("y.npy")             # shape: (samples, num_classes)
label_classes = np.load("label_classes.npy", allow_pickle=True)

print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
print(f"Total classes: {len(label_classes)}")

X_train = np.load("X_train.npy")             # shape: (samples, 30, features)
y_train = np.load("y_train.npy")             # shape: (samples, num_classes)
X_test = np.load("X_test.npy")             # shape: (samples, 30, features)
y_test = np.load("y_test.npy")             # shape: (samples, num_classes)

# Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    LSTM(128),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
checkpoint_cb = ModelCheckpoint("sign_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
earlystop_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Optional: Save training history
import json
with open("training_history.json", "w") as f:
    json.dump(history.history, f)

print("Training complete. Model saved as sign_model.h5")
