import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import wavfile

# Load and Preprocess Audio Files
def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return librosa.feature.mfcc(audio, sr=sr, n_mfcc=40).T  # Convert to MFCC

# Build CNN Model for Audio Classification
def build_audio_model():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(40, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')  # Output: Real or Deepfake
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# FGSM Attack to Fool Audio Model
def fgsm_attack(model, audio, label, epsilon=0.1):
    audio = tf.convert_to_tensor(audio.reshape((1, 40, 1)), dtype=tf.float32)
    label = tf.one_hot(label, depth=2)
    label = tf.reshape(label, (1, 2))

    with tf.GradientTape() as tape:
        tape.watch(audio)
        prediction = model(audio)
        loss = tf.keras.losses.CategoricalCrossentropy()(label, prediction)

    gradient = tape.gradient(loss, audio)
    adversarial_audio = audio + epsilon * tf.sign(gradient)
    adversarial_audio = tf.clip_by_value(adversarial_audio, -1, 1)  # Keep within valid audio range
    return adversarial_audio.numpy().reshape((40, 1))

# Generate Dummy Data (Real = 0, Deepfake = 1)
X_real = np.random.rand(100, 40, 1)  # Fake real samples
y_real = np.zeros(100)  # Label 0

X_fake = np.random.rand(100, 40, 1)  # Fake deepfake samples
y_fake = np.ones(100)  # Label 1

# Combine Data
X = np.concatenate((X_real, X_fake))
y = np.concatenate((y_real, y_fake))

# Encode Labels
y = tf.keras.utils.to_categorical(y, num_classes=2)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
audio_model = build_audio_model()
audio_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate on Clean Audio
y_pred_clean = np.argmax(audio_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Generate Adversarial Examples
X_test_adv = np.array([fgsm_attack(audio_model, x, y) for x, y in zip(X_test, y_true)])

# Evaluate on Adversarial Audio
y_pred_adv = np.argmax(audio_model.predict(X_test_adv), axis=1)

# Detect Attack
for i in range(len(y_true)):
    print(f"Original Prediction: {y_pred_clean[i]}, Adversarial Prediction: {y_pred_adv[i]}")
    if y_pred_clean[i] != y_pred_adv[i]:
        print("⚠️ Attack detected! Fake audio misclassified as real ⚠️")
    else:
        print("✅ No attack detected ✅")
