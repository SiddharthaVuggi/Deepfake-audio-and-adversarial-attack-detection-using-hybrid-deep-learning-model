from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
import os

app = Flask(__name__)

# Load the models
FGSM_MODEL_PATH = "E:/capstone/Deep_Fake/notebook/fake_audio_detector.h5"
CNN_MODEL_PATH = "E:/capstone/Deep_Fake/notebook/deepfake_audio_cnn.h5"

if not os.path.exists(CNN_MODEL_PATH) or not os.path.exists(FGSM_MODEL_PATH):
    raise FileNotFoundError("\u274c Model file(s) not found!")

print("\u2705 Loading models...")
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
fgsm_model = tf.keras.models.load_model(FGSM_MODEL_PATH)
print("\u2705 Models loaded successfully!")

# Function to extract audio features (Mel Spectrogram)
def extract_features(audio_path, max_pad_len=128):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=3)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ensure consistent shape (128, 128)
        pad_width = max_pad_len - mel_spec.shape[1]
        if pad_width > 0:
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :max_pad_len]
        
        return np.expand_dims(mel_spec, axis=[0, -1])  # Shape: (1, 128, 128, 1)
    except Exception as e:
        print(f"\u274c Feature extraction error: {e}")
        return None

# Route to serve HTML file
@app.route('/')
def home():
    return render_template("index.html")

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400

    attack_mode = request.form.get("attack_mode", "off").lower() == "on"

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file
    file_path = "temp_audio.wav"
    audio_file.save(file_path)

    print(f"\U0001F4E2 Attack Mode: {'ON' if attack_mode else 'OFF'}")
    print("\u2705 Audio file received and saved.")

    # Extract features
    features = extract_features(file_path)
    if features is None:
        return jsonify({"error": "Feature extraction failed"}), 500

    print(f"\U0001F50D Model Input Shape: {features.shape}")

    try:
        # Choose the model based on attack mode
        model = fgsm_model if attack_mode else cnn_model
        prediction = model.predict(features)
        print(f"\U0001F539 Raw Model Output: {prediction}")

        confidence_fake = float(prediction[0][0])
        confidence_real = float(prediction[0][1])

        result = "Deepfake" if confidence_fake > confidence_real else "Real"
        confidence = max(confidence_fake, confidence_real)
        print(f"\u2705 Prediction: {result} (Confidence: {confidence:.2f})")

        return jsonify({"prediction": result, "confidence": confidence, "attack_mode": attack_mode})

    except Exception as e:
        print(f"\u274c Error during prediction: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)