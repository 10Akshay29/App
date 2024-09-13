import streamlit as st
import os
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model
import sounddevice as sd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from keras.losses import MeanSquaredError


SVM_MODEL_PATH = 'ocsvm_model.pkl'
AUTOENCODER_MODEL_PATH = 'autoencoder_model.h5'
SCALER_PATH = 'scaler.pkl'



@st.cache_resource
def load_models():
    ocsvm = joblib.load(SVM_MODEL_PATH) if os.path.exists(SVM_MODEL_PATH) else None
    autoencoder = load_model(AUTOENCODER_MODEL_PATH, custom_objects={'mse': MeanSquaredError()}) if os.path.exists(AUTOENCODER_MODEL_PATH) else None
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return ocsvm, autoencoder, scaler



def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean



def record_audio(duration=5, fs=22050):
    st.write(f"Recording audio for {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio_data = audio_data.flatten()
    st.write("Recording complete.")
    return audio_data



def predict_parkinsons_svm(audio, scaler, ocsvm):
    features = extract_features(audio)
    features_scaled = scaler.transform([features])
    prediction = ocsvm.predict(features_scaled)
    if prediction == 1:
        return "The model predicts that the input voice is indicative of Parkinson's disease."
    else:
        return "The model predicts that the input voice is healthy."



def predict_anomaly_autoencoder(audio, scaler, autoencoder):
    features = extract_features(audio)
    features_scaled = scaler.transform([features])
    reconstruction = autoencoder.predict(features_scaled)
    reconstruction_error = np.mean(np.square(features_scaled - reconstruction))
    threshold = 0.05
    if reconstruction_error > threshold:
        return "Anomaly detected: This voice input might indicate Parkinson's disease."
    else:
        return "No anomaly detected: The voice input is likely healthy."



def main():
    st.title("Parkinson's Disease Detection from Voice")

    ocsvm, autoencoder, scaler = load_models()
    
    if not ocsvm or not autoencoder or not scaler:
        st.error("Models or scaler not found. Please train the models first.")
        st.stop()

    choice = st.sidebar.radio("Choose an option", ["Record Real-Time Voice", "Upload Audio File"])

    if choice == "Record Real-Time Voice":
        st.write("Speak into the microphone...")
        if st.button("Start Recording"):
            audio = record_audio(duration=5)
            if ocsvm and scaler:
                result = predict_parkinsons_svm(audio, scaler, ocsvm)
                st.write(result)
            else:
                st.error("OneClassSVM model or scaler is missing!")

    elif choice == "Upload Audio File":
        uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "flac"])
        if uploaded_file is not None:
            try:
                audio, sr = librosa.load(uploaded_file, sr=22050)
                if autoencoder and scaler:
                    result = predict_anomaly_autoencoder(audio, scaler, autoencoder)
                    st.write(result)
                else:
                    st.error("Autoencoder model or scaler is missing!")
            except Exception as e:
                st.error(f"Error loading the file: {e}")


if __name__ == "__main__":
    main()