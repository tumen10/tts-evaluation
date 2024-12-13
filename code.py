import os
import torchaudio
import torch
from transformers import pipeline
from jiwer import wer
import librosa
import numpy as np
from scipy.spatial.distance import euclidean

# Step 1: Create directories
PROJECT_DIR = "tts_evaluation"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
GENERATED_DIR = os.path.join(PROJECT_DIR, "generated")
EVAL_DIR = os.path.join(PROJECT_DIR, "evaluation")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# Step 2: Load the TTS model
print("Loading TTS model...")
tts_pipeline = pipeline("text-to-speech", model="parler-tts/parler-tts-mini-jenny-30H")

# Step 3: Define test sentences
sentences = [
    "Сегодня хорошая погода для прогулки.",
    "Широкозубый зверь шёл через шумный зал.",
    "Это была самая счастливая минута в моей жизни!",
    "Цена акций Tesla на 14 декабря — $658.45."
]

# Step 4: Generate audio files
print("Generating audio files...")
for i, text in enumerate(sentences):
    audio_path = os.path.join(GENERATED_DIR, f"sample_{i + 1}.wav")
    audio_data = tts_pipeline(text, return_tensors=True).audio_values
    torchaudio.save(audio_path, torch.tensor([audio_data]), sample_rate=24000)
    print(f"Generated: {audio_path}")

# Step 5: Define transcription function using SpeechRecognition
import speech_recognition as sr

def transcribe_audio(audio_path):
    """Transcribe audio to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        transcript = recognizer.recognize_google(audio, language="ru-RU")
        return transcript
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Error with transcription service: {e}")
        return ""

# Step 6: Calculate evaluation metrics

def calculate_wer(original_texts, generated_audio_dir):
    wer_scores = []
    for i, text in enumerate(original_texts):
        audio_path = os.path.join(generated_audio_dir, f"sample_{i + 1}.wav")
        transcript = transcribe_audio(audio_path)
        print(f"Original: {text}\nTranscript: {transcript}")
        score = wer(text, transcript)
        wer_scores.append(score)
    return wer_scores

def calculate_mcd(original_audio_path, generated_audio_path):
    """Calculate Mel Cepstral Distortion (MCD) between two audio files."""
    y_ref, sr_ref = librosa.load(original_audio_path, sr=None)
    y_gen, sr_gen = librosa.load(generated_audio_path, sr=None)

    # Ensure sampling rates match
    if sr_ref != sr_gen:
        raise ValueError("Sampling rates do not match between reference and generated audio.")

    # Extract mel-spectrograms
    mel_ref = librosa.feature.mfcc(y=y_ref, sr=sr_ref, n_mfcc=13)
    mel_gen = librosa.feature.mfcc(y=y_gen, sr=sr_gen, n_mfcc=13)

    # Calculate MCD
    mcd = np.mean([euclidean(m1, m2) for m1, m2 in zip(mel_ref.T, mel_gen.T)])
    return mcd

print("Evaluating WER...")
wer_results = calculate_wer(sentences, GENERATED_DIR)
print("WER Results:", wer_results)

# Step 7: Save evaluation results
results_path = os.path.join(EVAL_DIR, "evaluation_results.txt")
with open(results_path, "w") as f:
    f.write("WER Results:\n")
    for i, score in enumerate(wer_results):
        f.write(f"Sample {i + 1}: {score}\n")
    # Placeholder for additional metrics
    # f.write(f"MCD Example: {mcd_example}\n")

print(f"Results saved to {results_path}")

# Step 8: Implement placeholders for MOS and PESQ
# MOS (Mean Opinion Score) and PESQ (Perceptual Evaluation of Speech Quality)
# are not included in this script but can be integrated with third-party libraries or manual scoring processes.

# Note: This script now includes basic transcription and WER calculation.
