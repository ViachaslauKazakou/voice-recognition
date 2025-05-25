import sounddevice as sd
import wave
import numpy as np
import whisper
import os
import requests


class SoundManager:
    """
    Class to manage sound recording and speech recognition.
    """

    def __init__(self):
        # Initialize the Whisper model (you can choose a different size: "tiny", "base", "small", "medium", "large")
        # self.model = whisper.load_model("medium")
        # self.model = whisper.load_model("medium", device="cuda")  # Use GPU
        self.model = whisper.load_model("medium", device="cpu")  #  , in_memory=True)  # Use CPU

    def record(self):
        """
        Record audio from the microphone and save it to a WAV file."""
        print("Recording sound...")
        duration = 15  # seconds
        samplerate = 44100

        print("🎤 Запись началась...")
        audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()
        print("✅ Запись завершена.")

        # Save audio to WAV
        with wave.open("test.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())

    @timer
    def recognition(self, file_path=None):
        """
        Recognize speech from an audio file and return the transcribed text."""
        # Check if the model is loaded
        print("Recognizing speech...")

        # Use the recorded file if no file_path is provided
        if file_path is None:
            file_path = "test.wav"

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return None

        # Transcribe the audio file
        try:
            result = self.model.transcribe(file_path, language="ru")
            transcribed_text = result["text"]
            print(f"Transcribed text: {transcribed_text}")
            return transcribed_text
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None


if __name__ == "__main__":
    sound_manager = SoundManager()
    # sound_manager.record()
    text = sound_manager.recognition()
    text = "Напиши пример декоратора на Python."
    print(text)
    ai_manager = AiManager()
    response = ai_manager.ask_ollama(text)
    print("Use local HF model")
    hf = HFTransformer()
    # prompt = "Какой язык программирования лучше всего подходит для веб-разработки?"
    response = hf.generate(text)
    print("Generated response:")
    print(response)
