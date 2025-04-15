import sounddevice as sd
import wave
import numpy as np
import whisper
import os
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from ollama import chat
from ollama import ChatResponse

# model_id = "mistralai/Mistral-7B-Instruct-v0.2"  # можно заменить на другую


# модель, например "meta-llama/Llama-2-7b-chat-hf" или "google/flan-t5-base"
class AIModels:
    
    light: str = "google/flan-t5-base"
    medium: str = "google/flan-t5-large"
    heavy: str = "google/flan-t5-xl"
    light_1: str = "mistralai/Mistral-7B-Instruct-v0.2"
    distilgpt2: str = "distilgpt2"
    # heavy: str = "mistralai/Mistral-7B-Instruct-v0.2"
    # light: str = "meta-llama/Llama-2-7b-chat-hf"
    # medium: str = "meta-llama/Llama-2-7b-chat-hf"


class HFTransformer:
    def __init__(self):
        # Загружаем токенизатор и модель
        self.model_id = AIModels.distilgpt2
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")

    def generate(self, prompt):
        # Создаём pipeline
        generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        # Генерируем текст
        result = generator(
            prompt,
            max_length=200,
            truncation=True,
            do_sample=True,
            temperature=0.9,
            top_p=0.95, top_k=10,
            num_return_sequences=5
        ) 
        return result[0]["generated_text"]


class SoundManager:
    
    def __init__(self):
        # Initialize the Whisper model (you can choose a different size: "tiny", "base", "small", "medium", "large")
        # self.model = whisper.load_model("medium")
        # self.model = whisper.load_model("medium", device="cuda")  # Use GPU
        self.model = whisper.load_model("medium", device="cpu") #  , in_memory=True)  # Use CPU
    
    def record(self):
        print("Recording sound...")
        duration = 8  # seconds
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
            
    def recognition(self, file_path=None):
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
        
        
class AiManager:
    
    def ask_ollama_api(self, prompt):
        modified_prompt = f"You are a Senior software engineer with main skill Python, answer the next question: {prompt}"
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",  # или llama2, gemma и т.п.
                "prompt": modified_prompt,
                "stream": False
            }
        )
        print(response.json()["response"])
        return response.json()["response"]

    def ask_ollama(self, prompt):
        role = "You are a Senior software engineer with main skill Python, answer the next question:"
        modified_prompt = f"[INST]{role} {prompt}[/INST]"
        response: ChatResponse = chat(model='mistral', messages=[
            {
                'role': 'user',
                'content': modified_prompt,
            },
            ])
        print(response['message']['content'])
        # or access fields directly from the response object
        return response.message.content


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
