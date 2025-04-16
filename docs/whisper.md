# 🎙 Использование `openai-whisper` с `ffmpeg`

Whisper — это инструмент от OpenAI для распознавания речи. Он работает совместно с FFmpeg, что позволяет обрабатывать аудио- и видеофайлы различных форматов.

---

## 📦 Установка

### Whisper
```bash
pip install git+https://github.com/openai/whisper.git

pip install openai-whisper
```

### FFmpeg

#### macOS (через brew)
```bash
brew install ffmpeg
```

#### Ubuntu/Debian:
```bash
sudo apt update && sudo apt install ffmpeg
```

#### Windows
- Скачай FFmpeg с [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- Добавь путь к `ffmpeg.exe` в системную переменную `PATH`

---

## ✅ Проверка установки

```bash
ffmpeg -version
```

```bash
python -c "import whisper; print(whisper.__version__)"
```

---

## 🔍 Использование из командной строки

```bash
whisper my_audio.mp3 --language Russian
```

### Опции:
- `--language` — язык распознавания (например, Russian, English)
- `--model` — модель (`tiny`, `base`, `small`, `medium`, `large`)

Пример:
```bash
whisper my_audio.mp3 --language Russian --model medium
```

---

## 🧪 Использование из Python

```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("my_audio.mp3", language="ru")
print(result["text"])
```

---

## 🎧 Поддерживаемые форматы

Благодаря `ffmpeg`, поддерживаются:
- `.mp3`
- `.wav`
- `.m4a`
- `.mp4`
- `.ogg`
- и другие форматы (включая видео, из которых извлекается звук)

---

Whisper сам определяет язык, если не указан, но лучше задавать явно для лучшей точности.



