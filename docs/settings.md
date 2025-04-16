# Document settings
# 🎙️ Audio AI App (PyQt6 + Whisper + Ollama)

Приложение на PyQt6, которое позволяет:
- 🎤 Записывать аудио с микрофона
- 🧠 Распознавать речь с помощью Whisper
- 🤖 Отправлять запросы к LLM (Ollama / Transformers)
- 🪄 Получать пояснения к коду или вопросам на русском

---

## 🧩 Зависимости

- Python >= 3.8
- [Poetry](https://python-poetry.org/)
- [FFmpeg](https://ffmpeg.org/) (для Whisper)
- [Ollama](https://ollama.com/) (локальный LLM)

---

## 🛠 Установка на macOS

### 1. Установи Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
poetry --version
```

### 2. Создай виртуальное окружение
```bash
mkdir my-audio-ai-app && cd my-audio-ai-app
poetry init --no-interaction
poetry shell
```

### 3. Установи зависимости
```bash
poetry add \
  pyqt6 \
  sounddevice \
  numpy \
  transformers \
  torch \
  requests \
  pygments \
  git+https://github.com/openai/whisper.git \
  ollama
```

> 💡 Если проблемы с `torch`, попробуй:
```bash
poetry add torch --index-url https://download.pytorch.org/whl/cpu
```

### 4. Установи FFmpeg и PortAudio
```bash
brew install ffmpeg portaudio
```

---

## 🤖 Установка и запуск Ollama

### Установка:
```bash
brew install ollama
```

### Запуск сервера:
```bash
ollama serve
```

### Загрузка модели:
```bash
ollama run mistral
```

---

## 🚀 Запуск GUI приложения

Убедитесь, что у вас есть файлы `main.py` и `manager.py`. Затем запустите:
```bash
poetry run python gui.py
```

---

## ✅ Как пользоваться

1. Нажмите **"Start Record"** — начнётся запись голоса.
2. После остановки — аудио будет автоматически распознано.
3. Нажмите **"Explain"** — будет получен ответ от LLM (Ollama).

---

## 📁 Структура проекта
```
my-audio-ai-app/
├── gui.py               # PyQt интерфейс
├── manager.py           # SoundManager + AI logic
├── test.wav             # Временный файл записи
├── pyproject.toml       # Poetry конфигурация
└── README.md            # Документация
```

---

## 🧠 Модели

Используются:
- `openai/whisper` — для распознавания речи
- `mistral`, `llama2`, `gemma` — LLM модели через Ollama
- `distilgpt2`, `flan-t5-base` — через `transformers`

---



