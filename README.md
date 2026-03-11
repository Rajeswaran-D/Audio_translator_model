## AI-Based Multilingual Audio Translation and Dubbing System

This project presents an AI-based system designed to automatically translate spoken audio from one language to another while preserving the natural flow of speech.

The system accepts an input audio file such as a speech recording, dialogue, or movie audio and converts the spoken content into text using advanced speech recognition techniques. The extracted text is then translated into the user’s desired target language using a machine translation model. After translation, the system generates a new audio output using text-to-speech technology, producing a natural-sounding voice in the selected language.

The entire process is integrated into a web-based interface where users can upload an audio file, choose a target language, and receive the translated and synthesized audio output. The backend is implemented using Python and FastAPI, while AI models such as speech recognition and neural translation are used to process and convert the audio.

This system aims to simplify multilingual communication and enable automated dubbing for various types of audio content, including educational materials, podcasts, and entertainment media. By combining speech recognition, language translation, and voice synthesis, the project demonstrates how artificial intelligence can be used to create efficient and accessible multilingual audio conversion systems.

### Features

- **Automatic speech recognition (ASR)**: Convert spoken audio into source-language text.
- **Neural machine translation (NMT)**: Translate extracted text into a target language.
- **Text-to-speech (TTS)**: Generate natural-sounding speech in the translated language.
- **Web-based interface**: Upload audio, select target language, and download translated audio.
- **Scalable backend**: Python + FastAPI architecture suitable for deployment.

### Tech Stack

- **Backend**: Python, FastAPI
- **AI/ML**: Speech recognition, neural machine translation, text-to-speech
- **Audio processing**: Libraries like `pydub` and others as needed

### Project Structure (initial)

- `Audio_translator_model/` – Core audio processing and model-related utilities (e.g., `audio_splitter.py`).
- `README.md` – Project overview and usage.
- `requirements.txt` – Python dependencies (to be expanded as the project grows).

### Getting Started

1. **Create and activate a virtual environment (recommended)**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the FastAPI app (once created)**:

   ```bash
   uvicorn main:app --reload
   ```

### Future Work

- Integrate end-to-end pipeline: ASR → Translation → TTS.
- Build the FastAPI endpoints for uploading audio and returning translated audio.
- Add a frontend for easy interaction.
- Optimize performance for long-form content (lectures, movies, podcasts).

