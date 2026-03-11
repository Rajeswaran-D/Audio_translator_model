from gtts import gTTS
import uuid
import os

def generate_voice(text, lang="ta"):

    if not text or text.strip() == "":
        print("⚠ No speech detected from audio")
        return "No speech detected"

    filename = f"uploads/output_{uuid.uuid4().hex}.mp3"

    tts = gTTS(text=text, lang=lang)
    tts.save(filename)

    return filename