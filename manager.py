from speech_to_text import transcribe_audio
from translator import translate_text
from voice_generator import generate_voice

def process_audio(path, lang):

    text = transcribe_audio(path)

    if not text:
        return "No speech detected"

    translated = translate_text(text, lang)

    print("Translated text:", translated)

    voice = generate_voice(translated, lang)

    return voice