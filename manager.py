from speech_to_text import transcribe_audio, transcribe_audio_segments
from emotion_detector import detect_emotions_for_segments
from translator import translate_text
from voice_generator import generate_voice
import os
import uuid

def process_audio(path, lang):
    """Simple backward-compatible version."""
    from speech_to_text import transcribe_audio
    text = transcribe_audio(path)
    if not text:
        return "No speech detected"
    translated = translate_text(text, lang)
    print("Translated text:", translated)
    voice = generate_voice(translated, lang)
    return voice

def process_audio_detailed(path, lang):
    """
    Full pipeline processing:
    1. Segmented transcription
    2. Per-segment translation
    3. Emotion detection
    4. Metadata file generation
    5. Final voice generation
    """
    # 1. Transcription
    segments = transcribe_audio_segments(path)
    if not segments:
        return {"error": "No speech detected"}

    # 2. Emotion Detection
    segments = detect_emotions_for_segments(path, segments)

    # 3. Translation & Metadata Formatting
    full_translated_text = []
    metadata_lines = [
        "Index | Start (s) | End (s) | Duration (s) | Original Text | Translated Text | Emotion",
        "-" * 100
    ]

    for i, seg in enumerate(segments):
        original = seg.get('text', '')
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        duration = round(end - start, 2)
        emotion = seg.get('emotion', 'neutral')
        
        # Translate this segment
        translated = translate_text(original, lang)
        seg['translated_text'] = translated
        full_translated_text.append(translated)

        # Add to metadata file content
        line = f"{i+1:02d} | {start:>9.2f} | {end:>7.2f} | {duration:>12.2f} | {original[:30]:<30} | {translated[:30]:<30} | {emotion}"
        metadata_lines.append(line)

    # 4. Generate Combined Voice
    final_text = " ".join(full_translated_text)
    voice_file = generate_voice(final_text, lang)

    # 5. Save Metadata File
    metadata_filename = f"uploads/metadata_{uuid.uuid4().hex}.txt"
    with open(metadata_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    return {
        "status": "success",
        "audio_file": voice_file,
        "metadata_file": metadata_filename,
        "segments": segments
    }