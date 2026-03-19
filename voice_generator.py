import asyncio
import edge_tts
import uuid
import os
import re
from cloner_engine import ClonerEngine

# Initialize the global cloner
cloner = ClonerEngine()

# Mapping for neural voices (Male and Female options)
VOICE_MAPPING = {
    "ta": {
        "female": "ta-IN-PallaviNeural",
        "male": "ta-IN-ValluvarNeural"
    },
    "en": {
        "female": "en-US-AvaNeural",
        "male": "en-US-AndrewNeural"
    },
    "te": {
        "female": "te-IN-ShrutiNeural",
        "male": "te-IN-MohanNeural"
    },
    "hi": {
        "female": "hi-IN-SwaraNeural",
        "male": "hi-IN-MadhurNeural"
    }
}

# Emotion-to-Vocal Style adjustments (More dramatic for better "feeling")
EMOTION_STYLES = {
    "happy": {"rate": "+35%", "pitch": "+12Hz", "volume": "+10%"},
    "sad": {"rate": "-25%", "pitch": "-10Hz", "volume": "-15%"},
    "angry": {"rate": "+45%", "pitch": "+15Hz", "volume": "+25%"},
    "fear": {"rate": "+40%", "pitch": "+18Hz", "volume": "+5%"},
    "surprise": {"rate": "+25%", "pitch": "+20Hz", "volume": "+15%"},
    "neutral": {"rate": "+0%", "pitch": "+0Hz", "volume": "+0%"}
}

# Age-specific modifiers
AGE_MODIFIERS = {
    "child": {"rate_add": 15, "pitch_add": 25},
    "elderly": {"rate_add": -15, "pitch_add": -5},
    "adult": {"rate_add": 0, "pitch_add": 0}
}

async def generate_voice(text, lang="ta", emotion="neutral", gender="female", age_group="adult", intensity="moderate", reference_audio=None):
    """
    Generates expressive speech output.
    Director Level: Enforces pronunciation clarity and fallback intelligence.
    """
    if not text or text.strip() == "":
        return None

    # --- Director Rule 11: Fallback Intelligence ---
    # Attempt cloning first, but have a high-quality neural standby
    clone_path = None
    if reference_audio and os.path.exists(reference_audio):
        try:
            clone_path = await cloner.clone_voice_async(text, reference_audio, language=lang)
        except Exception as e:
            print(f"Director: Cloning encountered error {e}, initiating neural fallback.")

    if clone_path and os.path.exists(clone_path):
        return clone_path

    # --- Choice 2: High-Quality Neural TTS (Fallback) ---
    filename = f"uploads/output_{uuid.uuid4().hex}.mp3"
    
    # Get the correct voice based on language and gender
    lang_voices = VOICE_MAPPING.get(lang, VOICE_MAPPING["en"])
    voice = lang_voices.get(gender, list(lang_voices.values())[0])
    
    style = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])
    age_mod = AGE_MODIFIERS.get(age_group, AGE_MODIFIERS["adult"])

    # --- Director Rule 6: Pronunciation Perfection ---
    # Adjust base style based on intensity
    intensity_mod = {
        "high": {"rate": 10, "pitch": 5, "volume": "+15%"},
        "moderate": {"rate": 0, "pitch": 0, "volume": "+0%"},
        "quiet": {"rate": -10, "pitch": -5, "volume": "-15%"}
    }.get(intensity, {"rate": 0, "pitch": 0, "volume": "+0%"})

    # Ensure rate is never too fast to maintain articulation clarity
    base_rate = _parse_signed(style['rate']) + age_mod['rate_add'] + intensity_mod['rate']
    clamped_rate = max(-50, min(50, base_rate)) # Keep within human-readable range
    
    rate_val = clamped_rate
    pitch_val = _parse_signed(style['pitch']) + age_mod['pitch_add'] + intensity_mod['pitch']

    # --- Rule 7: Natural Speech Flow (SSML-lite) ---
    # We use punctuation-based pauses that Edge TTS respects naturally
    # Add subtle breaks after commas and periods if not already there
    ssml_text = text.replace(", ", ", ... ").replace(". ", ". ... ")

    # Edge TTS expects rates/pitches with a sign (e.g., '+50%')
    final_rate = f"{rate_val:+}%"
    final_pitch = f"{pitch_val:+}Hz"
    final_volume = style['volume']

    # Communicate supports rate, pitch, and VOLUME
    communicate = edge_tts.Communicate(
        ssml_text, 
        voice, 
        rate=final_rate, 
        pitch=final_pitch,
        volume=final_volume
    )
    await communicate.save(filename)
    return filename

def _parse_signed(val_str):
    """Parses strings like '+10Hz' or '-15%' into integers."""
    if not val_str:
        return 0
    match = re.search(r'([+-]?\d+)', val_str)
    return int(match.group(1)) if match else 0