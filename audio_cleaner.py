"""
Audio Cleaner Module — Speech/Noise isolation and volume normalization.

Provides:
  isolate_speech(audio_path) -> str (path to cleaned audio)
"""

import os
import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np
import uuid
from pydub import AudioSegment, effects

def isolate_speech(audio_path):
    """
    Isolates speech from background noise using noisereduce and normalization.
    """
    if not audio_path or not os.path.exists(audio_path):
        return None

    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)

        # Apply noise reduction
        # We use a stationary noise reduction as a baseline
        reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)

        # Save to a temporary file for pydub normalization
        temp_path = f"uploads/temp_clean_{uuid.uuid4().hex}.wav"
        sf.write(temp_path, reduced_noise, sr)

        # Use pydub for high-quality normalization
        raw_audio = AudioSegment.from_file(temp_path)
        normalized_audio = effects.normalize(raw_audio)
        
        final_path = f"uploads/cleaned_{uuid.uuid4().hex}.wav"
        normalized_audio.export(final_path, format="wav")

        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        return final_path

    except Exception as e:
        print(f"Error isolating speech: {e}")
        return audio_path # Return original if failure

def extract_reference(audio_path, start, end):
    """
    Extracts a short segment of audio to use as a cloning reference.
    Ideally 5-10 seconds of clear speech.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        # Convert seconds to milliseconds
        t1 = start * 1000
        t2 = end * 1000
        
        # Ensure at least a minimal length or adjust to max 10s for efficiency
        if (t2 - t1) > 10000:
            t2 = t1 + 10000
            
        segment = audio[t1:t2]
        
        ref_path = f"uploads/ref_{uuid.uuid4().hex}.wav"
        segment.export(ref_path, format="wav")
        return ref_path
    except Exception as e:
        print(f"Error extracting reference: {e}")
        return None
