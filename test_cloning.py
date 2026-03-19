"""
Verification script for the Voice Cloning transition.
"""
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from cloner_engine import ClonerEngine
from audio_cleaner import isolate_speech, extract_reference
from voice_generator import generate_voice

def test_cloning_pipeline():
    print("Verifying Voice Cloning Pipeline components...")
    
    # Check if cloner initializes
    # Note: Model loading might fail if dependencies aren't fully installed yet
    try:
        engine = ClonerEngine()
        print("✅ ClonerEngine class available.")
    except Exception as e:
        print(f"❌ ClonerEngine failed: {e}")

    # Check audio cleaner
    try:
        # Create a dummy silent wav for testing
        test_wav = "uploads/test_vocal.wav"
        from pydub import AudioSegment
        AudioSegment.silent(duration=1000).export(test_wav, format="wav")
        
        cleaned = isolate_speech(test_wav)
        assert cleaned is not None
        print("✅ Audio cleaner (isolate_speech) functional.")
        
        ref = extract_reference(test_wav, 0.1, 0.5)
        assert ref is not None
        print("✅ Audio cleaner (extract_reference) functional.")
        
        if os.path.exists(test_wav): os.remove(test_wav)
        if os.path.exists(cleaned) and cleaned != test_wav: os.remove(cleaned)
        if os.path.exists(ref): os.remove(ref)
    except Exception as e:
        print(f"❌ Audio cleaner failed: {e}")

if __name__ == "__main__":
    test_cloning_pipeline()
