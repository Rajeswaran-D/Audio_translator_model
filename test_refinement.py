import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from emotion_detector import _classify_gender, _classify_age, _classify_emotion, detect_emotions_for_segments
from voice_generator import generate_voice, VOICE_MAPPING
import asyncio

def test_speaker_analysis():
    print("Testing Speaker Analysis...")
    
    # Test Gender
    assert _classify_gender(120) == "male"
    assert _classify_gender(200) == "female"
    
    # Test Age
    assert _classify_age(260, 2.5) == "child"
    assert _classify_age(150, 1.2) == "elderly"
    assert _classify_age(150, 2.5) == "adult"
    
    # Test Emotion
    # High energy, high pitch, fast rate -> angry
    assert _classify_emotion(220, 0.08, 4.0, gender="female") == "angry"
    # Low energy, low pitch, slow rate -> sad
    assert _classify_emotion(90, 0.01, 1.0, gender="male") == "sad"
    
    print("✅ Speaker Analysis tests passed!")

def test_voice_gen_params():
    print("\nTesting Voice Generation (dry run for params)...")
    # This test assumes the logic in generate_voice correctly calculates final_rate/pitch
    # We can't easily check internal variables of the async function without modification,
    # but we can verify the function signature and execution doesn't crash.
    
    try:
        # Mock values
        path = generate_voice("Hello world", lang="en", emotion="happy", gender="female", age_group="child")
        print(f"Generated (mock/real) path: {path}")
        if path and os.path.exists(path):
            print("✅ Voice generation executed successfully.")
            # os.remove(path) # Keep for manual check if needed
    except Exception as e:
        print(f"❌ Voice generation failed: {e}")

if __name__ == "__main__":
    test_speaker_analysis()
    test_voice_gen_params()
