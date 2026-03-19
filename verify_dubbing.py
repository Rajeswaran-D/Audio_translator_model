import asyncio
from translator import translate_text
from emotion_detector import _calculate_intensity
from voice_generator import generate_voice
import os

def test_translator():
    print("--- Testing Translator Semantic Reconstruction ---")
    
    # Test Angry
    angry_text = translate_text("Please help me.", "en", emotion="angry")
    print(f"Original: 'Please help me.' | Angry: '{angry_text}'")
    assert "PLEASE" not in angry_text
    assert "HELP ME" in angry_text
    assert angry_text.endswith("!!")
    
    # Test Sad
    sad_text = translate_text("I am so sorry.", "en", emotion="sad")
    print(f"Original: 'I am so sorry.' | Sad: '{sad_text}'")
    assert sad_text.startswith("... ")
    assert sad_text.endswith("...")
    assert "i am so sorry" in sad_text.lower()
    
    # Test Fear
    fear_text = translate_text("Who is there?", "en", emotion="fear")
    print(f"Original: 'Who is there?' | Fear: '{fear_text}'")
    assert fear_text.startswith("W-W-Who")
    
    # Test Surprise
    surprise_text = translate_text("It is a ghost.", "en", emotion="surprise")
    print(f"Original: 'It is a ghost.' | Surprise: '{surprise_text}'")
    assert surprise_text.startswith("What?!")

    print("✅ Translator tests passed!\n")

def test_intensity_detection():
    print("--- Testing Intensity Detection ---")
    # _ENERGY_HIGH = 0.06
    assert _calculate_intensity(0.1) == "high"
    assert _calculate_intensity(0.05) == "moderate"
    assert _calculate_intensity(0.01) == "quiet"
    print("✅ Intensity detection tests passed!\n")

async def test_voice_generator_dry_run():
    print("--- Testing Voice Generator (Dry Run) ---")
    try:
        # Test with new intensity parameter
        path = await generate_voice(
            "This is a high intensity test.", 
            lang="en", 
            emotion="excited", 
            intensity="high"
        )
        if path and os.path.exists(path):
            print(f"✅ Voice generator (high intensity) successful: {path}")
            # os.remove(path)
            
        path_quiet = await generate_voice(
            "This is a quiet test.", 
            lang="en", 
            emotion="sad", 
            intensity="quiet"
        )
        if path_quiet and os.path.exists(path_quiet):
            print(f"✅ Voice generator (quiet) successful: {path_quiet}")
            # os.remove(path_quiet)
            
    except Exception as e:
        print(f"❌ Voice generator test failed: {e}")
        raise e

if __name__ == "__main__":
    test_translator()
    test_intensity_detection()
    asyncio.run(test_voice_generator_dry_run())
