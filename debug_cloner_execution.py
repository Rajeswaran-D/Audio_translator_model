import os
import sys
import time

def debug_tt_import():
    try:
        print("Attempting to import TTS...")
        start = time.time()
        from TTS.api import TTS
        print(f"Import successful in {time.time() - start:.2f}s")
        
        print("Attempting to initialize XTTS v2...")
        # Note: This usually asks for 'y' if not agreed before.
        # We can set an environment variable to skip if it exists.
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        start = time.time()
        # Initialize without downloading if possible, or just see if it hangs
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print(f"Model initialization successful in {time.time() - start:.2f}s")
        
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

if __name__ == "__main__":
    debug_tt_import()
