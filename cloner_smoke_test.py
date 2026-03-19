import asyncio
import sys
import torch
import transformers.utils
import transformers.utils.import_utils
from packaging import version
v = version.parse(torch.__version__.split('+')[0])
check_fn_12 = lambda: v >= version.parse("1.12")
check_fn_13 = lambda: v >= version.parse("1.13")
setattr(transformers.utils, "is_torch_greater_than_1_12", check_fn_12)
setattr(transformers.utils, "is_torch_greater_than_1_13", check_fn_13)
setattr(transformers.utils.import_utils, "is_torch_greater_than_1_12", check_fn_12)
setattr(transformers.utils.import_utils, "is_torch_greater_than_1_13", check_fn_13)

import os
import sys
from pydub import AudioSegment

# Add current directory to path
sys.path.append(os.getcwd())

from cloner_engine import ClonerEngine

async def smoke_test():
    print("Starting Cloner Smoke Test...")
    cloner = ClonerEngine()
    
    # Use a confirmed real reference audio file from the project (largest for best quality)
    ref_wav = os.path.join("uploads", "ref_cea5d80b0a214646a8f5dc7f565ce5ba.wav")
    
    if not os.path.exists(ref_wav):
        print(f"❌ FAILED: {ref_wav} not found. Please ensure at least one upload exists.")
        return
    
    print("Attempting to clone text...")
    try:
        # XTTS might fail on silence, but let's see the error
        path = await cloner.clone_voice_async("Hello testing cloning", ref_wav, language="en")
        print(f"Result Path: {path}")
        if path and os.path.exists(path):
            print("✅ SUCCESS: Cloning worked.")
        else:
            print("❌ FAILURE: Cloning returned None or file missing.")
    except Exception as e:
        print(f"❌ CRASH: {e}")
    finally:
        if os.path.exists(ref_wav): os.remove(ref_wav)

if __name__ == "__main__":
    asyncio.run(smoke_test())
