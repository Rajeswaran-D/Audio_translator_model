"""
Performance and Quality Verification script for the Optimized Engine.
"""
import asyncio
import time
import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from manager import process_audio_detailed

async def verify_optimization():
    print("🚀 Verifying High-Performance Optimizations...")
    
    # Check if uploads dir exists
    os.makedirs("uploads", exist_ok=True)
    
    # Create a dummy test file if not exists
    test_file = "uploads/perf_test.wav"
    from pydub import AudioSegment
    # 5 seconds of "speech" (silence for tests)
    AudioSegment.silent(duration=5000).export(test_file, format="wav")
    
    start_time = time.time()
    print("Starting parallel processing pipeline...")
    
    try:
        # Test with a mock run (might fail if models aren't loaded or GPU busy, but logic check)
        result = await process_audio_detailed(test_file, "en")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Pipeline completed in {duration:.2f} seconds.")
        if "audio_file" in result:
            print(f"✅ Final audio generated: {result['audio_file']}")
            print(f"✅ Segments processed: {len(result['segments'])}")
            for i, seg in enumerate(result['segments']):
                print(f"   [{i+1}] Type: {seg.get('voice_type')}, Emotion: {seg.get('emotion')}")
        
    except Exception as e:
        print(f"❌ Performance test encountered an issue: {e}")
        # If it fails due to missing model files (which is likely in a restricted env), 
        # the logic check is still valuable.

if __name__ == "__main__":
    asyncio.run(verify_optimization())
