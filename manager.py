from speech_to_text import transcribe_audio, transcribe_audio_segments
from emotion_detector import detect_emotions_for_segments, warm_up as warm_up_emotion
from translator import translate_text
from voice_generator import generate_voice
from merger import merge_audio_with_timing
from audio_cleaner import isolate_speech, extract_reference
import os
import uuid
import asyncio
from cloner_engine import ClonerEngine

def warm_up_engines():
    """Triggers loading of all heavy models to ensure first-run speed."""
    print("Director: Initializing global engine warm-up...")
    cloner = ClonerEngine()
    cloner.warm_up()
    warm_up_emotion()
    print("Director: Warm-up complete.")

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

async def process_audio_detailed(path, lang):
    """
    Full pipeline processing (Optimized Parallel Version):
    1. Segmented transcription
    2. Batch translation (preserves context)
    3. Per-segment voice generation
    4. Timing-aware audio merging
    5. Metadata file generation
    """
    # 0. Ensure engines are warm
    warm_up_engines()

    # 1. Speech Isolation & Cleaning
    cleaned_path = isolate_speech(path) or path
    
    # 2. Transcription
    segments = transcribe_audio_segments(cleaned_path)
    if not segments:
        return {"error": "No speech detected"}

    # 2. Emotion Detection
    segments = detect_emotions_for_segments(path, segments)

    # --- Director Rule 5: Global Speaker Reference Management ---
    # Select the longest/clearest segment as the "Global Reference" 
    # to ensure voice consistency across all segments.
    global_reference_audio = None
    if segments:
        best_seg = max(segments, key=lambda x: x.get('end', 0) - x.get('start', 0))
        global_reference_audio = extract_reference(cleaned_path, best_seg['start'], best_seg['end'])
        print(f"Director: Selected Global Reference from [{best_seg['start']}-{best_seg['end']}]")

    # 3. Batch Translation (Collect all text first)
    original_texts = [seg.get('text', '') for seg in segments]
    translated_texts = translate_text(original_texts, lang)
    
    # Ensure translated_texts is a list of the same length
    if not isinstance(translated_texts, list) or len(translated_texts) != len(segments):
        # Fallback if batch translation failed or returned unexpected result
        translated_texts = [translate_text(t, lang, emotion=segments[idx].get('emotion')) for idx, t in enumerate(original_texts)]

    # 4. Voice Generation per segment (PARALLEL)
    async def process_single_segment(seg, index):
        original = original_texts[index]
        translated = translated_texts[index]
        start = seg.get('start', 0.0)
        end = seg.get('end', 0.0)
        emotion = seg.get('emotion', 'neutral')
        gender = seg.get('gender', 'female')
        age_group = seg.get('age_group', 'adult')
        intensity = seg.get('intensity', 'moderate')
        
        seg['translated_text'] = translated

        # Director: Use the persistent Global Reference instead of per-segment extraction
        # reference_audio = extract_reference(cleaned_path, start, end)

        # Generate voice ASYNC
        seg_audio_path = await generate_voice(
            translated, 
            lang, 
            emotion=emotion, 
            gender=gender, 
            age_group=age_group,
            intensity=intensity,
            reference_audio=global_reference_audio
        )
        
        voice_type = "cloned" if "clone_" in (seg_audio_path or "") else "neural"
        seg['voice_type'] = voice_type
        
        return {
            'path': seg_audio_path, 
            'start': start, 
            'end': end, 
            'index': index,
            'original': original,
            'translated': translated,
            'emotion': emotion,
            'intensity': intensity,
            'age_group': age_group
        }

    # Execute all segments in parallel
    tasks = [process_single_segment(seg, i) for i, seg in enumerate(segments)]
    processed_audio_data = await asyncio.gather(*tasks)

    audio_segments_to_merge = []
    metadata_lines = [
        "Index | Start (s) | End (s) | Duration (s) | Original Text | Translated Text | Emotion | Age",
        "-" * 120
    ]

    for data in processed_audio_data:
        i = data['index']
        start = data['start']
        end = data['end']
        duration = round(end - start, 2)
        seg_audio_path = data['path']
        
        if seg_audio_path and os.path.exists(seg_audio_path):
            audio_segments_to_merge.append({
                'path': seg_audio_path, 
                'start': start,
                'text': data['original'] # Pass text for punctuation analysis in director mode
            })

        # Add to metadata file content
        line = f"{i+1:02d} | {start:>9.2f} | {end:>7.2f} | {duration:>12.2f} | {data['original'][:30]:<30} | {data['translated'][:30]:<30} | {data['emotion']:<8} | {data['intensity']:<8} | {data['age_group']}"
        metadata_lines.append(line)

    # 5. Merge audio with original timing
    final_output_path = f"uploads/final_{uuid.uuid4().hex}.mp3"
    voice_file = merge_audio_with_timing(audio_segments_to_merge, final_output_path)

    # 6. Save Metadata File
    metadata_filename = f"uploads/metadata_{uuid.uuid4().hex}.txt"
    with open(metadata_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    return {
        "status": "success",
        "audio_file": voice_file,
        "metadata_file": metadata_filename,
        "segments": segments
    }