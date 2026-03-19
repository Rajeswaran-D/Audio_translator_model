from pydub import AudioSegment, effects
import os

def merge_audio_with_timing(segments_data, output_path):
    """
    Merges audio segments with natural pauses based on director-level punctuation analysis.
    segments_data: list of dicts with {'path', 'start', 'end', 'text'}
    """
    if not segments_data:
        return None

    # Load and combine all segments
    final_audio = AudioSegment.silent(duration=0)
    current_time_ms = 0
    
    # Director Rule 7: Natural pauses based on punctuation
    PAUSE_SHORT = 300 # ms for commas/etc
    PAUSE_LONG = 600  # ms for periods/questions
    
    for seg in segments_data:
        if not os.path.exists(seg['path']):
            continue
            
        seg_audio = AudioSegment.from_file(seg['path'])
        
        # Add slight 50ms fade for smoothness
        seg_audio = seg_audio.fade_in(50).fade_out(50)
        
        start_ms = int(seg['start'] * 1000)
        
        # Calculate if we need actual silence or if we just append with a director-selected pause
        if start_ms > current_time_ms:
            silence_dur = start_ms - current_time_ms
            final_audio += AudioSegment.silent(duration=silence_dur)
            current_time_ms = start_ms
            
        final_audio += seg_audio
        current_time_ms += len(seg_audio)
        
        # Director: Add punctuation-based pause after the segment if text is available
        text = seg.get('text', '')
        if text.endswith(('.', '?', '!')):
            final_audio += AudioSegment.silent(duration=PAUSE_LONG)
            current_time_ms += PAUSE_LONG
        elif text.endswith((',', ';', ':')):
            final_audio += AudioSegment.silent(duration=PAUSE_SHORT)
            current_time_ms += PAUSE_SHORT

    # Normalize final output to prevent clipping and ensure consistent volume
    final_audio = effects.normalize(final_audio)

    # Ensure high-fidelity sample rate (fallback to 48kHz for professional clarity)
    final_audio = final_audio.set_frame_rate(48000)

    final_audio.export(output_path, format="mp3", bitrate="256k")
    return output_path