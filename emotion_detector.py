"""
Emotion Detector Module — acoustic-feature-based emotion classification.

Uses librosa to extract pitch (F0), energy (RMS), and speech-rate features
from audio segments, then maps the feature profile to one of six emotions:
  neutral, happy, sad, angry, fear, surprise

Provides:
  detect_emotion(audio_path)                           -> str   (backward-compatible)
  detect_emotions_for_segments(audio_path, segments)   -> list   (enriched segments)
  warm_up()                                            -> bool  (pre-loads models)
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded librosa (heavy import — only load when first needed)
# ---------------------------------------------------------------------------
_librosa = None


def _get_librosa():
    """Import librosa once and cache the module reference."""
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
        logger.info("librosa loaded successfully.")
    return _librosa


def warm_up():
    """Pre-loads heavy dependencies (librosa)."""
    print("Director: Warming up Emotion Detector...")
    return _get_librosa() is not None


# ---------------------------------------------------------------------------
# Audio cache — avoid re-reading the same file multiple times
# ---------------------------------------------------------------------------
_audio_cache: dict = {}


def _load_audio(audio_path: str, sr: int = 22050):
    """
    Load an audio file (cached). Returns (y, sr) or (None, None) on error.
    """
    cache_key = (audio_path, sr)
    if cache_key in _audio_cache:
        return _audio_cache[cache_key]

    librosa = _get_librosa()
    try:
        y, sr_out = librosa.load(audio_path, sr=sr)
        _audio_cache[cache_key] = (y, sr_out)
        return y, sr_out
    except Exception as exc:
        logger.error("Failed to load audio '%s': %s", audio_path, exc)
        _audio_cache[cache_key] = (None, None)
        return None, None


def clear_audio_cache():
    """Free memory occupied by cached audio arrays."""
    _audio_cache.clear()


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_pitch(y_segment, sr: int) -> float:
    """Return the median fundamental frequency (F0) in Hz, or 0.0."""
    librosa = _get_librosa()
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y_segment,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        if len(f0_voiced) == 0:
            return 0.0
        return float(np.median(f0_voiced))
    except Exception:
        return 0.0


def _extract_energy(y_segment) -> float:
    """Return the mean RMS energy of the segment."""
    librosa = _get_librosa()
    try:
        rms = librosa.feature.rms(y=y_segment)[0]
        return float(np.mean(rms))
    except Exception:
        return 0.0


def _estimate_speech_rate(text: str, duration: float) -> float:
    """Words per second — a rough proxy for speech rate."""
    if duration <= 0:
        return 0.0
    word_count = len(text.split())
    return word_count / duration


# ---------------------------------------------------------------------------
# Rule-based emotion classifier
# ---------------------------------------------------------------------------

# Thresholds (tuned for typical speech; can be refined with labelled data)
_PITCH_HIGH = 220.0    # Hz — above this is "high pitch"
_PITCH_LOW = 130.0     # Hz — below this is "low pitch"
_ENERGY_HIGH = 0.06    # RMS — above this is "loud"
_ENERGY_LOW = 0.015    # RMS — below this is "quiet"
_RATE_FAST = 3.5       # words/sec
_RATE_SLOW = 1.5       # words/sec


def _classify_gender(pitch: float, text: str = "") -> str:
    """
    Classifies gender based on median fundamental frequency (F0).
    Male typical: 85-155 Hz | Female typical: 165-255 Hz
    """
    if pitch == 0:
        return "male" 
    if pitch < 145:
        return "male"
    else:
        return "female"

def _classify_age(pitch: float, rate: float) -> str:
    """
    Classifies age group based on pitch and speech rate.
    Child typical: > 250 Hz
    Elderly typical: < 1.8 words/sec and slightly lower pitch than standard adult
    Adult: default
    """
    if pitch > 250:
        return "child"
    if rate > 0 and rate < 1.8:
        return "elderly"
    return "adult"

def _classify_emotion(pitch: float, energy: float, rate: float, gender: str = "male") -> str:
    """
    Map acoustic features to one of six emotions using gender-relative heuristics.
    """
    # Adjust pitch thresholds based on gender
    # Female standard: ~220Hz. Male standard: ~150Hz.
    PITCH_HI = 210.0 if gender == "female" else 155.0
    PITCH_LO = 130.0 if gender == "female" else 95.0

    # Log metrics for debugging emotionless results
    print(f"DEBUG: Emotion Check -> Pitch: {pitch:.1f} (Threshold: {PITCH_HI:.1f}), Energy: {energy:.4f} (Threshold: {_ENERGY_HIGH}), Rate: {rate:.1f}")

    if energy >= _ENERGY_HIGH and pitch >= PITCH_HI and rate >= _RATE_FAST:
        return "angry"
    if energy >= _ENERGY_HIGH and pitch >= PITCH_HI:
        return "happy"
    if pitch >= PITCH_HI * 1.3 and energy >= _ENERGY_LOW:
        return "surprise"
    if pitch >= PITCH_HI and energy < _ENERGY_LOW and rate >= _RATE_FAST:
        return "fear"
    if pitch < PITCH_LO and energy < _ENERGY_LOW and rate <= _RATE_SLOW:
        return "sad"
    return "neutral"

def _calculate_intensity(energy: float) -> str:
    """Return an intensity label: quiet, moderate, high."""
    if energy >= _ENERGY_HIGH * 1.5:
        return "high"
    if energy >= _ENERGY_HIGH * 0.8:
        return "moderate"
    return "quiet"

def detect_emotions_for_segments(audio_path: str, segments: list) -> list:
    """
    Detect emotion and gender for each speech segment using acoustic features.

    Parameters
    ----------
    audio_path : str
        Path to the original audio file.
    segments : list[dict]
        Each dict must have keys ``text`` (str), ``start`` (float), ``end`` (float).

    Returns
    -------
    list[dict]
        Same dicts with added ``emotion`` (str) and ``gender`` (str) keys.
    """
    if not segments:
        return []

    y, sr = _load_audio(audio_path)
    if y is None:
        return [{**seg, "emotion": "neutral", "gender": "female", "age_group": "adult"} for seg in segments]

    enriched = []
    print(f"\n--- Starting Audio Analysis for: {os.path.basename(audio_path)} ---")
    
    for seg in segments:
        try:
            start_sec = seg.get("start", 0.0)
            end_sec = seg.get("end", 0.0)
            text = seg.get("text", "")
            duration = end_sec - start_sec

            if duration < 0.1:
                enriched.append({**seg, "emotion": "neutral", "gender": "male", "age_group": "adult"})
                continue

            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            y_seg = y[start_sample:end_sample]

            if len(y_seg) == 0:
                enriched.append({**seg, "emotion": "neutral", "gender": "male", "age_group": "adult"})
                continue

            # Extract features
            pitch = _extract_pitch(y_seg, sr)
            energy = _extract_energy(y_seg)
            rate = _estimate_speech_rate(text, duration)

            gender = _classify_gender(pitch, text)
            age_group = _classify_age(pitch, rate)
            emotion = _classify_emotion(pitch, energy, rate, gender)
            intensity = _calculate_intensity(energy)
            
            print(f"DEBUG: Final Decision -> Gender: {gender}, Age: {age_group}, Emotion: {emotion}, Intensity: {intensity}\n")
            enriched.append({
                **seg, 
                "emotion": emotion, 
                "gender": gender, 
                "age_group": age_group,
                "intensity": intensity
            })

        except Exception as exc:
            logger.warning("Analysis failed for segment %s: %s", seg, exc)
            enriched.append({**seg, "emotion": "neutral", "gender": "male", "age_group": "adult"})

    print("--- Analysis Complete ---\n")
    clear_audio_cache()
    return enriched


def detect_emotion(audio_path: str) -> str:
    """
    Backward-compatible single-emotion detector.

    Analyses the *entire* file and returns one emotion label.
    Used by ``manager.py`` (if called) without any interface change.

    Parameters
    ----------
    audio_path : str
        Path to an audio file.

    Returns
    -------
    str
        One of: neutral, happy, sad, angry, fear, surprise.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return "neutral"

    y, sr = _load_audio(audio_path)
    if y is None:
        return "neutral"

    try:
        duration = len(y) / sr if sr else 0.0
        if duration < 0.1:
            return "neutral"

        pitch = _extract_pitch(y, sr)
        energy = _extract_energy(y)
        # No text available — assume average rate
        rate = 2.5
        emotion = _classify_emotion(pitch, energy, rate)

        clear_audio_cache()
        return emotion

    except Exception as exc:
        logger.error("Emotion detection failed: %s", exc)
        clear_audio_cache()
        return "neutral"