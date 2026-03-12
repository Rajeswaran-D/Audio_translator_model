"""
Emotion Detector Module — acoustic-feature-based emotion classification.

Uses librosa to extract pitch (F0), energy (RMS), and speech-rate features
from audio segments, then maps the feature profile to one of six emotions:
  neutral, happy, sad, angry, fear, surprise

Provides:
  detect_emotion(audio_path)                           -> str   (backward-compatible)
  detect_emotions_for_segments(audio_path, segments)   -> list   (enriched segments)
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


def _classify_emotion(pitch: float, energy: float, rate: float) -> str:
    """
    Map acoustic features to one of six emotions using a rule-based heuristic.

    The rules approximate well-known acoustic correlates of emotion:
      angry   — high energy + high pitch + fast rate
      happy   — high energy + high pitch
      fear    — high pitch  + low energy + fast rate
      surprise— very high pitch + moderate-to-high energy
      sad     — low pitch   + low energy + slow rate
      neutral — everything else
    """
    if energy >= _ENERGY_HIGH and pitch >= _PITCH_HIGH and rate >= _RATE_FAST:
        return "angry"
    if energy >= _ENERGY_HIGH and pitch >= _PITCH_HIGH:
        return "happy"
    if pitch >= _PITCH_HIGH * 1.3 and energy >= _ENERGY_LOW:
        return "surprise"
    if pitch >= _PITCH_HIGH and energy < _ENERGY_LOW and rate >= _RATE_FAST:
        return "fear"
    if pitch < _PITCH_LOW and energy < _ENERGY_LOW and rate <= _RATE_SLOW:
        return "sad"
    if pitch >= _PITCH_HIGH and energy < _ENERGY_HIGH:
        return "fear"
    return "neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_emotions_for_segments(audio_path: str, segments: list) -> list:
    """
    Detect emotion for each speech segment using acoustic features.

    Parameters
    ----------
    audio_path : str
        Path to the original audio file.
    segments : list[dict]
        Each dict must have keys ``text`` (str), ``start`` (float), ``end`` (float).

    Returns
    -------
    list[dict]
        Same dicts with an added ``emotion`` key (str).
        On any per-segment error the emotion defaults to ``"neutral"``.
    """
    if not segments:
        return []

    y, sr = _load_audio(audio_path)
    if y is None:
        # Cannot load audio — return all segments tagged as "neutral"
        return [{**seg, "emotion": "neutral"} for seg in segments]

    enriched = []
    for seg in segments:
        try:
            start_sec = seg.get("start", 0.0)
            end_sec = seg.get("end", 0.0)
            text = seg.get("text", "")
            duration = end_sec - start_sec

            # Guard: segment too short (< 0.1 s)
            if duration < 0.1:
                enriched.append({**seg, "emotion": "neutral"})
                continue

            # Slice the waveform for this segment
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)
            y_seg = y[start_sample:end_sample]

            if len(y_seg) == 0:
                enriched.append({**seg, "emotion": "neutral"})
                continue

            # Extract features
            pitch = _extract_pitch(y_seg, sr)
            energy = _extract_energy(y_seg)
            rate = _estimate_speech_rate(text, duration)

            emotion = _classify_emotion(pitch, energy, rate)
            enriched.append({**seg, "emotion": emotion})

        except Exception as exc:
            logger.warning("Emotion detection failed for segment %s: %s", seg, exc)
            enriched.append({**seg, "emotion": "neutral"})

    # Free cached audio now that we're done
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