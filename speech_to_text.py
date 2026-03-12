"""
Speech-to-Text Module — Whisper-based transcription with segmented output.

Provides:
  transcribe_audio(audio_path)          -> str   (backward-compatible, plain text)
  transcribe_audio_segments(audio_path) -> list   (segmented [{text, start, end}])
"""

import os
import logging
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded, cached Whisper model (singleton)
# ---------------------------------------------------------------------------
_whisper_model = None


def _get_model(model_name: str = "base"):
    """Load the Whisper model once and cache it for all future calls."""
    global _whisper_model
    if _whisper_model is None:
        import whisper

        logger.info("Loading Whisper '%s' model (first call — may take a moment)…", model_name)
        _whisper_model = whisper.load_model(model_name)
        logger.info("Whisper model loaded successfully.")
    return _whisper_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transcribe_audio_segments(audio_path: str, language: str = None) -> list:
    """
    Transcribe an audio file and return time-stamped segments.

    Parameters
    ----------
    audio_path : str
        Path to a WAV / MP3 / FLAC / etc. audio file.
    language : str, optional
        ISO-639-1 language code (e.g. "en", "ta"). ``None`` = auto-detect.

    Returns
    -------
    list[dict]
        Each dict has keys ``text`` (str), ``start`` (float), ``end`` (float).
        Returns an empty list on error.

    Example
    -------
    >>> transcribe_audio_segments("uploads/input.wav")
    [{"text": "Hello how are you", "start": 0.2, "end": 2.8}, ...]
    """
    # --- Guard: file exists? ---
    if not audio_path or not os.path.isfile(audio_path):
        logger.warning("Audio file not found: %s", audio_path)
        return []

    # --- Guard: file not empty / too small ---
    file_size = os.path.getsize(audio_path)
    if file_size < 1024:  # less than 1 KB is almost certainly silence / corrupt
        logger.warning("Audio file too small (%d bytes): %s", file_size, audio_path)
        return []

    try:
        model = _get_model()

        # Build transcribe kwargs
        transcribe_kwargs = {
            "word_timestamps": True,   # enables per-word timing
            "verbose": False,          # suppress Whisper's own prints
        }
        if language:
            transcribe_kwargs["language"] = language

        result = model.transcribe(audio_path, **transcribe_kwargs)

        segments = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if not text:
                continue  # skip empty segments
            segments.append({
                "text": text,
                "start": round(seg.get("start", 0.0), 3),
                "end": round(seg.get("end", 0.0), 3),
            })

        logger.info(
            "Transcribed %d segment(s) from '%s' [detected language: %s]",
            len(segments),
            audio_path,
            result.get("language", "unknown"),
        )
        return segments

    except Exception as exc:
        logger.error("Transcription failed for '%s': %s", audio_path, exc, exc_info=True)
        return []


def transcribe_audio(audio_path: str, language: str = None) -> str:
    """
    Backward-compatible wrapper — returns the full transcription as a single
    string, exactly as the old implementation did.

    Parameters
    ----------
    audio_path : str
        Path to an audio file.
    language : str, optional
        ISO-639-1 language code.  ``None`` = auto-detect.

    Returns
    -------
    str
        Concatenated transcription text (empty string on error).
    """
    segments = transcribe_audio_segments(audio_path, language=language)
    full_text = " ".join(seg["text"] for seg in segments)
    print("Detected text:", full_text)
    return full_text