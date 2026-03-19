import os
import sys
import torch

# --- Director Rule 14: Comprehensive Compatibility Polyfill ---
try:
    import transformers.pytorch_utils
    if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
        # Real signature: isin_mps_friendly(elements, test_elements)
        # Delegates to torch.isin which has the same semantics
        import torch as _torch
        def _isin_mps_friendly(elements, test_elements):
            return _torch.isin(elements, test_elements)
        transformers.pytorch_utils.isin_mps_friendly = _isin_mps_friendly
    
    import transformers
    if not hasattr(transformers, "GPT2PreTrainedModel"):
        try:
            from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
            transformers.GPT2PreTrainedModel = GPT2PreTrainedModel
        except ImportError:
            pass

    import transformers.utils
    import transformers.utils.import_utils
    import torch
    from packaging import version
    v = version.parse(torch.__version__.split('+')[0])
    check_fn_12 = lambda: v >= version.parse("1.12")
    check_fn_13 = lambda: v >= version.parse("1.13")
    
    # Inject missing version checks directly into the module dictionary
    for module in [transformers.utils, transformers.utils.import_utils]:
        if not hasattr(module, "is_torch_greater_than_1_12"):
            setattr(module, "is_torch_greater_than_1_12", check_fn_12)
        if not hasattr(module, "is_torch_greater_than_1_13"):
            setattr(module, "is_torch_greater_than_1_13", check_fn_13)
except Exception:
    pass

import uuid
import logging
import hashlib
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Prevent XTTS from blocking for manual TOS agreement
os.environ["COQUI_TOS_AGREED"] = "1"

# --- Director Rule 14: Version Check Polyfill ---
# Fix for "ImportError: cannot import name 'is_torch_greater_than_1_12' from 'transformers.utils'"
try:
    import transformers.utils
    import torch
    from packaging import version
    if not hasattr(transformers.utils, "is_torch_greater_than_1_12"):
        v = version.parse(torch.__version__.split('+')[0])
        transformers.utils.is_torch_greater_than_1_12 = v >= version.parse("1.12")
    if not hasattr(transformers.utils, "is_torch_greater_than_1_13"):
        v = version.parse(torch.__version__.split('+')[0])
        transformers.utils.is_torch_greater_than_1_13 = v >= version.parse("1.13")
except ImportError:
    pass

logger = logging.getLogger(__name__)

class ClonerEngine:
    _instance = None
    _model = None
    _latent_cache = {} # Cache for conditioning latents: {ref_hash: (gpt_cond_latent, speaker_embedding)}
    _executor = ThreadPoolExecutor(max_workers=1)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClonerEngine, cls).__new__(cls)
        return cls._instance

    def warm_up(self):
        """Forces model loading and device movement."""
        print("Director: Warming up Cloner Engine...")
        return self._get_model() is not None

    def _get_model(self):
        if self._model is None:
            try:
                from TTS.api import TTS
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    print("Director WARNING: CUDA NOT FOUND. Running cloner on CPU will be SIGNIFICANTLY slower (approx 20-30x slow).")
                print(f"Loading XTTS v2 on {device}...")
                
                # Optimized loading
                self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
                self._model.to(device)
                
                # If on GPU, we can use FP16 for ~2x speedup
                if device == "cuda":
                    # Note: XTTS v2 from TTS.api doesn't always expose internal precision easily,
                    # but the model itself is often loaded in fp16 by default if supported.
                    pass 
                    
            except Exception as e:
                logger.error(f"Failed to load cloning model: {e}")
                return None
        return self._model

    def _get_file_hash(self, file_path):
        """Generates a hash for the reference audio to use as a cache key."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    async def clone_voice_async(self, text, reference_audio, language="en"):
        """Async wrapper for clone_voice."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            self.clone_voice, 
            text, 
            reference_audio, 
            language
        )

    def clone_voice(self, text, reference_audio, language="en"):
        """
        Synthesizes text using the speaker identity from the reference_audio.
        Uses advanced caching for speaker latents to maximize performance.
        """
        model = self._get_model()
        if not model:
            return None

        output_path = f"uploads/clone_{uuid.uuid4().hex}.wav"
        
        try:
            # --- Director Optimization: Latent Caching ---
            ref_hash = self._get_file_hash(reference_audio)
            
            if ref_hash not in self._latent_cache:
                print(f"Director: Computing speaker latents for new reference {ref_hash[:8]}...")
                gpt_cond_latent, speaker_embedding = model.synthesizer.tts_model.get_conditioning_latents(
                    audio_path=[reference_audio]
                )
                self._latent_cache[ref_hash] = (gpt_cond_latent, speaker_embedding)
            else:
                gpt_cond_latent, speaker_embedding = self._latent_cache[ref_hash]

            # Use the low-level model.inference directly, which only accepts latents
            # (not speaker_wav). tts_to_file derives them from speaker_wav itself, so
            # it cannot accept pre-computed latents — we bypass it here.
            out = model.synthesizer.tts_model.inference(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
            )
            import soundfile as sf
            sf.write(output_path, out["wav"], 24000)
            return output_path

        except Exception as e:
            logger.error(f"Cloning synthesis failed: {e}")
            # Fallback: use the simple speaker_wav path via high-level API
            try:
                model.tts_to_file(
                    text=text,
                    speaker_wav=reference_audio,
                    language=language,
                    file_path=output_path
                )
                return output_path
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                return None
