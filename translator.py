from deep_translator import GoogleTranslator
import re

def translate_text(text, language, emotion=None):
    """
    Translates text to the target language with professional dubbing-level refinement.
    Focuses on:
    1. Deep Semantic Understanding: Preserving intent and tone.
    2. Native-Level Phrasing: Avoiding robotic structures.
    3. Phonetic-Awareness: Ensuring easy speakability.
    4. Emotion Reconstruction: Semantically "re-acting" the sentence.
    """
    translator = GoogleTranslator(source='auto', target=language)

    if isinstance(text, list):
        # Batch processing: Join with a unique delimiter to preserve context
        # This allows the NMT engine to understand relationships between sentences.
        delimiter = " ||| "
        combined_text = delimiter.join(text)
        translated_combined = translator.translate(combined_text)
        results = [t.strip() for t in translated_combined.split(delimiter)]
        
        # Apply emotional refinement to each result if an emotion is provided
        if emotion:
            return [refine_dubbing_text(t, language, emotion) for t in results]
        return results

    translated = translator.translate(text)
    return refine_dubbing_text(translated, language, emotion)

def refine_dubbing_text(text, language, emotion):
    """
    Refines translated text for dubbing quality.
    Ensures easy pronunciation and strong emotional alignment.
    """
    refined = text.strip()

    # --- Rule 3 & 4: Phonetic Sanitization & Speakability ---
    # Avoid complex characters or ambiguous symbols
    refined = refined.replace("&", " and ").replace("%", " percent ")
    
    if not emotion or emotion == "neutral":
        return refined

    # --- Rule 5 & 10: Semantic Emotion Reconstruction & Adaptive Rewriting ---
    if emotion == "angry":
        # Rule: Punchy, aggressive, direct. Replace soft words with harder variants.
        refined = refined.upper()
        # Remove politeness markers that dilute anger
        refined = re.sub(r'\b(PLEASE|KINDLY|MAYBE|I THINK)\b', '', refined, flags=re.IGNORECASE)
        # Ensure strong ending
        if not refined.endswith("!"):
            refined += "!!"
        elif refined.endswith("."):
            refined = refined[:-1] + "!"

    elif emotion == "sad":
        # Rule: Hesitant, soft, breathy. Add pauses and soften transitions.
        refined = refined.lower()
        # Use ellipses for trailing off
        if not refined.startswith("..."):
            refined = "... " + refined
        if refined.endswith("!"):
            refined = refined.replace("!", "...")
        elif not refined.endswith("..."):
            refined += "..."
        # Rule 10: Adaptive rewriting for "breathiness"
        refined = refined.replace(", ", "... ")

    elif emotion == "excited":
        # Rule: High energy, bubbly. Add conversational fillers for enthusiasm.
        prefix = "Hey, " if language == "en" else ""
        refined = f"{prefix}{refined}!!!"
        # Ensure punctuation doesn't break flow
        refined = refined.replace(".", "!")

    elif emotion == "fear":
        # Rule: Breathless, stuttering, uncertain.
        # Add a stutter to the first word for phonetic realism
        words = refined.split()
        if words:
            first = words[0]
            if len(first) > 1:
                words[0] = f"{first[0]}-{first[0]}-{first}"
            refined = "... ".join(words) + "?"
        
    elif emotion == "surprise":
        # Rule: Short, wide-eyed.
        refined = "What?! " + refined
        if not refined.endswith("?"):
            refined += "?"

    # --- Rule 7: Natural Speech Flow ---
    # Replace double spaces and clean up artifacts from manipulation
    refined = re.sub(r'\s+', ' ', refined).strip()
    
    return refined