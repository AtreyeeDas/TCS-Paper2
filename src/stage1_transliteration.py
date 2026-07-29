import re
from indic_transliteration import sanscript

class TransliterationEngine:
    def __init__(self):
        print("Initialized Phonetic Transliteration Engine (Devanagari -> ITRANS)")

    def process_text(self, text: str):
        """
        Takes raw Hinglish (Latin + Devanagari) and normalizes it to Roman script (ITRANS).
        Returns the clean transliterated text and the approximate word indices where language switches occur.
        """
        words = text.split()
        boundaries = []
        is_hindi = False
        
        # 1. Boundary Detection
        for i, word in enumerate(words):
            # Check if word contains Devanagari unicode characters
            if re.search(r'[\u0900-\u097F]', word):
                if not is_hindi:
                    boundaries.append(i) # Switch from English to Hindi detected
                is_hindi = True
            else:
                if is_hindi:
                    boundaries.append(i) # Switch from Hindi to English detected
                is_hindi = False
                
        # 2. Physical Transliteration
        # Convert the entire string. indic-transliteration safely ignores Latin English words
        # and only converts the Devanagari parts into readable Roman text (ITRANS format).
        romanized_text = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        
        # 3. Cleanup: ITRANS sometimes leaves artifact capitalization. We lower it for the TTS tokenizer.
        clean_romanized_text = romanized_text.lower()
        
        return clean_romanized_text, boundaries

# --- Quick Test Block (You can run this file directly to verify it works) ---
if __name__ == "__main__":
    engine = TransliterationEngine()
    test_sentence = "Doctor साहब me threshold values check kar raha hoon"
    clean_text, bounds = engine.process_text(test_sentence)
    print(f"Original: {test_sentence}")
    print(f"Transliterated: {clean_text}")
    print(f"Code-Switch Boundaries at word indices: {bounds}")
