import re
import string
import pandas as pd


def clean_text_for_linguistic(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_stylistic_features(texts):
    features = []

    for text in texts:
        text = str(text)
        words = text.split()
        sentences = re.split(r"[.!?]+", text)

        word_count = len(words)
        char_count = len(text)
        sentence_count = max(1, len([s for s in sentences if s.strip()]))

        exclamation_count = text.count("!")
        question_count = text.count("?")
        uppercase_count = sum(1 for c in text if c.isupper())
        punctuation_count = sum(1 for c in text if c in string.punctuation)

        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        uppercase_ratio = uppercase_count / char_count if char_count > 0 else 0
        punctuation_ratio = punctuation_count / char_count if char_count > 0 else 0

        features.append({
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "uppercase_ratio": uppercase_ratio,
            "punctuation_ratio": punctuation_ratio,
        })

    return pd.DataFrame(features)