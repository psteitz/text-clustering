"""
Text processing utilities, using NLTK.
"""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')


def get_sentences(text: str) -> list:
    """Split text into sentences and return list of unique sentences (no duplicates)."""
    # split text into sentences
    sentences = sent_tokenize(text)
    # remove duplicates
    sentences = list(set(sentences))
    # remove empty strings
    sentences = list(filter(None, sentences))
    return sentences


def word_count(sentence: str) -> int:
    """Count words in sentence."""
    return len(word_tokenize(sentence)) - 1  # subtract 1 for period


def get_all_sentences(texts: list) -> list:
    """Split combined texts into sentences and return list of unique sentences (no duplicates)."""
    # split texts into sentences
    sentences = []
    for text in texts:
        sentences += sent_tokenize(text)
    # remove duplicates
    sentences = list(set(sentences))
    # remove empty strings
    sentences = list(filter(None, sentences))
    return sentences


def contains_sentence(text: str, sentence: str) -> bool:
    """Return True if text contains sentence."""
    return sentence in sent_tokenize(text)
