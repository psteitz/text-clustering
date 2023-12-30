"""
Text processing utilities, using NLTK.
"""
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')


def get_sentences(text: str) -> list:
    """Split text into sentences and return list of unique sentences."""

    # split text into sentences
    sentences = sent_tokenize(text)

    # remove duplicates
    unique_sentences = list(set(sentences))

    # remove empty strings
    # sentences = list(filter(None, sentences))
    return list(filter(None, unique_sentences))


def word_count(sentence: str) -> int:
    """
    Length of token list returned by NLTK word tokenizer run on sentence.
    If sentence ends with a punctuation mark, subtract 1.
    """
    # If sentence is empty, return 0
    if sentence == "":
        return 0
    # If the sentence ends with a puctuation mark, subtract 1 from the word count
    last_char = sentence[-1]
    ret = len(word_tokenize(sentence))
    if last_char in [".", "?", "!"]:
        return ret - 1  # subtract 1 for punctuation mark.
    else:
        return ret


def get_all_sentences(texts: list) -> list:
    """
    Split combined texts into sentences and return list of unique sentences (no duplicates).
    """
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
    """
    Return True if text contains sentence.
    Run NLTK sentence tokenizer on text and check if sentence is in the list returned by the tokenizer.
    """
    return sentence in sent_tokenize(text)
