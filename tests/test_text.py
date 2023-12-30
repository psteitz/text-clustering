import context
from data import text_utils
# read text_texts.txt and split each line into sentences
with open("test_texts.txt", "r", encoding="UTF-8") as f:
    text = f.read()
    lines = text.splitlines()
    num_sentences = [2, 2, 5]
    for i, line in enumerate(lines):
        sentences = text_utils.get_sentences(lines[i])
        assert len(sentences) == num_sentences[i]
        if i == 0:
            assert "I drink your milkshake." in sentences
            assert "I drink it all up." in sentences
    print("Test splitting text into sentences passed.")

    combined_sentences = text_utils.get_all_sentences(lines)
    assert len(combined_sentences) == 8
    assert "No!" in combined_sentences
    print("Test splitting list of texts passed.")

    assert text_utils.word_count("I drink your milkshake.") == 4
    assert text_utils.word_count("I drink it all up.") == 5
    assert text_utils.word_count("No!") == 1
    assert text_utils.word_count("") == 0
    assert text_utils.word_count(":)") == 2
    assert text_utils.word_count("-") == 1
    print("Test counting words in sentence passed.")

# Test duplicate handling
text = "I drink your milkshake. I drink it all up. I drink your milkshake."
sentences = text_utils.get_sentences(text)
assert len(sentences) == 2
assert "I drink your milkshake." in sentences
assert "I drink it all up." in sentences

# Test contains_sentence
text = "I drink your milkshake. I drink it all up."
assert text_utils.contains_sentence(text, "I drink your milkshake.")
assert text_utils.contains_sentence(text, "I drink it all up.")
assert not text_utils.contains_sentence(
    text, "I drink your milkshake. I drink it all up.")
assert not text_utils.contains_sentence(text, "")

# Test empty string handling
text = ""
sentences = text_utils.get_sentences(text)
assert len(sentences) == 0
assert sentences == []
assert text_utils.word_count(text) == 0
assert not text_utils.contains_sentence(text, "")
