import re
from collections import Counter


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_vocab(texts, max_vocab_size=10000):
    counter = Counter()

    for text in texts:
        counter.update(text.split())

    most_common = counter.most_common(max_vocab_size - 2)

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx

    return vocab


def text_to_sequence(text, vocab):
    return [vocab.get(word, vocab["<UNK>"]) for word in text.split()]


def pad_sequence(sequence, max_len):
    if len(sequence) < max_len:
        sequence = sequence + [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence