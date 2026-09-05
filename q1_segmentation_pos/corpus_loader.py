import nltk
from nltk.corpus import brown
from collections import defaultdict
import random

def load_brown_corpus(split_ratio=0.8):
    tagged_sents = list(brown.tagged_sents(tagset='universal'))
    random.seed(42)
    random.shuffle(tagged_sents)
    split_idx = int(len(tagged_sents) * split_ratio)
    train_sents = tagged_sents[:split_idx]
    test_sents = tagged_sents[split_idx:]
    return train_sents, test_sents

def load_ud_spanish(train_path, dev_path, test_path):
    def parse_conllu(filepath):
        sentences = []
        with open(filepath, 'r', encoding='utf-8') as f:
            sent = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    if sent:
                        sentences.append(sent)
                        sent = []
                    continue
                parts = line.split('\t')
                if len(parts) >= 4 and parts[0].isdigit():
                    word = parts[1]
                    upos = parts[3]
                    sent.append((word, upos))
            if sent:
                sentences.append(sent)
        return sentences
    
    train_sents = parse_conllu(train_path)
    dev_sents = parse_conllu(dev_path)
    test_sents = parse_conllu(test_path)
    return train_sents, dev_sents, test_sents

def extract_words_tags(sentences):
    words = []
    tags = []
    for sent in sentences:
        for word, tag in sent:
            words.append(word.lower())
            tags.append(tag)
    return words, tags

def build_vocabulary(words):
    vocab = set(words)
    return vocab