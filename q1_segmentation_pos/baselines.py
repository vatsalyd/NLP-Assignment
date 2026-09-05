from collections import defaultdict

class GreedyLongestMatchSegmenter:
    def __init__(self, vocab, max_word_len=20):
        self.vocab = vocab
        self.max_word_len = max_word_len
    
    def segment(self, text):
        n = len(text)
        words = []
        i = 0
        while i < n:
            found = False
            for j in range(min(i + self.max_word_len, n), i, -1):
                word = text[i:j]
                if word in self.vocab:
                    words.append(word)
                    i = j
                    found = True
                    break
            if not found:
                words.append(text[i])
                i += 1
        return words

class MostFrequentTagger:
    def __init__(self):
        self.word_tag_counts = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)
        self.default_tag = None
    
    def train(self, tagged_sentences):
        for sent in tagged_sentences:
            for word, tag in sent:
                word = word.lower()
                self.word_tag_counts[word][tag] += 1
                self.tag_counts[tag] += 1
        self.default_tag = max(self.tag_counts, key=self.tag_counts.get) if self.tag_counts else 'NOUN'
    
    def tag(self, words):
        tags = []
        for word in words:
            word = word.lower()
            if word in self.word_tag_counts:
                best_tag = max(self.word_tag_counts[word], key=self.word_tag_counts[word].get)
                tags.append(best_tag)
            else:
                tags.append(self.default_tag)
        return tags