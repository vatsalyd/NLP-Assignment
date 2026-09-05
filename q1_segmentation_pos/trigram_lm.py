from collections import defaultdict
import math

class TrigramLanguageModel:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        self.total_unigrams = 0
        self.total_bigrams = 0
        self.total_trigrams = 0
        
    def train(self, words):
        for i, word in enumerate(words):
            self.unigram_counts[word] += 1
            self.total_unigrams += 1
            
            if i > 0:
                bigram = (words[i-1], word)
                self.bigram_counts[bigram] += 1
                self.total_bigrams += 1
                
            if i > 1:
                trigram = (words[i-2], words[i-1], word)
                self.trigram_counts[trigram] += 1
                self.total_trigrams += 1
    
    def unigram_prob(self, word, k=1.0):
        return (self.unigram_counts[word] + k) / (self.total_unigrams + k * len(self.vocab))
    
    def bigram_prob(self, prev_word, word, k=1.0):
        bigram = (prev_word, word)
        return (self.bigram_counts[bigram] + k) / (self.unigram_counts[prev_word] + k * len(self.vocab))
    
    def trigram_prob(self, prev2_word, prev_word, word, k=1.0):
        trigram = (prev2_word, prev_word, word)
        bigram = (prev2_word, prev_word)
        return (self.trigram_counts[trigram] + k) / (self.bigram_counts[bigram] + k * len(self.vocab))
    
    def sentence_log_prob(self, words, k=1.0):
        if not words:
            return float('-inf')
        log_prob = 0.0
        for i, word in enumerate(words):
            if i == 0:
                log_prob += math.log(self.unigram_prob(word, k))
            elif i == 1:
                log_prob += math.log(self.bigram_prob(words[i-1], word, k))
            else:
                log_prob += math.log(self.trigram_prob(words[i-2], words[i-1], word, k))
        return log_prob

def viterbi_segmentation(text, vocab, lm, max_word_len=20):
    n = len(text)
    dp = [float('-inf')] * (n + 1)
    backtrack = [None] * (n + 1)
    dp[0] = 0.0
    
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            word = text[j:i]
            if word in vocab:
                if j == 0:
                    prob = lm.unigram_prob(word)
                    if prob > 0 and dp[j] + math.log(prob) > dp[i]:
                        dp[i] = dp[j] + math.log(prob)
                        backtrack[i] = (j, word, '<START>', '<START>')
                elif j == 1:
                    first_word = text[0:1] if text[0:1] in vocab else text[0:j]
                    if first_word in vocab:
                        prob = lm.bigram_prob(first_word, word)
                        if prob > 0 and dp[j] + math.log(prob) > dp[i]:
                            dp[i] = dp[j] + math.log(prob)
                            backtrack[i] = (j, word, '<START>', first_word)
                else:
                    if backtrack[j] is not None:
                        prev_j, prev_word, prev2_tag, prev_tag = backtrack[j]
                        if prev_j == 0:
                            prob = lm.bigram_prob(prev_word, word)
                        else:
                            if backtrack[prev_j] is not None:
                                prev2_j, prev2_word, _, _ = backtrack[prev_j]
                                prob = lm.trigram_prob(prev2_word, prev_word, word)
                            else:
                                prob = lm.bigram_prob(prev_word, word)
                        if prob > 0 and dp[j] + math.log(prob) > dp[i]:
                            dp[i] = dp[j] + math.log(prob)
                            backtrack[i] = (j, word, prev_word, prev2_word if prev_j > 0 else '<START>')
    
    if dp[n] == float('-inf'):
        return [text]
    
    words = []
    i = n
    while i > 0:
        if backtrack[i] is None:
            words.append(text[:i])
            break
        j, word, _, _ = backtrack[i]
        words.append(word)
        i = j
    words.reverse()
    return words