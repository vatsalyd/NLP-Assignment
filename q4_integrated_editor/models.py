import nltk
from nltk.corpus import brown, treebank, gutenberg, reuters
from nltk import induce_pcfg, Nonterminal, Tree, ProbabilisticProduction
from collections import defaultdict, Counter
import math
import random
import sys

sys.path.append('D:\\projects\\NLP-Assignment\\q1_segmentation_pos')
from corpus_loader import load_brown_corpus, extract_words_tags, build_vocabulary
from pos_tagger import POSTagger

def load_shared_models():
    train_sents, test_sents = load_brown_corpus(0.8)
    train_words, train_tags = extract_words_tags(train_sents)
    
    vocab = build_vocabulary(train_words)
    unigram_counts = Counter(train_words)
    total_words = len(train_words)
    unigram_probs = {w: c/total_words for w, c in unigram_counts.items()}
    
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    
    for i in range(len(train_words)-1):
        bigram_counts[train_words[i]][train_words[i+1]] += 1
    for i in range(len(train_words)-2):
        trigram_counts[(train_words[i], train_words[i+1])][train_words[i+2]] += 1
    
    bigram_probs = {}
    for w1, counter in bigram_counts.items():
        total = sum(counter.values())
        bigram_probs[w1] = {w2: c/total for w2, c in counter.items()}
    
    trigram_probs = {}
    for w1w2, counter in trigram_counts.items():
        total = sum(counter.values())
        trigram_probs[w1w2] = {w3: c/total for w3, c in counter.items()}
    
    tags = sorted(set(train_tags))
    pos_tagger = POSTagger(tags)
    pos_tagger.train(train_sents)
    
    return {
        'vocab': vocab,
        'unigram_counts': unigram_counts,
        'unigram_probs': unigram_probs,
        'bigram_probs': bigram_probs,
        'trigram_probs': trigram_probs,
        'words': train_words,
        'pos_tagger': pos_tagger,
        'train_sents': train_sents
    }

def train_pcfg():
    """Train a simple PCFG from treebank tagged sentences."""
    productions = []
    
    # Collect POS tag sequences from treebank
    tag_sequences = []
    for sent in treebank.tagged_sents():
        tags = [tag for _, tag in sent]
        tag_sequences.append(tags)
    
    # Count POS n-grams for rule probabilities
    pos_unigrams = Counter()
    pos_bigrams = Counter()
    pos_trigrams = Counter()
    
    for tags in tag_sequences:
        for tag in tags:
            pos_unigrams[tag] += 1
        for i in range(len(tags)-1):
            pos_bigrams[(tags[i], tags[i+1])] += 1
        for i in range(len(tags)-2):
            pos_trigrams[(tags[i], tags[i+1], tags[i+2])] += 1
    
    # Add lexical rules: POS -> word (top 30 words per POS)
    word_pos_counts = defaultdict(Counter)
    for sent in treebank.tagged_sents():
        for word, tag in sent:
            word_pos_counts[tag][word.lower()] += 1
    
    for tag, counter in word_pos_counts.items():
        total = sum(counter.values())
        for word, count in counter.most_common(30):
            prob = count / total
            productions.append(ProbabilisticProduction(Nonterminal(tag), [word], prob=prob))
    
    # Add POS sequence rules using n-grams
    # S -> POS
    for tag, count in pos_unigrams.items():
        prob = count / sum(pos_unigrams.values())
        productions.append(ProbabilisticProduction(Nonterminal('S'), [Nonterminal(tag)], prob=prob * 0.1))
    
    # S -> POS POS
    for (t1, t2), count in pos_bigrams.items():
        prob = count / sum(pos_bigrams.values())
        productions.append(ProbabilisticProduction(Nonterminal('S'), [Nonterminal(t1), Nonterminal(t2)], prob=prob * 0.3))
    
    # S -> POS POS POS
    for (t1, t2, t3), count in pos_trigrams.items():
        prob = count / sum(pos_trigrams.values())
        productions.append(ProbabilisticProduction(Nonterminal('S'), [Nonterminal(t1), Nonterminal(t2), Nonterminal(t3)], prob=prob * 0.6))
    
    # S -> S S (recursive rule for longer sentences)
    productions.append(ProbabilisticProduction(Nonterminal('S'), [Nonterminal('S'), Nonterminal('S')], prob=0.5))
    
    # Add punctuation rules
    for tag in ['.', ',', ':', ';', '``', "''", '-LRB-', '-RRB-']:
        productions.append(ProbabilisticProduction(Nonterminal(tag), [tag], prob=1.0))
    
    start = Nonterminal('S')
    pcfg = induce_pcfg(start, productions)
    return pcfg

# Tagset reconciliation: Brown universal tags -> Penn Treebank tags
UNIVERSAL_TO_PTB = {
    'NOUN': 'NN', 'VERB': 'VB', 'ADJ': 'JJ', 'ADV': 'RB',
    'DET': 'DT', 'ADP': 'IN', 'PRON': 'PRP', 'CONJ': 'CC',
    'NUM': 'CD', 'PRT': 'RP', 'X': 'NN', '.': '.',
}

def universal_to_ptb(universal_tag):
    return UNIVERSAL_TO_PTB.get(universal_tag, 'NN')

class SmoothedNGramModel:
    def __init__(self, unigram_probs, bigram_probs, trigram_probs, vocab, k=0.1):
        self.unigram_probs = unigram_probs
        self.bigram_probs = bigram_probs
        self.trigram_probs = trigram_probs
        self.vocab = vocab
        self.k = k
        self.vocab_size = len(vocab)
    
    def bigram_prob(self, w1, w2):
        if w1 in self.bigram_probs and w2 in self.bigram_probs[w1]:
            return self.bigram_probs[w1][w2]
        return self.k / (self.k * self.vocab_size)
    
    def trigram_prob(self, w1, w2, w3):
        if (w1, w2) in self.trigram_probs and w3 in self.trigram_probs[(w1, w2)]:
            return self.trigram_probs[(w1, w2)][w3]
        return self.k / (self.k * self.vocab_size)
    
    def sentence_log_prob_bigram(self, words):
        if not words:
            return float('-inf')
        log_prob = math.log(self.unigram_probs.get(words[0], 1e-10))
        for i in range(1, len(words)):
            log_prob += math.log(self.bigram_prob(words[i-1], words[i]))
        return log_prob
    
    def sentence_log_prob_trigram(self, words):
        if not words:
            return float('-inf')
        if len(words) == 1:
            return math.log(self.unigram_probs.get(words[0], 1e-10))
        log_prob = math.log(self.unigram_probs.get(words[0], 1e-10))
        log_prob += math.log(self.bigram_prob(words[0], words[1]))
        for i in range(2, len(words)):
            log_prob += math.log(self.trigram_prob(words[i-2], words[i-1], words[i]))
        return log_prob
    
    def perplexity(self, words, n=3):
        if n == 2:
            log_prob = self.sentence_log_prob_bigram(words)
        else:
            log_prob = self.sentence_log_prob_trigram(words)
        return math.exp(-log_prob / len(words)) if words else float('inf')

def cky_parse(pcfg, tokens):
    n = len(tokens)
    if n == 0:
        return None
    
    table = [[{} for _ in range(n)] for _ in range(n)]
    
    for i, token in enumerate(tokens):
        for prod in pcfg.productions(rhs=token):
            table[i][i][prod.lhs()] = (prod.prob(), (prod,))
    
    for length in range(2, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for B, (prob_B, parse_B) in table[i][k].items():
                    for C, (prob_C, parse_C) in table[k+1][j].items():
                        for prod in pcfg.productions(rhs=(B, C)):
                            prob = prod.prob() * prob_B * prob_C
                            if prod.lhs() not in table[i][j] or prob > table[i][j][prod.lhs()][0]:
                                table[i][j][prod.lhs()] = (prob, (prod, parse_B, parse_C))
    
    start_symbol = pcfg.start()
    if start_symbol in table[0][n-1]:
        return table[0][n-1][start_symbol]
    return None

def pos_tag_and_parse(pcfg, pos_tagger, words):
    if not words:
        return None, []
    
    universal_tags = pos_tagger.viterbi_decode(words)
    ptb_tags = [universal_to_ptb(t) for t in universal_tags]
    
    result = cky_parse(pcfg, ptb_tags)
    return result, ptb_tags

def bigram_perplexity(words, bigram_probs, unigram_probs, k=0.1):
    if not words:
        return float('inf')
    vocab_size = len(unigram_probs)
    log_prob = math.log(unigram_probs.get(words[0], k/(k*vocab_size)))
    for i in range(1, len(words)):
        w1, w2 = words[i-1], words[i]
        prob = bigram_probs.get(w1, {}).get(w2, k/(k*vocab_size))
        log_prob += math.log(prob)
    return math.exp(-log_prob / len(words))

def trigram_perplexity(words, trigram_probs, bigram_probs, unigram_probs, k=0.1):
    if not words:
        return float('inf')
    vocab_size = len(unigram_probs)
    if len(words) == 1:
        return math.exp(-math.log(unigram_probs.get(words[0], k/(k*vocab_size))))
    log_prob = math.log(unigram_probs.get(words[0], k/(k*vocab_size)))
    log_prob += math.log(bigram_probs.get(words[0], {}).get(words[1], k/(k*vocab_size)))
    for i in range(2, len(words)):
        prob = trigram_probs.get((words[i-2], words[i-1]), {}).get(words[i], k/(k*vocab_size))
        log_prob += math.log(prob)
    return math.exp(-log_prob / len(words))

def viterbi_segmentation(text, vocab, trigram_probs, unigram_probs, max_word_len=20):
    n = len(text)
    dp = [float('-inf')] * (n + 1)
    backtrack = [None] * (n + 1)
    dp[0] = 0.0
    
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            word = text[j:i]
            if word in vocab:
                if j == 0:
                    prob = unigram_probs.get(word, 1e-10)
                    if dp[j] + math.log(prob) > dp[i]:
                        dp[i] = dp[j] + math.log(prob)
                        backtrack[i] = (j, word)
                elif j == 1:
                    first = text[0:1]
                    prob = trigram_probs.get((first,), {}).get(word, 1e-10)
                    if dp[j] + math.log(prob) > dp[i]:
                        dp[i] = dp[j] + math.log(prob)
                        backtrack[i] = (j, word)
                else:
                    if backtrack[j]:
                        prev_j, prev_word = backtrack[j]
                        if prev_j == 0:
                            prob = trigram_probs.get((prev_word,), {}).get(word, 1e-10)
                        else:
                            prev2_j, prev2_word = backtrack[prev_j] if backtrack[prev_j] else (0, '')
                            prob = trigram_probs.get((prev2_word, prev_word), {}).get(word, 1e-10)
                        if dp[j] + math.log(prob) > dp[i]:
                            dp[i] = dp[j] + math.log(prob)
                            backtrack[i] = (j, word)
    
    if dp[n] == float('-inf'):
        return [text]
    
    words = []
    i = n
    while i > 0:
        if backtrack[i] is None:
            words.append(text[:i])
            break
        j, word = backtrack[i]
        words.append(word)
        i = j
    words.reverse()
    return words

def get_random_passage():
    corpora = [gutenberg, reuters, brown]
    corpus = random.choice(corpora)
    files = corpus.fileids()
    file = random.choice(files)
    sents = corpus.sents(file)
    if len(sents) < 8:
        return get_random_passage()
    start = random.randint(0, len(sents) - 8)
    passage_sents = sents[start:start + random.randint(5, 8)]
    return ' '.join(' '.join(s) for s in passage_sents)

def introduce_merges(tokens, p=0.08):
    result = []
    i = 0
    while i < len(tokens):
        result.append(tokens[i])
        if i < len(tokens) - 1 and random.random() < p:
            result[-1] = result[-1] + tokens[i+1]
            i += 2
        else:
            i += 1
    return result

if __name__ == "__main__":
    print("Training shared models...")
    models = load_shared_models()
    print(f"Vocab size: {len(models['vocab'])}")
    
    print("Training PCFG...")
    pcfg = train_pcfg()
    print(f"PCFG productions: {len(pcfg.productions())}")
    
    # Test
    words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    result, tags = pos_tag_and_parse(pcfg, models['pos_tagger'], words)
    print(f"Words: {words}")
    print(f"Universal tags: {tags}")
    print(f"PTB tags: {[universal_to_ptb(t) for t in tags]}")
    print(f"PCFG parse: {'Success' if result else 'Failed'}")
    if result:
        print(f"PCFG log prob: {math.log(result[0])}")
    
    print("Done!")