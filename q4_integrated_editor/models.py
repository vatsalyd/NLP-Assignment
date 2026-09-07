import nltk

# Download required NLTK data (for Streamlit Cloud deployment)
def _download_nltk_data():
    for resource in ['brown', 'treebank', 'gutenberg', 'reuters', 'universal_tagset', 'punkt']:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

_download_nltk_data()

from nltk.corpus import brown, treebank, gutenberg, reuters
from nltk import induce_pcfg, Nonterminal, Tree, ProbabilisticProduction
from collections import defaultdict, Counter
import math
import random
import sys
import os

# Add project root to path for Q1 imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from q1_segmentation_pos.corpus_loader import load_brown_corpus, extract_words_tags, build_vocabulary
from q1_segmentation_pos.pos_tagger import POSTagger

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

def simplify_nt(nt):
    """Simplify Penn Treebank non-terminal to base category."""
    label = str(nt)
    # Remove functional tags and indices
    if '|' in label:
        label = label.split('|')[0]
    if '-' in label and not label.startswith('-'):
        base = label.split('-')[0]
        # Keep base categories
        if base in ['NP', 'VP', 'PP', 'ADJP', 'ADVP', 'SBAR', 'S', 'PRP', 'PRP$',
                      'DT', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                      'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'IN', 'CC', 'CD', 'DT',
                      'PRP', 'PRP$', 'WP', 'WP$', 'WRB', 'WDT', 'EX', 'FW', 'LS', 'MD',
                      'PDT', 'POS', 'RP', 'SYM', 'TO', 'UH', '``', "''", ',', '.', ':', ';',
                      '-LRB-', '-RRB-', '#', '$', 'POS', 'QP', 'INTJ', 'LST', 'ADVP',
                      'PRT', 'X', 'NAC', 'NX', 'QP', 'RRC', 'UCP', 'WHADVP', 'WHADJP',
                      'WHNP', 'WHPP', 'X', 'XX']:
            return base
    return label


def binarize_productions(productions):
    """Manually binarize productions without creating complex non-terminals."""
    binary_prods = []
    new_nt_counter = 0
    
    for prod in productions:
        lhs = prod.lhs()
        rhs = list(prod.rhs())
        prob = getattr(prod, 'prob', None)
        if prob is None:
            prob = 1.0  # Will be re-estimated by induce_pcfg
        
        if len(rhs) <= 2:
            binary_prods.append(ProbabilisticProduction(lhs, rhs, prob=prob))
        else:
            # Binarize: A -> B C D E  becomes  A -> B X1, X1 -> C X2, X2 -> D E
            current_lhs = lhs
            for i in range(len(rhs) - 2):
                new_nt_name = f"_BIN{new_nt_counter}"
                new_nt_counter += 1
                new_nt = Nonterminal(new_nt_name)
                binary_prods.append(ProbabilisticProduction(current_lhs, [rhs[i], new_nt], prob=prob))
                current_lhs = new_nt
            # Last pair
            binary_prods.append(ProbabilisticProduction(current_lhs, [rhs[-2], rhs[-1]], prob=prob))
    
    return binary_prods


def train_pcfg():
    """Train a PCFG from treebank parsed sentences with simplified non-terminals."""
    raw_productions = []
    
    # Use treebank parsed sentences with full constituency structure
    for tree in treebank.parsed_sents():
        # Simplify non-terminals in the tree
        def simplify_tree(t):
            if isinstance(t, nltk.Tree):
                t.set_label(simplify_nt(t.label()))
                for child in t:
                    simplify_tree(child)
        simplify_tree(tree)
        tree.collapse_unary(collapsePOS=True)
        # Don't use CNF - collect raw productions
        raw_productions.extend(tree.productions())
    
    # Manually binarize
    productions = binarize_productions(raw_productions)
    
    # Add lexical rules: POS -> word
    word_pos_counts = defaultdict(Counter)
    for sent in treebank.tagged_sents():
        for word, tag in sent:
            word_pos_counts[simplify_nt(tag)][word.lower()] += 1
    
    for tag, counter in word_pos_counts.items():
        total = sum(counter.values())
        for word, count in counter.most_common(30):
            prob = count / total
            productions.append(ProbabilisticProduction(Nonterminal(tag), [word], prob=prob))
    
    # Add identity productions for POS tags (allows CKY to use POS tags as terminals)
    pos_tags = ['DT', 'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'IN', 'CC', 'CD', 'PRP', 'PRP$',
                'WP', 'WP$', 'WRB', 'WDT', 'EX', 'FW', 'LS', 'MD', 'PDT', 'POS', 'RP',
                'SYM', 'TO', 'UH', '``', "''", ',', '.', ':', ';', '-LRB-', '-RRB-',
                '#', '$', 'QP', 'INTJ', 'NAC', 'NX', 'RRC', 'UCP', 'WHADVP', 'WHADJP',
                'WHNP', 'WHPP', 'X', 'XX']
    for tag in pos_tags:
        productions.append(ProbabilisticProduction(Nonterminal(tag), [Nonterminal(tag)], prob=1.0))
    
    # Add essential NP rules for pronouns
    productions.append(ProbabilisticProduction(Nonterminal('NP'), [Nonterminal('PRP')], prob=0.1))
    productions.append(ProbabilisticProduction(Nonterminal('NP'), [Nonterminal('PRP$')], prob=0.05))
    productions.append(ProbabilisticProduction(Nonterminal('NP'), [Nonterminal('WP')], prob=0.05))
    productions.append(ProbabilisticProduction(Nonterminal('NP'), [Nonterminal('EX')], prob=0.05))
    
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
    
    # Build index: (B, C) -> list of productions with that RHS
    binary_index = {}
    for prod in pcfg.productions():
        rhs = prod.rhs()
        if len(rhs) == 2 and all(isinstance(x, Nonterminal) for x in rhs):
            key = (rhs[0], rhs[1])
            if key not in binary_index:
                binary_index[key] = []
            binary_index[key].append(prod)
    
    # Also index unary productions for base case: A -> B (where B is Nonterminal)
    unary_index = {}
    for prod in pcfg.productions():
        rhs = prod.rhs()
        if len(rhs) == 1 and isinstance(rhs[0], Nonterminal):
            if rhs[0] not in unary_index:
                unary_index[rhs[0]] = []
            unary_index[rhs[0]].append(prod)
    
    # Convert string tokens to Nonterminal objects for PCFG lookup
    nt_tokens = [Nonterminal(t) for t in tokens]
    
    table = [[{} for _ in range(n)] for _ in range(n)]
    
    # Base case: span=1
    for i, nt_token in enumerate(nt_tokens):
        if nt_token in unary_index:
            for prod in unary_index[nt_token]:
                table[i][i][prod.lhs()] = (prod.prob(), (prod,))
    
    # Recursive case: span >= 2
    for length in range(2, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                for B, (prob_B, parse_B) in table[i][k].items():
                    for C, (prob_C, parse_C) in table[k+1][j].items():
                        key = (B, C)
                        if key in binary_index:
                            for prod in binary_index[key]:
                                prob = prod.prob() * prob_B * prob_C
                                if prod.lhs() not in table[i][j] or prob > table[i][j][prod.lhs()][0]:
                                    table[i][j][prod.lhs()] = (prob, (prod, parse_B, parse_C))
    
    start_symbol = pcfg.start()
    if start_symbol in table[0][n-1]:
        return table[0][n-1][start_symbol]
    return None

def pos_tag_and_parse(pcfg, pos_tagger, words, ngram_model=None):
    if not words:
        return None, [], None
    
    universal_tags = pos_tagger.viterbi_decode(words)
    ptb_tags = [universal_to_ptb(t) for t in universal_tags]
    
    result = cky_parse(pcfg, ptb_tags)
    
    # Fallback to n-gram perplexity if PCFG parse fails
    ngram_perp = None
    if result is None and ngram_model is not None:
        ngram_perp = ngram_model.perplexity(words, n=3)
    
    return result, ptb_tags, ngram_perp

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
    result, ptb_tags, ngram_perp = pos_tag_and_parse(pcfg, models['pos_tagger'], words)
    universal_tags = models['pos_tagger'].viterbi_decode(words)
    print(f"Words: {words}")
    print(f"Universal tags: {universal_tags}")
    print(f"PTB tags: {ptb_tags}")
    print(f"PCFG parse: {'Success' if result else 'Failed'}")
    if result:
        print(f"PCFG log prob: {math.log(result[0])}")
    if ngram_perp is not None:
        print(f"N-gram perplexity (fallback): {ngram_perp:.4f}")
    
    print("Done!")