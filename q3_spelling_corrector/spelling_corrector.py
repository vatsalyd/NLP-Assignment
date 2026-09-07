import nltk
from nltk.corpus import brown
from collections import defaultdict, Counter
import random
import math
import sys
sys.path.append('D:\\projects\\NLP-Assignment')
from model_utils import get_or_train, CHECKPOINT_DIR

FORCE_RETRAIN = False

def load_brown_corpus():
    sentences = list(brown.sents())
    words = [word.lower() for sent in sentences for word in sent]
    return sentences, words

def train_q3_models():
    sentences, words = load_brown_corpus()
    
    vocab = set(words)
    unigram_counts = Counter(words)
    total_words = len(words)
    unigram_probs = {w: c/total_words for w, c in unigram_counts.items()}
    
    bigram_counts = defaultdict(Counter)
    for i in range(len(words)-1):
        bigram_counts[words[i]][words[i+1]] += 1
    
    bigram_probs = {}
    for w1, counter in bigram_counts.items():
        total = sum(counter.values())
        bigram_probs[w1] = {w2: c/total for w2, c in counter.items()}
    
    sym_delete = SymmetricDeleteCorrector(vocab, unigram_counts)
    
    return {
        'vocab': vocab,
        'unigram_counts': unigram_counts,
        'unigram_probs': unigram_probs,
        'bigram_probs': bigram_probs,
        'sym_delete': sym_delete,
        'words': words,
        'sentences': sentences
    }

def build_models(words):
    vocab = set(words)
    unigram_counts = Counter(words)
    total_words = len(words)
    unigram_probs = {w: c/total_words for w, c in unigram_counts.items()}
    
    bigram_counts = defaultdict(Counter)
    for i in range(len(words)-1):
        bigram_counts[words[i]][words[i+1]] += 1
    
    bigram_probs = {}
    for w1, counter in bigram_counts.items():
        total = sum(counter.values())
        bigram_probs[w1] = {w2: c/total for w2, c in counter.items()}
    
    return vocab, unigram_counts, unigram_probs, bigram_probs

def edit_distance_1(word):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word)+1)]
    
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    
    return set(deletes + transposes + replaces + inserts)

class SymmetricDeleteCorrector:
    def __init__(self, vocab, unigram_counts):
        self.vocab = vocab
        self.unigram_counts = unigram_counts
        self.delete_dict = defaultdict(set)
        self._build_delete_dict()
    
    def _build_delete_dict(self):
        for word in self.vocab:
            for i in range(len(word)):
                deleted = word[:i] + word[i+1:]
                self.delete_dict[deleted].add(word)
    
    def get_candidates(self, word):
        candidates = set()
        for i in range(len(word)):
            deleted = word[:i] + word[i+1:]
            if deleted in self.delete_dict:
                candidates.update(self.delete_dict[deleted])
        return candidates

def correct_nonword(word, vocab, unigram_counts, method_a_candidates, method_b_candidates):
    if word in vocab:
        return word
    
    candidates_a = method_a_candidates(word)
    candidates_b = method_b_candidates.get_candidates(word)
    
    all_candidates = (candidates_a | candidates_b) & vocab
    
    if not all_candidates:
        return word
    
    best = max(all_candidates, key=lambda w: unigram_counts.get(w, 0))
    return best

def correct_realword(word, prev_word, vocab, unigram_counts, bigram_probs, method_a_candidates, method_b_candidates, threshold=1.5):
    if word not in vocab:
        return word
    
    candidates_a = method_a_candidates(word)
    candidates_b = method_b_candidates.get_candidates(word)
    all_candidates = (candidates_a | candidates_b) & vocab
    all_candidates.discard(word)
    
    if not all_candidates:
        return word
    
    orig_prob = bigram_probs.get(prev_word, {}).get(word, 1e-10)
    
    best_candidate = word
    best_prob = orig_prob
    
    for cand in all_candidates:
        cand_prob = bigram_probs.get(prev_word, {}).get(cand, 1e-10)
        if cand_prob > best_prob * threshold:
            best_prob = cand_prob
            best_candidate = cand
    
    return best_candidate

def generate_test_set(sentences, words, vocab, error_rate=0.1):
    test_sentences = random.sample(sentences, int(len(sentences) * 0.1))
    
    nonword_tests = []
    realword_tests = []
    
    for sent in test_sentences:
        if len(sent) < 3:
            continue
        idx = random.randint(0, len(sent)-1)
        orig_word = sent[idx].lower()
        
        if orig_word not in vocab:
            continue
        
        # Non-word error: introduce random edit
        edits = list(edit_distance_1(orig_word))
        nonword_candidates = [w for w in edits if w not in vocab]
        if nonword_candidates:
            error_word = random.choice(nonword_candidates)
            new_sent = sent.copy()
            new_sent[idx] = error_word
            nonword_tests.append((new_sent, idx, orig_word))
        
        # Real-word error: replace with another vocab word
        vocab_list = list(vocab)
        realword_candidates = [w for w in vocab_list if w != orig_word and len(w) == len(orig_word)]
        if realword_candidates:
            error_word = random.choice(realword_candidates)
            new_sent = sent.copy()
            new_sent[idx] = error_word
            realword_tests.append((new_sent, idx, orig_word))
    
    return nonword_tests, realword_tests

def evaluate_correction(test_cases, corrector_func, *extra_args):
    correct = 0
    for case in test_cases:
        if len(case) == 3:
            sent, idx, target = case
            corrected = corrector_func(sent[idx].lower(), *extra_args)
        else:
            word, target = case
            corrected = corrector_func(word, *extra_args)
        if corrected == target:
            correct += 1
    return correct / len(test_cases) if test_cases else 0

if __name__ == "__main__":
    print("Loading models...")
    models = get_or_train("q3_models", train_q3_models, force_retrain=FORCE_RETRAIN)
    print(f"Vocabulary size: {len(models['vocab'])}")
    
    print("\nGenerating test set...")
    nonword_tests, realword_tests = generate_test_set(models['sentences'], models['words'], models['vocab'])
    print(f"Non-word test cases: {len(nonword_tests)}")
    print(f"Real-word test cases: {len(realword_tests)}")
    
    print("\nEvaluating non-word correction...")
    nonword_acc = evaluate_correction(
        nonword_tests[:100],
        lambda w: correct_nonword(w, models['vocab'], models['unigram_counts'], edit_distance_1, models['sym_delete'])
    )
    print(f"Non-word accuracy: {nonword_acc:.4f}")
    
    print("\nEvaluating real-word correction...")
    realword_correct = 0
    for sent, idx, target in realword_tests[:100]:
        prev = sent[idx-1].lower() if idx > 0 else ''
        corrected = correct_realword(sent[idx].lower(), prev, models['vocab'], models['unigram_counts'], models['bigram_probs'], edit_distance_1, models['sym_delete'])
        if corrected == target:
            realword_correct += 1
    realword_acc = realword_correct / min(100, len(realword_tests))
    print(f"Real-word accuracy: {realword_acc:.4f}")
    
    print("\nSpeed benchmark...")
    import time
    
    # Generate 1000 misspelled words
    test_words = []
    for _ in range(1000):
        w = random.choice(models['words'])
        edits = list(edit_distance_1(w))
        nonword_edits = [e for e in edits if e not in models['vocab']]
        if nonword_edits:
            test_words.append(random.choice(nonword_edits))
        else:
            test_words.append(w)
    
    # Method A benchmark
    start = time.perf_counter()
    for w in test_words:
        _ = edit_distance_1(w) & models['vocab']
    method_a_time = time.perf_counter() - start
    
    # Method B benchmark
    start = time.perf_counter()
    for w in test_words:
        _ = models['sym_delete'].get_candidates(w)
    method_b_time = time.perf_counter() - start
    
    print(f"Method A (edit distance 1): {method_a_time:.6f}s")
    print(f"Method B (symmetric delete): {method_b_time:.6f}s")
    if method_b_time > 0:
        print(f"Speedup: {method_a_time/method_b_time:.2f}x")
    else:
        print("Method B too fast to measure accurately")