import streamlit as st
import random
import time
import math
from collections import Counter, defaultdict
import nltk
from nltk.corpus import brown, treebank, gutenberg, reuters
from nltk import induce_pcfg, Nonterminal

# Load models
@st.cache_resource
def load_models():
    sentences = list(brown.sents())
    words = [word.lower() for sent in sentences for word in sent]
    
    vocab = set(words)
    unigram_counts = Counter(words)
    total_words = len(words)
    unigram_probs = {w: c/total_words for w, c in unigram_counts.items()}
    
    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    
    for i in range(len(words)-1):
        bigram_counts[words[i]][words[i+1]] += 1
    for i in range(len(words)-2):
        trigram_counts[(words[i], words[i+1])][words[i+2]] += 1
    
    bigram_probs = {}
    for w1, counter in bigram_counts.items():
        total = sum(counter.values())
        bigram_probs[w1] = {w2: c/total for w2, c in counter.items()}
    
    trigram_probs = {}
    for w1w2, counter in trigram_counts.items():
        total = sum(counter.values())
        trigram_probs[w1w2] = {w3: c/total for w3, c in counter.items()}
    
    return {
        'vocab': vocab,
        'unigram_counts': unigram_counts,
        'unigram_probs': unigram_probs,
        'bigram_probs': bigram_probs,
        'trigram_probs': trigram_probs,
        'words': words
    }

@st.cache_resource
def load_pcfg():
    productions = []
    for tree in treebank.parsed_sents():
        tree.collapse_unary(collapsePOS=True)
        tree.chomsky_normal_form(horzMarkov=2)
        productions.extend(tree.productions())
    
    start = Nonterminal('S')
    pcfg = induce_pcfg(start, productions)
    return pcfg

@st.cache_resource
def load_spelling_models():
    import sys
    sys.path.append('D:\\projects\\NLP-Assignment\\q3_spelling_corrector')
    from spelling_corrector import SymmetricDeleteCorrector, edit_distance_1
    
    models = load_models()
    sym_delete = SymmetricDeleteCorrector(models['vocab'], models['unigram_counts'])
    return sym_delete, edit_distance_1

def correct_nonword(word, vocab, unigram_counts, edit_distance_1_func, sym_delete):
    if word in vocab:
        return word
    
    candidates_a = edit_distance_1_func(word)
    candidates_b = sym_delete.get_candidates(word)
    
    all_candidates = (candidates_a | candidates_b) & vocab
    
    if not all_candidates:
        return word
    
    best = max(all_candidates, key=lambda w: unigram_counts.get(w, 0))
    return best

def correct_realword(word, prev_word, vocab, unigram_counts, bigram_probs, edit_distance_1_func, sym_delete, threshold=1.5):
    if word not in vocab:
        return word
    
    candidates_a = edit_distance_1_func(word)
    candidates_b = sym_delete.get_candidates(word)
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
    for i, token in enumerate(tokens):
        result.append(token)
        if i < len(tokens) - 1 and random.random() < p:
            result[-1] = result[-1] + tokens[i+1]
            i += 1
    return result

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

def main():
    st.title("Integrated Background Editor")
    st.write("Live segmentation, spelling correction, and grammar checking")
    
    models = load_models()
    pcfg = load_pcfg()
    sym_delete, edit_distance_1 = load_spelling_models()
    
    vocab = models['vocab']
    unigram_counts = models['unigram_counts']
    unigram_probs = models['unigram_probs']
    bigram_probs = models['bigram_probs']
    trigram_probs = models['trigram_probs']
    
    # Session state
    if 'passage' not in st.session_state:
        st.session_state.passage = get_random_passage()
    if 'tokens' not in st.session_state:
        st.session_state.tokens = st.session_state.passage.split()
    if 'processed' not in st.session_state:
        st.session_state.processed = []
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'segmentation_count' not in st.session_state:
        st.session_state.segmentation_count = 0
    if 'spelling_count' not in st.session_state:
        st.session_state.spelling_count = 0
    
    st.write("### Live Typing Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Random Passage"):
            st.session_state.passage = get_random_passage()
            st.session_state.tokens = st.session_state.passage.split()
            st.session_state.processed = []
            st.session_state.alerts = []
            st.session_state.segmentation_count = 0
            st.session_state.spelling_count = 0
            st.rerun()
    
    with col2:
        if st.button("Start Live Processing"):
            st.session_state.run_live = True
    
    # Live processing
    if st.session_state.get('run_live', False):
        tokens = st.session_state.tokens
        merged_tokens = introduce_merges(tokens, 0.08)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        alert_container = st.empty()
        
        processed = []
        alerts = []
        
        N = 10
        
        for i, token in enumerate(merged_tokens):
            progress_bar.progress((i + 1) / len(merged_tokens))
            status_text.text(f"Processing token {i+1}/{len(merged_tokens)}: {token}")
            
            start = time.perf_counter()
            
            # SEGMENT-ALERT
            if token not in vocab and len(token) > 5:
                seg_words = viterbi_segmentation(token, vocab, trigram_probs, unigram_probs)
                if len(seg_words) > 1:
                    st.session_state.segmentation_count += 1
                    alerts.append(f"[SEGMENT-ALERT] '{token}' -> {' '.join(seg_words)}")
                    token = ' '.join(seg_words)
            
            # SPELL-ALERT
            word = token.split()[-1] if ' ' in token else token
            clean_word = word.strip('.,!?;:')
            if clean_word and clean_word not in vocab:
                corrected = correct_nonword(clean_word, vocab, unigram_counts, edit_distance_1, sym_delete)
                if corrected != clean_word:
                    st.session_state.spelling_count += 1
                    alerts.append(f"[SPELL-ALERT] '{clean_word}' -> '{corrected}'")
                    token = token.replace(clean_word, corrected)
            
            processed.append(token)
            
            # GRAMMAR-ALERT every N words
            if (i + 1) % N == 0:
                window = processed[max(0, i-N+1):i+1]
                flat_words = ' '.join(window).split()
                bi_perp = bigram_perplexity(flat_words, bigram_probs, unigram_probs)
                tri_perp = trigram_perplexity(flat_words, trigram_probs, bigram_probs, unigram_probs)
                
                if bi_perp > 500 or tri_perp > 500:
                    alerts.append(f"[GRAMMAR-ALERT] High perplexity: bigram={bi_perp:.1f}, trigram={tri_perp:.1f} in '{' '.join(window)}'")
            
            elapsed = (time.perf_counter() - start) * 1000
            if i % 5 == 0:
                alert_container.write(f"Latest alerts: {alerts[-3:] if alerts else 'None'}")
        
        st.session_state.processed = processed
        st.session_state.alerts = alerts
        st.session_state.run_live = False
        st.success("Live processing complete!")
    
    # Display results
    if st.session_state.processed:
        st.write("### Processed Passage")
        st.write(' '.join(st.session_state.processed))
        
        st.write("### Alerts")
        for alert in st.session_state.alerts:
            st.warning(alert)
        
        st.write(f"Segmentation corrections: {st.session_state.segmentation_count}")
        st.write(f"Spelling corrections: {st.session_state.spelling_count}")
        
        # Final analysis
        if st.button("Run Final Analysis"):
            full_text = ' '.join(st.session_state.processed)
            sentences = nltk.sent_tokenize(full_text)
            
            results = []
            for sent in sentences:
                tokens = nltk.word_tokenize(sent.lower())
                tokens = [t for t in tokens if t.isalpha()]
                
                pcfg_result = cky_parse(pcfg, tokens)
                pcfg_score = math.log(pcfg_result[0]) if pcfg_result else float('-inf')
                
                bi_perp = bigram_perplexity(tokens, bigram_probs, unigram_probs)
                tri_perp = trigram_perplexity(tokens, trigram_probs, bigram_probs, unigram_probs)
                
                # Decision rule
                if pcfg_result and pcfg_score > -100:
                    method = "PCFG"
                    verdict = "Grammatical" if pcfg_score > -50 else "Questionable"
                elif tri_perp < 200:
                    method = "Trigram"
                    verdict = "Grammatical" if tri_perp < 100 else "Questionable"
                else:
                    method = "Bigram"
                    verdict = "Grammatical" if bi_perp < 150 else "Questionable"
                
                results.append({
                    'sentence': sent,
                    'pcfg_score': f"{pcfg_score:.2f}" if pcfg_score != float('-inf') else "Unparseable",
                    'bigram_perp': f"{bi_perp:.2f}",
                    'trigram_perp': f"{tri_perp:.2f}",
                    'method': method,
                    'verdict': verdict
                })
            
            st.write("### Final Analysis")
            st.table(results)

if __name__ == "__main__":
    main()