import streamlit as st
import random
import time
import math
import sys
import os
import nltk
from nltk.corpus import brown, treebank, gutenberg, reuters

# Add path for Q1 and Q3 modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from q1_segmentation_pos.corpus_loader import load_brown_corpus as load_brown_corpus_q1, extract_words_tags, build_vocabulary
from q1_segmentation_pos.pos_tagger import POSTagger
from q3_spelling_corrector.spelling_corrector import SymmetricDeleteCorrector, edit_distance_1, load_brown_corpus as load_brown_corpus_q3
from config import Q4_CONFIG, Q3_CONFIG

# Import shared models and functions
from models import (
    load_shared_models,
    train_pcfg,
    pos_tag_and_parse,
    cky_parse,
    viterbi_segmentation,
    SmoothedNGramModel,
    get_random_passage,
    introduce_merges,
)

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

def run_speed_demon_benchmark(vocab, unigram_counts, bigram_probs, trigram_probs, 
                               unigram_probs, edit_distance_1_func, sym_delete, 
                               pos_tagger, pcfg, ngram_model):
    """Run Speed Demon benchmark for the full live-check pipeline."""
    import time
    
    # Generate 1000 simulated words as per assignment requirements
    _, words = load_brown_corpus_q3()
    test_words = []
    for _ in range(Q3_CONFIG["speed_demon_num_words"]):
        w = random.choice(words)
        test_words.append(w)
    
    # Benchmark 1: Per-token pipeline (segmentation + spelling) - NO PCFG
    start = time.perf_counter()
    seg_spell_latencies = []
    for word in test_words:
        token_start = time.perf_counter()
        
        # Segmentation check
        if word not in vocab and len(word) > 5:
            _ = viterbi_segmentation(word, vocab, trigram_probs, unigram_probs)
        
        # Spelling check
        clean_word = word.strip('.,!?;:')
        if clean_word and clean_word not in vocab:
            _ = correct_nonword(clean_word, vocab, unigram_counts, edit_distance_1_func, sym_delete)
        
        seg_spell_latencies.append((time.perf_counter() - token_start) * 1000)
    
    full_pipeline_time = time.perf_counter() - start
    avg_seg_spell = sum(seg_spell_latencies) / len(seg_spell_latencies)
    
    # Benchmark 2: Grammar trigger check only (every N words)
    N = Q4_CONFIG["trigger_interval"]
    trigger_latencies = []
    start = time.perf_counter()
    for i in range(0, len(test_words), N):
        window = test_words[i:i+N]
        if len(window) < 2:
            continue
        trigger_start = time.perf_counter()
        
        # PCFG parse with POS tagging
        _ = pos_tag_and_parse(pcfg, pos_tagger, window, ngram_model)
        
        trigger_latencies.append((time.perf_counter() - trigger_start) * 1000)
    
    grammar_time = time.perf_counter() - start
    avg_grammar = sum(trigger_latencies) / len(trigger_latencies) if trigger_latencies else 0
    
    return {
        'full_pipeline_total': full_pipeline_time,
        'avg_seg_spell_ms': avg_seg_spell,
        'grammar_check_total': grammar_time,
        'avg_grammar_ms': avg_grammar,
        'seg_spell_per_word': avg_seg_spell,
        'grammar_per_trigger': avg_grammar,
    }

@st.cache_resource
def load_all_models():
    """Load all shared models at once."""
    models = load_shared_models()
    
    # Build n-gram model for fallback
    ngram_model = SmoothedNGramModel(
        models['unigram_probs'],
        models['bigram_probs'],
        models['trigram_probs'],
        models['vocab']
    )
    
    # Train PCFG
    pcfg = train_pcfg()
    
    # Load spelling corrector
    sym_delete = SymmetricDeleteCorrector(models['vocab'], models['unigram_counts'])
    
    return {
        'models': models,
        'ngram_model': ngram_model,
        'pcfg': pcfg,
        'sym_delete': sym_delete,
        'pos_tagger': models['pos_tagger'],
    }

def main():
    st.title("Integrated Background Editor")
    st.write("Live segmentation, spelling correction, and grammar checking")
    
    # Sidebar for parameters
    with st.sidebar:
        st.write("### Parameters")
        st.write(f"**Merge probability p = {Q4_CONFIG['merge_probability']}**")
        st.write("Justification: Simulates realistic fast-typing spacebar miss rate (~8%)")
        st.write("")
        st.write(f"**Trigger interval N = {Q4_CONFIG['trigger_interval']}**")
        st.write("Justification: Balances latency (checking every 10 words) with responsiveness")
        st.write("")
        st.write(f"**Add-k smoothing k = {Q4_CONFIG['add_k_smoothing']}**")
        st.write("Justification: Standard small value for smoothing sparse n-grams")
        st.write("")
        st.write("**Tagset reconciliation**")
        st.write("Brown universal -> Penn Treebank via lookup table")
    
    all_models = load_all_models()
    models = all_models['models']
    ngram_model = all_models['ngram_model']
    pcfg = all_models['pcfg']
    sym_delete = all_models['sym_delete']
    pos_tagger = all_models['pos_tagger']
    
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
    if 'seg_latencies' not in st.session_state:
        st.session_state.seg_latencies = []
    if 'grammar_latencies' not in st.session_state:
        st.session_state.grammar_latencies = []
    if 'sentence_seg_counts' not in st.session_state:
        st.session_state.sentence_seg_counts = {}
    if 'sentence_spell_counts' not in st.session_state:
        st.session_state.sentence_spell_counts = {}
    
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
            st.session_state.seg_latencies = []
            st.session_state.grammar_latencies = []
            st.session_state.sentence_seg_counts = {}
            st.session_state.sentence_spell_counts = {}
            st.rerun()
    
    with col2:
        if st.button("Start Live Processing"):
            st.session_state.run_live = True
    
    # Live processing
    if st.session_state.get('run_live', False):
        tokens = st.session_state.tokens
        merged_tokens = introduce_merges(tokens, Q4_CONFIG["merge_probability"])
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        alert_container = st.empty()
        
        processed = []
        alerts = []
        sentence_seg_counts = {}
        sentence_spell_counts = {}
        current_sentence_idx = 0
        
        N = Q4_CONFIG["trigger_interval"]
        
        for i, token in enumerate(merged_tokens):
            progress_bar.progress((i + 1) / len(merged_tokens))
            status_text.text(f"Processing token {i+1}/{len(merged_tokens)}: {token}")
            
            token_start = time.perf_counter()
            
            # SEGMENT-ALERT
            seg_alert_fired = False
            if token not in vocab and len(token) > Q4_CONFIG["merge_min_token_len"]:
                seg_words = viterbi_segmentation(token, vocab, trigram_probs, unigram_probs)
                if len(seg_words) > 1:
                    st.session_state.segmentation_count += 1
                    sentence_seg_counts[current_sentence_idx] = sentence_seg_counts.get(current_sentence_idx, 0) + 1
                    alerts.append(f"[SEGMENT-ALERT] '{token}' -> {' '.join(seg_words)}")
                    token = ' '.join(seg_words)
                    seg_alert_fired = True
            
            # SPELL-ALERT
            spell_alert_fired = False
            word = token.split()[-1] if ' ' in token else token
            clean_word = word.strip('.,!?;:')
            if clean_word and clean_word not in vocab:
                corrected = correct_nonword(clean_word, vocab, unigram_counts, edit_distance_1, sym_delete)
                if corrected != clean_word:
                    st.session_state.spelling_count += 1
                    sentence_spell_counts[current_sentence_idx] = sentence_spell_counts.get(current_sentence_idx, 0) + 1
                    alerts.append(f"[SPELL-ALERT] '{clean_word}' -> '{corrected}'")
                    token = token.replace(clean_word, corrected)
                    spell_alert_fired = True
            
            processed.append(token)
            
            # Track sentence boundaries for per-sentence counts
            if token.endswith('.') or token.endswith('!') or token.endswith('?'):
                current_sentence_idx += 1
            
            # GRAMMAR-ALERT and REAL-WORD check every N words
            grammar_start = time.perf_counter()
            if (i + 1) % N == 0:
                window = processed[max(0, i-N+1):i+1]
                flat_words = ' '.join(window).split()
                
                # Try PCFG parse with POS tagging
                pcfg_result, ptb_tags, ngram_perp = pos_tag_and_parse(pcfg, pos_tagger, flat_words, ngram_model)
                
                grammar_alert_fired = False
                if pcfg_result:
                    pcfg_score = math.log(pcfg_result[0])
                    if pcfg_score < Q4_CONFIG["pcfg_parse_threshold"]:  # Very low probability
                        alerts.append(f"[GRAMMAR-ALERT] Low PCFG probability: {pcfg_score:.2f} in '{' '.join(window)}'")
                        grammar_alert_fired = True
                elif ngram_perp is not None:
                    # Fallback to n-gram perplexity
                    if ngram_perp > Q4_CONFIG["perplexity_high"]:
                        alerts.append(f"[GRAMMAR-ALERT] High trigram perplexity (fallback): {ngram_perp:.1f} in '{' '.join(window)}'")
                        grammar_alert_fired = True
                
                # REAL-WORD ERROR CHECK at same trigger interval
                if len(flat_words) >= 2:
                    for j in range(1, len(flat_words)):
                        prev = flat_words[j-1]
                        curr = flat_words[j]
                        if curr in vocab:
                            corrected = correct_realword(curr, prev, vocab, unigram_counts, 
                                                         bigram_probs, edit_distance_1, sym_delete,
                                                         Q3_CONFIG["realword_threshold"])
                            if corrected != curr:
                                alerts.append(f"[GRAMMAR-ALERT] Real-word: '{prev} {curr}' -> '{prev} {corrected}'")
            
            grammar_elapsed = (time.perf_counter() - grammar_start) * 1000
            token_elapsed = (time.perf_counter() - token_start) * 1000
            
            st.session_state.seg_latencies.append(token_elapsed)
            st.session_state.grammar_latencies.append(grammar_elapsed)
            
            if i % 5 == 0:
                alert_container.write(f"Latest alerts: {alerts[-3:] if alerts else 'None'}")
        
        st.session_state.processed = processed
        st.session_state.alerts = alerts
        st.session_state.sentence_seg_counts = sentence_seg_counts
        st.session_state.sentence_spell_counts = sentence_spell_counts
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
        
        # Latency report
        if st.session_state.seg_latencies:
            avg_seg = sum(st.session_state.seg_latencies) / len(st.session_state.seg_latencies)
            avg_gram = sum(st.session_state.grammar_latencies) / len(st.session_state.grammar_latencies) if st.session_state.grammar_latencies else 0
            st.write("### Latency Report")
            st.write(f"Avg per-token (seg+spell) latency: {avg_seg:.2f} ms")
            st.write(f"Avg per-trigger grammar latency: {avg_gram:.2f} ms")
        
        # Speed Demon Benchmark
        st.write("### Speed Demon Benchmark")
        if st.button("Run Speed Demon Benchmark (1000 words)"):
            with st.spinner("Running benchmark..."):
                bench_results = run_speed_demon_benchmark(
                    vocab, unigram_counts, bigram_probs, trigram_probs,
                    unigram_probs, edit_distance_1, sym_delete,
                    pos_tagger, pcfg, ngram_model
                )
            
            st.write(f"**Full Pipeline (seg+spell)**: Total={bench_results['full_pipeline_total']:.3f}s, Avg/word={bench_results['avg_seg_spell_ms']:.3f}ms")
            st.write(f"**Grammar Trigger Check**: Total={bench_results['grammar_check_total']:.3f}s, Avg/trigger={bench_results['avg_grammar_ms']:.3f}ms")
            st.write(f"**Overhead**: Seg+spell layer adds ~{bench_results['avg_seg_spell_ms'] - bench_results['avg_grammar_ms']:.3f}ms per word over grammar layer")
        
        # Final analysis
        if st.button("Run Final Analysis"):
            full_text = ' '.join(st.session_state.processed)
            sentences = nltk.sent_tokenize(full_text)
            
            results = []
            for sent_idx, sent in enumerate(sentences):
                tokens = nltk.word_tokenize(sent.lower())
                tokens = [t for t in tokens if t.isalpha()]
                
                if not tokens:
                    continue
                
                # PCFG parse with POS tagging and n-gram fallback
                pcfg_result, ptb_tags, ngram_perp = pos_tag_and_parse(pcfg, pos_tagger, tokens, ngram_model)
                pcfg_score = math.log(pcfg_result[0]) if pcfg_result else float('-inf')
                
                bi_perp = ngram_model.perplexity(tokens, n=2)
                tri_perp = ngram_model.perplexity(tokens, n=3)
                
                # Decision rule
                if pcfg_result and pcfg_score > Q4_CONFIG["pcfg_parse_threshold"]:
                    method = "PCFG"
                    verdict = "Grammatical" if pcfg_score > Q4_CONFIG["pcfg_grammatical_threshold"] else "Questionable"
                elif tri_perp < Q4_CONFIG["perplexity_trigram_adequate"]:
                    method = "Trigram"
                    verdict = "Grammatical" if tri_perp < Q4_CONFIG["perplexity_trigram_good"] else "Questionable"
                else:
                    method = "Bigram"
                    verdict = "Grammatical" if bi_perp < Q4_CONFIG["perplexity_bigram_good"] else "Questionable"
                
                seg_count = st.session_state.sentence_seg_counts.get(sent_idx, 0)
                spell_count = st.session_state.sentence_spell_counts.get(sent_idx, 0)
                
                results.append({
                    'sentence': sent[:Q4_CONFIG["sentence_display_max_chars"]] + ('...' if len(sent) > Q4_CONFIG["sentence_display_max_chars"] else ''),
                    'pcfg_score': f"{pcfg_score:.2f}" if pcfg_score != float('-inf') else "Unparseable",
                    'bigram_perp': f"{bi_perp:.2f}",
                    'trigram_perp': f"{tri_perp:.2f}",
                    'method': method,
                    'verdict': verdict,
                    'seg_corrections': seg_count,
                    'spell_corrections': spell_count
                })
            
            st.write("### Final Analysis - Per-Sentence Comparison Table")
            st.table(results)
            
            # Comparative analysis
            st.write("### Comparative Analysis Summary")
            pcfg_count = sum(1 for r in results if r['method'] == 'PCFG')
            trigram_count = sum(1 for r in results if r['method'] == 'Trigram')
            bigram_count = sum(1 for r in results if r['method'] == 'Bigram')
            grammatical = sum(1 for r in results if r['verdict'] == 'Grammatical')
            
            st.write(f"Sentences analyzed: {len(results)}")
            st.write(f"Method selection: PCFG={pcfg_count}, Trigram={trigram_count}, Bigram={bigram_count}")
            st.write(f"Grammatical: {grammatical}, Questionable: {len(results) - grammatical}")
            st.write(f"Total segmentation corrections: {st.session_state.segmentation_count}")
            st.write(f"Total spelling corrections: {st.session_state.spelling_count}")

if __name__ == "__main__":
    main()