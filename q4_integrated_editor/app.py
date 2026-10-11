import streamlit as st
import streamlit.components.v1 as components
import random
import time
import math
import sys
import os
import nltk

# Ensure NLTK data path is set BEFORE importing nltk.corpus
nltk.data.path.insert(0, os.path.expanduser("~/nltk_data"))

from nltk.corpus import brown, treebank, gutenberg, reuters

# Add path for Q1 and Q3 modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from q1_segmentation_pos.corpus_loader import load_brown_corpus as load_brown_corpus_q1, extract_words_tags, build_vocabulary
from q1_segmentation_pos.pos_tagger import POSTagger
from q3_spelling_corrector.spelling_corrector import SymmetricDeleteCorrector, edit_distance_1
from config import Q4_CONFIG, Q3_CONFIG
from model_utils import get_or_train

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

LIVE_TEXT_COMPONENT = components.declare_component(
    "q4_live_text",
    path=os.path.join(os.path.dirname(__file__), "live_text_component"),
)

def correct_nonword(word, vocab, unigram_counts, edit_distance_1_func, sym_delete):
    if word in vocab:
        return word
    
    candidates_a = edit_distance_1_func(word)
    candidates_b = sym_delete.get_candidates(word)
    
    all_candidates = {
        candidate for candidate in (candidates_a | candidates_b) & vocab
        if candidate.isalpha()
    }
    
    if not all_candidates:
        return word
    
    best = max(all_candidates, key=lambda w: unigram_counts.get(w, 0))
    return best

def correct_realword(word, prev_word, vocab, unigram_counts, bigram_probs, edit_distance_1_func, sym_delete, threshold=1.5):
    if word not in vocab:
        return word
    
    candidates_a = edit_distance_1_func(word)
    candidates_b = sym_delete.get_candidates(word)
    all_candidates = {
        candidate for candidate in (candidates_a | candidates_b) & vocab
        if candidate.isalpha()
    }
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
                               pos_tagger, pcfg, ngram_model, words):
    """Run Speed Demon benchmark for the full live-check pipeline."""
    import time

    # Generate 1000 simulated words as per assignment requirements.
    # `words` is the Brown corpus word list already loaded (and cached) by
    # load_all_models() - reusing it instead of reloading the whole corpus
    # here shaves ~6s off every benchmark run.
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
    
    # Train PCFG (or load from checkpoint - same pattern as Q1/Q2/Q3)
    pcfg = get_or_train("q4_pcfg", train_pcfg)
    
    # Load spelling corrector
    sym_delete = SymmetricDeleteCorrector(models['vocab'], models['unigram_counts'])
    
    return {
        'models': models,
        'ngram_model': ngram_model,
        'pcfg': pcfg,
        'sym_delete': sym_delete,
        'pos_tagger': models['pos_tagger'],
    }

def _add_live_alert(new_alerts, seen_keys, key, message):
    """Append an alert once while allowing the live text to be reprocessed."""
    if key not in seen_keys:
        seen_keys.add(key)
        new_alerts.append(message)
        return True
    return False


def _grammar_alerts(window, trigger_index, vocab, unigram_counts, bigram_probs,
                    edit_distance_1, sym_delete, pos_tagger, pcfg, ngram_model,
                    q4_config, new_alerts, seen_keys, grammar_latencies):
    """Run one grammar/real-word trigger and append any alerts it finds."""
    if not window:
        return

    grammar_start = time.perf_counter()
    normalized = [word.strip('.,!?;:').lower() for word in window]
    normalized = [word for word in normalized if word]

    try:
        pcfg_result, _, ngram_perp = pos_tag_and_parse(
            pcfg, pos_tagger, normalized, ngram_model
        )
    except Exception as exc:
        pcfg_result, ngram_perp = None, None
        _add_live_alert(
            new_alerts,
            seen_keys,
            f"grammar-error:{trigger_index}",
            f"[GRAMMAR-ALERT] Grammar check unavailable: {type(exc).__name__}",
        )

    if pcfg_result:
        pcfg_score = math.log(max(pcfg_result[0], 1e-300))
        if pcfg_score < q4_config["pcfg_parse_threshold"]:
            _add_live_alert(
                new_alerts,
                seen_keys,
                f"grammar-pcfg:{trigger_index}",
                f"[GRAMMAR-ALERT] Low PCFG probability: {pcfg_score:.2f} in '{' '.join(window)}'",
            )
    else:
        # A fixed N-word window is often only a phrase fragment (for example
        # "a common habit"), so PCFG failure alone is not a valid live alert.
        # Final sentence analysis still reports genuinely unparseable input.
        pass

    # Perplexity is checked independently of PCFG success.  The old code used
    # `elif`, so a successful PCFG parse could suppress the n-gram alert.
    # Require a meaningful margin above the warning threshold so a plausible
    # but unseen local trigram does not become a grammar error.
    if ngram_perp is not None and ngram_perp > q4_config["perplexity_high"] * 1.25:
        _add_live_alert(
            new_alerts,
            seen_keys,
            f"grammar-perplexity:{trigger_index}",
            f"[GRAMMAR-ALERT] High trigram perplexity: {ngram_perp:.1f} in '{' '.join(window)}'",
        )

    if len(normalized) >= 2:
        for index in range(1, len(normalized)):
            prev_word, current_word = normalized[index - 1:index + 1]
            if current_word not in vocab:
                continue
            corrected = correct_realword(
                current_word,
                prev_word,
                vocab,
                unigram_counts,
                bigram_probs,
                edit_distance_1,
                sym_delete,
                max(Q3_CONFIG["realword_threshold"], 3.0),
            )
            if corrected != current_word:
                _add_live_alert(
                    new_alerts,
                    seen_keys,
                    f"realword:{trigger_index}:{prev_word}:{current_word}:{corrected}",
                    f"[GRAMMAR-ALERT] Real-word: '{prev_word} {current_word}' -> '{prev_word} {corrected}'",
                )

    grammar_latencies.append((time.perf_counter() - grammar_start) * 1000)


def analyze_live_text(text, vocab, trigram_probs, unigram_probs, unigram_counts,
                      bigram_probs, edit_distance_1, sym_delete, pos_tagger, pcfg,
                      ngram_model, q4_config, seen_keys):
    """Incrementally analyze the current text for all live alert types.

    The function is deterministic for a given text and `seen_keys`, which
    makes it safe to call after every Streamlit widget update.
    """
    words = text.split()
    corrected_words = []
    new_alerts = []
    seg_count = 0
    spell_count = 0
    token_latencies = []
    grammar_latencies = []

    for index, original_word in enumerate(words):
        token_start = time.perf_counter()
        clean_word = original_word.strip('.,!?;:')
        lookup_word = clean_word.lower()
        replacement_words = [original_word]
        segmented = False

        if lookup_word and lookup_word not in vocab and len(lookup_word) > q4_config["merge_min_token_len"]:
            seg_words = viterbi_segmentation(
                lookup_word, vocab, trigram_probs, unigram_probs
            )
            if len(seg_words) > 1:
                if _add_live_alert(
                    new_alerts,
                    seen_keys,
                    f"segment:{index}:{original_word}",
                    f"[SEGMENT-ALERT] '{original_word}' -> {' '.join(seg_words)}",
                ):
                    seg_count += 1
                replacement_words = seg_words
                segmented = True

        if lookup_word and lookup_word not in vocab and not segmented:
            corrected = correct_nonword(
                lookup_word, vocab, unigram_counts, edit_distance_1, sym_delete
            )
            if corrected != lookup_word:
                if _add_live_alert(
                    new_alerts,
                    seen_keys,
                    f"spell:{index}:{original_word}",
                    f"[SPELL-ALERT] '{original_word}' -> '{corrected}'",
                ):
                    spell_count += 1
                replacement_words = [corrected]

        if lookup_word == 'i':
            if _add_live_alert(
                new_alerts,
                seen_keys,
                f"case:{index}",
                "[CASE-ALERT] 'i' -> 'I' (capitalize first-person pronoun)",
            ):
                spell_count += 1
            replacement_words = ['I']

        corrected_words.extend(replacement_words)
        token_latencies.append((time.perf_counter() - token_start) * 1000)

    normalized_words = [word.strip('.,!?;:').lower() for word in corrected_words]
    normalized_words = [word for word in normalized_words if word.isalpha()]
    trigger_interval = q4_config["trigger_interval"]
    for end in range(trigger_interval, len(normalized_words) + 1, trigger_interval):
        _grammar_alerts(
            normalized_words[end - trigger_interval:end],
            end,
            vocab,
            unigram_counts,
            bigram_probs,
            edit_distance_1,
            sym_delete,
            pos_tagger,
            pcfg,
            ngram_model,
            q4_config,
            new_alerts,
            seen_keys,
            grammar_latencies,
        )

    return {
        "alerts": new_alerts,
        "corrected_words": corrected_words,
        "segmentation_count": seg_count,
        "spelling_count": spell_count,
        "token_latencies": token_latencies,
        "grammar_latencies": grammar_latencies,
    }


def process_typing_alerts(text: str, vocab, trigram_probs, unigram_probs, unigram_counts,
                          edit_distance_1, sym_delete, pos_tagger, pcfg, ngram_model,
                          vocab_list, q4_config, session_state, bigram_probs=None):
    """Compatibility wrapper used by the UI and tests."""
    seen_keys = session_state.setdefault("alert_keys", set())
    result = analyze_live_text(
        text,
        vocab,
        trigram_probs,
        unigram_probs,
        unigram_counts,
        bigram_probs if bigram_probs is not None else session_state.get("bigram_probs", {}),
        edit_distance_1,
        sym_delete,
        pos_tagger,
        pcfg,
        ngram_model,
        q4_config,
        seen_keys,
    )
    session_state.setdefault("alerts", []).extend(result["alerts"])
    session_state["segmentation_count"] = session_state.get("segmentation_count", 0) + result["segmentation_count"]
    session_state["spelling_count"] = session_state.get("spelling_count", 0) + result["spelling_count"]
    session_state.setdefault("seg_latencies", []).extend(result["token_latencies"])
    session_state.setdefault("grammar_latencies", []).extend(result["grammar_latencies"])
    return result

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
        st.write(f"Justification: Checks every {Q4_CONFIG['trigger_interval']} words for responsive feedback")
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
    if 'live_mode' not in st.session_state:
        st.session_state.live_mode = False
    if 'live_index' not in st.session_state:
        st.session_state.live_index = 0
    if 'live_text' not in st.session_state:
        st.session_state.live_text = ""
    if 'live_char_index' not in st.session_state:
        st.session_state.live_char_index = 0
    if 'user_typing_mode' not in st.session_state:
        st.session_state.user_typing_mode = False
    if 'manual_input' not in st.session_state:
        st.session_state.manual_input = ""
    if 'manual_last_text' not in st.session_state:
        st.session_state.manual_last_text = ""
    if 'alert_keys' not in st.session_state:
        st.session_state.alert_keys = set()
    
    st.write("### Live Typing Simulation")
    st.caption(
        f"Batch passage: {len(st.session_state.tokens)} words "
        "(minimum 500 words)"
    )
    
    col1, col2, col3 = st.columns(3)
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
            st.session_state.live_mode = False
            st.session_state.live_index = 0
            st.session_state.live_text = ""
            st.session_state.live_char_index = 0
            st.session_state.user_typing_mode = False
            st.session_state.manual_input = ""
            st.session_state.manual_last_text = ""
            st.session_state.manual_corrected_text = ""
            st.session_state.alert_keys = set()
            st.query_params.clear()
            st.rerun()
    
    with col2:
        if st.button("Start Live Processing (Batch)"):
            st.session_state.run_live = True
            st.session_state.alerts = []
            st.session_state.segmentation_count = 0
            st.session_state.spelling_count = 0
            st.session_state.seg_latencies = []
            st.session_state.grammar_latencies = []
            st.session_state.alert_keys = set()
    
    with col3:
        if st.button("Manual Live Typing (User Input)"):
            st.session_state.user_typing_mode = True
            st.session_state.manual_input = ""
            st.session_state.manual_last_text = ""
            st.session_state.manual_corrected_text = ""
            st.session_state.alerts = []
            st.session_state.processed = []
            st.session_state.segmentation_count = 0
            st.session_state.spelling_count = 0
            st.session_state.seg_latencies = []
            st.session_state.grammar_latencies = []
            st.session_state.alert_keys = set()
            st.query_params.clear()
            st.session_state.typing_session_key = st.session_state.get('typing_session_key', 0) + 1
            st.rerun()
    
# Manual Live Typing - User types freely with REAL-TIME alerts on SPACE press (like Google search)
    # Legacy iframe implementation retained below for reference; the
    # Streamlit-native implementation is used further down.
    if False and st.session_state.get('user_typing_mode', False):
        st.write("### Manual Live Typing (User Input)")
        st.write("Type in the text area below. Alerts appear **when you press SPACE** (like Google search).")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Stop Typing Mode"):
                st.session_state.user_typing_mode = False
                st.rerun()
        with col2:
            show_alerts = st.checkbox("Show Alerts", value=True)
        
        # Initialize session state for typing
        if 'user_text' not in st.session_state:
            st.session_state.user_text = ""
        if 'last_space_count' not in st.session_state:
            st.session_state.last_space_count = 0
        
        def process_typing_alerts(text: str):
            """Process text and generate alerts - only trigger on NEW space presses"""
            st.session_state.user_text = text
            words = text.split()
            
            # Count spaces in current text
            space_count = text.count(' ')
            
            # Check for new space presses (only trigger alerts on NEW space presses)
            if space_count > st.session_state.last_space_count:
                st.session_state.last_space_count = space_count
                words = st.session_state.user_text.split()
                
                # SEGMENT-ALERT: Check for merged/unknown words
                for word in words:
                    if word not in vocab and len(word) > Q4_CONFIG["merge_min_token_len"]:
                        seg_words = viterbi_segmentation(word, vocab, trigram_probs, unigram_probs)
                        if len(seg_words) > 1:
                            alert_key = f"[SEGMENT-ALERT] '{word}'"
                            if alert_key not in st.session_state.alerts:
                                st.session_state.segmentation_count += 1
                                st.session_state.alerts.append(f"[SEGMENT-ALERT] '{word}' -> {' '.join(seg_words)}")
                
                # SPELL-ALERT: Check for unknown words
                for word in words:
                    clean_word = word.strip('.,!?;:')
                    if clean_word and clean_word not in vocab:
                        corrected = correct_nonword(clean_word, vocab, unigram_counts, edit_distance_1, sym_delete)
                        if corrected != clean_word:
                            alert_key = f"[SPELL-ALERT] '{clean_word}'"
                            if alert_key not in st.session_state.alerts:
                                st.session_state.spelling_count += 1
                                st.session_state.alerts.append(f"[SPELL-ALERT] '{clean_word}' -> '{corrected}'")
                
                # CASE-ALERT: Check for lowercase 'i' that should be capitalized 'I'
                for word in words:
                    if word == 'i':
                        alert_key = "[CASE-ALERT] 'i' -> 'I'"
                        if alert_key not in st.session_state.alerts:
                            st.session_state.spelling_count += 1
                            st.session_state.alerts.append("[CASE-ALERT] 'i' -> 'I' (capitalize first-person pronoun)")
                
                # GRAMMAR-ALERT: Check every N words (triggered on SPACE press)
                N = Q4_CONFIG["trigger_interval"]
                word_count = len([w for w in words if w.isalpha()])
                if word_count >= N and word_count % N == 0:
                    window = words[-N:]
                    pcfg_result, ptb_tags, ngram_perp = pos_tag_and_parse(pcfg, pos_tagger, window, ngram_model)
                    if pcfg_result:
                        pcfg_score = math.log(max(pcfg_result[0], 1e-300))
                        if pcfg_score < Q4_CONFIG["pcfg_parse_threshold"]:
                            alert_msg = f"[GRAMMAR-ALERT] Low PCFG probability: {pcfg_score:.2f}"
                            if alert_msg not in st.session_state.alerts:
                                st.session_state.alerts.append(alert_msg)
                    elif ngram_perp and ngram_perp > Q4_CONFIG["perplexity_high"]:
                        alert_msg = f"[GRAMMAR-ALERT] High trigram perplexity: {ngram_perp:.1f}"
                        if alert_msg not in st.session_state.alerts:
                            st.session_state.alerts.append(alert_msg)
                
                # Update last space count
                st.session_state.last_space_count = text.count(' ')
        
        # Initialize session state for typing
        if 'user_text' not in st.session_state:
            st.session_state.user_text = ""
        if 'last_space_count' not in st.session_state:
            st.session_state.last_space_count = 0
        
        # Custom HTML/JS component for REAL-TIME keystroke capture (SPACE detection)
        components.html(f"""
        <div style="margin: 10px 0;">
            <textarea 
                id="realtime-textarea" 
                style="width: 100%; height: 200px; padding: 10px; font-family: monospace; font-size: 14px; border: 1px solid #ccc; border-radius: 4px; resize: vertical;"
                placeholder="Start typing here... alerts appear when you press SPACE..."
                onkeydown="handleKeyDown(event)"
            ></textarea>
            <div id="status" style="margin-top: 5px; font-size: 12px; color: #666;">Ready - press SPACE to trigger alerts</div>
        </div>
        
        <script>
        let debounceTimer = null;
        
        function sendToStreamlit(text) {{
            // Update Streamlit via query params - triggers instant rerun
            const params = new URLSearchParams(window.location.search);
            params.set('user_text', text);
            params.set('trigger', Date.now().toString());
            window.location.search = params.toString();
        }}
        
        function handleKeyDown(event) {{
            if (event.key === ' ') {{
                // Get current text from textarea
                const text = event.target.value;
                sendToStreamlit(text);
            }}
        }}
        
        // Initialize with current text from URL
        const textarea = document.getElementById('realtime-textarea');
        const urlParams = new URLSearchParams(window.location.search);
        const savedText = urlParams.get('user_text');
        if (savedText) {{
            textarea.value = savedText;
        }}
        </script>
        """, height=250)
        
        # Read text from query params (set by JavaScript on SPACE press)
        query_params = st.query_params
        current_text = query_params.get('user_text', '')
        
        # Update session state when text changes
        if 'user_text' not in st.session_state:
            st.session_state.user_text = ""
        
        if current_text != st.session_state.user_text:
            st.session_state.user_text = current_text
            
            # Process alerts on the new text
            text = current_text
            words = text.split()
            
            # Count spaces in current text
            space_count = text.count(' ')
            
            # Check for new space presses (only trigger alerts on NEW space presses)
            if space_count > st.session_state.last_space_count:
                st.session_state.last_space_count = space_count
                words = st.session_state.user_text.split()
                
                # SEGMENT-ALERT: Check for merged/unknown words
                for word in words:
                    if word not in vocab and len(word) > Q4_CONFIG["merge_min_token_len"]:
                        seg_words = viterbi_segmentation(word, vocab, trigram_probs, unigram_probs)
                        if len(seg_words) > 1:
                            alert_key = f"[SEGMENT-ALERT] '{word}'"
                            if alert_key not in st.session_state.alerts:
                                st.session_state.segmentation_count += 1
                                st.session_state.alerts.append(f"[SEGMENT-ALERT] '{word}' -> {' '.join(seg_words)}")
                
                # SPELL-ALERT: Check for unknown words
                for word in words:
                    clean_word = word.strip('.,!?;:')
                    if clean_word and clean_word not in vocab:
                        corrected = correct_nonword(clean_word, vocab, unigram_counts, edit_distance_1, sym_delete)
                        if corrected != clean_word:
                            alert_key = f"[SPELL-ALERT] '{clean_word}'"
                            if alert_key not in st.session_state.alerts:
                                st.session_state.spelling_count += 1
                                st.session_state.alerts.append(f"[SPELL-ALERT] '{clean_word}' -> '{corrected}'")
                
                # CASE-ALERT: Check for lowercase 'i' that should be capitalized 'I'
                for word in words:
                    if word == 'i':
                        alert_key = "[CASE-ALERT] 'i' -> 'I'"
                        if alert_key not in st.session_state.alerts:
                            st.session_state.spelling_count += 1
                            st.session_state.alerts.append("[CASE-ALERT] 'i' -> 'I' (capitalize first-person pronoun)")
                
                # GRAMMAR-ALERT: Check every N words (triggered on SPACE press)
                N = Q4_CONFIG["trigger_interval"]
                word_count = len([w for w in words if w.isalpha()])
                if word_count >= N and word_count % N == 0:
                    window = words[-N:]
                    pcfg_result, ptb_tags, ngram_perp = pos_tag_and_parse(pcfg, pos_tagger, window, ngram_model)
                    if pcfg_result:
                        pcfg_score = math.log(max(pcfg_result[0], 1e-300))
                        if pcfg_score < Q4_CONFIG["pcfg_parse_threshold"]:
                            alert_msg = f"[GRAMMAR-ALERT] Low PCFG probability: {pcfg_score:.2f}"
                            if alert_msg not in st.session_state.alerts:
                                st.session_state.alerts.append(alert_msg)
                    elif ngram_perp and ngram_perp > Q4_CONFIG["perplexity_high"]:
                        alert_msg = f"[GRAMMAR-ALERT] High trigram perplexity: {ngram_perp:.1f}"
                        if alert_msg not in st.session_state.alerts:
                            st.session_state.alerts.append(alert_msg)
                
                # Update last space count
                st.session_state.last_space_count = text.count(' ')
        
        # Initialize session state for typing
        if 'user_text' not in st.session_state:
            st.session_state.user_text = ""
        if 'last_space_count' not in st.session_state:
            st.session_state.last_space_count = 0
        
        # Display live stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Words", len(st.session_state.user_text.split()))
        with col2:
            st.metric("Seg Alerts", st.session_state.segmentation_count)
        with col3:
            st.metric("Spell Alerts", st.session_state.spelling_count)
        
        # Show alerts
        if show_alerts and st.session_state.alerts:
            st.write("### Live Alerts")
            for alert in st.session_state.alerts[-10:]:
                st.warning(alert)
        
        # Display the text area (read-only display, actual input is in JS component)
        st.text_area(
            "Start typing here... alerts appear when you press SPACE (like Google search)",
            value=st.session_state.get('user_text', ''),
            height=200,
            key="typing_display",
            disabled=True
        )
        
        # DO NOT st.stop() - allow rest of UI to render
    # Streamlit-native manual typing. The same incremental pipeline is used
    # for simulated typing and manual input, so grammar alerts cannot be
    # skipped merely because segmentation did not fire.
    if st.session_state.get('user_typing_mode', False):
        st.write("### Manual Live Typing (User Input)")
        st.caption("Alerts are checked as the text widget updates; finish a word with a space to trigger its live checks.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Stop Typing Mode", key="stop_manual_typing"):
                st.session_state.user_typing_mode = False
                st.query_params.clear()
                st.rerun()
        with col2:
            show_alerts = st.checkbox("Show Alerts", value=True, key="show_manual_alerts")

        current_text = LIVE_TEXT_COMPONENT(
            value=st.session_state.manual_input,
            key="manual_live_component",
            default=st.session_state.manual_input,
        )
        if current_text is None:
            current_text = st.session_state.manual_input
        if current_text != st.session_state.manual_input:
            st.session_state.manual_input = current_text

        if current_text != st.session_state.manual_last_text:
            result = analyze_live_text(
                current_text,
                vocab,
                trigram_probs,
                unigram_probs,
                unigram_counts,
                bigram_probs,
                edit_distance_1,
                sym_delete,
                pos_tagger,
                pcfg,
                ngram_model,
                Q4_CONFIG,
                st.session_state.alert_keys,
            )
            st.session_state.alerts.extend(result["alerts"])
            st.session_state.segmentation_count += result["segmentation_count"]
            st.session_state.spelling_count += result["spelling_count"]
            st.session_state.seg_latencies.extend(result["token_latencies"])
            st.session_state.grammar_latencies.extend(result["grammar_latencies"])
            st.session_state.manual_corrected_text = " ".join(result["corrected_words"])
            st.session_state.processed = result["corrected_words"]
            st.session_state.manual_last_text = current_text

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Words", len(current_text.split()))
        with col2:
            st.metric("Seg Alerts", st.session_state.segmentation_count)
        with col3:
            st.metric("Spell Alerts", st.session_state.spelling_count)
        with col4:
            grammar_alerts = sum("[GRAMMAR-ALERT]" in alert for alert in st.session_state.alerts)
            st.metric("Grammar Alerts", grammar_alerts)

        if show_alerts and st.session_state.alerts:
            st.write("### Live Alerts")
            for alert in st.session_state.alerts[-10:]:
                st.warning(alert)
        if st.session_state.get("manual_corrected_text"):
            st.write("### Corrected Live Stream")
            st.info(st.session_state.manual_corrected_text)

    # Live processing
    if False and st.session_state.get('run_live', False):
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
                        pcfg_score = math.log(max(pcfg_result[0], 1e-300))
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
    
    # Corrected simulation path using the shared incremental checker.
    if st.session_state.get('run_live', False):
        tokens = st.session_state.tokens
        merged_tokens = introduce_merges(tokens, Q4_CONFIG["merge_probability"])
        progress_bar = st.progress(0)
        status_text = st.empty()
        alert_container = st.empty()
        raw_processed = []
        processed = []
        alerts = []
        seen_keys = st.session_state.alert_keys
        sentence_seg_counts = {}
        sentence_spell_counts = {}

        for index, token in enumerate(merged_tokens):
            raw_processed.append(token)
            progress_bar.progress((index + 1) / max(len(merged_tokens), 1))
            status_text.text(f"Processing token {index + 1}/{len(merged_tokens)}: {token}")
            result = analyze_live_text(
                " ".join(raw_processed),
                vocab,
                trigram_probs,
                unigram_probs,
                unigram_counts,
                bigram_probs,
                edit_distance_1,
                sym_delete,
                pos_tagger,
                pcfg,
                ngram_model,
                Q4_CONFIG,
                seen_keys,
            )
            processed = result["corrected_words"]
            alerts.extend(result["alerts"])
            st.session_state.segmentation_count += result["segmentation_count"]
            st.session_state.spelling_count += result["spelling_count"]
            st.session_state.seg_latencies.extend(result["token_latencies"])
            st.session_state.grammar_latencies.extend(result["grammar_latencies"])
            if result["alerts"]:
                alert_container.write(f"Latest alerts: {result['alerts'][-3:]}")

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
                    pos_tagger, pcfg, ngram_model, models['words']
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
                pcfg_score = (
                    math.log(max(pcfg_result[0], 1e-300))
                    if pcfg_result else float('-inf')
                )
                
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
