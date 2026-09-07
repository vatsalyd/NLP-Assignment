import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from q1_segmentation_pos.corpus_loader import load_brown_corpus, load_ud_spanish, extract_words_tags, build_vocabulary
from q1_segmentation_pos.trigram_lm import TrigramLanguageModel, viterbi_segmentation
from q1_segmentation_pos.pos_tagger import POSTagger, load_ud_spanish_morph
from q1_segmentation_pos.baselines import GreedyLongestMatchSegmenter, MostFrequentTagger
from q1_segmentation_pos.evaluation import compute_accuracy, compute_confusion_matrix, print_confusion_matrix, error_source_analysis, evaluate_segmentation
from model_utils import get_or_train, CHECKPOINT_DIR
from config import Q1_CONFIG, UD_SPANISH_TRAIN, UD_SPANISH_DEV, UD_SPANISH_TEST

FORCE_RETRAIN = False

def train_english_models():
    train_sents, test_sents = load_brown_corpus(Q1_CONFIG["brown_train_split"])
    train_words, train_tags = extract_words_tags(train_sents)
    
    vocab = build_vocabulary(train_words)
    lm = TrigramLanguageModel(vocab)
    lm.train(train_words)
    
    tags = sorted(set(train_tags))
    pos_tagger = POSTagger(tags)
    pos_tagger.train(train_sents)
    
    return {
        'vocab': vocab, 'lm': lm, 'pos_tagger': pos_tagger,
        'train_sents': train_sents, 'test_sents': test_sents,
        'train_words': train_words, 'train_tags': train_tags,
        'tags': tags
    }

def train_spanish_models():
    train_sents, dev_sents, test_sents = load_ud_spanish(
        UD_SPANISH_TRAIN, UD_SPANISH_DEV, UD_SPANISH_TEST
    )
    train_words, train_tags = extract_words_tags(train_sents)
    test_path = "D:/projects/NLP-Assignment/data/UD_Spanish-GSD/es_gsd-ud-test.conllu"
    
    train_sents, dev_sents, test_sents = load_ud_spanish(train_path, dev_path, test_path)
    train_words, train_tags = extract_words_tags(train_sents)
    
    vocab = build_vocabulary(train_words)
    lm = TrigramLanguageModel(vocab)
    lm.train(train_words)
    
    tags = sorted(set(train_tags))
    pos_tagger = POSTagger(tags)
    pos_tagger.train(train_sents)
    
    return {
        'vocab': vocab, 'lm': lm, 'pos_tagger': pos_tagger,
        'train_sents': train_sents, 'dev_sents': dev_sents, 'test_sents': test_sents,
        'train_words': train_words, 'train_tags': train_tags,
        'tags': tags
    }

def train_spanish_morph_models():
    train_sents, dev_sents, test_sents = load_ud_spanish_morph(
        UD_SPANISH_TRAIN, UD_SPANISH_DEV, UD_SPANISH_TEST
    )
    train_words, train_tags = extract_words_tags(train_sents)
    
    vocab = build_vocabulary(train_words)
    lm = TrigramLanguageModel(vocab)
    lm.train(train_words)
    
    tags = sorted(set(train_tags))
    pos_tagger = POSTagger(tags)
    pos_tagger.train(train_sents)
    
    return {
        'vocab': vocab, 'lm': lm, 'pos_tagger': pos_tagger,
        'train_sents': train_sents, 'dev_sents': dev_sents, 'test_sents': test_sents,
        'train_words': train_words, 'train_tags': train_tags,
        'tags': tags
    }

def run_english():
    print("=" * 60)
    print("ENGLISH - Question 1")
    print("=" * 60)
    
    models = get_or_train(
        "q1_english_models",
        train_english_models,
        force_retrain=FORCE_RETRAIN
    )
    
    vocab = models['vocab']
    lm = models['lm']
    pos_tagger = models['pos_tagger']
    train_sents = models['train_sents']
    test_sents = models['test_sents']
    train_words = models['train_words']
    train_tags = models['train_tags']
    tags = models['tags']
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")
    
    print("\n--- Sample Test Strings ---")
    test_strings = [
        "thequickbrownfoxjumpsoverthelazydog",
    ]
    
    for text in test_strings:
        print(f"\nInput: {text}")
        seg_words = viterbi_segmentation(text, vocab, lm)
        print(f"Segmented: {seg_words}")
        pos_tags = pos_tagger.viterbi_decode(seg_words)
        print(f"POS tags: {list(zip(seg_words, pos_tags))}")
    
    eval_size = Q1_CONFIG["eval_sample_size"]
    
    all_pred_words = []
    all_gold_words = []
    all_pred_tags = []
    all_gold_tags = []
    
    for sent in test_sents[:eval_size]:
        gold_words = [w.lower() for w, _ in sent]
        gold_tags = [t for _, t in sent]
        text = ''.join(gold_words)
        
        pred_words = viterbi_segmentation(text, vocab, lm)
        pred_tags = pos_tagger.viterbi_decode(pred_words)
        
        all_pred_words.extend(pred_words)
        all_gold_words.extend(gold_words)
        all_pred_tags.extend(pred_tags)
        all_gold_tags.extend(gold_tags[:len(pred_tags)])
    
    seg_acc = evaluate_segmentation(all_pred_words, all_gold_words)
    tag_acc = compute_accuracy(all_pred_tags, all_gold_tags[:len(all_pred_tags)])
    print(f"Segmentation Accuracy: {seg_acc:.4f}")
    print(f"POS Tagging Accuracy: {tag_acc:.4f}")
    
    cm, tags_list = compute_confusion_matrix(all_pred_tags, all_gold_tags[:len(all_pred_tags)], tags)
    print_confusion_matrix(cm, tags_list)
    
    seg_err, tag_err = error_source_analysis(all_pred_words, all_pred_tags, all_gold_words, all_gold_tags)
    print(f"\nSegmentation-induced errors: {seg_err}")
    print(f"Genuine tagging errors: {tag_err}")
    
    print("\n--- Baselines ---")
    greedy_seg = GreedyLongestMatchSegmenter(vocab, Q1_CONFIG["baseline_greedy_max_len"])
    mft_tagger = MostFrequentTagger()
    mft_tagger.train(train_sents)
    
    baseline_pred_words = []
    baseline_pred_tags = []
    for sent in test_sents[:eval_size]:
        gold_words = [w.lower() for w, _ in sent]
        gold_tags = [t for _, t in sent]
        text = ''.join(gold_words)
        
        pred_words = greedy_seg.segment(text)
        pred_tags = mft_tagger.tag(pred_words)
        
        baseline_pred_words.extend(pred_words)
        baseline_pred_tags.extend(pred_tags)
    
    baseline_seg_acc = evaluate_segmentation(baseline_pred_words, all_gold_words)
    baseline_tag_acc = compute_accuracy(baseline_pred_tags, all_gold_tags[:len(baseline_pred_tags)])
    print(f"Baseline Segmentation Accuracy: {baseline_seg_acc:.4f}")
    print(f"Baseline POS Tagging Accuracy: {baseline_tag_acc:.4f}")
    print(f"Improvement (Seg): {seg_acc - baseline_seg_acc:.4f}")
    print(f"Improvement (Tag): {tag_acc - baseline_tag_acc:.4f}")

def run_spanish():
    print("\n" + "=" * 60)
    print("SPANISH - Question 1")
    print("=" * 60)
    
    models = get_or_train(
        "q1_spanish_models",
        train_spanish_models,
        force_retrain=FORCE_RETRAIN
    )
    
    vocab = models['vocab']
    lm = models['lm']
    pos_tagger = models['pos_tagger']
    train_sents = models['train_sents']
    dev_sents = models['dev_sents']
    test_sents = models['test_sents']
    train_words = models['train_words']
    train_tags = models['train_tags']
    tags = models['tags']
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train sentences: {len(train_sents)}")
    
    print("\n--- Sample Test Strings ---")
    test_strings = [
        "mispadrespuedenviajar",
        "elcielodespejadoesazul",
    ]
    
    for text in test_strings:
        print(f"\nInput: {text}")
        seg_words = viterbi_segmentation(text, vocab, lm)
        print(f"Segmented: {seg_words}")
        pos_tags = pos_tagger.viterbi_decode(seg_words)
        print(f"POS tags: {list(zip(seg_words, pos_tags))}")
    
    eval_size = Q1_CONFIG["eval_sample_size"]
    
    all_pred_words = []
    all_gold_words = []
    all_pred_tags = []
    all_gold_tags = []
    
    for sent in dev_sents[:eval_size]:
        gold_words = [w.lower() for w, _ in sent]
        gold_tags = [t for _, t in sent]
        text = ''.join(gold_words)
        
        pred_words = viterbi_segmentation(text, vocab, lm)
        pred_tags = pos_tagger.viterbi_decode(pred_words)
        
        all_pred_words.extend(pred_words)
        all_gold_words.extend(gold_words)
        all_pred_tags.extend(pred_tags)
        all_gold_tags.extend(gold_tags[:len(pred_tags)])
    
    seg_acc = evaluate_segmentation(all_pred_words, all_gold_words)
    tag_acc = compute_accuracy(all_pred_tags, all_gold_tags[:len(all_pred_tags)])
    print(f"Segmentation Accuracy: {seg_acc:.4f}")
    print(f"POS Tagging Accuracy: {tag_acc:.4f}")
    
    cm, tags_list = compute_confusion_matrix(all_pred_tags, all_gold_tags[:len(all_pred_tags)], tags)
    print_confusion_matrix(cm, tags_list)
    
    seg_err, tag_err = error_source_analysis(all_pred_words, all_pred_tags, all_gold_words, all_gold_tags)
    print(f"\nSegmentation-induced errors: {seg_err}")
    print(f"Genuine tagging errors: {tag_err}")
    
    print("\n--- Baselines ---")
    greedy_seg = GreedyLongestMatchSegmenter(vocab, Q1_CONFIG["baseline_greedy_max_len"])
    mft_tagger = MostFrequentTagger()
    mft_tagger.train(train_sents)
    
    baseline_pred_words = []
    baseline_pred_tags = []
    for sent in dev_sents[:eval_size]:
        gold_words = [w.lower() for w, _ in sent]
        gold_tags = [t for _, t in sent]
        text = ''.join(gold_words)
        
        pred_words = greedy_seg.segment(text)
        pred_tags = mft_tagger.tag(pred_words)
        
        baseline_pred_words.extend(pred_words)
        baseline_pred_tags.extend(pred_tags)
    
    baseline_seg_acc = evaluate_segmentation(baseline_pred_words, all_gold_words)
    baseline_tag_acc = compute_accuracy(baseline_pred_tags, all_gold_tags[:len(baseline_pred_tags)])
    print(f"Baseline Segmentation Accuracy: {baseline_seg_acc:.4f}")
    print(f"Baseline POS Tagging Accuracy: {baseline_tag_acc:.4f}")
    print(f"Improvement (Seg): {seg_acc - baseline_seg_acc:.4f}")
    print(f"Improvement (Tag): {tag_acc - baseline_tag_acc:.4f}")

def run_spanish_morph():
    print("\n" + "=" * 60)
    print("SPANISH MORPHOLOGY-AWARE - Question 1 Part 3")
    print("=" * 60)
    
    models = get_or_train(
        "q1_spanish_morph_models",
        train_spanish_morph_models,
        force_retrain=FORCE_RETRAIN
    )
    
    vocab = models['vocab']
    lm = models['lm']
    pos_tagger = models['pos_tagger']
    train_sents = models['train_sents']
    dev_sents = models['dev_sents']
    test_sents = models['test_sents']
    train_words = models['train_words']
    train_tags = models['train_tags']
    tags = models['tags']
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Unique tags: {len(set(train_tags))}")
    
    print("\n--- Sample Test Strings ---")
    test_strings = [
        "mispadrespuedenviajar",
        "elcielodespejadoesazul",
    ]
    
    for text in test_strings:
        print(f"\nInput: {text}")
        seg_words = viterbi_segmentation(text, vocab, lm)
        print(f"Segmented: {seg_words}")
        pos_tags = pos_tagger.viterbi_decode(seg_words)
        print(f"POS tags: {list(zip(seg_words, pos_tags))}")
    
    eval_size = Q1_CONFIG["eval_sample_size"]
    
    all_pred_words = []
    all_gold_words = []
    all_pred_tags = []
    all_gold_tags = []
    
    for sent in dev_sents[:eval_size]:
        gold_words = [w.lower() for w, _ in sent]
        gold_tags = [t for _, t in sent]
        text = ''.join(gold_words)
        
        pred_words = viterbi_segmentation(text, vocab, lm)
        pred_tags = pos_tagger.viterbi_decode(pred_words)
        
        all_pred_words.extend(pred_words)
        all_gold_words.extend(gold_words)
        all_pred_tags.extend(pred_tags)
        all_gold_tags.extend(gold_tags[:len(pred_tags)])
    
    seg_acc = evaluate_segmentation(all_pred_words, all_gold_words)
    tag_acc = compute_accuracy(all_pred_tags, all_gold_tags[:len(all_pred_tags)])
    print(f"Segmentation Accuracy: {seg_acc:.4f}")
    print(f"POS Tagging Accuracy: {tag_acc:.4f}")

if __name__ == "__main__":
    run_english()
    run_spanish()
    run_spanish_morph()
    run_spanish_morph()
    run_spanish_morph()