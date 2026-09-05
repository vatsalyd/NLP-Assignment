import sys
sys.path.append('D:\\projects\\NLP-Assignment')

from corpus_loader import load_brown_corpus, load_ud_spanish, extract_words_tags, build_vocabulary
from trigram_lm import TrigramLanguageModel, viterbi_segmentation
from pos_tagger import POSTagger, load_ud_spanish_morph
from baselines import GreedyLongestMatchSegmenter, MostFrequentTagger
from evaluation import compute_accuracy, compute_confusion_matrix, print_confusion_matrix, error_source_analysis, evaluate_segmentation

def run_english():
    print("=" * 60)
    print("ENGLISH - Question 1")
    print("=" * 60)
    
    train_sents, test_sents = load_brown_corpus(0.8)
    train_words, train_tags = extract_words_tags(train_sents)
    test_words, test_tags = extract_words_tags(test_sents)
    
    vocab = build_vocabulary(train_words)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train sentences: {len(train_sents)}, Test sentences: {len(test_sents)}")
    
    lm = TrigramLanguageModel(vocab)
    lm.train(train_words)
    
    tags = sorted(set(train_tags))
    pos_tagger = POSTagger(tags)
    pos_tagger.train(train_sents)
    
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
    
    print("\n--- Evaluation on Test Set ---")
    all_pred_words = []
    all_gold_words = []
    all_pred_tags = []
    all_gold_tags = []
    
    for sent in test_sents[:100]:
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
    greedy_seg = GreedyLongestMatchSegmenter(vocab)
    mft_tagger = MostFrequentTagger()
    mft_tagger.train(train_sents)
    
    baseline_pred_words = []
    baseline_pred_tags = []
    for sent in test_sents[:100]:
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
    
    train_path = "D:/projects/NLP-Assignment/data/UD_Spanish-GSD/es_gsd-ud-train.conllu"
    dev_path = "D:/projects/NLP-Assignment/data/UD_Spanish-GSD/es_gsd-ud-dev.conllu"
    test_path = "D:/projects/NLP-Assignment/data/UD_Spanish-GSD/es_gsd-ud-test.conllu"
    
    train_sents, dev_sents, test_sents = load_ud_spanish(train_path, dev_path, test_path)
    train_words, train_tags = extract_words_tags(train_sents)
    
    vocab = build_vocabulary(train_words)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train sentences: {len(train_sents)}")
    
    lm = TrigramLanguageModel(vocab)
    lm.train(train_words)
    
    tags = sorted(set(train_tags))
    pos_tagger = POSTagger(tags)
    pos_tagger.train(train_sents)
    
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
    
    print("\n--- Evaluation on Dev Set ---")
    all_pred_words = []
    all_gold_words = []
    all_pred_tags = []
    all_gold_tags = []
    
    for sent in dev_sents[:100]:
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
    greedy_seg = GreedyLongestMatchSegmenter(vocab)
    mft_tagger = MostFrequentTagger()
    mft_tagger.train(train_sents)
    
    baseline_pred_words = []
    baseline_pred_tags = []
    for sent in dev_sents[:100]:
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
    
    train_path = "D:/projects/NLP-Assignment/data/UD_Spanish-GSD/es_gsd-ud-train.conllu"
    dev_path = "D:/projects/NLP-Assignment/data/UD_Spanish-GSD/es_gsd-ud-dev.conllu"
    test_path = "D:/projects/NLP-Assignment/data/UD_Spanish-GSD/es_gsd-ud-test.conllu"
    
    train_sents, dev_sents, test_sents = load_ud_spanish_morph(train_path, dev_path, test_path)
    train_words, train_tags = extract_words_tags(train_sents)
    
    vocab = build_vocabulary(train_words)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Unique tags: {len(set(train_tags))}")
    
    lm = TrigramLanguageModel(vocab)
    lm.train(train_words)
    
    tags = sorted(set(train_tags))
    pos_tagger = POSTagger(tags)
    pos_tagger.train(train_sents)
    
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
    
    print("\n--- Evaluation on Dev Set ---")
    all_pred_words = []
    all_gold_words = []
    all_pred_tags = []
    all_gold_tags = []
    
    for sent in dev_sents[:100]:
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