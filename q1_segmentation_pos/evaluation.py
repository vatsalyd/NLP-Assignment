from collections import defaultdict
import numpy as np

def compute_accuracy(pred_tags, gold_tags):
    correct = sum(1 for p, g in zip(pred_tags, gold_tags) if p == g)
    total = len(gold_tags)
    return correct / total if total > 0 else 0.0

def compute_confusion_matrix(pred_tags, gold_tags, tag_set):
    tag_to_idx = {tag: i for i, tag in enumerate(tag_set)}
    n = len(tag_set)
    cm = np.zeros((n, n), dtype=int)
    for p, g in zip(pred_tags, gold_tags):
        if p in tag_to_idx and g in tag_to_idx:
            cm[tag_to_idx[g], tag_to_idx[p]] += 1
    return cm, tag_set

def print_confusion_matrix(cm, tags):
    print("Confusion Matrix:")
    print("Actual \\ Predicted", end="")
    for tag in tags:
        print(f"  {tag[:8]:>8}", end="")
    print()
    for i, tag in enumerate(tags):
        print(f"{tag[:8]:>8}", end="")
        for j in range(len(tags)):
            print(f"  {cm[i,j]:>8}", end="")
        print()

def error_source_analysis(pred_words, pred_tags, gold_words, gold_tags):
    seg_errors = 0
    tag_errors = 0
    
    i = j = 0
    while i < len(gold_words) and j < len(pred_words):
        if pred_words[j] == gold_words[i]:
            if j < len(pred_tags) and i < len(gold_tags) and pred_tags[j] != gold_tags[i]:
                tag_errors += 1
            i += 1
            j += 1
        else:
            seg_errors += 1
            k = j + 1
            merged = pred_words[j]
            while k < len(pred_words) and merged != gold_words[i]:
                merged += pred_words[k]
                k += 1
            if merged == gold_words[i]:
                j = k
            else:
                j += 1
            i += 1
    
    seg_errors += abs(len(pred_words) - len(gold_words))
    return seg_errors, tag_errors

def evaluate_segmentation(pred_words, gold_words):
    correct = sum(1 for p, g in zip(pred_words, gold_words) if p == g)
    total = len(gold_words)
    return correct / total if total > 0 else 0.0