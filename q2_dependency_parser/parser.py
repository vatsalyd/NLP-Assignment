from typing import List, Dict, Tuple, Set
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from model_utils import get_or_train, CHECKPOINT_DIR
from config import Q2_CONFIG, UD_ENGLISH_TRAIN, UD_ENGLISH_DEV
from q2_dependency_parser.conllu_parser import Sentence, Token, parse_conllu, get_gold_arcs
from q2_dependency_parser.transition_system import (
    Configuration,
    Transition,
    TransitionType,
    apply_transition,
    is_transition_legal,
)
from q2_dependency_parser.features import extract_features, transition_to_label, label_to_transition, prepare_training_data

CHECKPOINT_NAME = "q2_models_arcstandard_v2"
FORCE_RETRAIN = False

class DependencyParser:
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)
        lr_config = Q2_CONFIG["logistic_regression"]
        self.classifier = LogisticRegression(
            max_iter=lr_config["max_iter"],
            C=lr_config["C"],
            solver=lr_config["solver"]
        )
        self.classes_ = None
    
    def train(self, sentences: List[Sentence]):
        X, y = prepare_training_data(sentences)
        X_vec = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vec, y)
        self.classes_ = self.classifier.classes_
        print(f"Training completed. Classes: {len(self.classes_)}")
        print(f"Training instances: {len(X)}")
    
    def predict_transition(self, config: Configuration, sentence: Sentence) -> Transition:
        features = extract_features(config, sentence)
        X = self.vectorizer.transform([features])
        # Greedy classification is retained, but only structurally legal
        # transitions are allowed to mutate the configuration.
        probabilities = self.classifier.predict_proba(X)[0]
        for class_index in np.argsort(probabilities)[::-1]:
            candidate = label_to_transition(self.classifier.classes_[class_index])
            if is_transition_legal(config, candidate):
                return candidate
        raise RuntimeError("Classifier produced no legal transition")
    
    def parse(self, sentence: Sentence) -> List[Tuple[int, int, str]]:
        config = Configuration(
            stack=[0],
            buffer=[token.id for token in sentence.tokens],
            arcs=[]
        )

        max_steps = 2 * len(sentence.tokens) + 1
        steps = 0
        while config.buffer or len(config.stack) > 1:
            if steps >= max_steps:
                raise RuntimeError("Parser exceeded the maximum arc-standard steps")
            trans = self.predict_transition(config, sentence)
            config = apply_transition(config, trans, sentence)
            steps += 1

        return config.arcs

def compute_las(predicted_arcs: List[Tuple[int, int, str]], gold_arcs: List[Tuple[int, int, str]]) -> float:
    predicted_by_dependent = {dependent: (head, label) for head, dependent, label in predicted_arcs}
    if not gold_arcs:
        return 1.0
    correct = sum(
        predicted_by_dependent.get(dependent) == (head, label)
        for head, dependent, label in gold_arcs
    )
    total = len(gold_arcs)
    return correct / total

def evaluate_on_dev(parser: DependencyParser, dev_sentences, n=None):
    if n is None:
        n = len(dev_sentences)
    print(f"\nEvaluating on {n} dev sentences...")
    total_correct = 0
    total_tokens = 0
    
    for i, sentence in enumerate(dev_sentences[:n]):
        pred_arcs = parser.parse(sentence)
        gold_arcs = get_gold_arcs(sentence)
        las = compute_las(pred_arcs, gold_arcs)
        pred_by_dep = {dependent: (head, label) for head, dependent, label in pred_arcs}
        total_correct += sum(
            pred_by_dep.get(dependent) == (head, label)
            for head, dependent, label in gold_arcs
        )
        total_tokens += len(gold_arcs)
        if i < 3:
            print(f"  Sentence: {sentence.text}")
            print(f"  Pred: {pred_arcs[:5]}...")
            print(f"  Gold: {gold_arcs[:5]}...")
            print(f"  LAS: {las:.4f}")
    
    corpus_las = total_correct / total_tokens if total_tokens else 0.0
    print(f"Corpus LAS on {total_tokens} tokens: {corpus_las:.4f}")
    return corpus_las

def train_q2_models():
    print("Loading training data...")
    train_sentences = parse_conllu(UD_ENGLISH_TRAIN)
    print(f"Loaded {len(train_sentences)} training sentences")
    
    print("Loading dev data...")
    dev_sentences = parse_conllu(UD_ENGLISH_DEV)
    print(f"Loaded {len(dev_sentences)} dev sentences")
    
    print("Training parser...")
    parser = DependencyParser()
    train_limit = Q2_CONFIG.get("train_subset_size")
    parser.train(train_sentences if train_limit is None else train_sentences[:train_limit])
    
    print("Evaluating on dev set...")
    las = evaluate_on_dev(parser, dev_sentences)
    print(f"LAS on dev set: {las:.4f}")
    
    return {
        'parser': parser,
        'dev_sentences': dev_sentences,
        'las': las
    }


def make_demo_sentence(words: List[str], upos_tags: List[str]) -> Sentence:
    """Build a sentence for the three assignment examples."""
    tokens = [
        Token(i, word, "_", upos, "_", "_", 0, "_", "_", "_")
        for i, (word, upos) in enumerate(zip(words, upos_tags), start=1)
    ]
    return Sentence(tokens=tokens, text=" ".join(words))


def run_demo_sentences(parser: DependencyParser):
    demos = [
        make_demo_sentence(
            ["The", "cat", "sat", "on", "the", "mat", "."],
            ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
        ),
        make_demo_sentence(
            ["She", "eats", "a", "green", "salad", "."],
            ["PRON", "VERB", "DET", "ADJ", "NOUN", "PUNCT"],
        ),
        make_demo_sentence(
            ["I", "saw", "the", "man", "with", "a", "telescope", "."],
            ["PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
        ),
    ]
    print("\nDemo sentence predictions:")
    for sentence in demos:
        print(f"  {sentence.text}")
        print(f"    {parser.parse(sentence)}")

if __name__ == "__main__":
    # Alias the script module before checkpointing so the pickle can also be
    # loaded by importing q2_dependency_parser.parser later.
    if __name__ == "__main__":
        sys.modules.setdefault("q2_dependency_parser.parser", sys.modules[__name__])
        DependencyParser.__module__ = "q2_dependency_parser.parser"
    models = get_or_train(CHECKPOINT_NAME, train_q2_models, force_retrain=FORCE_RETRAIN)
    print(f"LAS on dev set: {models['las']:.4f}")
    run_demo_sentences(models['parser'])
