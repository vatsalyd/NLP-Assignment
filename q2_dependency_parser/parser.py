from typing import List, Dict, Tuple, Set
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import sys
sys.path.append('D:\\projects\\NLP-Assignment')
from model_utils import get_or_train, CHECKPOINT_DIR
from config import Q2_CONFIG, UD_ENGLISH_TRAIN, UD_ENGLISH_DEV
from conllu_parser import Sentence, parse_conllu, get_gold_arcs
from transition_system import Configuration, Transition, TransitionType, apply_transition, get_oracle_transition
from features import extract_features, transition_to_label, label_to_transition, prepare_training_data

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
        pred_label = self.classifier.predict(X)[0]
        return label_to_transition(pred_label)
    
    def parse(self, sentence: Sentence) -> List[Tuple[int, int, str]]:
        config = Configuration(
            stack=[],
            buffer=[token.id for token in sentence.tokens],
            arcs=[]
        )
        
        while config.buffer or len(config.stack) > 1:
            trans = self.predict_transition(config, sentence)
            config = apply_transition(config, trans, sentence)
        
        return config.arcs

def compute_las(predicted_arcs: List[Tuple[int, int, str]], gold_arcs: List[Tuple[int, int, str]]) -> float:
    pred_set = set(predicted_arcs)
    gold_set = set(gold_arcs)
    
    if not gold_set:
        return 1.0
    
    correct = len(pred_set & gold_set)
    total = len(gold_set)
    return correct / total

def evaluate_on_dev(parser: DependencyParser, dev_sentences, n=None):
    if n is None:
        n = Q2_CONFIG["eval_sample_size"]
    print(f"\nEvaluating on {n} dev sentences...")
    total_las = 0.0
    count = 0
    
    for i, sentence in enumerate(dev_sentences[:n]):
        pred_arcs = parser.parse(sentence)
        gold_arcs = get_gold_arcs(sentence)
        las = compute_las(pred_arcs, gold_arcs)
        total_las += las
        count += 1
        if i < 3:
            print(f"  Sentence: {sentence.text}")
            print(f"  Pred: {pred_arcs[:5]}...")
            print(f"  Gold: {gold_arcs[:5]}...")
            print(f"  LAS: {las:.4f}")
    
    avg_las = total_las / count if count > 0 else 0.0
    print(f"Average LAS on {count} dev sentences: {avg_las:.4f}")
    return avg_las

def train_q2_models():
    print("Loading training data...")
    train_sentences = parse_conllu(UD_ENGLISH_TRAIN)
    print(f"Loaded {len(train_sentences)} training sentences")
    
    print("Loading dev data...")
    dev_sentences = parse_conllu(UD_ENGLISH_DEV)
    print(f"Loaded {len(dev_sentences)} dev sentences")
    
    print("Training parser...")
    parser = DependencyParser()
    parser.train(train_sentences[:Q2_CONFIG["train_subset_size"]])
    
    print("Evaluating on dev set...")
    las = evaluate_on_dev(parser, dev_sentences)
    print(f"LAS on dev set: {las:.4f}")
    
    return {
        'parser': parser,
        'dev_sentences': dev_sentences,
        'las': las
    }

if __name__ == "__main__":
    models = get_or_train("q2_models", train_q2_models, force_retrain=FORCE_RETRAIN)
    print(f"LAS on dev set: {models['las']:.4f}")