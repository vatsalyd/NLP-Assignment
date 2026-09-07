from typing import List, Dict, Tuple
from q2_dependency_parser.conllu_parser import Sentence
from q2_dependency_parser.transition_system import Configuration, Transition, TransitionType

def extract_features(config: Configuration, sentence: Sentence) -> Dict[str, str]:
    features = {}
    
    if config.stack:
        top_idx = config.stack[-1] - 1
        if top_idx < len(sentence.tokens):
            features['stack_top_pos'] = sentence.tokens[top_idx].upos
            features['stack_top_form'] = sentence.tokens[top_idx].form.lower()
    else:
        features['stack_top_pos'] = 'NIL'
        features['stack_top_form'] = 'NIL'
    
    if len(config.stack) >= 2:
        second_idx = config.stack[-2] - 1
        if second_idx < len(sentence.tokens):
            features['stack_second_pos'] = sentence.tokens[second_idx].upos
            features['stack_second_form'] = sentence.tokens[second_idx].form.lower()
    else:
        features['stack_second_pos'] = 'NIL'
        features['stack_second_form'] = 'NIL'
    
    if config.buffer:
        first_idx = config.buffer[0] - 1
        if first_idx < len(sentence.tokens):
            features['buffer_first_pos'] = sentence.tokens[first_idx].upos
            features['buffer_first_form'] = sentence.tokens[first_idx].form.lower()
    else:
        features['buffer_first_pos'] = 'NIL'
        features['buffer_first_form'] = 'NIL'
    
    if len(config.buffer) >= 2:
        second_idx = config.buffer[1] - 1
        if second_idx < len(sentence.tokens):
            features['buffer_second_pos'] = sentence.tokens[second_idx].upos
            features['buffer_second_form'] = sentence.tokens[second_idx].form.lower()
    else:
        features['buffer_second_pos'] = 'NIL'
        features['buffer_second_form'] = 'NIL'
    
    # Distance features
    if config.stack and config.buffer:
        features['dist_top_buffer'] = str(config.buffer[0] - config.stack[-1])
    
    return features

def transition_to_label(trans: Transition) -> str:
    if trans.type == TransitionType.SHIFT:
        return 'SHIFT'
    elif trans.type == TransitionType.LEFT_ARC:
        return f'LEFT_ARC_{trans.label}' if trans.label else 'LEFT_ARC'
    elif trans.type == TransitionType.RIGHT_ARC:
        return f'RIGHT_ARC_{trans.label}' if trans.label else 'RIGHT_ARC'
    return 'SHIFT'

def label_to_transition(label: str) -> Transition:
    if label == 'SHIFT':
        return Transition(TransitionType.SHIFT)
    elif label.startswith('LEFT_ARC_'):
        return Transition(TransitionType.LEFT_ARC, label[9:])  # Remove "LEFT_ARC_"
    elif label.startswith('RIGHT_ARC_'):
        return Transition(TransitionType.RIGHT_ARC, label[10:])  # Remove "RIGHT_ARC_"
    return Transition(TransitionType.SHIFT)

def prepare_training_data(sentences: List[Sentence]):
    X = []
    y = []
    for i, sentence in enumerate(sentences):
        if i % 100 == 0:
            print(f"Processing sentence {i}/{len(sentences)}")
        instances = simulate_oracle(sentence)
        for config, trans in instances:
            features = extract_features(config, sentence)
            X.append(features)
            y.append(transition_to_label(trans))
    return X, y

from q2_dependency_parser.transition_system import simulate_oracle