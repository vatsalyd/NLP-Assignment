from typing import List, Dict, Tuple
from q2_dependency_parser.conllu_parser import Sentence
from q2_dependency_parser.transition_system import (
    Configuration,
    Transition,
    TransitionType,
    is_projective,
    simulate_oracle,
)


def _pos(sentence: Sentence, token_id: int) -> str:
    """Return a POS value for a token id, including the artificial ROOT."""
    if token_id == 0:
        return 'ROOT'
    return sentence.tokens[token_id - 1].upos

def extract_features(config: Configuration, sentence: Sentence) -> Dict[str, str]:
    features = {}
    
    features['stack_top_pos'] = _pos(sentence, config.stack[-1]) if config.stack else 'NIL'
    
    if len(config.stack) >= 2:
        features['stack_second_pos'] = _pos(sentence, config.stack[-2])
    else:
        features['stack_second_pos'] = 'NIL'
    
    if config.buffer:
        features['buffer_first_pos'] = _pos(sentence, config.buffer[0])
    else:
        features['buffer_first_pos'] = 'NIL'
    
    if len(config.buffer) >= 2:
        features['buffer_second_pos'] = _pos(sentence, config.buffer[1])
    else:
        features['buffer_second_pos'] = 'NIL'
    
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
    skipped_nonprojective = 0
    for i, sentence in enumerate(sentences):
        if i % 100 == 0:
            print(f"Processing sentence {i}/{len(sentences)}")
        if not is_projective(sentence):
            skipped_nonprojective += 1
            continue
        instances = simulate_oracle(sentence)
        for config, trans in instances:
            features = extract_features(config, sentence)
            X.append(features)
            y.append(transition_to_label(trans))
    if skipped_nonprojective:
        print(f"Skipped {skipped_nonprojective} non-projective sentences (unsupported by arc-standard transitions)")
    return X, y
