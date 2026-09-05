from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from enum import Enum
from conllu_parser import Sentence, Token, get_gold_arcs

class TransitionType(Enum):
    SHIFT = "SHIFT"
    LEFT_ARC = "LEFT_ARC"
    RIGHT_ARC = "RIGHT_ARC"

@dataclass
class Transition:
    type: TransitionType
    label: Optional[str] = None
    
    def __str__(self):
        if self.label:
            return f"{self.type.value}({self.label})"
        return self.type.value

@dataclass
class Configuration:
    stack: List[int] = field(default_factory=list)
    buffer: List[int] = field(default_factory=list)
    arcs: List[Tuple[int, int, str]] = field(default_factory=list)
    
    def copy(self):
        return Configuration(
            stack=self.stack.copy(),
            buffer=self.buffer.copy(),
            arcs=self.arcs.copy()
        )

def is_valid_left_arc(config: Configuration, sentence: Sentence) -> bool:
    if len(config.stack) < 2:
        return False
    top = config.stack[-1]
    second = config.stack[-2]
    return sentence.tokens[second - 1].head == top

def is_valid_right_arc(config: Configuration, sentence: Sentence) -> bool:
    if len(config.stack) < 2:
        return False
    top = config.stack[-1]
    second = config.stack[-2]
    return sentence.tokens[top - 1].head == second

def apply_transition(config: Configuration, trans: Transition, sentence: Sentence) -> Configuration:
    new_config = config.copy()
    
    if trans.type == TransitionType.SHIFT:
        if new_config.buffer:
            new_config.stack.append(new_config.buffer.pop(0))
    elif trans.type == TransitionType.LEFT_ARC:
        if len(new_config.stack) >= 2:
            top = new_config.stack.pop()
            second = new_config.stack[-1]
            label = trans.label or sentence.tokens[second - 1].deprel
            new_config.arcs.append((top, second, label))
    elif trans.type == TransitionType.RIGHT_ARC:
        if len(new_config.stack) >= 2:
            top = new_config.stack[-1]
            second = new_config.stack[-2]
            new_config.stack.pop()
            label = trans.label or sentence.tokens[top - 1].deprel
            new_config.arcs.append((second, top, label))
    
    return new_config

def get_oracle_transition(config: Configuration, sentence: Sentence, gold_arcs: Set[Tuple[int, int, str]]) -> Transition:
    gold_set = set(gold_arcs)
    
    if len(config.stack) >= 2:
        top = config.stack[-1]
        second = config.stack[-2]
        
        if (top, second, sentence.tokens[second - 1].deprel) in gold_set:
            if not any(h == second and d != top for h, d, _ in gold_set):
                return Transition(TransitionType.LEFT_ARC, sentence.tokens[second - 1].deprel)
        
        if (second, top, sentence.tokens[top - 1].deprel) in gold_set:
            if not any(h == top and d != second for h, d, _ in gold_set):
                if not config.buffer or not any(h == config.buffer[0] and d == top for h, d, _ in gold_set):
                    return Transition(TransitionType.RIGHT_ARC, sentence.tokens[top - 1].deprel)
    
    if config.buffer:
        return Transition(TransitionType.SHIFT)
    elif len(config.stack) >= 2:
        # No buffer left, must do an arc
        top = config.stack[-1]
        second = config.stack[-2]
        if (top, second, sentence.tokens[second - 1].deprel) in gold_set:
            return Transition(TransitionType.LEFT_ARC, sentence.tokens[second - 1].deprel)
        return Transition(TransitionType.RIGHT_ARC, sentence.tokens[top - 1].deprel)
    
    return Transition(TransitionType.SHIFT)

def simulate_oracle(sentence: Sentence) -> List[Tuple[Configuration, Transition]]:
    gold_arcs = get_gold_arcs(sentence)
    gold_set = set(gold_arcs)
    
    config = Configuration(
        stack=[],
        buffer=[token.id for token in sentence.tokens],
        arcs=[]
    )
    
    training_instances = []
    
    while config.buffer or len(config.stack) > 1:
        trans = get_oracle_transition(config, sentence, gold_set)
        training_instances.append((config.copy(), trans))
        config = apply_transition(config, trans, sentence)
    
    return training_instances