from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from enum import Enum
from q2_dependency_parser.conllu_parser import Sentence, Token, get_gold_arcs

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


def is_projective(sentence: Sentence) -> bool:
    """Check whether the gold dependency arcs are projective.

    Arc-standard parsing with only SHIFT/LEFT-ARC/RIGHT-ARC can derive
    projective trees.  UD-EWT contains a small number of non-projective
    trees, so those sentences are reported and omitted from oracle training.
    """
    arcs = [(token.head, token.id) for token in sentence.tokens]
    for i, (head_a, dep_a) in enumerate(arcs):
        left_a, right_a = sorted((head_a, dep_a))
        for head_b, dep_b in arcs[i + 1:]:
            left_b, right_b = sorted((head_b, dep_b))
            if {head_a, dep_a} & {head_b, dep_b}:
                continue
            if (left_a < left_b < right_a < right_b) or (left_b < left_a < right_b < right_a):
                return False
    return True

def is_valid_left_arc(config: Configuration, sentence: Sentence) -> bool:
    if len(config.stack) < 2:
        return False
    top = config.stack[-1]
    second = config.stack[-2]
    # ROOT must never be removed from the stack.
    return second != 0 and sentence.tokens[second - 1].head == top

def is_valid_right_arc(config: Configuration, sentence: Sentence) -> bool:
    if len(config.stack) < 2:
        return False
    top = config.stack[-1]
    second = config.stack[-2]
    return top != 0 and sentence.tokens[top - 1].head == second


def is_transition_legal(config: Configuration, trans: Transition) -> bool:
    """Return whether a transition is structurally legal in this configuration."""
    if trans.type == TransitionType.SHIFT:
        return bool(config.buffer)
    if len(config.stack) < 2:
        return False
    dependent = config.stack[-2] if trans.type == TransitionType.LEFT_ARC else config.stack[-1]
    if dependent == 0:
        return False
    # Each token may receive only one head.
    return not any(dep == dependent for _, dep, _ in config.arcs)

def apply_transition(config: Configuration, trans: Transition, sentence: Sentence) -> Configuration:
    new_config = config.copy()
    
    if trans.type == TransitionType.SHIFT:
        if not new_config.buffer:
            raise ValueError("SHIFT requires a non-empty buffer")
        new_config.stack.append(new_config.buffer.pop(0))
    elif trans.type == TransitionType.LEFT_ARC:
        if len(new_config.stack) < 2:
            raise ValueError("LEFT-ARC requires at least two stack items")
        top = new_config.stack[-1]
        second = new_config.stack[-2]
        if second == 0:
            raise ValueError("LEFT-ARC cannot remove ROOT")
        label = trans.label or sentence.tokens[second - 1].deprel
        # Arc-standard LEFT-ARC pops the dependent (the second item), not
        # the head at the top of the stack.
        new_config.stack.pop(-2)
        new_config.arcs.append((top, second, label))
    elif trans.type == TransitionType.RIGHT_ARC:
        if len(new_config.stack) < 2:
            raise ValueError("RIGHT-ARC requires at least two stack items")
        top = new_config.stack[-1]
        second = new_config.stack[-2]
        if top == 0:
            raise ValueError("RIGHT-ARC cannot remove ROOT")
        label = trans.label or sentence.tokens[top - 1].deprel
        new_config.stack.pop()
        new_config.arcs.append((second, top, label))
    
    return new_config

def get_oracle_transition(config: Configuration, sentence: Sentence, gold_arcs: Set[Tuple[int, int, str]]) -> Transition:
    gold_set = set(gold_arcs)

    def has_unattached_child(head: int, except_dependent: int) -> bool:
        return any(
            h == head and d != except_dependent and (h, d, label) not in config.arcs
            for h, d, label in gold_set
        )
    
    if len(config.stack) >= 2:
        top = config.stack[-1]
        second = config.stack[-2]
        
        if second != 0 and (top, second, sentence.tokens[second - 1].deprel) in gold_set:
            if not has_unattached_child(second, top):
                return Transition(TransitionType.LEFT_ARC, sentence.tokens[second - 1].deprel)
        
        if top != 0 and (second, top, sentence.tokens[top - 1].deprel) in gold_set:
            if not has_unattached_child(top, second):
                return Transition(TransitionType.RIGHT_ARC, sentence.tokens[top - 1].deprel)
    
    if config.buffer:
        return Transition(TransitionType.SHIFT)
    elif len(config.stack) >= 2:
        # No buffer remains; a valid projective gold tree must permit an arc.
        top = config.stack[-1]
        second = config.stack[-2]
        if second != 0 and (top, second, sentence.tokens[second - 1].deprel) in gold_set:
            return Transition(TransitionType.LEFT_ARC, sentence.tokens[second - 1].deprel)
        if top != 0 and (second, top, sentence.tokens[top - 1].deprel) in gold_set:
            return Transition(TransitionType.RIGHT_ARC, sentence.tokens[top - 1].deprel)
        raise ValueError("No valid oracle transition for this configuration")
    
    raise ValueError("No valid oracle transition for a completed configuration")

def simulate_oracle(sentence: Sentence) -> List[Tuple[Configuration, Transition]]:
    gold_arcs = get_gold_arcs(sentence)
    gold_set = set(gold_arcs)
    
    config = Configuration(
        stack=[0],
        buffer=[token.id for token in sentence.tokens],
        arcs=[]
    )
    
    training_instances = []
    
    max_steps = 2 * len(sentence.tokens) + 1
    while config.buffer or len(config.stack) > 1:
        if len(training_instances) >= max_steps:
            raise RuntimeError("Oracle exceeded the maximum arc-standard steps")
        trans = get_oracle_transition(config, sentence, gold_set)
        training_instances.append((config.copy(), trans))
        config = apply_transition(config, trans, sentence)
    
    return training_instances
