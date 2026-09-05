from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

@dataclass
class Token:
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str

@dataclass
class Sentence:
    tokens: List[Token]
    text: str = ""

def parse_conllu(filepath: str) -> List[Sentence]:
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        tokens = []
        sent_text = ""
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append(Sentence(tokens=tokens, text=sent_text))
                    tokens = []
                    sent_text = ""
                continue
            if line.startswith('#'):
                if line.startswith('# text ='):
                    sent_text = line[8:].strip()
                continue
            parts = line.split('\t')
            if len(parts) >= 10 and parts[0].isdigit():
                token = Token(
                    id=int(parts[0]),
                    form=parts[1],
                    lemma=parts[2],
                    upos=parts[3],
                    xpos=parts[4],
                    feats=parts[5],
                    head=int(parts[6]),
                    deprel=parts[7],
                    deps=parts[8],
                    misc=parts[9]
                )
                tokens.append(token)
        if tokens:
            sentences.append(Sentence(tokens=tokens, text=sent_text))
    return sentences

def get_gold_arcs(sentence: Sentence) -> List[Tuple[int, int, str]]:
    arcs = []
    for token in sentence.tokens:
        if token.head > 0:
            arcs.append((token.head, token.id, token.deprel))
    return arcs