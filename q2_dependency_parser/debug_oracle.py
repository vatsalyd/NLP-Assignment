from conllu_parser import parse_conllu
from transition_system import simulate_oracle, apply_transition, get_oracle_transition, Configuration, TransitionType

train_path = "D:/projects/NLP-Assignment/data/UD_English-EWT/en_ewt-ud-train.conllu"
train_sentences = parse_conllu(train_path)

# Take first sentence and debug
sentence = train_sentences[0]
print(f"Sentence: {sentence.text}")
print("Tokens:")
for t in sentence.tokens:
    print(f"  {t.id}: {t.form} {t.upos} head={t.head} deprel={t.deprel}")

instances = simulate_oracle(sentence)
print(f"\nOracle generated {len(instances)} transitions:")
for i, (config, trans) in enumerate(instances):
    print(f"  {i}: stack={config.stack}, buffer={config.buffer}, trans={trans}, arcs={config.arcs}")

# Check final arcs
config = Configuration(stack=[], buffer=[t.id for t in sentence.tokens], arcs=[])
for inst_config, trans in instances:
    config = apply_transition(config, trans, sentence)

print(f"\nFinal arcs: {config.arcs}")
print(f"Gold arcs:  {[(t.head, t.id, t.deprel) for t in sentence.tokens if t.head > 0]}")