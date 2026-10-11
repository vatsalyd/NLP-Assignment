# Central configuration for NLP Assignment
# All magic numbers and paths should be defined here

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Corpus paths
UD_SPANISH_GSD_DIR = DATA_DIR / "UD_Spanish-GSD"
UD_ENGLISH_EWT_DIR = DATA_DIR / "UD_English-EWT"
UD_SPANISH_TRAIN = UD_SPANISH_GSD_DIR / "es_gsd-ud-train.conllu"
UD_SPANISH_DEV = UD_SPANISH_GSD_DIR / "es_gsd-ud-dev.conllu"
UD_SPANISH_TEST = UD_SPANISH_GSD_DIR / "es_gsd-ud-test.conllu"
UD_ENGLISH_TRAIN = UD_ENGLISH_EWT_DIR / "en_ewt-ud-train.conllu"
UD_ENGLISH_DEV = UD_ENGLISH_EWT_DIR / "en_ewt-ud-dev.conllu"

# Q1: Segmentation & POS Tagging
Q1_CONFIG = {
    # Data splits
    "brown_train_split": 0.8,
    "random_seed": 42,
    
    # Segmentation
    "max_word_len": 20,
    
    # Evaluation
    "eval_sample_size": 100,
    
    # Baselines
    "baseline_greedy_max_len": 20,
    
    # Morphology-aware tags
    "spanish_morph_gender_keys": ["Gender=Fem", "Gender=Masc"],
    "spanish_morph_number_keys": ["Number=Sing", "Number=Plur"],
}

# Q2: Dependency Parser
Q2_CONFIG = {
    # Training
    # None means use the complete official train/dev split.
    "train_subset_size": None,
    "eval_sample_size": None,
    
    # Classifier
    "logistic_regression": {
        "max_iter": 1000,
        "C": 1.0,
        "solver": "lbfgs",
    },
    
    # Oracle
    "transition_types": ["SHIFT", "LEFT_ARC", "RIGHT_ARC"],
}

# Q3: Spelling Corrector
Q3_CONFIG = {
    # Corpus
    "brown_test_split": 0.1,
    
    # Correction
    "realword_threshold": 1.5,
    "edit_distance": 1,
    
    # Speed Demon
    "speed_demon_num_words": 1000,
    "benchmark_test_slice": 100,
    
    # CLI
    "cli_exit_command": "exit",
}

# Q4: Integrated Editor
Q4_CONFIG = {
    # Typing simulation
    "merge_probability": 0.08,
    "merge_min_token_len": 5,
    
    # Grammar checking
    "trigger_interval": 3,
    "grammar_check_every_n": 3,
    
    # PCFG thresholds (log probabilities)
    "pcfg_parse_threshold": -200,
    "pcfg_grammatical_threshold": -150,
    
    # Perplexity thresholds (based on actual observed values: good ~6K, bad ~45K)
    "perplexity_high": 40000,
    "perplexity_trigram_adequate": 20000,
    "perplexity_trigram_good": 10000,
    "perplexity_bigram_good": 5000,
    
    # N-gram smoothing
    "add_k_smoothing": 0.1,
    
    # Speed Demon
    "speed_demon_num_words": 1000,
    
    # Display
    "sentence_display_max_chars": 80,
}

# Shared N-gram
NGRAM_CONFIG = {
    "add_k": 0.1,
}

# Tagset mapping
UNIVERSAL_TO_PTB = {
    'NOUN': 'NN', 'VERB': 'VB', 'ADJ': 'JJ', 'ADV': 'RB',
    'DET': 'DT', 'ADP': 'IN', 'PRON': 'PRP', 'CONJ': 'CC',
    'NUM': 'CD', 'PRT': 'RP', 'X': 'NN', '.': '.',
}
