# NLP Group Assignment - Contribution Split

## Team Members
1. **Vatsal Yadav**
2. **Rudra Mehul Dudhat**
3. **Mohak Arya**
4. **Kanishk Nandeshwar**
5. **Arnav Mishra**

---

## Question 1: Word Segmentation & POS Tagging — **Arnav Mishra**

| Member | Responsibility | Deliverables |
|--------|---------------|--------------|
| **Arnav Mishra** | Complete Q1 Implementation | `trigram_lm.py` - Trigram LM with add-k smoothing, Viterbi segmentation; `pos_tagger.py` - Feature-based POS tagger with trigram HMM transitions, Viterbi decoding; `corpus_loader.py` - Brown/UD Spanish loaders; `baselines.py` - Greedy longest-match & most-frequent-tag baselines; `evaluation.py` - Accuracy metrics, confusion matrix, error source analysis (98.6% cascade finding); `main.py` - Full pipeline, Spanish morphology-aware tagging (86 tags), checkpointing integration |
| **Arnav Mishra** | Q1 Documentation | `Q1_REPORT.md` - Results, error analysis, English vs Spanish comparison, morphology assessment |

---

## Question 2: Transition-Based Dependency Parser — **Kanishk Nandeshwar & Mohak Arya**

| Member | Responsibility | Deliverables |
|--------|---------------|--------------|
| **Kanishk Nandeshwar** | CoNLL-U Parser & Transition System | `conllu_parser.py` - UD format parser, sentence/token data structures; `transition_system.py` - SHIFT/LEFT-ARC/RIGHT-ARC, oracle simulator, configuration class |
| **Mohak Arya** | Features, Classifier & Parser Loop | `features.py` - POS tag features (s0, s1, b0, b1), DictVectorizer, 77,604 training instances; `parser.py` - LogisticRegression (C=1.0, lbfgs), arc-standard parsing loop, LAS evaluation (30.3%) |
| **Kanishk Nandeshwar** | Q2 Documentation | `Q2_REPORT.md` - LAS results, error analysis, design decisions, future improvements |
| **Mohak Arya** | Q2 Checkpointing & Config | Model persistence via `model_utils.py`, Q2_CONFIG in `config.py`, training subset management |

---

## Question 3: Spelling Corrector — **Rudra Mehul Dudhat**

| Member | Responsibility | Deliverables |
|--------|---------------|--------------|
| **Rudra Mehul Dudhat** | Complete Q3 Implementation | `spelling_corrector.py` - Method A (Edit Distance 1: deletes, transposes, replaces, inserts) + Method B (Symmetric Delete with precomputed deletion dictionary, 20x speedup); Non-word correction (83%), real-word correction with bigram context (13%), threshold tuning; `cli.py` - Live terminal with highlighting, Speed Demon benchmark (1000 words), latency reporting; Model persistence, Q3_CONFIG (speed_demon_num_words=1000), config-driven parameters |
| **Rudra Mehul Dudhat** | Infrastructure | `model_utils.py` - `save_model`, `load_model`, `checkpoint_exists`, `get_or_train()` |

---

## Question 4: Integrated Background Editor — **Vatsal Yadav**

| Member | Responsibility | Deliverables |
|--------|---------------|--------------|
| **Vatsal Yadav** | Complete Q4 Implementation | `models.py:train_pcfg()` - Penn Treebank induction (70K productions), `cky_parse()` - Custom binary index, binarization; `UNIVERSAL_TO_PTB` lookup table, `SmoothedNGramModel` class (add-k=0.1), bigram/trigram perplexity; `introduce_merges(p=0.08)`, three alert types: SEGMENT/SPELL/GRAMMAR, trigger interval N=10; `app.py` - Live processing, per-sentence comparison table, decision rule, latency reporting, sidebar config; Full pipeline benchmark (1000 words), integration of Q1/Q3 components, config-driven all parameters |
| **Vatsal Yadav** | Central Configuration | `config.py` - All Q1-Q4 configs, NGRAM_CONFIG, UNIVERSAL_TO_PTB |
| **Vatsal Yadav** | Comparative Documentation | `COMPARATIVE_ANALYSIS_REPORT.md` - 2 sample runs, sub-system interaction table, design justifications |

---

## Git Workflow

| Member | Branches Created | PRs Raised |
|--------|------------------|------------|
| Vatsal Yadav | `pr/infrastructure`, `pr/q4-improvements`, `pr/final-integration` | 3 |
| Rudra Mehul Dudhat | `pr/q3-improvements` | 1 |
| Mohak Arya | `pr/q2-improvements` | 1 |
| Kanishk Nandeshwar | `pr/q2-improvements` | 1 |
| Arnav Mishra | `pr/q1-improvements`, `pr/documentation` | 2 |

---

## Verification

All code verified working:
- ✅ Q1: Segmentation + POS tagging (English/Spanish) with checkpointing
- ✅ Q2: Dependency parser LAS 30.3% with checkpointing  
- ✅ Q3: Spelling corrector 1000-word Speed Demon with checkpointing
- ✅ Q4: Streamlit editor with live alerts, PCFG parsing, n-gram fallback, comparative analysis
- ✅ All hardcoded values externalized to `config.py`
- ✅ Model checkpointing via `model_utils.py` for Q1, Q2, Q3
- ✅ Reports: `Q1_REPORT.md`, `Q2_REPORT.md`, `COMPARATIVE_ANALYSIS_REPORT.md`

---

## Contribution Summary

| Member | Primary Question | Docs | Infrastructure | Approx. % |
|--------|-----------------|------|----------------|-----------|
| **Arnav Mishra** | Q1 (complete) | Q1_REPORT.md | — | ~20% |
| **Kanishk Nandeshwar** | Q2 (half) | Q2_REPORT.md | — | ~20% |
| **Mohak Arya** | Q2 (half) | — | Q2 checkpointing/config | ~20% |
| **Rudra Mehul Dudhat** | Q3 (complete) | — | model_utils.py | ~20% |
| **Vatsal Yadav** | Q4 (complete) | COMPARATIVE_ANALYSIS_REPORT.md | config.py | ~20% |

All members contributed equally (~20% each) across code, testing, documentation, and infrastructure. All design decisions, code reviews, and final integration were collaborative.