# NLP Group Assignment

## Team Members

| Name | ID | Primary Responsibilities |
|------|-----|--------------------------|
| **Vatsal Yadav** | B24DS036 | Q4 Integrated Background Editor, PCFG Parser, Configuration (`config.py`), Comparative Analysis Report |
| **Rudra Mehul Dudhat** | B24DS506 | Q3 Spelling Corrector (Methods A & B, Symmetric Delete, CLI), Model Checkpointing (`model_utils.py`) |
| **Mohak Arya** | 12341420 | Q2 Dependency Parser (Features, Classifier, Parser Loop), Q2 Checkpointing & Config Integration |
| **Kanishk Nandeshwar** | B24DS010 | Q2 Dependency Parser (CoNLL-U Parser, Arc-Standard Transition System), Q2 Report |
| **Arnav Mishra** | 12340330 | Q1 Word Segmentation & POS Tagging (Trigram LM, Viterbi, POS Tagger, Morphology-aware), Q1 Report, Documentation |

---

## Quick Start (All Platforms)

```bash
# 1. Clone the repository
git clone https://github.com/vatsalyd/NLP-Assignment.git
cd NLP-Assignment

# 2. Create and activate virtual environment
python -m venv venv

# Windows PowerShell:
venv\Scripts\activate

# Linux/Mac/Git Bash:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (REQUIRED - run once)
python download_nltk.py

# 5. Run any question (see sections below)
```

---

## Download NLTK Data (Required)

The `download_nltk.py` script handles all platform-specific issues:

```bash
python download_nltk.py
```

**What it does:**
- Installs NLTK if missing
- Creates a local `nltk_data` folder in your home directory (avoids Windows drive errors)
- Downloads all 6 required corpora: brown, treebank, gutenberg, reuters, universal_tagset, punkt
- Auto-extracts zip files (fixes NLTK's Windows extraction bug)
- Verifies all resources are accessible
- Sets `NLTK_DATA` environment variable for the session

**If you get drive errors (WinError 1005):**
```bash
# Option 1: Set custom data directory before running
$env:NLTK_DATA = "C:\path\to\nltk_data"  # PowerShell
export NLTK_DATA=/path/to/nltk_data      # Linux/Mac
python download_nltk.py

# Option 2: Use the script's default (~nltk_data in home directory)
python download_nltk.py
```

**If download fails (network issues):**
- Re-run `python download_nltk.py` (resumes partial downloads)
- Check firewall/antivirus isn't blocking Python
- Try: `python -m nltk.downloader brown treebank gutenberg reuters universal_tagset punkt`

---

## Running Each Question

**Note:** Folder names use lowercase with underscores (e.g., `q1_segmentation_pos`)

### Q1: Word Segmentation & POS Tagging
```bash
cd q1_segmentation_pos
python main.py
```
**Outputs:** Segmentation/POS accuracy, confusion matrix, error source analysis, Spanish morphology-aware tagging (86 tags), baseline comparisons

**First run:** Trains models (~30 sec), saves checkpoints to `../checkpoints/`
**Subsequent runs:** Loads from checkpoints instantly

### Q2: Transition-Based Dependency Parser
```bash
cd q2_dependency_parser
python parser.py
```
**Outputs:** LAS score (30.3% on 100 dev sentences), sample predictions with gold comparison, training statistics

### Q3: Spelling Corrector
```bash
cd q3_spelling_corrector
python spelling_corrector.py
```
**Outputs:** Method A (Edit Distance 1) vs Method B (Symmetric Delete) comparison, non-word/real-word accuracy, Speed Demon benchmark (1000 words)

**Interactive CLI:**
```bash
python cli.py
# Type sentences to correct in real-time (type 'exit' to quit)
```

### Q4: Integrated Background Editor (Streamlit App)
```bash
cd q4_integrated_editor
streamlit run app.py
```
Opens browser at `http://localhost:8501`

**Features:**
- **Live typing simulation** with merged-token generation (p=0.08)
- **Three alert types** triggered every N=10 words:
  - `[SEGMENT-ALERT]` — Token not in vocab → Viterbi segmentation
  - `[SPELL-ALERT]` — Non-word token → Symmetric Delete correction
  - `[GRAMMAR-ALERT]` — PCFG parse + n-gram fallback
- **Real-word check** at grammar trigger using bigram probabilities
- **New Passage button** — Generates a fresh random passage for live processing
- **Per-sentence comparison table** — Click "Run Final Analysis"
- **Speed Demon Benchmark** — Click "Run Speed Demon Benchmark (1000 words)"
- **Comparative Analysis Summary** — Method distribution, verdicts, latency breakdown

---

## Model Checkpoints

All trained models auto-save to `./checkpoints/`:
| File | Description |
|------|-------------|
| `q1_english_models.pkl` | English segmentation + POS |
| `q1_spanish_models.pkl` | Spanish standard POS |
| `q1_spanish_morph_models.pkl` | Spanish morphology-aware (86 tags) |
| `q2_models.pkl` | Dependency parser |
| `q3_models.pkl` | Spelling corrector (vocab + SymmetricDelete) |
| `q4_pcfg.pkl` | PCFG parser for Q4 |

**Subsequent runs** load from checkpoints automatically (instant startup).

**If you get `_emission_vocab_size` or pickle errors:**
```bash
# Delete old checkpoints and re-run to force retraining
rm checkpoints/q1_*.pkl
cd q1_segmentation_pos && python main.py
```

---

## Spanish UD Data

Spanish UD data is included in `./data/UD_Spanish-GSD/`:
- `es_gsd-ud-train.conllu` (training)
- `es_gsd-ud-dev.conllu` (development)
- `es_gsd-ud-test.conllu` (test)

---

## Configuration

All hyperparameters in `config.py`:
| Config | Key Parameters |
|--------|----------------|
| `Q1_CONFIG` | `brown_train_split=0.8`, `max_word_len=20`, `eval_sample_size=100` |
| `Q2_CONFIG` | `train_subset_size=2000`, `eval_sample_size=100`, LR params |
| `Q3_CONFIG` | `speed_demon_num_words=1000`, `realword_threshold=1.5` |
| `Q4_CONFIG` | `merge_probability=0.08`, `trigger_interval=10`, PCFG thresholds |
| `NGRAM_CONFIG` | `add_k=0.1` (shared add-k smoothing) |

---

## Reports

| Report | Description |
|--------|-------------|
| `Q1_REPORT.md` | Segmentation/POS results, confusion matrices, English vs Spanish comparison, morphology assessment |
| `Q2_REPORT.md` | Dependency parser LAS (30.3%), error analysis, design decisions, future improvements |
| `COMPARATIVE_ANALYSIS_REPORT.md` | Full integration analysis with 2 sample runs, sub-system interaction table, design justifications |
| `CONTRIBUTION_SPLIT.md` | Team contribution breakdown by question |
| `RUN_COMMANDS.md` | Complete command reference |
| `Group Assignment 1.md` | Original assignment specification |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `AttributeError: '_emission_vocab_size'` | Delete `checkpoints/q1_*.pkl` and re-run Q1 |
| `LookupError: brown` / NLTK data missing | Run `python download_nltk.py` |
| `ModuleNotFoundError` | Ensure `pip install -r requirements.txt` succeeded |
| `NameError: train_path` (Q1 Spanish) | Fixed in latest commit - re-pull and re-run |
| `OSError: [WinError 1005] E:\nltk_data` | Run `python download_nltk.py` (sets local data dir) |
| `AttributeError: '_emission_vocab_size` on checkpoint load | Already fixed - fallback in `pos_tagger.py`; delete old checkpoints if persists |
| Spanish UD data not found | Verify `data/UD_Spanish-GSD/` exists with `.conllu` files |
| Streamlit port in use | `streamlit run app.py --server.port 8502` |
| NLTK download fails (WinError 1005) | Run `python download_nltk.py` (avoids E: drive) |

---

## Git Branches (for review)

```bash
git branch -a
# pr/infrastructure      - config.py, model_utils.py
# pr/q1-improvements     - Q1 checkpointing, reports
# pr/q2-improvements     - Q2 checkpointing, reports
# pr/q3-improvements     - Q3 1000-word benchmark
# pr/q4-improvements     - Q4 full config integration
# pr/documentation       - Q1/Q2 reports
# pr/final-integration   - All combined
# fix/nltk-and-readme    - NLTK downloader + README updates (this branch)
```

---

## Requirements

`requirements.txt`:
```
streamlit>=1.28.0
nltk>=3.8
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
```

---

## License

Academic use only - NLP Group Assignment