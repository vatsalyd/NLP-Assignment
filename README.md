# NLP Group Assignment

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/vatsalyd/NLP-Assignment.git
cd NLP-Assignment

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('brown'); nltk.download('treebank'); nltk.download('gutenberg'); nltk.download('reuters'); nltk.download('universal_tagset'); nltk.download('punkt')"
```

## Requirements

Create `requirements.txt`:
```
streamlit
nltk
scikit-learn
numpy
pandas
```

## Running Each Question

### Q1: Word Segmentation & POS Tagging
```bash
cd q1_segmentation_pos
python main.py
```
Outputs: Segmentation/POS accuracy, confusion matrix, error analysis, Spanish morphology-aware tagging

### Q2: Transition-Based Dependency Parser
```bash
cd q2_dependency_parser
python parser.py
```
Outputs: LAS score (30.3% on 100 dev sentences), sample predictions, training statistics

### Q3: Spelling Corrector
```bash
cd q3_spelling_corrector
python spelling_corrector.py
```
Outputs: Method A vs B comparison, non-word/real-word accuracy, Speed Demon benchmark (1000 words)

Interactive CLI:
```bash
python cli.py
```

### Q4: Integrated Background Editor (Streamlit App)
```bash
cd q4_integrated_editor
streamlit run app.py
```
Then open http://localhost:8501 in browser

Features:
- Live typing simulation with merged-token generation (p=0.08)
- Three alert types: [SEGMENT-ALERT], [SPELL-ALERT], [GRAMMAR-ALERT]
- Trigger interval N=10 words for grammar/real-word checks
- PCFG constituency parsing with n-gram fallback
- Per-sentence comparison table with decision rule
- Speed Demon benchmark (1000 words)
- Comparative analysis summary

## Model Checkpoints

All trained models are auto-saved to `./checkpoints/`:
- `q1_english_models.pkl`
- `q1_spanish_models.pkl`
- `q1_spanish_morph_models.pkl`
- `q2_models.pkl`
- `q3_models.pkl`

Subsequent runs load from checkpoints automatically.

## Configuration

All hyperparameters in `config.py`:
- `Q1_CONFIG` - Segmentation/POS settings
- `Q2_CONFIG` - Dependency parser settings
- `Q3_CONFIG` - Spelling corrector settings
- `Q4_CONFIG` - Integrated editor settings
- `NGRAM_CONFIG` - Shared add-k smoothing (k=0.1)

## Reports

- `Q1_REPORT.md` - Segmentation/POS results & analysis
- `Q2_REPORT.md` - Dependency parser LAS & error analysis
- `COMPARATIVE_ANALYSIS_REPORT.md` - Full integration analysis with 2 sample runs

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
```