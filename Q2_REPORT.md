# Question 2: Transition-Based Dependency Parser - Report

## Executive Summary
This report presents the implementation and evaluation of a transition-based dependency parser using the arc-standard system, trained on the Universal Dependencies English-EWT corpus.

## 1. System Overview

### Transition System: Arc-Standard
- **Data Structures**: Stack, Buffer, Arc Set
- **Transitions**:
  - **SHIFT**: Move first word from buffer to stack
  - **LEFT-ARC(label)**: Stack top becomes head of second word; second word popped
  - **RIGHT-ARC(label)**: Second word becomes head of stack top; stack top popped

### Training Data Generation
- **Oracle Simulator**: For each gold parse, simulate parsing to generate (configuration, correct_transition) pairs
- **Training Instances**: 392,242 (full training split, excluding unsupported non-projective trees)
- **Non-projective training sentences skipped**: 287

## 2. Feature Extraction
Features extracted from parser configuration:
1. POS tag of word on top of stack (s0)
2. POS tag of second word on stack (s1) - if exists
3. POS tag of first word in buffer (b0) - if exists
4. POS tag of second word in buffer (b1) - if exists

## 3. Model Training
- **Classifier**: Logistic Regression (scikit-learn)
- **Hyperparameters**: 
  - max_iter=1000
  - C=1.0
  - solver='lbfgs'
- **Vectorization**: DictVectorizer (sparse)
- **Training Classes**: 82 labeled transition types
- **Training Instances**: 392,242

## 4. Evaluation Results

### Labeled Attachment Score (LAS)
| Dataset | Sentences | LAS |
|---------|-----------|-----|
| Dev (UD English-EWT) | 2,001 | **56.92%** |

### Sample Predictions
| Sentence | Predicted (top 5) | Gold (top 5) | LAS |
|----------|-------------------|--------------|-----|
| "From the AP comes this story" | (3,2,det), (3,1,case), (4,3,nsubj), (6,5,det), (4,6,obj) | (3,1,case), (3,2,det), (4,3,obl), (0,4,root), (6,5,det) | 71.43% |
| "President Bush on Tuesday..." | (4,3,case), (5,4,nsubj), (5,2,nsubj), (5,1,nsubj), (7,6,nummod) | (2,1,nmod:desc), (5,2,nsubj), (4,3,case), (5,4,obl), (0,5,root) | 73.68% |

## 5. Error Analysis

### Common Error Types
1. **Attachment Errors**: Wrong head assignment (most common)
2. **Label Errors**: Correct head but wrong dependency relation
3. **Long-distance Dependencies**: Difficulty with non-local attachments
3. **Coordination**: Conjunction structures poorly handled

### Root Causes
1. **Limited Features**: Only POS tags of s0, s1, b0, b1 - no lexical or arc-history features
2. **Non-projective trees**: 287 training sentences cannot be represented by the required transitions
3. **No Beam Search**: Greedy decoding - no exploration of alternatives
4. **Arc-Standard Limitations**: Cannot handle non-projective trees

## 6. Design Decisions

### Feature Set Choice
- **Minimal POS-only features**: Matches the assignment specification, but limits expressivity
- **No lexical features**: Word forms not used - limits disambiguation
- **No arc labels in history**: Cannot use previously predicted arc labels

### Arc-Standard Choice
- **Pros**: Simple, efficient O(n) parsing, well-studied
- **Cons**: Cannot handle non-projective dependencies; unsupported trees are skipped during oracle training

### Classifier Choice
- **Logistic Regression**: Fast training, probabilistic output, works well with sparse features
- **Alternative**: Could use neural networks for better feature combinations

## 7. Performance Assessment

### Strengths
- Clean, modular implementation
- Correct arc-standard implementation
- Proper oracle simulation
- Reasonable baseline LAS for minimal features

### Weaknesses
- LAS (56.92%) remains below richer state-of-the-art systems
- Non-projective trees are excluded from oracle training
- Greedy decoding - no search
- No non-projective handling

## 8. Comparative Analysis

| System | Features | Training Data | LAS (Dev) |
|--------|----------|---------------|-----------|
| **Our System** | POS tags (s0,s1,b0,b1) | Full split, projective trees | 56.92% |
| **Baseline (random)** | - | - | ~10% |
| **Stanford CoreNLP** | Rich features + neural | Full UD | ~88% |
| **UDPipe** | Neural + rich features | Full UD | ~85% |

## 8. Future Improvements

### Immediate (High Impact)
1. **Add a SWAP transition** to train on non-projective trees
2. **Add beam search** (beam width 5-10)
3. **Add lexical features** only if features beyond POS are permitted

### Medium-term
4. **Add arc labels to features** (previously predicted labels)
5. **Distance features** (stack/buffer distances)
6. **Valency features** (number of left/right dependents)

### Long-term
7. **Neural parser** (BiLSTM/Transformer encoder + MLP classifier)
8. **Non-projective handling** (swap transitions or graph-based)
9. **Joint POS tagging + parsing** (multi-task learning)

## 9. Checkpointing and Reproducibility

### Model Persistence
- **Checkpointing**: Implemented via pickle (model_utils.py)
- **Cache Location**: ./checkpoints/q2_models_arcstandard_v2.pkl
- **Contents**: Trained DependencyParser (vectorizer + classifier)

### Configuration
All hyperparameters externalized to config.py:
- Training subset size: None (full split)
- Eval sample size: None (full split)
- LogisticRegression: max_iter=1000, C=1.0, solver='lbfgs'

## 10. Conclusions

The implementation demonstrates the core components of a transition-based dependency parser:
1. ✅ CoNLL-U parsing with proper data structures
2. ✅ Arc-standard oracle simulation
3. ✅ Feature extraction from parser configurations
4. ✅ LogisticRegression classifier training
5. ✅ Arc-standard parsing loop with greedy decoding
6. ✅ LAS evaluation metric

The corrected **LAS of 56.92%** is a stronger baseline. The primary bottlenecks are:
1. **Non-projective training trees** (287 are unsupported by plain arc-standard)
2. **Minimal feature set** (POS tags only)
3. **Greedy decoding** (no search)
4. **No lexical information**

Adding SWAP, beam search, or richer permitted features could improve this baseline further.

---
