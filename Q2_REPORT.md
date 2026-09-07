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
- **Training Instances**: 77,604 (from 2,000 training sentences)

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
- **Training Classes**: 73 transition types
- **Training Instances**: 77,604

## 4. Evaluation Results

### Labeled Attachment Score (LAS)
| Dataset | Sentences | LAS |
|---------|-----------|-----|
| Dev (UD English-EWT) | 100 | **30.30%** |

### Sample Predictions
| Sentence | Predicted (top 5) | Gold (top 5) | LAS |
|----------|-------------------|--------------|-----|
| "From the AP comes this story" | (3,2,det), (6,5,det), (5,7,punct), (4,5,det), (2,4,root) | (3,1,case), (3,2,det), (4,3,obl), (6,5,det), (4,6,nsubj) | 33.33% |
| "President Bush on Tuesday..." | (2,1,nmod:desc), (4,3,case), (7,6,nummod), (9,8,mark), (10,8,mark) | (2,1,nmod:desc), (5,2,nsubj), (4,3,case), (5,4,obl), (7,6,nummod) | 33.33% |

## 5. Error Analysis

### Common Error Types
1. **Attachment Errors**: Wrong head assignment (most common)
2. **Label Errors**: Correct head but wrong dependency relation
3. **Long-distance Dependencies**: Difficulty with non-local attachments
3. **Coordination**: Conjunction structures poorly handled

### Root Causes
1. **Limited Features**: Only POS tags of s0, s1, b0, b1 - no lexical features, distances, or arc labels
2. **Training Subset**: Only 2,000 of ~12,000 training sentences used
3. **No Beam Search**: Greedy decoding - no exploration of alternatives
4. **Arc-Standard Limitations**: Cannot handle non-projective trees

## 6. Design Decisions

### Feature Set Choice
- **Minimal POS-only features**: Fast extraction, but limited expressivity
- **No lexical features**: Word forms not used - limits disambiguation
- **No arc labels in history**: Cannot use previously predicted arc labels

### Arc-Standard Choice
- **Pros**: Simple, efficient O(n) parsing, well-studied
- **Cons**: Cannot handle non-projective dependencies (~5-10% in English)

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
- Low LAS (30.3%) - far from state-of-the-art (~90%+)
- Only 2,000 training sentences (subset)
- Greedy decoding - no search
- No non-projective handling

## 8. Comparative Analysis

| System | Features | Training Data | LAS (Dev) |
|--------|----------|---------------|-----------|
| **Our System** | POS tags (s0,s1,b0,b1) | 2,000 sent | 30.3% |
| **Baseline (random)** | - | - | ~10% |
| **Stanford CoreNLP** | Rich features + neural | Full UD | ~88% |
| **UDPipe** | Neural + rich features | Full UD | ~85% |

## 8. Future Improvements

### Immediate (High Impact)
1. **Use full training data** (~12k sentences) - expect +15-20% LAS
2. **Add beam search** (beam width 5-10) - expect +5-10% LAS
3. **Add lexical features** (word forms of s0, s1, b0, b1) - expect +5% LAS

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
- **Cache Location**: ./checkpoints/q2_models.pkl
- **Contents**: Trained DependencyParser (vectorizer + classifier)

### Configuration
All hyperparameters externalized to config.py:
- Training subset size: 2,000
- Eval sample size: 100
- LogisticRegression: max_iter=1000, C=1.0, solver='lbfgs'

## 10. Conclusions

The implementation correctly demonstrates the core components of a transition-based dependency parser:
1. ✅ CoNLL-U parsing with proper data structures
2. ✅ Arc-standard oracle simulation
3. ✅ Feature extraction from parser configurations
4. ✅ LogisticRegression classifier training
5. ✅ Arc-standard parsing loop with greedy decoding
6. ✅ LAS evaluation metric

However, the **LAS of 30.3%** indicates significant room for improvement. The primary bottlenecks are:
1. **Training data subset** (2k vs 12k available)
2. **Minimal feature set** (POS tags only)
3. **Greedy decoding** (no search)
4. **No lexical information**

With full training data and enriched features, we expect LAS to reach 50-60%. With neural models and beam search, 75%+ should be achievable.

---
