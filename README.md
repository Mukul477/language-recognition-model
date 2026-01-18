# Language Recognition Model (From Scratch)

A simple language recognition system built from scratch using classical machine learning techniques.  
The project focuses on understanding **feature engineering, model behavior, and evaluation**, rather than relying on pretrained deep learning models.

Languages supported:
- English
- French
- German

---

## Motivation

The goal of this project is to understand how far **simple models + good features** can go for language identification, and to build intuition by implementing models **from scratch** before moving to higher-level libraries.

---

## Dataset

- Custom sentence-level dataset
- ~30–40 sentences per language
- Languages: English, French, German
- Data stored as plain text files (`english.txt`, `french.txt`, `german.txt`)
- Train/test split: 80% / 20%
- Sentences are shuffled before splitting to avoid ordering bias

> Note: The dataset is intentionally small to highlight the impact of feature engineering and model choice.

---

## Feature Engineering

Each sentence is converted into a numerical feature vector.  
The following features are used:

1. **Number of words**  
2. **Average word length**  
3. **Total number of characters**  
4. **Repeated character ratio**  
   - Measures how often characters repeat within a sentence  
5. **Vowel ratio**  
   - Fraction of vowels in the sentence  
6. **Stopword ratio**  
   - Language-specific stopword frequency  

All features are **standardized (z-score normalization)** using training-set statistics to ensure stable optimization and fair distance calculations.

---

## Models Implemented

### 1. Softmax Regression (From Scratch)

- Multiclass linear classifier
- Implemented using NumPy
- Cross-entropy loss
- Trained with gradient descent
- Model parameters:
  - Weight matrix
  - Bias vector
- Supports saving and loading learned weights

### 2. k-Nearest Neighbors (From Scratch)

- Distance-based classifier
- Euclidean distance
- Majority voting among k nearest samples
- Used as a baseline to compare against softmax regression

---

## Results

| Model        | Train Accuracy | Test Accuracy |
|-------------|---------------|---------------|
| Softmax     | 100%          | ~91–100%     |
| k-NN        | ~98%          | ~88-98%      |

Additional “hard test” sentences (not seen during training) were also evaluated to test generalization.

> Due to the small dataset and strong language-specific features, high accuracy is expected. Results should not be interpreted as production-level performance.

---

## Key Learnings

- Feature engineering can significantly outperform complex models on small datasets
- Normalization is critical for gradient-based and distance-based models
- Simple baselines (k-NN) are strong and should not be ignored
- High accuracy does not always imply generalization
- Implementing models from scratch builds strong debugging and intuition skills

---

## Limitations

- Small dataset size
- Hand-engineered features may encode language-specific heuristics
- Limited language coverage
- No character n-grams or word embeddings used

---

## Future Work

- Expand dataset size and language coverage
- Compare with sklearn implementations
- Add n-gram or TF-IDF features
- Explore Decision Trees and Random Forests
- Transition to PyTorch for neural models

---

## How to Run

```bash
python -m model.train

