# Video Game Review Sentiment Analysis

Sentiment classification of 1.6 million Metacritic video game reviews using a fine-tuned DistilBERT transformer model built with TensorFlow and Hugging Face. The project compares traditional machine learning (logistic regression) against a deep learning transformer approach for multi-class sentiment classification.

## Overview

Video game reviews contain nuanced, domain-specific language that makes sentiment classification challenging. This project builds and evaluates a pipeline that processes raw review text, extracts meaningful features, and classifies sentiment as positive, neutral, or negative — providing insights that could influence game development and marketing strategies.

## Repository Structure

```
video-game-sentiment-analysis/
│
├── 0_EDA.ipynb                        # Exploratory data analysis
├── 1. Data_loading_and_filtering.ipynb # Data loading, language filtering, labeling
├── 2. Model_Selection.ipynb            # Benchmarking and model selection
├── 3. Distilbert_Sample.ipynb          # Initial DistilBERT experiments on data subset
├── 4. Logistic_Regression.ipynb        # Logistic regression baseline model
├── 5. Distilbert_Layer.ipynb           # Final DistilBERT model with custom layers
├── Evaluation_and_Metrics.ipynb        # Model evaluation and comparison
└── README.md
```

## Dataset

- **Source:** Metacritic video game reviews
- **Size:** 1.6 million reviews
- **Features:** Review text, score (0–100), platform, date, reviewer type (user/critic)
- **Languages:** Multi-language dataset; filtered to English reviews using `langdetect`
- **Target:** Sentiment — scores bucketed into `negative`, `neutral`, and `positive` categories

## Methodology

### 1. Exploratory Data Analysis
- Analyzed review score distributions across platforms and reviewer types
- Identified class imbalance between sentiment categories
- Explored review length distributions and language composition

### 2. Data Preparation

**Logistic Regression pipeline:**
- Lowercased text and removed punctuation
- Tokenized words, removed stop words, and applied lemmatization
- Applied TF-IDF vectorization for feature extraction

**DistilBERT pipeline:**
- Minimal cleaning (removed extra whitespace) — transformers handle context natively
- Tokenized using `distilbert-base-uncased` tokenizer with:
  - `MAX_LEN` = 128
  - Truncation and padding to max length
  - Attention masks to distinguish real vs padded tokens
  - Special tokens [CLS, SEP] for sequence classification
- Batched tokenization to avoid memory crashes on large dataset
- Data formatted as `tf.data.Dataset` with `input_ids` and `attention_mask`

### 3. Model Selection & Benchmarking
Multiple models were benchmarked on a data subset evaluating both accuracy and training time before selecting the final approach. Initial DistilBERT for sequence classification experiments crashed on full dataset training, leading to a custom architecture with better memory management.

### 4. Model Architecture (DistilBERT)

```
Input Layer (input_ids + attention_mask)
        ↓
DistilBERT Base Layer (pretrained, outputs hidden states)
        ↓
Dropout Layer (regularization to reduce overfitting)
        ↓
Global Average Pooling Layer (dimensionality reduction)
        ↓
Dense Layer with Softmax (3-class sentiment output)
```

### 5. Training
- Early stopping callback monitoring `val_accuracy` with patience of 2 epochs
- Best weights restored automatically on stopping
- Batch size of 16 to manage memory constraints
- Train/validation/test split using `train_test_split`

### 6. Baseline Comparison
A logistic regression model was trained as a baseline using TF-IDF features to compare against the transformer approach on classification metrics.

## Results

| Model | Approach |
|---|---|
| Logistic Regression | TF-IDF + traditional ML baseline |
| DistilBERT + Custom Layers | Transformer + dropout + pooling + dense |

The DistilBERT model demonstrated stronger performance on nuanced and context-dependent reviews where traditional bag-of-words approaches struggled.

## Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Hugging Face Transformers
- **Model:** DistilBERT (`distilbert-base-uncased`)
- **Traditional ML:** scikit-learn (Logistic Regression, TF-IDF)
- **Data Processing:** pandas, numpy, langdetect, NLTK
- **Visualization:** matplotlib, seaborn

## Key Challenges

- **Scale:** 1.6M reviews required batched tokenization and careful memory management
- **Language filtering:** Dataset contained reviews in multiple languages requiring detection and filtering before modeling
- **Overfitting:** Custom dropout and early stopping layers added to address transformer overfitting on the classification task
- **Compute constraints:** Full dataset training required architectural adjustments after initial model crashed on local hardware

## Future Work

- Fine-tune on a GPU/cloud environment to train on the full dataset without constraints
- Experiment with other transformer architectures (BERT, RoBERTa)
- Build a simple inference API to classify new reviews in real time
- Extend analysis to critic vs user sentiment comparison by platform
