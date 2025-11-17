# IMDb Sentiment Analysis: Dataset & Preprocessing Report

## 1. Dataset Description

### 1.1 Basic Information

- **Dataset**: IMDb Movie Reviews Dataset
- **Source**: Kaggle IMDb Dataset
- **Task**: Binary Sentiment Classification
- **Format**: CSV file
- **Total Samples**: 50,000 movie reviews
- **Distribution**: 25,000 positive (50%) + 25,000 negative (50%)
- **Split**: Train 80% (40K) / Validation 10% (5K) / Test 10% (5K)

### 1.2 Text Statistics

| Metric         | Value             |
| -------------- | ----------------- |
| Average Length | 1,309 characters  |
| Min Length     | 32 characters     |
| Max Length     | 13,704 characters |
| Median         | 970 characters    |

### 1.3 Data Quality Issues

1. **HTML Tags**: Contains `<br />`, `<p>`, `<div>` tags
2. **Case Inconsistency**: Mixed uppercase/lowercase
3. **Special Characters**: Punctuation, emojis, URLs
4. **Length Variation**: Wide range (32-13,704 chars)
5. **Spelling Errors**: User-generated content

---

## 2. Preprocessing Pipeline

```
Raw CSV Data
    ↓
1. Text Cleaning (HTML removal, lowercase)
    ↓
2. Tokenization
    ↓
3. Vocabulary Building
    ↓
4. Text Encoding
    ↓
5. Padding/Truncation
    ↓
6. Label Encoding
    ↓
7. Train/Val/Test Split
    ↓
8. DataLoader Creation
    ↓
Model Training
```

---

## 3. Detailed Preprocessing Steps

### Step 1: Text Cleaning

**Implementation**: Python `re` library with regex pattern `r"<[^>]+>"`

```python
def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))  # Remove HTML
    text = text.lower().strip()                 # Lowercase & trim
    return text
```

**Purpose**: Remove HTML tags, standardize format, reduce noise

---

### Step 2: Text Normalization

**Implementation**:

```python
text = text.lower()  # Convert to lowercase
text = text.strip()   # Remove whitespace
```

**Purpose**: Treat "Good", "GOOD", "good" as same word

**Impact**:

- Smaller embedding layer
- Better generalization

---

### Step 3: Tokenization

#### 3.1 RNN/LSTM Tokenization

**Implementation**: Regex pattern `\b\w+\b` for word boundaries

```python
def tokenize_basic(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text)
```

**Example**: `"I love this movie!"` → `["i", "love", "this", "movie"]`

#### 3.2 BERT Tokenization

**Implementation:** HuggingFace Transformers `AutoTokenizer`

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize(text)
```

**Advantages**:

- WordPiece algorithm handles unknown words
- Subword splitting: "unhappiness" → ["un", "##happiness"]
- Compatible with BERT pre-trained vocabulary

---

### Step 4: Vocabulary Building

**Implementation**: Python `collections.Counter`

```python
def build_vocab(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize_basic(text))
    
    vocab = {"<pad>": 0, "<unk>": 1}  # Special tokens
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    
    return vocab
```

**Frequency Filtering** (min_freq=2):

- Filters noise (spelling errors)
- Controls vocabulary size
- Prevents overfitting

**Special Tokens**:

| Token   | ID   | Purpose           |
| ------- | ---- | ----------------- |
| `<pad>` | 0    | Padding sequences |
| `<unk>` | 1    | Unknown words     |

**Impact**:

- Embedding size: vocab_size × embedding_dim
- Smaller vocab = fewer parameters = faster training

---

### Step 5: Sequence Encoding

**Implementation**: NumPy arrays, PyTorch tensors

```python
def encode_text(tokens: List[str], vocab: Dict[str, int], max_length: int) -> List[int]:
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    ids = ids[:max_length]  # Truncate
    if len(ids) < max_length:
        ids += [vocab["<pad>"]] * (max_length - len(ids))  # Pad
    return ids
```

**Sequence Length**:

- RNN/LSTM: `max_length = 200` (covers ~75% samples)
- BERT: `max_length = 256` (BERT max is 512)

**Truncation**: Keep first N tokens (reviews express key points early)

**Padding**: Append zeros at the end (doesn't affect semantics)

**Impact**:

- Enables batch processing
- GPU parallelization
- Controlled computational complexity O(sequence_length)

---

### Step 6: Attention Mask (BERT Only)

**Implementation**:

```python
def create_attention_mask(input_ids: List[int], pad_token_id: int = 0) -> List[int]:
    return [1 if token_id != pad_token_id else 0 for token_id in input_ids]
```

**Example**:

```
input_ids:      [101, 2023, 3185, 2003, 0, 0, 0]
attention_mask: [1,   1,    1,    1,   0, 0, 0]
                 ↑    Real tokens    ↑ Padding
```

**Purpose**: Tells BERT which positions are padding in self-attention mechanism

**Impact**: Improves accuracy, required for BERT

---

### Step 7: Label Encoding

**Implementation**:

```python
label = 1 if sentiment == "positive" else 0
```

| Original   | Encoded | Meaning         |
| ---------- | ------- | --------------- |
| "positive" | 1       | Positive review |
| "negative" | 0       | Negative review |

**Purpose**: PyTorch CrossEntropyLoss requires integer labels

---

### Step 8: Data Splitting

**Implementation**: Pandas `sample()`, `iloc[]`

```python
def split_df(df: pd.DataFrame, split: List[float], seed: int):
    train_ratio, val_ratio, test_ratio = split
    df = df.sample(frac=1.0, random_state=seed)  # Shuffle
    
    n_train = int(len(df) * train_ratio)
    n_val = int(len(df) * val_ratio)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    return train_df, val_df, test_df
```

**Split Ratio**: `[0.8, 0.1, 0.1]` → 40K / 5K / 5K samples

**Random Seed** (seed=42): Ensures reproducibility and fair comparison

**Purpose**:

- Train set: Learn patterns
- Validation set: Hyperparameter tuning and early stopping
- Test set: Unbiased performance evaluation

---

### Step 9: Batch DataLoader

**Implementation**:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=32,      # LSTM: 32, BERT: 16
    shuffle=True,       # Shuffle training data
    pin_memory=True     # Accelerate GPU transfer
)
```

**Batch Size**:

- LSTM: 32 (~10M parameters)
- BERT: 16 (~110M parameters, memory constraint)

**Shuffle**:

- Train: `True` (prevents memorization)
- Val/Test: `False` (consistency)

**Impact**:

- GPU parallel processing
- Average gradients over batch
- 2.5× training speedup

**Technology**: PyTorch DataLoader

---

## 4. Technology Stack

### Data Processing

| Library               | Purpose                      |
| --------------------- | ---------------------------- |
| Pandas                | Data loading & manipulation  |
| NumPy                 | Numerical operations         |
| Python `re`           | Text cleaning & tokenization |
| `collections.Counter` | Word frequency statistics    |

### Deep Learning

| Library                       | Purpose                      |
| ----------------------------- | ---------------------------- |
| PyTorch 2.0+                  | DataLoader, Dataset, Tensors |
| HuggingFace Transformers 4.x+ | BERT tokenizer & models      |

---

## 5. Summary

### Key Achievements

✅ Processed 50,000 movie reviews  
✅ Implemented 9 preprocessing steps  
✅ Support for RNN/LSTM and BERT models  

### Technical Highlights

- Modular design for easy extension
- Comprehensive special token handling
- Efficient batch processing mechanism
- Reproducible random seed management