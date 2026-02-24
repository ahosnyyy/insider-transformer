# InsiderTransformerAE: Model Pipeline Documentation

**Architecture:** Transformer Autoencoder for insider threat detection  
**Dataset:** CERT R4.2 (~1000 users, ~501 days, ~32M events)  
**Approach:** Session-aware daily feature engineering with 60-day sequences  

---

## 1. Model Architecture

### 1.1 Transformer Autoencoder

Our model is a **full reconstruction Transformer Autoencoder** that learns normal user behavior patterns from 60-day sequences of daily feature vectors.

```
Input: [batch_size, 60, 57]  # 60-day sequences, 57 features per day
    │
    ▼
Positional Encoding
    │
    ▼
Transformer Encoder (4 layers, 6 heads, d_model=192)
    │
    ▼
Linear Projection → [batch_size, 60, 57]  # Reconstruct input
    │
    ▼
Loss: MSE on behavioral features (masked during training)
```

### 1.2 Model Configuration

```yaml
model:
  type: "transformer_autoencoder"
  lookback: 60                  # sequence length (days)
  min_days: 10                  # skip users with <10 active days
  n_heads: 6                    # attention heads
  n_layers: 4                    # transformer layers
  d_model: 192                   # model dimension
  d_ff: 512                      # feed-forward dimension
  dropout: 0.2                   # dropout rate

# Note: Users with <10 active days are excluded to avoid excessive zero-padding
# (e.g., 5 days → 55 zeros + 5 real days = 91% padded data)
```

### 1.3 Full Reconstruction Training

During training, the model learns to reconstruct **all input features**:

```python
# Forward pass - full reconstruction
predictions = model(x_cont, x_cat)

# Loss on behavioral features only
loss = reconstruction_loss(predictions, x_cont, behavioral_indices)
```

**Why full reconstruction?**
- Simpler training objective
- Learns complete behavioral patterns
- Easier to interpret reconstruction errors
- Proven effective for anomaly detection

### 1.4 Context Features

Certain features are **context-only** and not included in reconstruction loss:

```python
context_only_features = ['openness', 'conscientiousness', 'extraversion', 
                        'agreeableness', 'neuroticism']
```

These provide user context but aren't behavioral, so the model shouldn't be penalized for not reconstructing them.

---

## 2. Training Pipeline

### 2.1 Training Script: `scripts/03_train.py`

The training script orchestrates the entire training process:

```bash
python scripts/03_train.py [--config config/config.yaml] [--seed 42]
```

### 2.2 Training Flow

```
1. Load Configuration
    │
    ▼
2. Feature Engineering (if needed)
    │
    ▼
3. Sequence Creation
    │   - Split users: train/val/test (70/15/15)
    │   - Create 60-day sliding windows
    │   - Apply data augmentation (insider sequences only)
    │
    ▼
4. Model Initialization
    │   - Create Transformer architecture
    │   - Initialize weights
    │   - Setup optimizer and scheduler
    │
    ▼
5. Training Loop (200 epochs)
    │   - Forward pass (full reconstruction)
    │   - Compute MSE loss on behavioral features
    │   - Backward pass + gradient clipping
    │   - Learning rate scheduling
    │   - Validation evaluation
    │   - Model checkpointing
    │
    ▼
6. Final Model Selection
    │   - Choose best validation model
    │   - Save final checkpoint
```

### 2.3 Training Configuration

```yaml
training:
  loss: "mse"                   # MSE on behavioral features
  epochs: 200
  batch_size: 64
  learning_rate: 0.0001
  optimizer: "adamw"
  weight_decay: 0.05
  clip_grad_norm: 1.0
  warmup_epochs: 5              # linear warmup
  seed: 42
```

### 2.4 Data Splitting Strategy

```python
# User-based splitting (no data leakage)
train_users = 70% of users
val_users   = 15% of users  
test_users  = 15% of users

# Only normal users in training
insider_users = excluded from training
```

**Why user-based splitting?**
- Prevents data leakage between sequences
- Ensures model learns generalizable patterns
- Mimics real-world deployment (unseen users)

### 2.5 Data Augmentation

For insider sequences in test set, we apply augmentation to improve robustness:

```yaml
augmentation:
  enabled: true
  insider_multiplier: 5         # 5 augmented copies per insider sequence
  strategies:
    jittering:
      enabled: true
      noise_std: 0.05
    feature_dropout:
      enabled: true
      dropout_rate: 0.1
    scaling:
      enabled: true
      scale_range: [0.9, 1.1]
```

---

## 3. Evaluation Pipeline

### 3.1 Evaluation Script: `scripts/04_evaluate.py`

Comprehensive evaluation of the trained model:

```bash
python scripts/04_evaluate.py [--config config/config.yaml] \
                            [--checkpoint checkpoints/best_model.pt] \
                            [--use-augmented] \
                            [--threshold-method best_f1]
```

### 3.2 Evaluation Process

```
1. Load Trained Model
    │
    ▼
2. Load Test Data
    │   - Feature vectors
    │   - Labels (insider vs normal)
    │   - Session data (for session-level eval)
    │
    ▼
3. Anomaly Scoring
    │   - Forward pass on all sequences
    │   - Compute reconstruction error per day
    │   - Aggregate to user-level scores
    │
    ▼
4. Threshold Selection
    │   - Multiple methods: best_f1, percentile_99, etc.
    │   - Choose optimal threshold on validation set
    │
    ▼
5. Metrics Computation
    │   - User-level: det_rate, precision, recall, F1
    │   - Sequence-level: AUC, AUPRC, confusion matrix
    │   - Session-level: localization accuracy
    │
    ▼
6. SOC Report Generation
    │   - Detailed incident reports
    │   - Session drill-down
    │   - Risk indicators
    │
    ▼
7. Results Export
    │   - evaluation_results.json
    │   - soc_report.json
    │   - test_anomaly_scores.npy
```

### 3.3 Anomaly Scoring Methods

```python
# Primary scoring method
def score_dataset(model, sequences):
    """Compute mean + max reconstruction error per sequence"""
    with torch.no_grad():
        reconstructions = model(sequences)
        mse_per_feature = (reconstructions - sequences) ** 2
        
        # Behavioral features only (exclude context-only)
        behavioral_idx = get_behavioral_feature_indices()
        mse_behavioral = mse_per_feature[:, :, behavioral_idx]
        
        # Mean + max combination
        mean_error = mse_behavioral.mean(dim=(1, 2))
        max_error = mse_behavioral.max(dim=(1, 2))[0]
        
        return 0.5 * mean_error + 0.5 * max_error
```

### 3.4 Threshold Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `best_f1` | Threshold maximizing F1 on validation set | **Default** - optimal balance |
| `percentile_99` | 99th percentile of normal reconstruction errors | Fixed false positive rate |
| `percentile_95` | 95th percentile of normal reconstruction errors | More sensitive |
| `mean_plus_3std` | Mean + 3×std of normal errors | Statistical outlier |
| `fixed_recall_95` | Threshold achieving 95% recall | High recall requirement |
| `fixed_recall_99` | Threshold achieving 99% recall | Very high recall |

### 3.5 Evaluation Metrics

#### User-Level Metrics
```python
metrics = {
    'det_rate': true_positives / total_insiders,      # Detection rate
    'precision': true_positives / (true_positives + false_positives),
    'recall': true_positives / (true_positives + false_negatives),
    'f1': 2 * (precision * recall) / (precision + recall),
    'fpr': false_positives / total_normals,           # False positive rate
    'fnr': false_negatives / total_insiders           # False negative rate
}
```

#### Sequence-Level Metrics
```python
metrics = {
    'auroc': area_under_roc_curve,                    # Overall ranking
    'auprc': area_under_pr_curve,                     # Imbalanced performance
    'detection_latency': mean_time_to_detection,       # Hours from threat start
    'user_detection_rate': detected_insiders / total_insiders
}
```

#### Session-Level Metrics
```python
metrics = {
    'session_precision': correct_sessions / flagged_sessions,
    'session_recall': correct_sessions / total_malicious_sessions,
    'session_f1': 2 * (precision * recall) / (precision + recall),
    'localization_accuracy': days_with_correct_session / total_flagged_days
}
```

**Why evaluate sessions only during malicious periods?**

1. **Ground Truth Limitation**: CERT dataset only labels insider periods, not individual sessions
2. **Precision Focus**: Analysts care most about "when model flags a session, is it actually malicious?"
3. **Practical Relevance**: False positives in normal periods are less critical than missed threats
4. **Resource Allocation**: Security teams need to know which flagged sessions warrant investigation
5. **Threat Localization**: Helps identify the specific time windows and activities that triggered alerts

This approach evaluates the model's precision at the session level - crucial for operational deployment where analysts must investigate flagged sessions.

---

## 4. Visualization Pipeline

### 4.1 Plotting Script: `scripts/05_plot.py`

Comprehensive visualization of model performance and data:

```bash
python scripts/05_plot.py [--config config/config.yaml] \
                         [--use-augmented] \
                         [--output-dir plots/]
```

### 4.2 Generated Plots

The script generates **11 plots** organized into categories:

#### Training Progress (2 plots)
1. **Loss Curve** - Training and validation loss over epochs
2. **Learning Rate Schedule** - LR changes during training

#### Performance Analysis (3 plots)
3. **ROC Curve** - True positive vs false positive rate
4. **Precision-Recall Curve** - Precision vs recall at different thresholds
5. **Confusion Matrix** - User-level classification results

#### Anomaly Analysis (3 plots)
6. **Score Distribution** - Normal vs insider anomaly scores
7. **Detection Timeline** - When insiders are detected over time
8. **Per-Scenario Performance** - Detection rate by insider scenario

#### Session-Level Analysis (2 plots)
9. **Session Confusion Matrix** - Session-level classification (malicious periods only)
   - Evaluates model's ability to identify specific suspicious sessions within insider periods
   - Focuses only on sessions during known malicious time windows for precise threat localization
   - Cannot evaluate true negatives (normal sessions) as ground truth only covers insider periods
10. **Session Duration Analysis** - Duration of flagged vs normal sessions

#### Feature Analysis (1 plot)
11. **Feature Importance** - Top contributing features to anomalies

### 4.3 Plot Features

#### Auto-Detection
```python
# Automatically detects augmented vs non-augmented data
if len(scores) != len(labels):
    # Try the other label file
    alt_labels = load_alternative_labels()
    if len(alt_labels) == len(scores):
        labels = alt_labels
```

#### Individual Timeline Plots
For each detected insider, generates detailed timeline:
```
User: ACM2278 (Scenario 1 - USB Theft)
├── Day 150: Normal (score: 0.12)
├── Day 151: Normal (score: 0.15)
├── Day 152: **ANOMALY** (score: 0.87) ← USB activity detected
└── Day 153: **ANOMALY** (score: 0.91) ← Continued suspicious activity
```

#### Session Drill-Down
Session-level plots show:
- Session start/end times
- Activity breakdown per session
- Risk indicators and anomalies

### 4.4 Plot Customization

```python
# Plot configuration
plot_config = {
    'figure_size': (12, 8),
    'dpi': 300,
    'style': 'seaborn-v0_8',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
    'save_formats': ['png', 'pdf'],
    'individual_timelines': True  # Generate per-insider timelines
}
```

---

## 5. Inference Pipeline

### 5.1 Inference Script: `scripts/06_inference.py`

Real-time inference on new data:

```bash
python scripts/06_inference.py [--config config/config.yaml] \
                              [--checkpoint checkpoints/best_model.pt] \
                              [--users "user1,user2"] \
                              [--start-date 2011-01-01] \
                              [--end-date 2011-01-31] \
                              [--threshold-method best_f1] \
                              [--top-k 10]
```

### 5.2 Inference Features

#### User Filtering
```python
# Specific users
--users "ACM2278,BDT1839"

# All users (default)
# No flag needed
```

#### Date Range Filtering
```python
# Process specific time window
--start-date 2011-01-01
--end-date 2011-01-31

# All dates (default)
# No flags needed
```

#### Threshold Selection
```python
# Multiple options
--threshold-method best_f1          # Default
--threshold-method percentile_99
--threshold-method 0.85            # Fixed threshold
```

#### Output Options
```python
# Top K anomalous users
--top-k 10                         # Default: all users
```

### 5.3 Inference Output

```
=== INFERENCE RESULTS ===
Processed: 1,000 users
Date range: 2011-01-01 to 2011-01-31
Threshold method: best_f1 (0.82)

Top 10 Anomalous Users:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ User ID     │ Risk Score  │ Status      │ Details     │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ ACM2278     │ 0.94        │ ALERT       │ USB activity│
│ BDT1839     │ 0.87        │ ALERT       │ After-hours │
│ ...         │ ...         │ ...         │ ...         │
└─────────────┴─────────────┴─────────────┴─────────────┘

Alerts generated: 23 users
High confidence: 8 users
Medium confidence: 12 users
Low confidence: 3 users
```

### 5.4 Real-Time Processing

The inference pipeline supports:
- **Incremental updates** - Process new days without recomputing everything
- **Streaming mode** - Handle live data feeds
- **Batch processing** - Process large date ranges efficiently
- **Caching** - Reuse computed features for repeated queries

---

## 7. Configuration Management

### 7.1 Central Configuration

All pipeline components use `config/config.yaml`:

```yaml
# Model architecture
model:
  lookback: 60
  n_heads: 6
  n_layers: 4
  d_model: 192

# Training hyperparameters  
training:
  epochs: 200
  batch_size: 64
  learning_rate: 0.0001

# Feature definitions
features:
  continuous: [...]
  session: [...]
  categorical: [...]

# Evaluation settings
evaluation:
  primary_metric: "auprc"
  threshold_methods: [...]
  default_threshold: "best_f1"
```

## 8. File Structure

```
insider-transformer/
├── scripts/
│   ├── 01_download_data.py      # Data download
│   ├── 02_feature_engineering.py # Feature computation
│   ├── 03_train.py              # Model training
│   ├── 04_evaluate.py           # Model evaluation
│   ├── 05_plot.py               # Visualization
│   └── 06_inference.py          # Real-time inference
├── src/
│   ├── data/
│   │   └── feature_engineering.py # SQL feature pipeline
│   ├── models/
│   │   └── transformer.py       # Transformer architecture
│   ├── evaluation/
│   │   ├── scoring.py           # Anomaly scoring
│   │   └── helpers.py           # Evaluation utilities
│   └── sequence_creation.py     # Sequence generation
├── config/
│   └── config.yaml              # Central configuration
├── checkpoints/                 # Model checkpoints
├── data/processed/              # Processed features
├── plots/                       # Generated visualizations
└── results/                     # Evaluation outputs
```
