# Rebuild Plan: Daily Autoencoder Transformer for Insider Threat Detection

**Dataset:** CERT R4.2 (~1000 users, ~501 days, ~32M events)
**Objective:** Novelty-based insider threat detection using reconstruction error

---

## 1. Core Design Decisions

### 1.1 Granularity: Daily Aggregation

Aggregate all events per `(user_id, date)` into a single feature vector.

**Why daily over 15-min windows:**
- Insider scenarios in CERT R4.2 unfold over days/weeks, not minutes
- Daily tokens are dense (many events per day) → less noise
- Enables 60-day lookback (captures weekly cycles + behavioral drift)
- Far fewer tokens → faster training, no O(n²) attention issues

**No explicit session boundaries.** Session count is captured as a feature
(`num_sessions`), not as a structural element. Logon/Logoff events are
counted but not used to define boundaries.

### 1.2 Architecture: Encoder-Only Transformer Autoencoder

```
Input: (batch, 60, F_cont) + (batch, 60, F_cat)
  |
  v
Continuous:  Linear(F_cont -> d_model)
Categorical: Embed -> Linear(total_embed -> d_model)
  |
  v
Fuse: Linear(2*d_model -> d_model) + LayerNorm + Dropout(0.2)
  |
  v
+ Learned Positional Encoding (60 positions)
  |
  v
Transformer Encoder (4 layers, 8 heads, d_ff=512, dropout=0.2, GELU, pre-norm)
  |
  v
Reconstruction Head: LayerNorm -> Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> F_cont)
  |
  v
Output: (batch, 60, F_cont)
```

**No masking.** Full autoencoder reconstruction. No [CLS] token. No [MASK] token.

**Embedding extraction:** The encoder output `(batch, 60, d_model)` can be
mean-pooled to `(batch, d_model)` via `get_embeddings()` for downstream
tasks like FAISS-based user baselining. No retraining required.

**Why no masking:**
- Eliminates train/inference mismatch (masked training vs LOO inference)
- Single forward pass scoring instead of O(n²) leave-one-out
- Simpler architecture, fewer hyperparameters
- Identity mapping prevented by dropout + moderate capacity

**Reconstruction target:** Continuous features only. Categorical features
serve as context input but are not reconstructed.

### 1.3 Training Strategy

- Train ONLY on normal (benign) user data
- No insider samples in train or validation sets
- Insider users go exclusively to the test set
- Model learns the distribution of normal daily behavior

### 1.4 Anomaly Scoring

Single forward pass per sequence:

```
per_day_error[t] = MSE(input[t], reconstruction[t])  # over behavioral features
sequence_score   = 0.5 * mean(per_day_error) + 0.5 * max(per_day_error)
```

The mean captures sustained anomalies; the max captures spike anomalies.

### 1.5 Detection Levels

| Level | What it answers | How |
|-------|----------------|-----|
| **User-level** (primary) | "Is user X an insider?" | User is insider if ANY sequence score > threshold |
| **Sequence-level** | "When did the anomaly occur?" | Which 60-day windows were flagged |
| **Per-scenario** | "Which attack types are caught?" | Breakdown by CERT R4.2 scenario (1, 2, 3) |

### 1.6 Confusion Matrix: User-Level

|                     | Predicted Normal | Predicted Insider |
|---------------------|-----------------|-------------------|
| **Actual Normal**   | TN              | FP                |
| **Actual Insider**  | FN              | TP                |

A user is classified as insider if any of their sequences exceeds the threshold.

---

## 2. Feature Engineering

### 2.1 Daily Aggregation SQL

Source: `events` table in DuckDB, grouped by `(user_id, CAST(datetime AS DATE))`.

### 2.2 Feature Set (~45 continuous + 5 categorical)

#### Activity Counts (13 features, log1p-transformed)

| Feature | SQL |
|---------|-----|
| `http_request_count` | `COUNT(CASE WHEN activity='Http' ...)` |
| `unique_domains` | `COUNT(DISTINCT domain)` |
| `emails_sent` | `COUNT(CASE WHEN activity='Email' ...)` |
| `total_attachments` | `SUM(attachments)` |
| `external_email_count` | `COUNT(... NOT LIKE '%@dtaa.com%')` |
| `unique_recipients` | `COUNT(DISTINCT to_addr)` |
| `file_copy_count` | `COUNT(CASE WHEN activity='File' ...)` |
| `exe_copy_count` | `COUNT(... LIKE '%.exe' ...)` |
| `doc_copy_count` | `COUNT(... LIKE '%.doc' ...)` |
| `archive_copy_count` | `COUNT(... LIKE '%.zip' ...)` |
| `usb_connect_count` | `COUNT(CASE WHEN activity='Connect' ...)` |
| `usb_disconnect_count` | `COUNT(CASE WHEN activity='Disconnect' ...)` |
| `total_actions` | `COUNT(*)` |

#### Intra-Day Structure (5 features)

| Feature | Description |
|---------|-------------|
| `num_sessions` | Count of Logon events that day |
| `active_hours` | Number of distinct hours with any activity |
| `after_hours_ratio` | Fraction of events outside 8:00-18:00 |
| `after_hours_actions` | Count of events outside 8:00-18:00 (log1p) |
| `peak_hour_concentration` | Fraction of events in the busiest hour |

#### PC Usage (1 feature)

| Feature | Description |
|---------|-------------|
| `num_distinct_pcs` | Number of distinct PCs used that day |

#### Interaction Features (4 features, log1p)

| Feature | Formula |
|---------|---------|
| `file_usb_interaction` | `file_copy_count * (usb_connect > usb_disconnect)` |
| `external_email_attachments` | `external_email_count * total_attachments` |
| `http_after_hours` | `http_request_count * is_after_hours_flag` |
| `email_after_hours` | `emails_sent * is_after_hours_flag` |

Note: interaction features use the raw counts (before log1p) for the
multiplication, then apply log1p to the product.

#### Per-User Z-Scores + Rolling Stats (12 features)

For each of: `http_request_count`, `emails_sent`, `file_copy_count`, `total_actions`:
- `{col}_zscore` — (value - user_mean) / user_std
- `{col}_rolling_mean` — rolling mean over last 20 days
- `{col}_rolling_std` — rolling std over last 20 days

#### Temporal (3 features)

| Feature | Description |
|---------|-------------|
| `day_of_week_sin` | `sin(2*pi*dow/7)` |
| `day_of_week_cos` | `cos(2*pi*dow/7)` |
| `is_weekend` | 1 if Saturday/Sunday |

Note: No `days_since_last_active` — after review, this would require
handling inactive days (zero-event days) which complicates the pipeline.
The rolling stats already capture activity gaps implicitly.

#### Static / Slow-Changing (9 features)

| Feature | Source |
|---------|--------|
| `openness` | psychometric table |
| `conscientiousness` | psychometric table |
| `extraversion` | psychometric table |
| `agreeableness` | psychometric table |
| `neuroticism` | psychometric table |
| `role_changed` | user_changes table (binary, that month) |
| `dept_changed` | user_changes table (binary, that month) |
| `functional_unit_changed` | user_changes table (binary, that month) |
| `terminated_flag` | terminated_users table (binary, on/after termination month) |

#### Categorical (5 features, embedded)

| Feature | Embedding Dim |
|---------|--------------|
| `user_id` | 16 |
| `pc` | 16 (MODE — most-used PC that day) |
| `role` | 8 |
| `department` | 4 |
| `functional_unit` | 4 |

### 2.3 Normalization

1. **log1p** on all count features (before z-scores)
2. **Per-user z-scores** on selected features (captures individual baseline deviation)
3. **StandardScaler** fitted on train users only (prevents data leakage)
4. Cyclical features and binary flags are NOT scaled

### 2.4 Context-Only Features

Personality traits (O, C, E, A, N) are included as model input but
excluded from the reconstruction loss via `behavioral_indices`. The model
can use them as context but is not penalized for failing to reconstruct
static features.

---

## 3. Sequence Creation

### 3.1 Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lookback` | 60 days | Covers weekly cycles + behavioral drift |
| `stride` | 5 days | Good overlap, manageable sequence count |
| `min_days` | 10 | Skip users with very sparse activity |

### 3.2 Process

For each user (sorted by date):
1. Get their daily feature vectors (only days with activity)
2. Create sliding windows: `[day_i : day_i+60]` with stride 5
3. If user has < 60 active days, zero-pad from the left
4. Label: 1 if ANY day in the window overlaps with insider activity period

### 3.3 Split

| Split | Users | Purpose |
|-------|-------|---------|
| Train (70%) | Normal users only | Fit the autoencoder |
| Val (15%) | Normal users only | Early stopping, threshold calibration |
| Test (15% + all insiders) | Normal + all insider users | Evaluation |

Insider users are NEVER in train or val.

---

## 4. Model Configuration

```yaml
model:
  type: "transformer_autoencoder"
  lookback: 60
  n_heads: 8
  n_layers: 4
  d_model: 128
  d_ff: 512
  dropout: 0.2

training:
  loss: "mse"
  epochs: 200
  batch_size: 128
  learning_rate: 0.0003
  optimizer: "adamw"
  weight_decay: 0.01
  clip_grad_norm: 1.0
  warmup_epochs: 5
  seed: 42
```

---

## 5. Scoring & Evaluation

### 5.1 Anomaly Score

```python
per_day_error = ((x_cont - reconstruction) ** 2).mean(dim=-1)  # (batch, 60)
score = 0.5 * per_day_error.mean(dim=1) + 0.5 * per_day_error.max(dim=1)
```

Only behavioral features (excluding personality traits) contribute to the error.

### 5.2 Threshold Methods

| Method | Description |
|--------|-------------|
| `best_f1` | Maximize F1 on test set (oracle, not deployable) |
| `percentile_99` | 99th percentile of validation scores |
| `percentile_95` | 95th percentile of validation scores |
| `mean_plus_3std` | Mean + 3*std of validation scores |

### 5.3 Metrics

**Threshold-free (primary):**
- AUPRC (Area Under Precision-Recall Curve)
- AUROC

**Threshold-dependent (user-level, at best-F1 threshold):**
- Precision, Recall, F1
- FPR (False Positive Rate) — normal users falsely flagged
- FNR (False Negative Rate) — insider users missed

**Operational:**
- Detection latency (days from insider start to first alert)
- User detection rate (fraction of insider users caught)
- Per-scenario breakdown (Scenario 1, 2, 3 recall)

---

## 6. Implementation Phases

### Phase 1: Config
Update `config.yaml` with new feature lists, model params, daily windowing.

### Phase 2: Feature Engineering
Rewrite `src/data/feature_engineering.py`:
- Daily aggregation SQL (GROUP BY user_id, date)
- Intra-day features (num_sessions, active_hours, after_hours_ratio, etc.)
- Interaction features
- Per-user z-scores + rolling stats
- StandardScaler (train-only fit)
- Export: features_continuous.npy, features_categorical.npy, labels.npy, etc.

### Phase 3: Sequence Creation
Rewrite `src/data/sequence_creation.py`:
- 60-day sliding window, stride=5
- Per-user processing
- Train/val/test split (insiders → test only)
- Zero-pad short users

### Phase 4: Model
Rewrite `src/models/transformer.py`:
- `InsiderTransformerAE` — no masking, no [CLS], full reconstruction
- `get_anomaly_scores()` — single forward pass, mean+max scoring
- `get_embeddings()` — mean-pool encoder output for FAISS (Phase 9)
- Keep `behavioral_indices` for context-only feature exclusion
- Keep `CategoricalEmbedding` (works as-is)

### Phase 5: Training
Rewrite `scripts/03_train.py`:
- Full MSE loss on all positions (no masking logic)
- AdamW + cosine LR with warmup
- Early stopping on val loss
- Test monitoring with AUC/AUPRC logging

### Phase 6: Scoring & Evaluation
Adapt `src/evaluation/scoring.py` and `src/evaluation/helpers.py`:
- Single-pass scoring (no LOO)
- User-level aggregation and confusion matrix
- FNR metric
- Keep: per-scenario breakdown, latency computation, SOC report

### Phase 7: Augmentation
Update `src/data/augmentation.py` for daily sequences (same strategies,
adapted to daily feature structure).

### Phase 8: End-to-End Verification
Dry-run the full pipeline to verify shapes, no crashes, reasonable outputs.

### Phase 9: FAISS User Baselining (Future)

Use encoder embeddings to build per-user behavioral baselines for
nearest-neighbor anomaly detection.

**Embedding extraction:**
```python
model.get_embeddings(x_cont, x_cat, pooling="mean")  # (batch, d_model=128)
```

Mean-pools the Transformer encoder output across all 60 time steps.
Alternative: `pooling="last"` uses the final time step's hidden state.
No retraining required — works with existing trained weights.

**FAISS workflow:**
1. Extract embeddings for all normal users' train sequences
2. Build a FAISS index (IVFFlat or HNSW) on normal embeddings
3. At inference, embed a new sequence → query k-nearest neighbors
4. Distance to nearest normal neighbors = neighborhood anomaly score
5. Combine with reconstruction score for a hybrid anomaly score:
   ```
   hybrid_score = α * reconstruction_score + (1-α) * faiss_distance_score
   ```

**Per-user baseline:**
- Aggregate a user's normal embeddings into a centroid (mean vector)
- Track centroid drift over time for behavioral change detection
- Flag users whose recent embeddings diverge from their own baseline

**Implementation files (to be created):**
| File | Purpose |
|------|---------|
| `src/evaluation/embeddings.py` | Extract & cache embeddings from trained model |
| `src/evaluation/faiss_index.py` | Build/query FAISS index on normal embeddings |
| `scripts/06_build_index.py` | Pipeline script: extract → build → save index |
| `scripts/07_hybrid_eval.py` | Evaluate hybrid (reconstruction + FAISS) scoring |

---

## 7. Open Concerns

### 7.1 Inactive Days
Some users have sparse activity (not every day). Current plan: only include
days with at least 1 event. This means the "60-day lookback" is 60 active
days, not 60 calendar days. This preserves signal density but loses
information about gaps. The rolling stats partially compensate.

**Alternative:** Include zero-filled rows for inactive calendar days. This
preserves gap information but adds many zero-vectors that dilute the signal.

**Recommendation:** Start with active-days-only. If results are weak, try
calendar-day filling as an experiment.

### 7.2 Sequence Label Definition — DECIDED: Any-Overlap
A sequence is labeled insider=1 if ANY day in the 60-day window overlaps
with the insider's ground-truth activity period. This means some "insider"
sequences may contain mostly normal days with only a few anomalous days at
the end. This is correct because:
- Labels are only used at evaluation, not training (novelty detection)
- Early detection is the goal — we want to reward catching insiders early
- CERT R4.2 ground truth is coarse (broad start/end window)
- Matches user-level aggregation (ANY sequence > threshold = insider)

### 7.3 User-Level Aggregation Strategy
Current plan: user is insider if ANY sequence exceeds threshold. This is
aggressive (high recall, potentially lower precision). Alternative: require
N sequences or a sustained score above threshold. Start with ANY, tune later.

---

## 8. Files Modified

| File | Action |
|------|--------|
| `config/config.yaml` | Rewrite — new features, model, windowing |
| `src/data/feature_engineering.py` | Rewrite — daily aggregation |
| `src/data/sequence_creation.py` | Rewrite — 60-day windows |
| `src/data/augmentation.py` | Adapt — daily sequences |
| `src/models/transformer.py` | Rewrite — autoencoder (no masking) |
| `scripts/03_train.py` | Rewrite — full MSE training |
| `src/evaluation/scoring.py` | Adapt — single-pass scoring |
| `src/evaluation/helpers.py` | Adapt — user-level metrics, FNR |
| `scripts/02_feature_engineering.py` | Minor — remove window-size args |
| `scripts/04_evaluate.py` | Adapt — new scoring flow |

**NOT modified:**
| File | Reason |
|------|--------|
| `scripts/01_prepare_data.py` | DuckDB schema unchanged |
| `src/data/parquet_to_duckdb.py` | Data ingestion unchanged |
| `src/data/csv_to_parquet.py` | Raw data conversion unchanged |

**Phase 9 (future):**
| File | Action |
|------|---------|
| `src/models/transformer.py` | `get_embeddings()` added ✔ |
| `src/evaluation/embeddings.py` | New — extract & cache embeddings |
| `src/evaluation/faiss_index.py` | New — FAISS index build/query |
| `scripts/06_build_index.py` | New — pipeline script |
| `scripts/07_hybrid_eval.py` | New — hybrid evaluation |
