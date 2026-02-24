# InsiderTransformerAE: Session-Aware Architecture Documentation

**Dataset:** CERT R4.2 (~1000 users, ~501 days, ~32M events)  
**Approach:** Session-aware daily feature engineering with sub-day temporal granularity

---

## 1. Architecture Overview

Our approach captures both **daily activity patterns** and **intra-day session dynamics** to provide comprehensive behavioral modeling for insider threat detection. By extracting per-session features first, then aggregating them to daily level, we preserve temporal structure that pure daily aggregation loses.

**Key insight:** A user who copies 50 files in one normal 8-hour session should look different from a user who copies 5 files in a suspicious 30-minute late-night session plus 45 files during normal work. Our session-aware features capture this distinction.

---

## 2. Feature Engineering Pipeline

```
Raw events (logon, device, email, file, http)
    │
    ▼
┌─────────────────────────────────┐
│  Session Detection              │
│  Pair Logon → Logoff per user   │
│  per PC per day                 │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Per-Session Feature Extraction │
│  ~10 features per session       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Daily Feature Aggregation      │
│  Daily activity counts (18)     │
│  + Session-derived features (18)│
│  + Interaction features (4)      │
│  + Cyclical/binary features (7) │
│  + Personality traits (5)        │
│  = ~52 continuous features      │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  60-day Sliding Window          │
│  Per-user sequential modeling   │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Transformer Autoencoder       │
│  Learns normal behavior patterns│
└─────────────────────────────────┘
```

**Result:** The Transformer AE receives comprehensive daily vectors with both aggregate counts and session-level temporal structure, enabling detection of subtle behavioral anomalies.

---

## 3. Feature Configuration

The feature definitions are centralized in `config/config.yaml` to enable easy modification and experimentation:

```yaml
features:
  continuous:
    # Activity counts (log1p)
    - http_request_count
    - unique_domains
    - emails_sent
    - total_attachments
    - external_email_count
    - unique_recipients
    - file_copy_count
    - exe_copy_count
    - doc_copy_count
    - archive_copy_count
    - usb_connect_count
    - usb_disconnect_count
    - total_actions
    # Intra-day structure
    - num_sessions
    - active_hours
    - after_hours_ratio
    - after_hours_actions
    - peak_hour_concentration
    # PC usage
    - num_distinct_pcs
    # Personality traits (context-only, not reconstructed)
    - openness
    - conscientiousness
    - extraversion
    - agreeableness
    - neuroticism

  interaction:
    # Interaction features (log1p)
    - file_usb_interaction
    - external_email_attachments
    - http_after_hours
    - email_after_hours

  session:
    # Session-derived features (log1p on counts, raw on ratios/stats)
    - sess_count
    - mean_session_duration
    - max_session_duration
    - std_session_duration
    - total_session_time
    - num_after_hours_sessions
    - after_hours_session_ratio
    - earliest_session_hour
    - latest_session_hour
    - session_hour_spread
    - mean_session_event_count
    - max_session_event_count
    - mean_session_entropy
    - num_usb_sessions
    - num_exe_sessions
    - longest_inter_session_gap
    - usb_after_hours_sessions
    - file_heavy_sessions
    - short_usb_sessions

  cyclical:
    - day_of_week_sin
    - day_of_week_cos

  binary:
    - is_weekend
    - role_changed
    - dept_changed
    - functional_unit_changed
    - terminated_flag

  context_only:
    - openness
    - conscientiousness
    - extraversion
    - agreeableness
    - neuroticism

  categorical:
    - user_id
    - pc
    - role
    - department
    - functional_unit

  # Features to compute per-user z-scores and rolling stats for
  user_normalized:
    - http_request_count
    - emails_sent
    - file_copy_count
    - total_actions
    - sess_count
    - mean_session_duration
    - total_session_time
```

**Configuration Benefits:**
- **Modular Design:** Features can be enabled/disabled by commenting out lines
- **Easy Experimentation:** New features can be added without code changes
- **Clear Organization:** Features grouped by type and computation method
- **Reproducibility:** Same configuration across training, evaluation, and inference

---

## 4. Feature Type Clarification

### Session vs Sequence vs Daily Features

| Feature Type | Time Granularity | Purpose | Examples |
|--------------|------------------|---------|----------|
| **Session Features** | Minutes to hours (within a single login session) | Capture intra-day behavioral patterns and timing anomalies | `mean_session_duration`, `short_usb_sessions`, `session_hour_spread` |
| **Daily Features** | 24-hour periods (calendar days) | Provide aggregate activity volume and patterns | `http_request_count`, `emails_sent`, `total_actions` |
| **Sequence Features** | 60-day sliding windows (for model input) | Enable temporal modeling and pattern learning across days | All daily features concatenated into sequences |
| **Context Features** | Static per user | Provide user-specific baseline information | `role`, `department`, personality traits |

### Data Flow Hierarchy

```
Raw Events (timestamped)
    │
    ▼
[Session Detection] → Sessions (minutes-hours)
    │
    ▼
[Session Aggregation] → Session Features (per session)
    │
    ▼
[Daily Aggregation] → Daily Features (per day)
    │
    ▼
[Sequence Creation] → Sequences (60-day windows)
    │
    ▼
[Transformer Model] → Learned patterns
```

### Why This Hierarchy Matters

1. **Session Features** detect **when** and **how** activities occur within a day
2. **Daily Features** detect **how much** activity occurs per day
3. **Sequence Features** enable the model to learn **patterns over time**
4. **Context Features** help the model understand **who** is performing the actions

### Feature Processing Pipeline

| Stage | Input | Output | Processing |
|-------|-------|--------|------------|
| Raw Events | Event logs | Sessions | SQL: Pair Logon/Logoff |
| Sessions | Sessions + Events | Session Features | SQL: Per-session aggregation |
| Session Features | Session Features | Daily Features | SQL: Aggregate to daily level |
| Daily Features | Daily Features + Context | Featured Table | SQL: log1p, interactions, cyclical |
| Featured Table | Featured Table | Sequences | Python: Sliding windows |
| Sequences | Sequences | Model Input | Python: Scaling, batching |

---

## 5. Feature Vector Structure

### Daily Feature Vector

Each day is represented by **one feature vector** of shape `[57]` containing:

#### Continuous Features (~52)

```
# Daily Activity Counts (18)
[http_request_count, unique_domains, emails_sent, total_attachments, 
 external_email_count, unique_recipients, file_copy_count, exe_copy_count,
 doc_copy_count, archive_copy_count, usb_connect_count, usb_disconnect_count,
 total_actions, num_sessions, active_hours, after_hours_ratio,
 after_hours_actions, peak_hour_concentration, num_distinct_pcs]

# Session-Derived Features (18) 
[sess_count, mean_session_duration, max_session_duration, std_session_duration,
 total_session_time, num_after_hours_sessions, after_hours_session_ratio,
 earliest_session_hour, latest_session_hour, session_hour_spread,
 mean_session_event_count, max_session_event_count, mean_session_entropy,
 num_usb_sessions, num_exe_sessions, longest_inter_session_gap,
 usb_after_hours_sessions, file_heavy_sessions, short_usb_sessions]

# Interaction Features (4)
[file_usb_interaction, external_email_attachments, http_after_hours, email_after_hours]

# Cyclical Features (2)
[day_of_week_sin, day_of_week_cos]

# Personality Traits (5) - context-only
[openness, conscientiousness, extraversion, agreeableness, neuroticism]

# Binary Features (5)
[is_weekend, role_changed, dept_changed, functional_unit_changed, terminated_flag]
```

#### Categorical Features (5) - Embedded

```
[user_id_encoded, pc_encoded, role_encoded, department_encoded, functional_unit_encoded]
```

### Feature Vector Example

For a single day, the feature vector might look like:

```python
# Day vector: shape (57,)
day_vector = np.array([
    # Continuous features (52) - after log1p + StandardScaler
    2.398, 1.098, 1.609, 0.000,  # Daily counts
    0.693, 1.386, 1.098, 0.000,  # ... more daily features
    1.386, 0.693, 3.401, 2.197,  # Session features
    0.000, 0.234, 0.567,         # Session ratios/stats (raw)
    0.000, 1.098, 0.000,         # Interaction features
    0.785, -0.707,               # Cyclical (sin/cos)
    3.2, 4.1, 2.8, 3.9, 2.3,    # Personality traits (scaled)
    0, 0, 0, 0, 0,               # Binary flags (scaled)
    
    # Categorical embeddings (5) - integer IDs
    147, 23, 4, 2, 1              # Encoded IDs
])
```

### Data Organization

```
User A:
├── Day 1 (2011-01-01) → Feature vector [57]
├── Day 2 (2011-01-02) → Feature vector [57]
├── Day 3 (2011-01-03) → Feature vector [57]
└── ...
└── Day 501 (2011-05-17) → Feature vector [57]

User B:
├── Day 1 (2011-01-01) → Feature vector [57]
├── Day 2 (2011-01-02) → Feature vector [57]
└── ...
```

### From Daily Vectors to Sequences

The Transformer processes **60-day sliding windows**:

```
User A Sequences:
├── Sequence 1: [Day 1, Day 2, ..., Day 60] → Shape [60, 57]
├── Sequence 2: [Day 2, Day 3, ..., Day 61] → Shape [60, 57]
├── Sequence 3: [Day 3, Day 4, ..., Day 62] → Shape [60, 57]
└── ... (sliding by stride of 5 days by default)
```

### Key Properties

1. **One vector per day** - Each day's activities compressed into ~57 numbers
2. **User-specific** - Each user has their own time series of daily vectors
3. **Variable length** - Users with fewer days have fewer sequences
4. **Sliding windows** - Model learns from overlapping 60-day contexts
5. **Prediction target** - Model tries to reconstruct the last day in each sequence

### Feature Processing Pipeline

| Stage | Input | Output | Shape | Processing |
|-------|-------|--------|-------|------------|
| Raw Events | Event logs | Daily aggregations | `[n_days, 18]` | SQL aggregation |
| Session Features | Sessions + Events | Session stats | `[n_days, 18]` | SQL + aggregation |
| Feature Engineering | Daily + Session | Featured table | `[n_days, 52]` | log1p, interactions |
| Context Join | Featured + Context | Complete vectors | `[n_days, 57]` | Joins, encoding |
| Scaling | Complete vectors | Scaled vectors | `[n_days, 57]` | StandardScaler |
| Sequencing | Scaled vectors | Sequences | `[n_seqs, 60, 57]` | Sliding windows |

---

## 6. Feature Categories

The model uses **~52 continuous features** organized into five categories:

- **Daily activity counts** (18 features) — Raw event counts aggregated per day
- **Session-derived features** (18 features) — Per-session behavioral statistics aggregated to daily level
- **Interaction features** (4 features) — Cross-feature combinations
- **Cyclical & binary features** (7 features) — Temporal encoding and flags
- **Personality traits** (5 features) — Big Five psychometric scores (context-only)

Plus **5 categorical features** (user_id, pc, role, department, functional_unit) embedded as context.

**Note:** Per-user z-scores and rolling statistics are computed for select features but are not used as direct model inputs — they help with anomaly detection internally.

---

## 4. Daily Activity Counts (18 features)

These features are computed directly from raw events grouped by `(user_id, date)`.

| Feature | Description | Computation | Transform |
|---------|-------------|------------|-----------|
| `http_request_count` | Number of HTTP requests per day | `COUNT(CASE WHEN activity='Http' THEN 1 END)` | log1p |
| `unique_domains` | Distinct domains visited via HTTP | `COUNT(DISTINCT SPLIT_PART(url, '/', 3))` | log1p |
| `emails_sent` | Number of emails sent per day | `COUNT(CASE WHEN activity='Email' THEN 1 END)` | log1p |
| `total_attachments` | Total email attachments | `SUM(CASE WHEN activity='Email' THEN attachments ELSE 0 END)` | log1p |
| `external_email_count` | Emails to external domains | `COUNT(CASE WHEN to_addr NOT LIKE '%@dtaa.com%' THEN 1 END)` | log1p |
| `unique_recipients` | Distinct email recipients | `COUNT(DISTINCT to_addr)` | log1p |
| `file_copy_count` | Total file copy operations | `COUNT(CASE WHEN activity='File' THEN 1 END)` | log1p |
| `exe_copy_count` | Executable files copied (.exe, .msi, .bat, .cmd) | `COUNT(CASE WHEN filename LIKE '%.exe%' THEN 1 END)` | log1p |
| `doc_copy_count` | Document files copied (.doc, .pdf, .xls, .ppt, .txt, .csv) | `COUNT(CASE WHEN filename LIKE '%.doc%' THEN 1 END)` | log1p |
| `archive_copy_count` | Archive files copied (.zip, .rar, .7z, .tar, .gz) | `COUNT(CASE WHEN filename LIKE '%.zip%' THEN 1 END)` | log1p |
| `usb_connect_count` | USB device connect events | `COUNT(CASE WHEN activity='Connect' THEN 1 END)` | log1p |
| `usb_disconnect_count` | USB device disconnect events | `COUNT(CASE WHEN activity='Disconnect' THEN 1 END)` | log1p |
| `total_actions` | Total events across all activity types | `COUNT(*)` | log1p |
| `num_sessions` | Number of login sessions per day | `COUNT(CASE WHEN activity='Logon' THEN 1 END)` | log1p |
| `active_hours` | Distinct hours with activity | `COUNT(DISTINCT EXTRACT(HOUR FROM datetime))` | raw |
| `after_hours_ratio` | Fraction of events outside 8am-6pm | `after_hours_actions / total_actions` | raw |
| `after_hours_actions` | Events outside 8am-6pm | `COUNT(*) FILTER (WHERE hour < 8 OR hour >= 18)` | log1p |
| `peak_hour_concentration` | Fraction of events in busiest hour | `MAX(hourly_count) / total_actions` | raw |
| `num_distinct_pcs` | Number of different PCs used | `COUNT(DISTINCT pc)` | raw |

**Insider threat relevance:**
- Data exfiltration via USB (`usb_connect_count`, `file_copy_count`, `exe_copy_count`)
- Data theft via email (`external_email_count`, `total_attachments`)
- Unusual timing patterns (`after_hours_ratio`, `peak_hour_concentration`)
- Abnormal activity volume (`total_actions`, `http_request_count`)

---

## 5. Session Detection

### 5.1 Boundary Rules

```sql
-- A session starts with a Logon event and ends with:
--   (a) The next Logoff event for the same user on the same PC, OR
--   (b) The next Logon event for the same user (implicit logoff), OR
--   (c) End of day (23:59:59) if no logoff found

WITH session_boundaries AS (
    SELECT
        user_id,
        pc,
        datetime AS session_start,
        LEAD(datetime) OVER (
            PARTITION BY user_id, pc
            ORDER BY datetime
        ) AS next_event_time,
        activity,
        LEAD(activity) OVER (
            PARTITION BY user_id, pc
            ORDER BY datetime
        ) AS next_activity
    FROM events
    WHERE activity IN ('Logon', 'Logoff')
)
```

### 5.2 Edge Cases

| Case | Resolution |
|------|------------|
| Missing Logoff | End session at next Logon or end of day |
| Multiple Logons without Logoff | Each Logon starts a new session |
| Logoff without Logon | Ignore (orphan event) |
| Overnight session (crosses midnight) | Assign to the start date |
| Shared PC (multiple users) | Sessions are per (user_id, pc) |
| Very short sessions (< 1 min) | Keep — may indicate automated/scripted behavior |
| Very long sessions (> 12 hours) | Cap duration at 12h for feature computation |

### 5.3 Session Table Schema

```sql
CREATE TABLE sessions AS (
    session_id      INTEGER,        -- auto-increment
    user_id         VARCHAR,
    pc              VARCHAR,
    session_date    DATE,           -- date of session start
    session_start   TIMESTAMP,
    session_end     TIMESTAMP,
    duration_min    FLOAT,          -- capped at 720 (12h)
    is_after_hours  BOOLEAN,        -- started outside 8:00-18:00
    hour_of_start   INTEGER         -- 0-23
)
```

---

## 6. Per-Session Features

### 6.1 Activity Features (per session)

| Feature | Description |
|---------|-------------|
| `session_duration_min` | Minutes from logon to logoff (capped 720) |
| `session_event_count` | Total events within session |
| `session_file_count` | File copy events in session |
| `session_email_count` | Emails sent in session |
| `session_http_count` | HTTP requests in session |
| `session_usb_used` | 1 if any USB connect in session |
| `session_external_email` | 1 if any external email in session |
| `session_exe_copied` | 1 if any .exe/.msi/.bat copied |
| `session_unique_domains` | Distinct domains visited in session |
| `session_attachment_count` | Total email attachments in session |

### 6.2 Temporal Features (per session)

| Feature | Description |
|---------|-------------|
| `session_hour_start` | Hour of day (0-23) when session started |
| `session_is_after_hours` | Started outside 8:00-18:00 |
| `session_is_weekend` | Started on Saturday/Sunday |

### 6.3 Behavioral Features (per session)

| Feature | Description |
|---------|-------------|
| `session_event_entropy` | Shannon entropy of event type distribution |

---

## 7. Daily Aggregation of Session Features (18 features)

For each `(user_id, date)`, aggregate all sessions into daily statistics:

### 7.1 Session Count & Duration Stats

| Daily Feature | Aggregation |
|---------------|-------------|
| `sess_count` | COUNT of sessions |
| `mean_session_duration` | AVG(session_duration_min) |
| `max_session_duration` | MAX(session_duration_min) |
| `std_session_duration` | STDDEV(session_duration_min) |
| `total_session_time` | SUM(session_duration_min) |

### 7.2 Session Timing Stats

| Daily Feature | Aggregation |
|---------------|-------------|
| `num_after_hours_sessions` | COUNT where is_after_hours = true |
| `after_hours_session_ratio` | num_after_hours_sessions / num_sessions |
| `earliest_session_hour` | MIN(session_hour_start) |
| `latest_session_hour` | MAX(session_hour_start) |
| `session_hour_spread` | latest - earliest |

### 7.3 Session Behavior Stats

| Daily Feature | Aggregation |
|---------------|-------------|
| `mean_session_event_count` | AVG(session_event_count) |
| `max_session_event_count` | MAX(session_event_count) |
| `mean_session_entropy` | AVG(session_event_entropy) |
| `num_usb_sessions` | COUNT where session_usb_used = 1 |
| `num_exe_sessions` | COUNT where session_exe_copied = 1 |
| `longest_inter_session_gap` | MAX gap between consecutive sessions (minutes) |

### 7.4 Session Interaction Features

| Daily Feature | Formula |
|---------------|---------|
| `usb_after_hours_sessions` | COUNT(usb_used AND is_after_hours) |
| `file_heavy_sessions` | COUNT(session_file_count > 10) |
| `short_usb_sessions` | COUNT(duration < 15 AND usb_used) — quick USB grabs |

**Insider threat relevance:**
- Quick USB grabs (`short_usb_sessions`) — suspicious short sessions with USB activity
- After-hours data exfiltration (`usb_after_hours_sessions`)
- Abnormal session patterns (`mean_session_duration`, `session_hour_spread`)
- Session entropy spikes (`mean_session_entropy`) — unusual activity distributions

---

## 8. Interaction Features (4 features)

Cross-feature combinations that capture suspicious co-occurrence patterns.

| Feature | Description | Computation | Transform |
|---------|-------------|------------|-----------|
| `file_usb_interaction` | File copies with USB present | `LN(1 + file_copy_count * (usb_connect > usb_disconnect))` | log1p |
| `external_email_attachments` | External emails with attachments | `LN(1 + external_email_count * total_attachments)` | log1p |
| `http_after_hours` | HTTP requests during after-hours | `LN(1 + http_request_count * (after_hours_actions > 0))` | log1p |
| `email_after_hours` | Emails during after-hours | `LN(1 + emails_sent * (after_hours_actions > 0))` | log1p |

**Insider threat relevance:**
- Data exfiltration via USB (`file_usb_interaction`)
- Sensitive data via email (`external_email_attachments`)
- Suspicious timing (`http_after_hours`, `email_after_hours`)

---

## 9. Cyclical & Binary Features (7 features)

Temporal encoding and binary flags.

| Feature | Description | Computation | Transform |
|---------|-------------|------------|-----------|
| `day_of_week_sin` | Cyclical encoding of day of week (sine) | `SIN(2*PI*dow/7)` | raw |
| `day_of_week_cos` | Cyclical encoding of day of week (cosine) | `COS(2*PI*dow/7)` | raw |
| `is_weekend` | Binary: Saturday or Sunday | `CASE WHEN dow >= 5 THEN 1 ELSE 0 END` | binary |
| `role_changed` | Binary: role changed in current month | `COALESCE(uc.role_changed, 0)` | binary |
| `dept_changed` | Binary: department changed in current month | `COALESCE(uc.dept_changed, 0)` | binary |
| `functional_unit_changed` | Binary: FU changed in current month | `COALESCE(uc.fu_changed, 0)` | binary |
| `terminated_flag` | Binary: user terminated this month | `CASE WHEN date >= termination_month THEN 1 ELSE 0 END` | binary |

**Insider threat relevance:**
- Weekend activity (`is_weekend`) — suspicious when combined with other anomalies
- Role changes (`role_changed`) — insider threat often precedes or follows role changes
- Terminated users (`terminated_flag`) — post-termination activity (if any)

---

## 10. Personality Traits (5 features) — Context-Only

Big Five personality scores from psychometric assessment. Used as **context-only** features (not reconstructed in loss function).

| Feature | Description | Source | Transform |
|---------|-------------|--------|-----------|
| `openness` | Openness to experience | psychometric.csv | raw |
| `conscientiousness` | Conscientiousness | psychometric.csv | raw |
| `extraversion` | Extraversion | psychometric.csv | raw |
| `agreeableness` | Agreeableness | psychometric.csv | raw |
| `neuroticism` | Neuroticism | psychometric.csv | raw |

**Why context-only:** Personality traits are static per user and not behavioral. Including them in reconstruction loss would force the model to learn a user's personality, which is not the goal. They serve as contextual embeddings that help the model understand baseline behavior.

**Insider threat relevance:**
- Some insider threat scenarios correlate with personality traits (e.g., high neuroticism, low conscientiousness)

---

## 11. Categorical Features (5 features) — Embedded

Categorical features embedded as context vectors.

| Feature | Description | Cardinality | Embedding Dim |
|---------|-------------|-------------|---------------|
| `user_id` | User identifier | ~1000 | 16 |
| `pc` | PC identifier | ~1000 | 16 |
| `role` | Job role | ~20 | 8 |
| `department` | Department | ~10 | 4 |
| `functional_unit` | Functional unit | 6 | 4 |

**Encoding:** `DENSE_RANK() OVER (ORDER BY ...)` → integer IDs → learned embeddings.

**Why embedded:** These provide context (who is acting, from where, in what role) but are not reconstructed. They help the model understand user-specific baselines.

---

## 12. Feature Transformations

### Log1p Transform

**Why log1p?** Applied to all count-based features (skewed distributions):

```python
log1p(x) = ln(1 + x)
```

**Reasons:**

1. **Handles skewed distributions** — Event counts (emails, file copies, HTTP requests) follow power laws: most days have low activity, some days have huge spikes. log1p compresses the long tail, making the distribution more Gaussian-like for the model.

2. **Handles zeros gracefully** — `ln(1+x)` is defined at x=0 (ln(1)=0), whereas `ln(x)` would be undefined.

3. **Emphasizes relative changes** — Going from 1→2 emails is more anomalous than 100→101. log1p captures this: ln(2)=0.69 vs ln(101)=4.62, difference is 0.69 vs 0.01.

4. **Gradient stability** — Large raw values (e.g., 10,000 events) can cause exploding gradients in neural networks. log1p keeps values bounded.

5. **Better for StandardScaler** — After log1p, features have tighter distributions, so z-score normalization is more meaningful.

**Example:**
```
Raw: [0, 1, 5, 10, 100, 1000]  →  highly skewed
log1p: [0, 0.69, 1.79, 2.40, 4.62, 6.91]  →  more uniform
```

Applied to: all activity counts, session counts, session durations, interaction features.

### StandardScaler (Z-Score Normalization)

Fitted **only on training users** to prevent data leakage:

```python
z = (x - mean) / std
```

Applied to all continuous features after log1p transform.

### Per-User Z-Scores & Rolling Stats

For selected features (`user_normalized` list), compute per-user z-scores and 20-day rolling statistics:

```sql
-- Per-user z-score
(col - AVG(col) OVER (PARTITION BY user_id)) / 
    STDDEV(col) OVER (PARTITION BY user_id)

-- Rolling mean (20-day window)
AVG(col) OVER (PARTITION BY user_id ORDER BY date 
    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)

-- Rolling std (20-day window)
STDDEV(col) OVER (PARTITION BY user_id ORDER BY date 
    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
```

Features with per-user normalization:
- `http_request_count`
- `emails_sent`
- `file_copy_count`
- `total_actions`
- `sess_count`
- `mean_session_duration`
- `total_session_time`

**Why:** Captures deviations from a user's own historical baseline, not just population averages.

---

## 13. Feature Computation Pipeline

```
Raw Events (logon, device, email, file, http)
    │
    ▼
[Stage 1a] Daily Aggregation
    → Activity counts per (user_id, date)
    → 18 daily features
    │
    ▼
[Stage 1b] Session Detection
    → Pair Logon/Logoff events → sessions table
    │
    ▼
[Stage 1c] Per-Session Features
    → Compute ~10 features per session
    → Aggregate to daily level → 18 session-derived features
    │
    ▼
[Stage 1d] Merge Session Stats
    → LEFT JOIN session stats onto daily table
    │
    ▼
[Stage 2] Enrich with Static Features
    → Join users, psychometric, LDAP changes, terminations, insider labels
    │
    ▼
[Stage 3] Compute Features
    → log1p transforms on count features
    → Interaction features (cross-feature products)
    → Z-scores + rolling stats (per-user)
    → Cyclical encoding (day of week)
    → Binary flags
    │
    ▼
[Stage 4] StandardScaler + Export
    → Fit scaler on train users only
    → Export to numpy arrays
```

---

## 14. Detection Capabilities

### Insider Threat Scenario Coverage

| Scenario | Key Features | Detection Strength |
|----------|---------------|-------------------|
| **Data Theft via USB** | `usb_connect_count`, `file_copy_count`, `short_usb_sessions`, `usb_after_hours_sessions` | Strong - captures quick USB grabs and after-hours activity |
| **Data Theft via Email** | `external_email_count`, `total_attachments`, `external_email_attachments` | Strong - identifies external communication with attachments |
| **Sabotage** | `after_hours_actions`, `mean_session_entropy`, `session_hour_spread` | Strong - detects unusual timing and activity patterns |
| **Unauthorized Access** | `num_sessions`, `active_hours`, `num_distinct_pcs` | Strong - flags abnormal access patterns |
| **Credential Abuse** | `total_actions`, `peak_hour_concentration`, `mean_session_duration` | Strong - identifies volume and timing anomalies |

### Session-Level Granularity Benefits

1. **Temporal Precision:** Detects suspicious short sessions (e.g., 15-minute USB grabs) that daily aggregation masks
2. **Pattern Recognition:** Identifies unusual session timing (after-hours, weekend activity)
3. **Behavioral Context:** Session entropy reveals abnormal activity distributions
4. **Investigation Focus:** Enables SOC analysts to drill down to specific suspicious sessions

---

## 15. Session-Level Evaluation Metrics

Our approach enables **session-level** evaluation, measuring whether the model can pinpoint specific suspicious sessions within a flagged day.

### Evaluation Process

1. Model flags a day as anomalous (high reconstruction error)
2. Compute **per-feature contribution** to that day's error
3. Map high-contribution session-derived features back to originating sessions
4. Compare flagged sessions against ground-truth insider activity timestamps

### Session-Level Metrics

| Metric | Description |
|--------|-------------|
| **Session Precision** | Of all sessions flagged suspicious, how many overlap with actual insider activity? |
| **Session Recall** | Of all sessions with actual insider activity, how many were flagged? |
| **Session F1** | Harmonic mean of session precision and recall |
| **Session Localization Accuracy** | % of flagged days where the correct session was identified as most suspicious |
| **Mean Session Latency** | Time from insider session start to first flagged session (hours) |

### Multi-Level Evaluation Framework

```
Evaluation Levels:
  ├── User-Level     → det_rate, user precision/recall/F1
  ├── Sequence-Level → AUC, AUPRC, FPR, FNR, confusion matrix
  └── Session-Level  → session precision/recall/F1, localization accuracy
```

---

## 16. SOC Session-Level Incident Report

Our SOC report (`soc_report.json`) includes session-level drill-down for each flagged user, providing analysts with the **exact session(s)** to investigate, not just the day.

### Report Structure

```json
{
  "alert_id": "ALERT-2024-001",
  "user_id": "ACM2278",
  "risk_score": 0.87,
  "flagged_days": [
    {
      "date": "2011-05-18",
      "day_anomaly_score": 0.82,
      "top_contributing_features": [
        "file_usb_interaction",
        "num_usb_sessions",
        "short_usb_sessions"
      ],
      "sessions": [
        {
          "session_id": 14523,
          "start": "2011-05-18T09:02:00",
          "end": "2011-05-18T17:15:00",
          "duration_min": 493,
          "risk": "low",
          "summary": "Normal work session, 120 HTTP requests, 8 emails"
        },
        {
          "session_id": 14524,
          "start": "2011-05-18T22:45:00",
          "end": "2011-05-18T23:12:00",
          "duration_min": 27,
          "risk": "high",
          "summary": "After-hours session: 15 file copies to USB, 3 .exe files",
          "indicators": [
            "After-hours USB session (22:45)",
            "Executable files copied to removable media",
            "Short session with high file activity (burst_score=0.89)"
          ],
          "events": [
            {"time": "22:46:00", "type": "device", "detail": "USB connect"},
            {"time": "22:47:00", "type": "file", "detail": "copy project_specs.docx to E:\\"},
            {"time": "22:48:00", "type": "file", "detail": "copy deploy_tool.exe to E:\\"},
            {"time": "22:50:00", "type": "file", "detail": "copy client_db.zip to E:\\"},
            {"time": "23:10:00", "type": "device", "detail": "USB disconnect"},
            {"time": "23:12:00", "type": "logon", "detail": "Logoff"}
          ]
        }
      ]
    }
  ],
  "recommendation": "Investigate session #14524 — after-hours USB data exfiltration pattern",
  "confidence": "high"
}
```

### SOC Investigation Workflow

```
Model flags user → SOC report generated
    │
    ▼
Analyst opens report
    │
    ▼
Sees flagged DAY with anomaly score
    │
    ▼
Drills into flagged SESSION(s) within that day
    │
    ▼
Sees raw events within the session
    │
    ▼
Takes action (escalate / dismiss / monitor)
```

### SOC Investigation Benefits

| Benefit | Our Session-Level Approach |
|---------|---------------------------|
| **Alert context** | "User X had a suspicious 27-min USB session at 10:45pm on May 18" |
| **Investigation time** | Analyst reviews 1 session, ~6 events instead of full day |
| **False alarm triage** | Easy verification — "flagged session was scheduled backup" |
| **Evidence for escalation** | Specific events with timestamps and session context |
| **Mean time to respond** | Minutes (session pre-filtered) vs hours (manual review) |

---

## 17. Design Rationale

**Session-aware approach chosen because:**
- **Temporal Granularity:** Captures intra-day behavioral patterns critical for insider threat detection
- **Investigation Focus:** Enables SOC analysts to drill down to specific suspicious sessions
- **Comprehensive Coverage:** Addresses multiple insider threat scenarios (USB theft, email exfiltration, sabotage)
- **Scalable Implementation:** Efficient SQL-based processing suitable for large datasets
- **Actionable Intelligence:** Provides precise timestamps and context for security response

**Architecture Benefits:**
- Preserves daily aggregation for long-term pattern modeling
- Adds session-level features for short-term anomaly detection
- Maintains model simplicity (same Transformer AE architecture)
- Enables multi-level evaluation (user, sequence, session)

---

## 19. Feature Count Summary

| Category | Count | Details |
|----------|-------|---------|
| Daily activity counts | 18 | Raw event aggregations |
| Session-derived | 18 | Per-session stats → daily aggregation |
| Interaction | 4 | Cross-feature products |
| Cyclical | 2 | Day of week sin/cos |
| Binary | 5 | Weekend, role changes, termination |
| Personality (context-only) | 5 | Big Five traits |
| **Total continuous** | **~52** | After scaling and transformations |
| Categorical (embedded) | 5 | user_id, pc, role, department, functional_unit |
| **Grand total** | **~57** | Input to Transformer AE |
