# Hybrid Plan: Session → Daily → Window Transformer for Insider Threat Detection

**Dataset:** CERT R4.2 (~1000 users, ~501 days, ~32M events)
**Prerequisite:** Complete Phase 8 (daily pipeline working end-to-end)
**Objective:** Enrich daily feature vectors with session-level behavioral statistics

---

## 1. Motivation

The current daily aggregation pipeline achieves strong results (AUC ~0.998,
AUPRC ~0.93) but loses intra-day temporal structure. A user who copies 50 files
in one normal 8-hour session looks identical to a user who copies 5 files in a
suspicious 30-minute late-night session plus 45 files during normal work.

The hybrid approach preserves this signal by extracting **per-session features
first**, then aggregating them into daily vectors alongside existing daily counts.

---

## 2. Architecture Overview

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
│  ~15 features per session       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Daily Aggregation              │
│  Existing ~45 features          │
│  + ~20 session-derived features │
│  = ~65 continuous features      │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  60-day Sliding Window          │
│  Same as current pipeline       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Transformer AE (unchanged)     │
│  Just n_continuous increases     │
└─────────────────────────────────┘
```

**Key insight:** The model architecture does not change. Only the feature
engineering pipeline changes. The Transformer AE receives a wider daily
feature vector (~65 instead of ~45 continuous features).

---

## 3. Session Detection

### 3.1 Boundary Rules

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

### 3.2 Edge Cases

| Case | Resolution |
|------|------------|
| Missing Logoff | End session at next Logon or end of day |
| Multiple Logons without Logoff | Each Logon starts a new session |
| Logoff without Logon | Ignore (orphan event) |
| Overnight session (crosses midnight) | Assign to the start date |
| Shared PC (multiple users) | Sessions are per (user_id, pc) |
| Very short sessions (< 1 min) | Keep — may indicate automated/scripted behavior |
| Very long sessions (> 12 hours) | Cap duration at 12h for feature computation |

### 3.3 Session Table Schema

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

## 4. Per-Session Features

### 4.1 Activity Features (per session)

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

### 4.2 Temporal Features (per session)

| Feature | Description |
|---------|-------------|
| `session_hour_start` | Hour of day (0-23) when session started |
| `session_is_after_hours` | Started outside 8:00-18:00 |
| `session_is_weekend` | Started on Saturday/Sunday |

### 4.3 Behavioral Features (per session)

| Feature | Description |
|---------|-------------|
| `session_event_entropy` | Shannon entropy of event type distribution |
| `session_burst_score` | Max events in any 5-min window / total events |

---

## 5. Daily Aggregation of Session Features

For each `(user_id, date)`, aggregate all sessions into daily statistics:

### 5.1 Session Count & Duration Stats

| Daily Feature | Aggregation |
|---------------|-------------|
| `num_sessions` | COUNT of sessions (already exists, now more accurate) |
| `mean_session_duration` | AVG(session_duration_min) |
| `max_session_duration` | MAX(session_duration_min) |
| `std_session_duration` | STDDEV(session_duration_min) |
| `total_session_time` | SUM(session_duration_min) |

### 5.2 Session Timing Stats

| Daily Feature | Aggregation |
|---------------|-------------|
| `num_after_hours_sessions` | COUNT where is_after_hours = true |
| `after_hours_session_ratio` | num_after_hours_sessions / num_sessions |
| `earliest_session_hour` | MIN(session_hour_start) |
| `latest_session_hour` | MAX(session_hour_start) |
| `session_hour_spread` | latest - earliest |

### 5.3 Session Behavior Stats

| Daily Feature | Aggregation |
|---------------|-------------|
| `mean_session_event_count` | AVG(session_event_count) |
| `max_session_event_count` | MAX(session_event_count) |
| `mean_session_entropy` | AVG(session_event_entropy) |
| `num_usb_sessions` | COUNT where session_usb_used = 1 |
| `num_exe_sessions` | COUNT where session_exe_copied = 1 |
| `max_session_burst` | MAX(session_burst_score) |
| `longest_inter_session_gap` | MAX gap between consecutive sessions (minutes) |

### 5.4 Session Interaction Features

| Daily Feature | Formula |
|---------------|---------|
| `usb_after_hours_sessions` | COUNT(usb_used AND is_after_hours) |
| `file_heavy_sessions` | COUNT(session_file_count > 10) |
| `short_usb_sessions` | COUNT(duration < 15 AND usb_used) — quick USB grabs |

### 5.5 Total New Features: ~20

Combined with existing ~45 → **~65 continuous features** per daily vector.

---

## 6. What Changes in the Pipeline

| Component | Change |
|-----------|--------|
| `src/data/feature_engineering.py` | Add session detection + session feature extraction SQL |
| `src/data/sequence_creation.py` | No change (just wider vectors) |
| `src/models/transformer.py` | No change (n_continuous passed dynamically) |
| `scripts/03_train.py` | No change |
| `config/config.yaml` | Add session features to continuous list, update behavioral_indices |
| `REBUILD_PLAN.md` | Reference this plan |

### What does NOT change
- Model architecture (same Transformer AE)
- Training strategy (normal users only)
- Anomaly scoring (mean + max MSE)
- Evaluation pipeline
- FAISS embedding extraction

---

## 7. Implementation Steps

### Step 1: Session Detection SQL
Add `_create_sessions_table(conn)` to `feature_engineering.py`:
- Pair Logon/Logoff events
- Handle edge cases (missing logoff, overnight, shared PC)
- Create `sessions` table in DuckDB

### Step 2: Per-Session Features SQL
Add `_compute_session_features(conn)` to `feature_engineering.py`:
- Join events to sessions by user_id, pc, time range
- Compute per-session activity counts, entropy, burst score
- Create `session_features` table

### Step 3: Daily Aggregation of Session Features
Modify `_create_daily_aggregation(conn)` to:
- LEFT JOIN session stats onto existing daily aggregation
- Add ~20 new columns
- Handle days with 0 sessions (NULL → 0)

### Step 4: Update Config
```yaml
features:
  continuous:
    - ... existing 45 ...
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
    - max_session_burst
    - longest_inter_session_gap
    - usb_after_hours_sessions
    - file_heavy_sessions
    - short_usb_sessions
```

### Step 5: Update behavioral_indices
All new session features are behavioral (not static), so they should be
included in the reconstruction loss.

### Step 6: Retrain
```bash
python scripts/03_train.py
```
No architecture changes needed — `n_continuous` is read from data shape.

### Step 7: Compare Results
Run `04_evaluate.py` and compare:
- AUPRC (primary metric)
- det_rate and latency
- Per-scenario breakdown (which scenarios improved?)

---

## 8. Expected Impact

| Metric | Daily Only | Hybrid (expected) | Why |
|--------|-----------|-------------------|-----|
| **AUPRC** | ~0.93 | ~0.94-0.96 | Session patterns add discriminative signal |
| **det_rate** | ~96% | ~96-98% | May catch edge cases missed by daily |
| **latency** | ~15h | ~10-15h | Session timing features enable faster detection |
| **FPR** | ~0.33% | ~0.25-0.35% | More features = better normal modeling |

### Which scenarios benefit most

| Scenario | Current Signal | Hybrid Adds |
|----------|---------------|-------------|
| **Scenario 1** (data theft via USB) | file_copy_count, usb_connect_count | short_usb_sessions, usb_after_hours_sessions |
| **Scenario 2** (data theft via email) | external_email_count, total_attachments | session with external email + attachments co-occurrence |
| **Scenario 3** (sabotage) | after_hours_actions | session entropy spike, unusual session duration patterns |

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Session boundary detection errors | Validate against known session counts; log orphan events |
| Feature explosion (65 features) | Monitor reconstruction loss; may need larger d_model |
| Overfitting on new features | Same regularization (dropout, weight decay); augmentation |
| Longer feature engineering time | Session SQL is O(n log n) — manageable for 32M events |
| No improvement | Keep daily-only as fallback; A/B compare results |

---

## 10. Decision Criteria

**Proceed with hybrid if:**
- Daily pipeline is stable and fully evaluated (Phase 8 complete)
- AUPRC is below 0.95 after tuning
- Specific insider scenarios are consistently missed
- Sub-day detection latency is required

**Stay with daily if:**
- AUPRC > 0.95 with daily features alone
- All insider scenarios detected at acceptable latency
- Simplicity is preferred for deployment

---

## 11. Estimated Effort

| Task | Time |
|------|------|
| Session detection SQL + edge cases | 1-2 days |
| Per-session feature extraction | 1 day |
| Daily aggregation integration | 0.5 day |
| Config + pipeline testing | 0.5 day |
| Retraining + evaluation | 1 day |
| Comparison analysis | 0.5 day |
| **Total** | **~5 days** |
