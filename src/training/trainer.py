"""
Training Module — InsiderTransformerAE
=======================================
Trainer class for the session-aware Transformer autoencoder.

Encapsulates all training logic:
- Data loading (session-aware features)
- Model creation with categorical embeddings
- Training loop with AMP, gradient clipping, cosine LR schedule
- Validation and test-set monitoring (AUC, AUPRC, latency)
- Checkpointing (best, latest, final) and resume support
- TensorBoard logging
- Early stopping on validation loss

Used by scripts/03_train.py as a thin CLI wrapper.
"""

import gc
import json
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve
from torch.amp import GradScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.transformer import create_model, count_parameters
from evaluation.scoring import score_dataset, compute_ranking_metrics
from utils import load_config, set_seed, get_device


# =========================================================================
# Standalone helpers (used by Trainer, but pure functions)
# =========================================================================

def full_reconstruction_loss(predictions, targets, behavioral_indices=None):
    """MSE on all positions, optionally restricted to behavioral features.

    Args:
        predictions: (batch, seq_len, n_continuous)
        targets:     (batch, seq_len, n_continuous)
        behavioral_indices: optional (n_behavioral,) long tensor — feature indices for loss
    Returns:
        Scalar loss
    """
    if behavioral_indices is not None:
        diff = (predictions[:, :, behavioral_indices] - targets[:, :, behavioral_indices]) ** 2
    else:
        diff = (predictions - targets) ** 2
    return diff.mean()


def create_scheduler(optimizer, total_epochs, warmup_epochs, n_train_batches):
    """Linear warmup + cosine decay."""
    warmup_steps = warmup_epochs * n_train_batches
    total_steps = total_epochs * n_train_batches

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================================================================
# Trainer
# =========================================================================

class Trainer:
    """
    Trainer for InsiderTransformerAE.

    Manages the full training lifecycle: data loading, model creation,
    training loop, validation, test monitoring, checkpointing, and resume.

    Args:
        config: Configuration dictionary (from config.yaml)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        patience: Early stopping patience
        use_amp: Enable mixed precision training
        eval_interval: Run test evaluation every N epochs
        checkpoint_interval: Save latest checkpoint every N epochs
        dry_run: If True, run 1 epoch with limited steps
        dry_run_steps: Max batches per phase in dry run
        resume: Resume from best_model.pt checkpoint
        output_dir: Directory for outputs (checkpoints, logs, history)
        data_dir: Directory containing feature arrays
    """

    def __init__(
        self,
        config: dict,
        epochs: int,
        batch_size: int,
        lr: float,
        patience: int = 20,
        use_amp: bool = False,
        eval_interval: int = 10,
        checkpoint_interval: int = 10,
        dry_run: bool = False,
        dry_run_steps: int = 5,
        resume: bool = False,
        output_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ):
        self.config = config
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.use_amp = use_amp
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval
        self.dry_run = dry_run
        self.dry_run_steps = dry_run_steps if dry_run else None
        self.resume = resume

        project_root = Path(__file__).parent.parent.parent
        self.data_dir = data_dir or project_root / config['data']['processed_dir']
        self.output_dir = output_dir or project_root / 'outputs'
        if self.dry_run:
            self.epochs = 1
            self.output_dir = self.output_dir / 'dry_run'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.weight_decay = config['training'].get('weight_decay', 0.01)
        self.clip_norm = config['training'].get('clip_grad_norm', 1.0)
        self.seed = config['training'].get('seed', 42)
        self.warmup_epochs = config['training'].get('warmup_epochs', 5)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_train_val(self):
        """Load training and validation datasets."""
        d = self.data_dir
        X_train_cont = np.load(d / 'X_train_continuous.npy')
        X_train_cat = np.load(d / 'X_train_categorical.npy')
        X_val_cont = np.load(d / 'X_val_continuous.npy')
        X_val_cat = np.load(d / 'X_val_categorical.npy')

        print(f"  Train: {X_train_cont.shape[0]:,} sequences, "
              f"cont={X_train_cont.shape}, cat={X_train_cat.shape}")
        print(f"  Val:   {X_val_cont.shape[0]:,} sequences, "
              f"cont={X_val_cont.shape}, cat={X_val_cat.shape}")

        train_dataset = TensorDataset(
            torch.tensor(X_train_cont, dtype=torch.float32),
            torch.tensor(X_train_cat, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_cont, dtype=torch.float32),
            torch.tensor(X_val_cat, dtype=torch.long),
        )

        del X_train_cont, X_train_cat, X_val_cont, X_val_cat
        gc.collect()
        return train_dataset, val_dataset

    def _load_test(self):
        """Load test set for monitoring."""
        d = self.data_dir
        X_test_cont = np.load(d / 'X_test_continuous.npy')
        X_test_cat = np.load(d / 'X_test_categorical.npy')
        y_test = np.load(d / 'y_test.npy')
        user_ids_test = np.load(d / 'user_ids_test.npy', allow_pickle=True)

        ts_path = d / 'dates_test.npy'
        timestamps_test = np.load(ts_path, allow_pickle=True) if ts_path.exists() else None

        dataset = TensorDataset(
            torch.tensor(X_test_cont, dtype=torch.float32),
            torch.tensor(X_test_cat, dtype=torch.long),
        )
        return dataset, y_test, user_ids_test, timestamps_test

    # ------------------------------------------------------------------
    # Model setup helpers
    # ------------------------------------------------------------------

    def _get_cat_cardinalities(self):
        """Get cardinality for each categorical feature from preprocessing artifacts."""
        with open(self.data_dir / 'preprocessing_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)

        cat_names = self.config['features']['categorical']
        label_mappings = artifacts.get('label_mappings', {})

        cardinalities = {}
        for name in cat_names:
            if name not in label_mappings:
                raise KeyError(
                    f"Categorical feature '{name}' not found in label_mappings. "
                    f"Available: {list(label_mappings.keys())}. "
                    f"Re-run feature engineering."
                )
            cardinalities[name] = len(label_mappings[name])

        print(f"  Categorical cardinalities: {cardinalities}")
        return cardinalities

    def _get_behavioral_indices(self):
        """Compute indices of behavioral (non-context-only) continuous features.

        Returns None if no context_only features are configured.
        """
        context_only = self.config.get('features', {}).get('context_only', [])
        if not context_only:
            return None

        with open(self.data_dir / 'preprocessing_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)

        continuous_columns = artifacts['continuous_columns']
        context_set = set(context_only)

        missing = context_set - set(continuous_columns)
        if missing:
            raise ValueError(
                f"context_only features not found in continuous_columns: {missing}. "
                f"Available: {continuous_columns}"
            )

        behavioral = [i for i, col in enumerate(continuous_columns) if col not in context_set]

        excluded = [col for col in continuous_columns if col in context_set]
        print(f"  Context-only features excluded from loss: {excluded}")
        print(f"  Behavioral features: {len(behavioral)}/{len(continuous_columns)}")
        return behavioral

    # ------------------------------------------------------------------
    # Train / Validate (single epoch)
    # ------------------------------------------------------------------

    def _train_epoch(self, model, loader, optimizer, scheduler, scaler, device, writer, epoch):
        """Train for one epoch with optional AMP."""
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        n_batches = 0
        use_amp = scaler is not None
        behavioral_idx = model.behavioral_indices

        loop = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for i, (x_cont, x_cat) in enumerate(loop):
            if self.dry_run_steps is not None and n_batches >= self.dry_run_steps:
                break
            global_step = (epoch - 1) * len(loader) + i
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                predictions = model(x_cont, x_cat)
                loss = full_reconstruction_loss(predictions, x_cont, behavioral_idx)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_norm)
                optimizer.step()

            scheduler.step()

            if i % 10 == 0:
                writer.add_scalar('Train/Loss_Step', loss.item(), global_step)
                writer.add_scalar('Train/GradNorm_Step', grad_norm.item(), global_step)
                writer.add_scalar('Train/LR_Step', optimizer.param_groups[0]['lr'], global_step)

            total_loss += loss.item()
            total_grad_norm += grad_norm.item()
            n_batches += 1
            loop.set_postfix(loss=loss.item(), grad=grad_norm.item())

        return total_loss / max(n_batches, 1), total_grad_norm / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self, model, loader, device):
        """Validation reconstruction error."""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        for x_cont, x_cat in loader:
            if self.dry_run_steps is not None and n_batches >= self.dry_run_steps:
                break
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            predictions = model(x_cont, x_cat)
            loss = full_reconstruction_loss(predictions, x_cont, model.behavioral_indices)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Test monitoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_detection_latency(y_test, test_scores, user_ids, timestamps=None):
        """Lightweight latency computation for training-time monitoring."""
        precisions, recalls, thresholds = precision_recall_curve(y_test, test_scores)
        f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
        best_thresh = thresholds[np.argmax(f1s)]
        y_pred = (test_scores >= best_thresh).astype(int)

        insider_users = np.unique(user_ids[y_test == 1])
        latencies_seq = []
        latencies_hours = []
        caught = 0

        for u in insider_users:
            mask = user_ids == u
            u_labels = y_test[mask]
            u_preds = y_pred[mask]

            insider_idx = np.where(u_labels == 1)[0]
            detected_idx = np.where((u_preds == 1) & (u_labels == 1))[0]

            if len(detected_idx) > 0:
                caught += 1
                lat_seq = int(detected_idx[0] - insider_idx[0])
                latencies_seq.append(lat_seq)

                if timestamps is not None:
                    u_ts = timestamps[mask]
                    try:
                        fi_ts = u_ts[insider_idx[0]]
                        fd_ts = u_ts[detected_idx[0]]
                        delta_h = float((fd_ts - fi_ts) / np.timedelta64(1, 'h'))
                        latencies_hours.append(delta_h)
                    except (ValueError, TypeError):
                        pass

        result = {
            'threshold': float(best_thresh),
            'user_detection_rate': caught / max(len(insider_users), 1),
        }
        if latencies_seq:
            result['mean_latency_seq'] = float(np.mean(latencies_seq))
        if latencies_hours:
            result['mean_latency_hours'] = float(np.mean(latencies_hours))
            result['median_latency_hours'] = float(np.median(latencies_hours))

        return result

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _make_checkpoint(self, model, optimizer, scheduler, scaler,
                         epoch, val_loss, n_continuous, cat_cardinalities,
                         behavioral_indices):
        """Build a checkpoint dict."""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'n_continuous': n_continuous,
            'cat_cardinalities': cat_cardinalities,
            'behavioral_indices': behavioral_indices,
        }
        if scaler is not None:
            ckpt['scaler_state_dict'] = scaler.state_dict()
        return ckpt

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """Execute the full training pipeline."""
        print("=" * 60)
        print("INSIDER TRANSFORMER AE — TRAINING")
        print("=" * 60)
        if self.dry_run:
            print(f"  DRY RUN: 1 epoch, up to {self.dry_run_steps} steps per phase. "
                  f"Outputs under {self.output_dir}.")
        print(f"  Epochs: {self.epochs}, Batch: {self.batch_size}, LR: {self.lr}")
        print(f"  Weight decay: {self.weight_decay}, Grad clip: {self.clip_norm}, "
              f"Patience: {self.patience}")

        set_seed(self.seed)
        device = get_device()
        use_amp = torch.cuda.is_available() and self.use_amp
        if use_amp:
            print(f"  Mixed precision (AMP): enabled")
        else:
            print(f"  Mixed precision (AMP): disabled")

        # [1] Load data
        print("\n[1] Loading data...")
        train_dataset, val_dataset = self._load_train_val()

        pin = device.type == 'cuda'
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  shuffle=True, drop_last=True,
                                  num_workers=0, pin_memory=pin)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=0, pin_memory=pin)

        n_continuous = train_dataset[0][0].shape[-1]
        print("  Loading test data for monitoring...")
        test_dataset, y_test, user_ids_test, timestamps_test = self._load_test()
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=0, pin_memory=pin)
        if timestamps_test is None:
            print("  [info] timestamps_test.npy not found — time-based latency skipped")

        cat_cardinalities = self._get_cat_cardinalities()
        behavioral_indices = self._get_behavioral_indices()

        # [2] Create model
        print("\n[2] Creating model...")
        model = create_model(self.config, n_continuous, cat_cardinalities, behavioral_indices)
        model = model.to(device)
        print(f"  Parameters: {count_parameters(model):,}")
        print(f"  d_model={model.d_model}, n_continuous={n_continuous}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)
        scheduler = create_scheduler(optimizer, self.epochs, self.warmup_epochs,
                                     len(train_loader))
        scaler = GradScaler('cuda') if use_amp else None

        writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

        # Resume from checkpoint
        start_epoch = 1
        best_val_loss = float('inf')
        best_epoch = 0
        history = {'train_loss': [], 'val_loss': [], 'lr': [], 'grad_norm': []}

        if self.resume and not self.dry_run:
            ckpt_path = self.output_dir / 'best_model.pt'
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                if 'optimizer_state_dict' in ckpt:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                if scaler is not None and 'scaler_state_dict' in ckpt:
                    scaler.load_state_dict(ckpt['scaler_state_dict'])
                start_epoch = ckpt['epoch'] + 1
                best_val_loss = ckpt['val_loss']
                best_epoch = ckpt['epoch']
                print(f"  Resumed from epoch {ckpt['epoch']} (val_loss={best_val_loss:.6f})")

                hist_path = self.output_dir / 'training_history.json'
                if hist_path.exists():
                    with open(hist_path, 'r') as f:
                        history = json.load(f)
                        if 'grad_norm' not in history:
                            history['grad_norm'] = [0.0] * len(history['train_loss'])
            else:
                print(f"  No checkpoint found, starting fresh")

        # [3] Training loop
        print(f"\n[3] Training (epochs {start_epoch}-{self.epochs})...")
        epochs_no_improve = 0
        epoch = start_epoch - 1
        val_loss = best_val_loss
        t_total = time.time()

        try:
            for epoch in range(start_epoch, self.epochs + 1):
                t0 = time.time()
                train_loss, grad_norm = self._train_epoch(
                    model, train_loader, optimizer, scheduler, scaler, device, writer, epoch,
                )
                val_loss = self._validate(model, val_loader, device)
                current_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - t0

                global_step_end = epoch * len(train_loader)
                writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
                writer.add_scalar('Val/Loss_Epoch', val_loss, epoch)
                writer.add_scalar('Val/Loss_Step', val_loss, global_step_end)
                writer.add_scalar('Train/GradNorm_Epoch', grad_norm, epoch)
                writer.add_scalar('Train/LR_Epoch', current_lr, epoch)

                if epoch % self.eval_interval == 0 and not self.dry_run:
                    print(f"\n  [Test Eval] Running scoring on {len(y_test)} sequences...")
                    test_scores = score_dataset(model, test_loader, device)
                    test_metrics = compute_ranking_metrics(y_test, test_scores)

                    normal_mask = y_test == 0
                    insider_mask = y_test == 1
                    normal_mean = float(test_scores[normal_mask].mean())
                    insider_mean = float(test_scores[insider_mask].mean())

                    print(f"  [Test Eval] AUC: {test_metrics['auc']:.4f}, "
                          f"AUPRC: {test_metrics['auprc']:.4f}")
                    writer.add_scalar('Test/AUC', test_metrics['auc'], epoch)
                    writer.add_scalar('Test/AUPRC', test_metrics['auprc'], epoch)
                    writer.add_scalar('Test/ScoreMean_Normal', normal_mean, epoch)
                    writer.add_scalar('Test/ScoreMean_Insider', insider_mean, epoch)
                    writer.add_scalar('Test/ScoreRatio',
                                      insider_mean / max(normal_mean, 1e-10), epoch)

                    lat = self._compute_detection_latency(
                        y_test, test_scores, user_ids_test, timestamps_test
                    )
                    writer.add_scalar('Test/BestF1_Threshold', lat['threshold'], epoch)
                    writer.add_scalar('Test/UserDetectionRate', lat['user_detection_rate'], epoch)
                    if 'mean_latency_seq' in lat:
                        writer.add_scalar('Test/MeanLatency_Sequences', lat['mean_latency_seq'], epoch)
                    if 'mean_latency_hours' in lat:
                        writer.add_scalar('Test/MeanLatency_Hours', lat['mean_latency_hours'], epoch)
                        writer.add_scalar('Test/MedianLatency_Hours', lat['median_latency_hours'], epoch)

                    det_str = f"det_rate={lat['user_detection_rate']:.0%}"
                    if 'mean_latency_hours' in lat:
                        det_str += f", latency={lat['mean_latency_hours']:.1f}h"
                    elif 'mean_latency_seq' in lat:
                        det_str += f", latency={lat['mean_latency_seq']:.1f} seqs"
                    print(f"  [Test Eval] {det_str}")

                    threshold = lat.get('threshold', 0.0)
                    y_pred = (test_scores >= threshold).astype(int)
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                    print(f"  [Test Eval] Confusion Matrix (Insider=positive):")
                    print(f"    TN: {tn:5d} | FP: {fp:5d}")
                    print(f"    FN: {fn:5d} | TP: {tp:5d}")
                    print(f"    FPR: {fpr:.2%} | FNR: {fnr:.2%}")
                    writer.add_scalar('Test/CM_TN', tn, epoch)
                    writer.add_scalar('Test/CM_FP', fp, epoch)
                    writer.add_scalar('Test/CM_FN', fn, epoch)
                    writer.add_scalar('Test/CM_TP', tp, epoch)
                    writer.add_scalar('Test/FPR', fpr, epoch)
                    writer.add_scalar('Test/FNR', fnr, epoch)

                    model.train()

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['lr'].append(current_lr)
                history['grad_norm'].append(grad_norm)

                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_no_improve = 0
                    ckpt_dict = self._make_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, val_loss,
                        n_continuous, cat_cardinalities, behavioral_indices,
                    )
                    torch.save(ckpt_dict, self.output_dir / 'best_model.pt')
                    with open(self.output_dir / 'training_history.json', 'w') as f:
                        json.dump(history, f)
                else:
                    epochs_no_improve += 1

                if self.checkpoint_interval > 0 and epoch % self.checkpoint_interval == 0:
                    latest_ckpt = self._make_checkpoint(
                        model, optimizer, scheduler, scaler, epoch, val_loss,
                        n_continuous, cat_cardinalities, behavioral_indices,
                    )
                    torch.save(latest_ckpt, self.output_dir / 'latest_model.pt')

                if epoch <= 10 or epoch % 5 == 0 or improved:
                    star = " ★" if improved else ""
                    print(f"  Epoch {epoch:3d}/{self.epochs} | "
                          f"train={train_loss:.6f} val={val_loss:.6f} | "
                          f"lr={current_lr:.2e} gnorm={grad_norm:.2f} | "
                          f"{elapsed:.1f}s{star}")

                if epochs_no_improve >= self.patience:
                    print(f"\n  Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        finally:
            history['last_epoch'] = epoch
            with open(self.output_dir / 'training_history.json', 'w') as f:
                json.dump(history, f)
            writer.close()

        total_time = time.time() - t_total
        print(f"\n  Best: epoch {best_epoch}, val_loss={best_val_loss:.6f}")
        print(f"  Total training time: {total_time:.1f}s "
              f"({total_time/60:.1f}m)")

        # [4] Save final model
        print(f"\n[4] Saving outputs...")
        final_ckpt = self._make_checkpoint(
            model, optimizer, scheduler, scaler, epoch, val_loss,
            n_continuous, cat_cardinalities, behavioral_indices,
        )
        torch.save(final_ckpt, self.output_dir / 'final_model.pt')

        print(f"  Saved: best_model.pt (epoch {best_epoch})")
        print(f"  Saved: final_model.pt (epoch {epoch})")
        if self.checkpoint_interval > 0:
            latest_epoch = (epoch // self.checkpoint_interval) * self.checkpoint_interval
            if latest_epoch > 0:
                print(f"  Saved: latest_model.pt (epoch {latest_epoch})")
        print(f"  Saved: training_history.json ({len(history['train_loss'])} epochs)")
        if self.dry_run:
            print("\nDry run complete. Checkpoints under outputs/dry_run/.")
        print("\nTraining complete! Next: python scripts/04_evaluate.py")
