"""
Training Module
================
Training loop for LSTM-Autoencoder with TensorBoard logging.

Key features:
- Train on normal sequences only
- MSE loss for reconstruction
- Adam optimizer with paper's hyperparameters
- Early stopping on validation loss
- TensorBoard logging
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.lstm_autoencoder import LSTMAutoencoder, create_model, count_parameters


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train, val, test data and labels."""
    project_root = Path(__file__).parent.parent.parent
    processed_dir = project_root / config['data']['processed_dir']
    
    X_train = np.load(processed_dir / "X_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_test = np.load(processed_dir / "y_test.npy")
    
    return X_train, X_val, X_test, y_test


def create_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training and validation."""
    
    # Convert to tensors
    train_tensor = torch.FloatTensor(X_train)
    val_tensor = torch.FloatTensor(X_val)
    
    # For autoencoder, target = input
    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset = TensorDataset(val_tensor, val_tensor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader


class Trainer:
    """
    Trainer for LSTM-Autoencoder.
    
    Args:
        model: LSTM-Autoencoder model
        config: Configuration dictionary
        device: torch device (cuda or cpu)
    """
    
    def __init__(
        self,
        model: LSTMAutoencoder,
        config: dict,
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Paper hyperparameters
        self.epochs = config['training']['epochs']
        self.lr = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        
        # Setup output directory
        project_root = Path(__file__).parent.parent.parent
        self.output_dir = project_root / "outputs"
        self.model_dir = self.output_dir / "models"
        self.log_dir = self.output_dir / "logs"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(self.log_dir / f"run_{int(time.time())}")
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int = None, total_epochs: int = None) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        
        # Batch-level progress bar
        if epoch and total_epochs:
            desc = f"Epoch {epoch}/{total_epochs}"
        elif epoch:
            desc = f"Epoch {epoch}"
        else:
            desc = "Training"
        pbar = tqdm(train_loader, desc=desc, leave=False, unit="batch")
        
        for batch_x, _ in pbar:
            batch_x = batch_x.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            reconstructed = self.model(batch_x)
            loss = self.criterion(reconstructed, batch_x)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
            # Update progress bar with current loss
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(train_loader.dataset)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, dict]:
        """Validate model. Returns average loss and reconstruction error stats."""
        self.model.eval()
        total_loss = 0.0
        all_errors = []
        
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(self.device)
                reconstructed = self.model(batch_x)
                loss = self.criterion(reconstructed, batch_x)
                total_loss += loss.item() * batch_x.size(0)
                
                # Calculate per-sample reconstruction errors
                errors = self.model.get_reconstruction_error(batch_x)
                all_errors.append(errors.cpu().numpy())
        
        all_errors = np.concatenate(all_errors)
        
        recon_stats = {
            'mean': float(np.mean(all_errors)),
            'std': float(np.std(all_errors)),
            'min': float(np.min(all_errors)),
            'max': float(np.max(all_errors)),
            'p95': float(np.percentile(all_errors, 95)),
            'p99': float(np.percentile(all_errors, 99)),
        }
        
        return total_loss / len(val_loader.dataset), recon_stats
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.model_dir / "checkpoint_latest.pt")
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.model_dir / "checkpoint_best.pt")
    
    def evaluate_test(
        self, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        batch_size: int = 256
    ) -> dict:
        """
        Evaluate on test set for MONITORING ONLY.
        
        This does NOT influence training decisions (model selection uses val loss).
        Metrics are logged to TensorBoard under 'TestMonitor/'.
        
        Returns:
            Dictionary of test metrics
        """
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(X_test), batch_size):
                batch = torch.FloatTensor(X_test[i:i + batch_size]).to(self.device)
                error = self.model.get_reconstruction_error(batch)
                errors.append(error.cpu().numpy())
        
        errors = np.concatenate(errors)
        
        # Find threshold (using F1-maximizing on this test set - for monitoring only)
        thresholds = np.percentile(errors, np.linspace(80, 99.9, 50))
        best_f1 = 0
        best_threshold = thresholds[0]
        
        for threshold in thresholds:
            y_pred = (errors > threshold).astype(int)
            if y_pred.sum() > 0:
                f1 = f1_score(y_test, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        # Compute metrics at best threshold
        y_pred = (errors > best_threshold).astype(int)
        
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': best_f1,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'threshold': best_threshold,
            'error_normal_mean': errors[y_test == 0].mean(),
            'error_insider_mean': errors[y_test == 1].mean(),
        }
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        early_stopping_patience: int = 20,
        test_eval_interval: int = 10
    ) -> dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            X_test: Test features (optional, for monitoring only)
            y_test: Test labels (optional, for monitoring only)
            early_stopping_patience: Patience for early stopping
            test_eval_interval: Evaluate on test every N epochs (monitoring only)
        
        Returns:
            Training history dictionary
        """
        print("\n" + "=" * 60)
        print("TRAINING LSTM-AUTOENCODER")
        print("=" * 60)
        print(f"  Device: {self.device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Training samples: {len(train_loader.dataset):,}")
        print(f"  Validation samples: {len(val_loader.dataset):,}")
        print(f"  Model parameters: {count_parameters(self.model):,}")
        if X_test is not None:
            print(f"  Test monitoring: every {test_eval_interval} epochs")
        
        history = {'train_loss': [], 'val_loss': [], 'recon_stats': [], 'test_metrics': []}
        patience_counter = 0
        
        for epoch in range(1, self.epochs + 1):
            # Train (with batch-level progress bar)
            train_loss = self.train_epoch(train_loader, epoch, self.epochs)
            val_loss, recon_stats = self.validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['recon_stats'].append(recon_stats)
            
            # Log losses to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Log reconstruction error stats to TensorBoard
            self.writer.add_scalar('ReconError/mean', recon_stats['mean'], epoch)
            self.writer.add_scalar('ReconError/std', recon_stats['std'], epoch)
            self.writer.add_scalar('ReconError/p95', recon_stats['p95'], epoch)
            self.writer.add_scalar('ReconError/p99', recon_stats['p99'], epoch)
            
            # Test monitoring (every N epochs) - FOR MONITORING ONLY
            if X_test is not None and epoch % test_eval_interval == 0:
                test_metrics = self.evaluate_test(X_test, y_test)
                history['test_metrics'].append({'epoch': epoch, **test_metrics})
                
                # Log to TensorBoard under 'TestMonitor/' prefix
                self.writer.add_scalar('TestMonitor/Accuracy', test_metrics['accuracy'], epoch)
                self.writer.add_scalar('TestMonitor/Precision', test_metrics['precision'], epoch)
                self.writer.add_scalar('TestMonitor/Recall', test_metrics['recall'], epoch)
                self.writer.add_scalar('TestMonitor/F1', test_metrics['f1'], epoch)
                self.writer.add_scalar('TestMonitor/FPR', test_metrics['fpr'], epoch)
                self.writer.add_scalar('TestMonitor/ErrorNormalMean', test_metrics['error_normal_mean'], epoch)
                self.writer.add_scalar('TestMonitor/ErrorInsiderMean', test_metrics['error_insider_mean'], epoch)
            
            # Check for best model (ONLY uses val loss, NOT test metrics)
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Print epoch summary
            best_marker = " *best" if is_best else ""
            print(f"  → Train: {train_loss:.4f}, Val: {val_loss:.4f}{best_marker}")
            
            # Early stopping (based on val loss, NOT test metrics)
            if patience_counter >= early_stopping_patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(best: {self.best_epoch}, val_loss: {self.best_val_loss:.6f})")
                break

        
        self.writer.close()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.6f}")
        print(f"  Model saved: {self.model_dir / 'checkpoint_best.pt'}")
        
        return history


def main():
    """Main training entry point."""
    config = load_config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    X_train, X_val, X_test, y_test = load_data(config)
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Test insiders: {y_test.sum():,}")
    print(f"  Test normals: {(y_test == 0).sum():,}")
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader, val_loader = create_dataloaders(X_train, X_val, batch_size)
    
    # Create model
    model = create_model(config)
    print(f"\nModel created: {count_parameters(model):,} parameters")
    
    # Create trainer and train
    # Test data passed for MONITORING ONLY - does not influence model selection
    trainer = Trainer(model, config, device)
    history = trainer.train(
        train_loader, 
        val_loader,
        X_test=X_test,
        y_test=y_test,
        test_eval_interval=10  # Monitor test metrics every 10 epochs
    )
    
    # Save training history
    project_root = Path(__file__).parent.parent.parent
    np.save(project_root / "outputs" / "training_history.npy", history)
    print(f"\nTraining history saved to outputs/training_history.npy")


if __name__ == "__main__":
    main()
