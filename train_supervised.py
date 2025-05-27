"""
Supervised Learning Training for CARLA Driving
Train Vision Transformer on expert demonstrations
"""

import argparse
import os
import time
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
import matplotlib.pyplot as plt

from vision_transformer import create_vit_driving_model

console = Console()

class CARLADrivingDataset(Dataset):
    """Dataset for CARLA driving demonstrations"""
    
    def __init__(self, data_files, use_lidar=True, augment=False):
        self.data_files = data_files
        self.use_lidar = use_lidar
        self.augment = augment
        
        # Load all data
        self.images = []
        self.states = []
        self.actions = []
        self.lidar_points = []
        self.rewards = []
        
        self._load_data()
        
        console.log(f"[green]Loaded dataset with {len(self.images)} samples[/green]")
        
    def _load_data(self):
        """Load data from HDF5 files"""
        for data_file in self.data_files:
            console.log(f"[cyan]Loading data from {data_file}[/cyan]")
            
            with h5py.File(data_file, 'r') as f:
                # Load basic data
                self.images.append(f['images'][:])
                self.states.append(f['states'][:])
                self.actions.append(f['actions'][:])
                self.rewards.append(f['rewards'][:])
                  # Load LiDAR data if available
                if 'lidar_points' in f and self.use_lidar:
                    lidar_data = f['lidar_points'][:]
                    # Standardize LiDAR point count to 2048
                    num_samples, current_points, features = lidar_data.shape
                    target_points = 2048
                    
                    if current_points > target_points:
                        # Downsample by taking every nth point
                        step = current_points // target_points
                        lidar_data = lidar_data[:, ::step, :][:, :target_points, :]
                    elif current_points < target_points:
                        # Pad with zeros
                        padding = np.zeros((num_samples, target_points - current_points, features), dtype=lidar_data.dtype)
                        lidar_data = np.concatenate([lidar_data, padding], axis=1)
                    
                    self.lidar_points.append(lidar_data)
                else:
                    # Create dummy LiDAR data if not available
                    num_samples = len(f['images'])
                    dummy_lidar = np.zeros((num_samples, 2048, 4), dtype=np.float32)
                    self.lidar_points.append(dummy_lidar)
        
        # Concatenate all data
        self.images = np.concatenate(self.images, axis=0)
        self.states = np.concatenate(self.states, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        self.lidar_points = np.concatenate(self.lidar_points, axis=0)
        self.rewards = np.concatenate(self.rewards, axis=0)
        
        # Filter out low-quality samples (low rewards, indicating poor driving)
        good_samples = self.rewards > -5.0  # Adjust threshold as needed
        self.images = self.images[good_samples]
        self.states = self.states[good_samples]
        self.actions = self.actions[good_samples]
        self.lidar_points = self.lidar_points[good_samples]
        self.rewards = self.rewards[good_samples]
        
        console.log(f"[yellow]Filtered to {len(self.images)} good quality samples[/yellow]")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0  # Normalize to [0, 1]
        state = self.states[idx].astype(np.float32)
        action = self.actions[idx].astype(np.float32)
        lidar = self.lidar_points[idx].astype(np.float32) if self.use_lidar else None
        
        # Data augmentation
        if self.augment:
            image, state, action, lidar = self._augment_data(image, state, action, lidar)
        
        # Convert to torch tensors and adjust dimensions
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
        state = torch.from_numpy(state)
        action = torch.from_numpy(action)
        
        if lidar is not None:
            lidar = torch.from_numpy(lidar)
            return image, state, lidar, action
        else:
            return image, state, action
    
    def _augment_data(self, image, state, action, lidar):
        """Apply data augmentation"""
        # Random horizontal flip (with steering adjustment)
        if np.random.random() < 0.3:
            image = np.flip(image, axis=1).copy()
            action[0] = -action[0]  # Flip steering
            if lidar is not None:
                lidar[:, 1] = -lidar[:, 1]  # Flip y-coordinate
        
        # Random brightness adjustment
        if np.random.random() < 0.3:
            brightness_factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 1)
        
        # Random noise
        if np.random.random() < 0.2:
            noise = np.random.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        # Small steering perturbations
        if np.random.random() < 0.2:
            action[0] += np.random.normal(0, 0.05)
            action[0] = np.clip(action[0], -1.0, 1.0)
        
        return image, state, action, lidar

def create_data_loaders(data_dir, batch_size=8, val_split=0.2, use_lidar=True):
    """Create training and validation data loaders"""
    
    # Find all data files
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                  if f.endswith('.h5')]
    
    if not data_files:
        raise ValueError(f"No data files found in {data_dir}")
    
    console.log(f"[cyan]Found {len(data_files)} data files[/cyan]")
    
    # Split files into train and validation
    train_files, val_files = train_test_split(data_files, test_size=val_split, random_state=42)
    
    # Create datasets
    train_dataset = CARLADrivingDataset(train_files, use_lidar=use_lidar, augment=True)
    val_dataset = CARLADrivingDataset(val_files, use_lidar=use_lidar, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def compute_loss(predictions, targets, loss_weights=None):
    """Compute weighted loss for driving actions"""
    if loss_weights is None:
        loss_weights = [2.0, 1.0, 1.0]  # Weight steering more heavily
    
    mse_loss = nn.MSELoss(reduction='none')
    losses = mse_loss(predictions, targets)
    
    # Apply weights
    weighted_losses = losses * torch.tensor(loss_weights, device=predictions.device)
    
    return weighted_losses.mean()

def validate_model(model, val_loader, device, use_lidar=True):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    steering_errors = []
    throttle_errors = []
    brake_errors = []
    
    with torch.no_grad():
        for batch in val_loader:
            if use_lidar:
                images, states, lidar, actions = batch
                lidar = lidar.to(device).float()
            else:
                images, states, actions = batch
                lidar = None
            
            images = images.to(device).float()
            states = states.to(device).float()
            actions = actions.to(device).float()
            
            # Forward pass
            predictions = model(images, states, lidar)
            
            # Compute loss
            loss = compute_loss(predictions, actions)
            
            total_loss += loss.item() * len(images)
            total_samples += len(images)
            
            # Compute individual action errors
            steering_errors.extend(torch.abs(predictions[:, 0] - actions[:, 0]).cpu().numpy())
            throttle_errors.extend(torch.abs(predictions[:, 1] - actions[:, 1]).cpu().numpy())
            brake_errors.extend(torch.abs(predictions[:, 2] - actions[:, 2]).cpu().numpy())
    
    avg_loss = total_loss / total_samples
    avg_steering_error = np.mean(steering_errors)
    avg_throttle_error = np.mean(throttle_errors)
    avg_brake_error = np.mean(brake_errors)
    
    return avg_loss, avg_steering_error, avg_throttle_error, avg_brake_error

def train_supervised_model(data_dir, epochs=100, batch_size=8, learning_rate=0.001, 
                          use_lidar=True, save_interval=10):
    """Train the Vision Transformer model with supervised learning"""
    
    console.rule("[bold green]Supervised Learning Training")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.log(f"[cyan]Using device: {device}[/cyan]")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(data_dir, batch_size, use_lidar=use_lidar)
      # Create model
    model_config = {
        'img_size': 224,  # Updated for high-quality camera
        'patch_size': 16,  # Optimal patch size for 224x224
        'embed_dim': 512,
        'depth': 6,
        'num_heads': 8,
        'state_dim': 5,
        'max_lidar_points': 2048 if use_lidar else 0
    }
    
    model = create_vit_driving_model(model_config)
    model = model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Setup logging
    log_dir = f"supervised_logs/{int(time.time())}"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    ) as progress:
        
        task = progress.add_task("[green]Training...", total=epochs)
        
        for epoch in range(epochs):
            model.train()
            total_train_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if use_lidar:
                    images, states, lidar, actions = batch
                    lidar = lidar.to(device).float()
                else:
                    images, states, actions = batch
                    lidar = None
                
                images = images.to(device).float()
                states = states.to(device).float()
                actions = actions.to(device).float()
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(images, states, lidar)
                
                # Compute loss
                loss = compute_loss(predictions, actions)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
                
                # Log batch loss
                if batch_idx % 50 == 0:
                    writer.add_scalar('Loss/Train_Batch', loss.item(), 
                                    epoch * len(train_loader) + batch_idx)
            
            # Validation
            val_loss, steer_err, throttle_err, brake_err = validate_model(
                model, val_loader, device, use_lidar)
            
            avg_train_loss = total_train_loss / num_batches
            
            # Log epoch metrics
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Error/Steering', steer_err, epoch)
            writer.add_scalar('Error/Throttle', throttle_err, epoch)
            writer.add_scalar('Error/Brake', brake_err, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Update learning rate
            scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': model_config
                }, f"{log_dir}/best_model.pth")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': model_config
                }, f"{log_dir}/checkpoint_epoch_{epoch+1}.pth")
            
            # Log progress
            console.log(f"[cyan]Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, "
                       f"Steer Err: {steer_err:.4f}[/cyan]")
            
            progress.update(task, advance=1)
    
    writer.close()
    
    console.log(f"[green]Training completed! Best validation loss: {best_val_loss:.4f}[/green]")
    console.log(f"[green]Model saved to {log_dir}/best_model.pth[/green]")
    
    return f"{log_dir}/best_model.pth"

def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer for CARLA driving")
    parser.add_argument('--data_dir', type=str, default='carla_dataset',
                       help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training (optimized for VRAM)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--no_lidar', action='store_true',
                       help='Disable LiDAR data usage')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        console.log(f"[red]Data directory {args.data_dir} does not exist![/red]")
        console.log("[yellow]Please run 'make dataset' first to collect training data[/yellow]")
        return
    
    # Start training
    model_path = train_supervised_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_lidar=not args.no_lidar,
        save_interval=args.save_interval
    )
    
    console.log(f"[bold green]Supervised training complete![/bold green]")
    console.log(f"[green]Best model saved to: {model_path}[/green]")

if __name__ == "__main__":
    main()
