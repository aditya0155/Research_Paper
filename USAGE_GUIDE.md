# CARLA Hybrid Learning - Usage Examples and Troubleshooting Guide

## 📋 Quick Usage Examples

### Example 1: Complete Hybrid Training Pipeline

```bash
# Step 1: Start CARLA Simulator
# Download and run CARLA 0.9.10+, disable ray tracing for performance

# Step 2: Quick hybrid training (using make commands)
make dataset          # Collect 100 episodes of expert data
make train-supervised # Train Vision Transformer for 50 epochs  
make train-hybrid     # Train hybrid model for 500k steps

# Step 3: Validate results
python advanced_testing.py --test all --output_dir ./validation_results
```

### Example 2: Custom Data Collection

```bash
# Collect data from multiple towns with different weather
python collect_data.py --num_episodes 50 --town Town01 --weather sunny --save_path ./data/town01_sunny.h5
python collect_data.py --num_episodes 50 --town Town02 --weather rainy --save_path ./data/town02_rainy.h5
python collect_data.py --num_episodes 50 --town Town03 --weather foggy --save_path ./data/town03_foggy.h5

# Combine datasets for training
python train_supervised.py --data_paths ./data/town01_sunny.h5 ./data/town02_rainy.h5 ./data/town03_foggy.h5
```

### Example 3: Memory-Optimized Training (for 8GB GPU)

```bash
# Low VRAM configuration
python train_sac_carla.py \
  --hybrid \
  --pretrained_vit ./models/vision_transformer_best.pth \
  --use_int4_quantization \
  --batch_size 16 \
  --buffer_size 30000 \
  --total_timesteps 300000
```

### Example 4: Advanced Supervised Training

```bash
# Train with custom architecture and augmentation
python train_supervised.py \
  --data_path ./data/expert_data.h5 \
  --embed_dim 768 \
  --depth 8 \
  --num_heads 12 \
  --batch_size 32 \
  --epochs 100 \
  --learning_rate 1e-4 \
  --weight_decay 1e-2 \
  --use_augmentation \
  --validation_split 0.2
```

### Example 5: Model Evaluation and Testing

```bash
# Comprehensive testing
python test_system.py                    # Run all unit tests
python advanced_testing.py --test model  # Validate model architecture
python advanced_testing.py --test performance --output_dir ./benchmarks

# Test specific model checkpoint
python advanced_testing.py --test model --model_path ./models/vision_transformer_best.pth
```

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. CARLA Connection Issues

**Problem**: `RuntimeError: time-out of 2000ms while waiting for the simulator`

**Solutions**:
```bash
# Check if CARLA is running
netstat -an | findstr :2000

# If CARLA is stuck, restart it
taskkill /f /im CarlaUE4.exe
# Then restart CARLA

# If using different port
python train_sac_carla.py --carla_port 2002
```

**Problem**: `carla.libcarla.NoSuchDeviceError`

**Solutions**:
- Ensure CARLA version compatibility (0.9.10+)
- Check if multiple CARLA instances are running
- Verify Python CARLA package matches CARLA version:
```bash
pip uninstall carla
pip install carla==0.9.15  # Match your CARLA version
```

#### 2. Memory and GPU Issues

**Problem**: `CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size and buffer size
python train_sac_carla.py --batch_size 16 --buffer_size 25000

# Enable memory optimizations
python train_sac_carla.py --use_int4_quantization --gradient_checkpointing

# Monitor GPU memory
python -c "import torch; print(f'Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

**Problem**: `RuntimeError: Expected all tensors to be on the same device`

**Solutions**:
```python
# Check device consistency in training
python train_sac_carla.py --device cuda  # or --device cpu

# For mixed precision training
python train_sac_carla.py --mixed_precision
```

#### 3. Data Collection Issues

**Problem**: `No expert data collected` or `Empty dataset`

**Solutions**:
```bash
# Check NPC spawning
python collect_data.py --num_vehicles 50 --num_pedestrians 100

# Verify data quality threshold
python collect_data.py --quality_threshold 0.5  # Lower threshold

# Debug mode
python collect_data.py --debug --num_episodes 5
```

**Problem**: `HDF5 file corruption`

**Solutions**:
```bash
# Verify HDF5 file
python -c "import h5py; f=h5py.File('./data/expert_data.h5', 'r'); print(list(f.keys()))"

# Recreate corrupted file
rm ./data/expert_data.h5
python collect_data.py --num_episodes 100
```

#### 4. Training Issues

**Problem**: `Loss explodes` or `NaN values in loss`

**Solutions**:
```python
# Reduce learning rate
python train_supervised.py --learning_rate 1e-5

# Add gradient clipping
python train_sac_carla.py --grad_clip 0.5

# Check data normalization
python train_supervised.py --normalize_data
```

**Problem**: `Model not improving`

**Solutions**:
```bash
# Increase model capacity
python train_supervised.py --embed_dim 768 --depth 12

# More training data
python collect_data.py --num_episodes 500

# Better data quality
python collect_data.py --quality_threshold 0.9

# Learning rate scheduling
python train_supervised.py --use_scheduler --lr_decay 0.95
```

#### 5. Hybrid Training Issues

**Problem**: `Pre-trained model not loading`

**Solutions**:
```bash
# Verify model file exists
ls -la ./models/vision_transformer_best.pth

# Check model compatibility
python advanced_testing.py --test model --model_path ./models/vision_transformer_best.pth

# Retrain supervised model if needed
python train_supervised.py --epochs 50
```

**Problem**: `RL training unstable after ViT integration`

**Solutions**:
```bash
# Freeze ViT longer
python train_sac_carla.py --hybrid --freeze_vit_epochs 20

# Lower RL learning rate
python train_sac_carla.py --hybrid --learning_rate 1e-4

# Progressive unfreezing
python train_sac_carla.py --hybrid --progressive_unfreezing
```

### Performance Optimization Tips

#### For Training Speed
```bash
# Use multiple workers for data loading
python train_supervised.py --num_workers 4

# Enable mixed precision
python train_sac_carla.py --mixed_precision

# Optimize CARLA settings
# In CARLA: Settings -> Quality -> Low, disable ray tracing

# Use SSD for data storage
# Move datasets to SSD drive
```

#### For Memory Efficiency
```bash
# Gradient accumulation instead of large batches
python train_supervised.py --batch_size 8 --gradient_accumulation_steps 4

# Model parallelism for large models
python train_sac_carla.py --model_parallel

# Clear cache regularly
python -c "import torch; torch.cuda.empty_cache()"
```

#### For Model Quality
```bash
# Data augmentation
python train_supervised.py --use_augmentation --augmentation_prob 0.7

# Regularization
python train_supervised.py --weight_decay 1e-2 --dropout 0.1

# Ensemble training
python train_supervised.py --ensemble_size 3
```

## 🧪 Debugging and Validation

### Debug Mode Training
```bash
# Enable debug logging
python train_sac_carla.py --debug --log_level DEBUG

# Visualize training data
python collect_data.py --visualize --save_images ./debug_images/

# Monitor gradients
python train_supervised.py --log_gradients --tensorboard_dir ./debug_logs/
```

### Comprehensive Testing
```bash
# Full system validation
python advanced_testing.py --test all --output_dir ./test_results

# Performance benchmarking
python advanced_testing.py --test performance

# Model architecture validation
python test_system.py
```

### Monitoring Training
```bash
# TensorBoard
tensorboard --logdir ./logs --port 6006

# Real-time monitoring
python -c "
import torch
import time
while True:
    if torch.cuda.is_available():
        print(f'GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
    time.sleep(5)
"
```

## 📊 Performance Benchmarks

### Expected Performance (RTX 3080 12GB)

| Configuration | Training Speed | Inference FPS | Memory Usage |
|--------------|----------------|---------------|--------------|
| Small ViT | 15 episodes/min | 45 FPS | 4.2 GB |
| Medium ViT | 12 episodes/min | 32 FPS | 6.8 GB |
| Large ViT | 8 episodes/min | 22 FPS | 9.1 GB |

### Scaling Guidelines

| GPU Memory | Recommended Config | Batch Size | Buffer Size |
|------------|-------------------|------------|-------------|
| 6 GB | Small ViT | 8 | 25k |
| 8 GB | Small ViT | 16 | 50k |
| 12 GB | Medium ViT | 32 | 100k |
| 16+ GB | Large ViT | 64 | 200k |

## 🔍 Advanced Diagnostics

### Check System Dependencies
```bash
python -c "
import torch
import stable_baselines3
import cv2
import h5py
print('PyTorch:', torch.__version__)
print('SB3:', stable_baselines3.__version__)
print('OpenCV:', cv2.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name())
"
```

### Validate Model Architecture
```python
# test_architecture.py
from vision_transformer import create_vit_driving_model
import torch

model = create_vit_driving_model()
print("Model created successfully")

# Test forward pass
x_img = torch.randn(1, 3, 84, 84)
x_state = torch.randn(1, 5)
x_lidar = torch.randn(1, 2048, 4)

with torch.no_grad():
    output = model(x_img, x_state, x_lidar)
    print(f"Output shape: {output.shape}")
    print("Architecture validation passed!")
```

### Data Quality Check
```python
# validate_data.py
import h5py
import numpy as np

with h5py.File('./data/expert_data.h5', 'r') as f:
    print("Dataset keys:", list(f.keys()))
    
    for key in f.keys():
        data = f[key][:]
        print(f"{key}: shape={data.shape}, dtype={data.dtype}")
        print(f"  Range: [{np.min(data):.3f}, {np.max(data):.3f}]")
        print(f"  Mean: {np.mean(data):.3f}, Std: {np.std(data):.3f}")
```

## 🚀 Deployment Tips

### Model Export
```bash
# Export trained model for deployment
python -c "
from train_sac_carla import export_model
export_model('./models/sac_carla_hybrid_final.zip', './deployment/model.onnx')
"
```

### Real-time Inference
```python
# inference_example.py
import torch
from vision_transformer import create_vit_driving_model

# Load model
model = create_vit_driving_model()
checkpoint = torch.load('./models/vision_transformer_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Real-time inference function
def predict_action(image, state, lidar):
    with torch.no_grad():
        action = model(image, state, lidar)
    return action.cpu().numpy()
```

---

For additional support, check the [GitHub Issues](https://github.com/your-repo/issues) or refer to the detailed documentation in `README.md`.
