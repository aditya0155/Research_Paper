# CARLA Hybrid Learning Self-Driving Car

This project implements a hybrid supervised learning + reinforcement learning approach to train a self-driving car in the CARLA simulator. The system combines Vision Transformers for supervised pre-training with SAC reinforcement learning for optimal policy learning.

## 🚀 Key Features

### **Hybrid Training Architecture**
- **Vision Transformer (ViT)** for multi-modal perception (Camera + LiDAR + Vehicle State)
- **Expert NPC Data Collection** with intelligent waypoint navigation
- **Progressive Training Pipeline**: Supervised pre-training → RL fine-tuning
- **Memory Optimized**: Int4 quantization support and reduced VRAM usage

### **Advanced Perception System**
- **Enhanced LiDAR**: 64 channels, 100m range, 360° coverage
- **Multi-Modal Fusion**: Cross-attention between vision, LiDAR, and state
- **Data Augmentation**: Robust training with varied conditions
- **Quality Filtering**: Automatic filtering of low-quality driving samples

### **Training Modes**
1. **Expert Data Collection**: Record NPC driving behavior
2. **Supervised Learning**: Pre-train Vision Transformer on expert data
3. **Hybrid Training**: Combine supervised and RL approaches
4. **Pure RL Training**: Traditional reinforcement learning (backward compatibility)

### **Development Features**
- **Command-line Interface**: Easy training with `make` commands
- **Comprehensive Logging**: Training metrics, validation, and monitoring
- **Checkpointing**: Save and resume training progress
- **Model Export**: Save trained models for deployment

## 📋 Prerequisites

### Software Requirements
- **CARLA Simulator 0.9.10+**: Download from [CARLA releases](https://github.com/carla-simulator/carla/releases)
- **Python 3.8+**: Required for all dependencies
- **CUDA-capable GPU**: Recommended for training (8GB+ VRAM)

### Python Dependencies
```bash
pip install torch torchvision torchaudio
pip install stable-baselines3[extra]
pip install carla-client
pip install opencv-python
pip install h5py
pip install tqdm
pip install tensorboard
pip install optuna
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd carla1
```

2. **Install dependencies**
```bash
pip install -r requirements.txt  # If available, or install manually as above
```

3. **Setup CARLA**
   - Download CARLA 0.9.10+ 
   - Extract to your preferred location
   - **Important**: Disable ray tracing in CARLA settings for better performance

## 🚗 Quick Start

### **Method 1: Using Make Commands (Recommended)**

```bash
# 1. Collect expert driving data (10,000 samples)
make dataset

# 2. Train Vision Transformer on expert data
make train-supervised

# 3. Train hybrid model (ViT + RL)
make train-hybrid

# 4. Traditional RL training (optional)
make train-rl
```

### **Method 2: Manual Commands**

```bash
# 1. Start CARLA simulator first
./CarlaUE4.exe  # Windows
./CarlaUE4.sh   # Linux

# 2. Collect expert data
python collect_data.py --num_episodes 100 --save_path ./data/expert_data.h5

# 3. Train supervised model
python train_supervised.py --data_path ./data/expert_data.h5 --epochs 50

# 4. Train hybrid model
python train_sac_carla.py --hybrid --pretrained_vit ./models/vision_transformer_best.pth

# 5. Traditional RL training
python train_sac_carla.py --total_timesteps 1000000
```

## 📊 Training Options

### **Data Collection Parameters**
```bash
python collect_data.py \
    --num_episodes 100 \
    --save_path ./data/expert_data.h5 \
    --town Town01 \
    --weather random \
    --quality_threshold 0.8
```

### **Supervised Learning Parameters**
```bash
python train_supervised.py \
    --data_path ./data/expert_data.h5 \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 1e-4 \
    --validation_split 0.2
```

### **Hybrid Training Parameters**
```bash
python train_sac_carla.py \
    --hybrid \
    --pretrained_vit ./models/vision_transformer_best.pth \
    --total_timesteps 500000 \
    --learning_rate 3e-4 \
    --batch_size 64 \
    --freeze_vit_epochs 10
```

### **Memory Optimization**
```bash
# For lower VRAM usage
python train_sac_carla.py \
    --hybrid \
    --pretrained_vit ./models/vision_transformer_best.pth \
    --use_int4_quantization \
    --batch_size 32 \
    --buffer_size 50000
```

## 📁 Project Structure

```
carla1/
├── carla_env.py              # CARLA environment with enhanced LiDAR
├── vision_transformer.py     # ViT architecture with multi-modal fusion
├── train_supervised.py       # Supervised learning training script
├── train_sac_carla.py       # RL and hybrid training script
├── collect_data.py          # Expert data collection script
├── npc_expert.py            # Expert NPC driver implementation
├── hybrid_training.py       # Hybrid training utilities
├── test_system.py           # Testing and validation scripts
├── make.bat                 # Command-line interface
├── plan.md                  # Development roadmap
└── README.md                # This file

data/
├── expert_data.h5           # Collected expert driving data
└── processed/               # Processed datasets

models/
├── vision_transformer_best.pth    # Pre-trained ViT model
├── sac_carla_hybrid_final.zip     # Trained hybrid model
└── checkpoints/             # Training checkpoints

logs/
├── supervised_training/     # Supervised learning logs
├── hybrid_training/         # Hybrid training logs
└── tensorboard/            # TensorBoard logs
```

## 🧪 Testing and Validation

### **Run System Tests**
```bash
python test_system.py --test_all
```

### **Individual Component Tests**
```bash
# Test data collection
python test_system.py --test_data_collection

# Test supervised training
python test_system.py --test_supervised

# Test hybrid training
python test_system.py --test_hybrid

# Test model inference
python test_system.py --test_inference --model_path ./models/sac_carla_hybrid_final.zip
```

### **Performance Benchmarking**
```bash
# Compare pure RL vs hybrid training
python test_system.py --benchmark --episodes 50
```

## 📈 Monitoring Training

### **TensorBoard Logs**
```bash
tensorboard --logdir ./logs/tensorboard
```

### **Key Metrics to Monitor**
- **Supervised Training**: Loss, accuracy, validation metrics
- **RL Training**: Episode reward, success rate, collision rate
- **Hybrid Training**: Combined loss, policy performance, feature alignment

## 🔧 Troubleshooting

### **Common Issues**

**1. CARLA Connection Failed**
- Ensure CARLA is running before starting training
- Check if port 2000 is available
- Try restarting CARLA simulator

**2. CUDA Out of Memory**
- Reduce batch size: `--batch_size 16`
- Enable quantization: `--use_int4_quantization`
- Reduce buffer size: `--buffer_size 50000`

**3. Training Crashes**
- Check data quality: Look for NaN values in dataset
- Verify model paths: Ensure pre-trained models exist
- Monitor GPU memory usage

**4. Poor Performance**
- Increase data collection episodes
- Improve data quality filtering
- Tune hyperparameters
- Extend training duration

### **Performance Tips**

**For Better Training Speed:**
- Use SSD for data storage
- Enable mixed precision training
- Use multiple CPU cores for data loading
- Optimize CARLA settings (disable ray tracing)

**For Better Results:**
- Collect diverse training data (various weather, towns)
- Use quality filtering for expert data
- Implement progressive training schedule
- Fine-tune reward function

## 📚 Advanced Usage

### **Custom Data Collection**
```python
from collect_data import ExpertDataCollector

collector = ExpertDataCollector(
    save_path="custom_data.h5",
    quality_threshold=0.9,
    town="Town02",
    weather_conditions=["sunny", "rainy"]
)
collector.collect(num_episodes=200)
```

### **Custom ViT Architecture**
```python
from vision_transformer import VisionTransformer

model = VisionTransformer(
    image_size=224,
    patch_size=16,
    embed_dim=768,
    num_layers=12,
    num_heads=12,
    action_dim=2
)
```

### **Resume Training**
```bash
# Resume supervised training
python train_supervised.py --resume ./models/vision_transformer_checkpoint.pth

# Resume hybrid training
python train_sac_carla.py --hybrid --resume ./models/sac_carla_100000_steps.zip
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a Pull Request

### **Development Roadmap**
- [ ] Multi-agent training support
- [ ] Real-world deployment pipeline
- [ ] Advanced perception sensors (radar, cameras)
- [ ] Improved reward shaping
- [ ] Model compression and optimization

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@misc{carla-hybrid-learning,
  title={CARLA Hybrid Learning Self-Driving Car},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/carla-hybrid-learning}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [CARLA Simulator](https://carla.org/) team for the excellent autonomous driving simulator
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for reinforcement learning implementations
- Vision Transformer implementations and research community
- Contributors and the open-source community

---

## 🎥 Demo Videos

### Original Project Videos
- [Training Demo](https://www.youtube.com/watch?v=uS3r_8r4riY&t=10s)
- [Performance Showcase](https://www.youtube.com/watch?v=lIOpiagK0PU)

### New Hybrid Training Features
- Expert Data Collection Demo (Coming Soon)
- Vision Transformer Training (Coming Soon)
- Hybrid Training Comparison (Coming Soon)

---

**Note**: This is an evolved version of the original CARLA RL project, now featuring hybrid supervised + reinforcement learning with Vision Transformers and enhanced perception systems.
