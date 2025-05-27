import os
import argparse
import gym
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rich.console import Console
import optuna
from stable_baselines3.common.logger import configure  
import serial
import time
    
from carla_env import CarlaEnv, CustomSAC
from vision_transformer import VisionTransformerDriving
from hybrid_training import (
    HybridTrainingCallback, ProgressiveTrainer, RewardShaper,
    setup_hybrid_training, validate_pretrained_model, export_hybrid_model
)



console = Console()



# NEW: Configure a logger that outputs to stdout, CSV, and tensorboard.

new_logger = configure("./sac_tensorboard/", ["stdout", "csv", "tensorboard"])



from stable_baselines3.common.callbacks import BaseCallback



class EntropyLoggingCallback(BaseCallback):

    def __init__(self, log_interval: int = 1000, verbose: int = 1):

        super(EntropyLoggingCallback, self).__init__(verbose)

        self.log_interval = log_interval



    def _on_step(self) -> bool:

        if self.n_calls % self.log_interval == 0:

            # Directly access the entropy coefficient from the model.

            current_ent_coef = torch.exp(self.model.log_ent_coef).item()

            print(f"[INFO] Step {self.n_calls}: Entropy Coefficient = {current_ent_coef}")

        return True





# --- Define a Residual Block for the CNN ---

class ResidualBlock(nn.Module):

    def __init__(self, channels: int):

        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

        

    def forward(self, x):

        residual = x

        out = self.relu(self.conv1(x))

        out = self.conv2(out)

        return self.relu(out + residual)



# --- Enhanced Feature Extractor with Attention Fusion ---

class CombinedExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1024):

        # Expect state to be 5-dim now.

        super(CombinedExtractor, self).__init__(observation_space, features_dim)

        # Enhanced CNN for higher resolution (224x224)
        self.cnn = nn.Sequential(
            # First block: 224x224 -> 56x56
            nn.Conv2d(3, 64, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            ResidualBlock(64),
            
            # Second block: 56x56 -> 28x28 
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128),
            
            # Third block: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(256),
            
            # Fourth block: 14x14 -> 7x7
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(512),
            
            # Final layers: 7x7 -> 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Determine the CNN output dimension.

        with torch.no_grad():

            dummy_image = torch.zeros(1, 3, 224, 224)  # Updated for 224x224

            cnn_out_dim = self.cnn(dummy_image).shape[1]

        

        state_dim = observation_space.spaces["state"].shape[0]  # now 5

        # MLP for state.

        self.mlp = nn.Sequential(

            nn.Linear(state_dim, 128),

            nn.ReLU(),

            nn.Linear(128, 128),

            nn.ReLU()

        )



        self.attention_layer = nn.Linear(cnn_out_dim, 128)

        

        combined_dim = cnn_out_dim + 128

        self.fc = nn.Sequential(

            nn.Linear(combined_dim, features_dim),

            nn.ReLU()

        )

        self._features_dim = features_dim



    def forward(self, observations):

        # Process image.

        if observations["image"].ndim == 4 and observations["image"].shape[1] == 3:

            image = observations["image"].float() / 255.0

        else:

            image = observations["image"].permute(0, 3, 1, 2).float() / 255.0

        

        image_features = self.cnn(image)  # shape: [batch, cnn_out_dim]

        state_features = self.mlp(observations["state"].float())  # shape: [batch, 128]

        



        attn_weights = torch.sigmoid(self.attention_layer(image_features))  # shape: [batch, 128]

        fused_state = state_features * attn_weights

        

        concatenated = torch.cat([image_features, fused_state], dim=1)



        concatenated = torch.nan_to_num(concatenated, nan=0.0, posinf=1e3, neginf=-1e3)

        return self.fc(concatenated)







# --- NEW: Hybrid Feature Extractor with Pre-trained ViT ---

class HybridVisionExtractor(BaseFeaturesExtractor):

    """Hybrid feature extractor using pre-trained Vision Transformer"""

    

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1024, 

                 pretrained_vit_path: str = None, freeze_vit: bool = False):

        super(HybridVisionExtractor, self).__init__(observation_space, features_dim)

        

        self.pretrained_vit_path = pretrained_vit_path

        self.freeze_vit = freeze_vit

            # Create Vision Transformer backbone

        self.vit_model = VisionTransformerDriving(

            img_size=224,  # Updated for high-quality camera

            patch_size=16,  # Optimal patch size for 224x224 

            embed_dim=512,

            depth=6,

            num_heads=8,

            state_dim=observation_space.spaces["state"].shape[0],

            max_lidar_points=2048,

            num_actions=3

        )

        

        # Load pre-trained weights if available

        if pretrained_vit_path and os.path.exists(pretrained_vit_path):

            console.log(f"[cyan]Loading pre-trained ViT weights from {pretrained_vit_path}[/cyan]")

            checkpoint = torch.load(pretrained_vit_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:

                self.vit_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            else:

                self.vit_model.load_state_dict(checkpoint, strict=False)

            console.log("[green]✓ Pre-trained weights loaded successfully[/green]")

        

        # Optionally freeze ViT parameters

        if freeze_vit:

            for param in self.vit_model.parameters():

                param.requires_grad = False

            console.log("[yellow]ViT parameters frozen for transfer learning[/yellow]")

        

        # Remove the action head from ViT (we only want features)

        self.vit_features = nn.Sequential(*list(self.vit_model.children())[:-1])

        

        # Calculate ViT feature dimension (without action head)

        vit_feature_dim = self.vit_model.embed_dim

        

        # Adaptation layers for RL

        self.feature_adapter = nn.Sequential(

            nn.Linear(vit_feature_dim, features_dim),

            nn.ReLU(),

            nn.Dropout(0.1),

            nn.Linear(features_dim, features_dim),

            nn.ReLU()

        )

        

        self._features_dim = features_dim

        

    def forward(self, observations):

        """Extract features using pre-trained ViT"""

        # Prepare inputs

        images = observations["image"]

        states = observations["state"]

        

        # Handle image format

        if images.ndim == 4 and images.shape[1] == 3:

            images = images.float() / 255.0

        else:

            images = images.permute(0, 3, 1, 2).float() / 255.0

        

        # Get LiDAR if available (optional for now)

        lidar_points = observations.get("lidar", None)

        

        # Extract features using ViT backbone

        with torch.set_grad_enabled(not self.freeze_vit):

            # Get image features through ViT processing pipeline

            batch_size = images.shape[0]

            

            # Process images

            image_patches = self.vit_model.patch_embedding(images)

            cls_tokens = self.vit_model.cls_token.expand(batch_size, -1, -1)

            image_tokens = torch.cat([cls_tokens, image_patches], dim=1)

            image_tokens += self.vit_model.pos_embedding

            

            # Apply transformer blocks

            for block in self.vit_model.transformer_blocks:

                image_tokens = block(image_tokens)

            

            # Extract class token (global image representation)

            image_features = image_tokens[:, 0]  # (batch_size, embed_dim)

            

            # Process other modalities

            state_features = self.vit_model.state_encoder(states)

            

            if lidar_points is not None:

                lidar_features = self.vit_model.lidar_encoder(lidar_points)

            else:

                lidar_features = torch.zeros(batch_size, self.vit_model.embed_dim // 2, 

                                           device=images.device)

            

            # Fuse modalities

            fused_features = torch.cat([image_features, state_features, lidar_features], dim=1)

            vit_features = self.vit_model.modal_fusion(fused_features)

        

        # Adapt features for RL

        adapted_features = self.feature_adapter(vit_features)

        

        return adapted_features



def make_env():
    def _init():
        # Create CARLA environment with high-quality camera
        env = CarlaEnv(
            num_npcs=1, 
            frame_skip=8, 
            visualize=True,
            fixed_delta_seconds=0.05,
            camera_width=224,  # Improved resolution for better feature detection
            camera_height=224,  # Improved resolution for better feature detection
            model=None
        )
        return env
    return _init

# --- Memory Optimization Functions ---
def apply_int4_quantization(model):
    """Apply int4 quantization to reduce memory usage"""
    try:
        # Import quantization libraries
        import torch.quantization
        
        console.log("[cyan]Applying int4 quantization optimization...[/cyan]")
        
        # Apply quantization to the Vision Transformer components if available
        if hasattr(model.policy, 'features_extractor'):
            features_extractor = model.policy.features_extractor
            
            if hasattr(features_extractor, 'vit_model'):
                vit_model = features_extractor.vit_model
                
                # Quantize patch embedding
                if hasattr(vit_model, 'patch_embedding'):
                    vit_model.patch_embedding = torch.quantization.quantize_dynamic(
                        vit_model.patch_embedding, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                    )
                
                # Quantize transformer blocks
                if hasattr(vit_model, 'transformer_blocks'):
                    for i, block in enumerate(vit_model.transformer_blocks):
                        vit_model.transformer_blocks[i] = torch.quantization.quantize_dynamic(
                            block, {torch.nn.Linear}, dtype=torch.qint8
                        )
                
                # Quantize encoders
                for encoder_name in ['state_encoder', 'lidar_encoder']:
                    if hasattr(vit_model, encoder_name):
                        encoder = getattr(vit_model, encoder_name)
                        setattr(vit_model, encoder_name, 
                               torch.quantization.quantize_dynamic(encoder, {torch.nn.Linear}, dtype=torch.qint8))
                
                console.log("[green]✓ Applied dynamic quantization to ViT components[/green]")
            
            # Convert remaining components to half precision for additional memory savings
            if hasattr(features_extractor, 'feature_adapter'):
                features_extractor.feature_adapter = features_extractor.feature_adapter.half()
                console.log("[green]✓ Applied half precision to feature adapter[/green]")
        
        # Apply half precision to value and policy networks
        if hasattr(model.policy, 'value_net'):
            model.policy.value_net = model.policy.value_net.half()
        if hasattr(model.policy, 'action_net'):
            model.policy.action_net = model.policy.action_net.half()
            
        console.log("[green]✓ Int4/Dynamic quantization applied successfully[/green]")
        
        return model
        
    except Exception as e:
        console.log(f"[yellow]Quantization failed, using float16 fallback: {e}[/yellow]")
        
        # Fallback to half precision if quantization fails
        if hasattr(model.policy, 'features_extractor'):
            if hasattr(model.policy.features_extractor, 'vit_model'):
                model.policy.features_extractor.vit_model.half()
                console.log("[green]Applied half precision to ViT components as fallback[/green]")
        
        return model
        
        return model
    except Exception as e:
        console.log(f"[red]Quantization failed: {e}[/red]")
        return model

def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3     # GB
        return allocated, cached
    return 0, 0

def get_adaptive_buffer_size():
    """Get adaptive buffer size based on available GPU memory"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        # Conservative buffer sizing based on total GPU memory
        if total_memory >= 12:  # High-end GPU
            return 100000
        elif total_memory >= 8:  # Mid-range GPU  
            return 50000
        elif total_memory >= 6:  # Budget GPU
            return 25000
        else:  # Low memory
            return 10000
    else:
        return 25000  # Conservative default for CPU

class CustomCheckpointCallback(CheckpointCallback):

    def _on_training_start(self) -> None:

        if hasattr(self.model, "num_timesteps"):

            self.n_calls = self.model.num_timesteps

        super()._on_training_start()





def objective(trial):

    lr = 2e-4

    batch_size = trial.suggest_categorical('batch_size', [8])

    tau = 0.002191

    

    policy_kwargs = dict(

        features_extractor_class=CombinedExtractor, 

        features_extractor_kwargs=dict(features_dim=1024),

        net_arch=dict(pi=[1024, 1024], qf=[1024, 1024])

    )

    env = DummyVecEnv([make_env() for _ in range(1)])

    model = SAC(

        "MultiInputPolicy",

        env,

        verbose=1,

        tensorboard_log="./sac_tensorboard/",

        device="cuda" if torch.cuda.is_available() else "cpu",        learning_rate=lr,

        buffer_size=get_adaptive_buffer_size(),

        learning_starts=1000,

        batch_size=batch_size,

        tau=tau,

        ent_coef=0.5,  # Fixed entropy coefficient during optimization

        policy_kwargs=policy_kwargs

    )



    # Train for a short trial.

    model.learn(total_timesteps=10000)

    # Evaluate performance over 1000 steps.

    rewards = []

    obs = env.reset()   

    for _ in range(1000):

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, _ = env.step(action)

        rewards.append(reward)

        if done:

            obs = env.reset()

    avg_reward = np.mean(rewards)

    return avg_reward



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train SAC on CARLA environment with advanced features")
    
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint/model file to resume training from")

    parser.add_argument("--total_timesteps", type=int, default=150000, help="Total timesteps for training")

    parser.add_argument("--optimize", action="store_true", help="Run hyperparameter optimization before training")
    
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid training with pre-trained Vision Transformer")
    
    parser.add_argument("--pretrained_vit", type=str, default="models/vit_driving_best.pth", 
                        help="Path to pre-trained ViT model weights")

    args = parser.parse_args()



    if args.optimize:

        console.log("[bold yellow]Starting hyperparameter optimization...[/bold yellow]")

        study = optuna.create_study(direction='maximize')

        study.optimize(objective, n_trials=10)

        best_params = study.best_trial.params

        console.log(f"[bold green]Best hyperparameters: {best_params}[/bold green]")

        learning_rate = best_params['learning_rate']

        batch_size = best_params['batch_size']

        tau = best_params['tau']

    else:

        learning_rate = 2e-4

        batch_size = 8

        tau = 0.004

    console.rule("[bold green]Starting Training")

    env = DummyVecEnv([make_env() for _ in range(1)])

    # Choose feature extractor based on hybrid mode
    if args.hybrid:
        console.log("[bold cyan]Using Hybrid Vision Transformer feature extractor[/bold cyan]")
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Memory optimization for hybrid training
        features_dim = 512  # Reduced from 1024 for memory efficiency
        policy_kwargs = dict(
            features_extractor_class=HybridVisionExtractor,
            features_extractor_kwargs=dict(
                features_dim=features_dim,
                pretrained_vit_path=args.pretrained_vit,
                freeze_vit=False  # Allow fine-tuning
            ),
            net_arch=dict(pi=[512, 256], qf=[512, 256])  # Smaller networks
        )
        
        # Reduce batch size for memory efficiency
        batch_size = min(batch_size, 8)
        console.log(f"[yellow]Reduced batch size to {batch_size} for hybrid training[/yellow]")
        
    else:
        console.log("[bold blue]Using standard CNN feature extractor[/bold blue]")
        policy_kwargs = dict(
            features_extractor_class=CombinedExtractor,
            features_extractor_kwargs=dict(features_dim=1024),
            net_arch=dict(pi=[1024, 1024], qf=[1024, 1024])
        )



    checkpoint_callback = CustomCheckpointCallback(save_freq=2000, save_path='./checkpoints/', name_prefix='sac_carla')



    class LossLoggingCallback(BaseCallback):

        def __init__(self, log_interval: int = 1000, verbose: int = 1):

            super(LossLoggingCallback, self).__init__(verbose)

            self.log_interval = log_interval



        def _on_step(self) -> bool:

            if self.n_calls % self.log_interval == 0:

                self.logger.record("custom/progress", self.n_calls)

                if self.verbose > 0:

                    print(f"Step: {self.n_calls}")

            return True



    class StuckDetectionCallback(BaseCallback):     

        def __init__(self, verbose: int = 1):

            super(StuckDetectionCallback, self).__init__(verbose)



        def _on_step(self) -> bool: 

            infos = self.locals.get("infos", [])

            for info in infos:

                if isinstance(info, dict) and info.get("stuck", False):

                    print("[yellow][STUCK CALLBACK] Vehicle was respawned due to being stuck.[/yellow]")

            return True

        

    loss_logging_callback = LossLoggingCallback(log_interval=1000, verbose=1)

    stuck_detection_callback = StuckDetectionCallback(verbose=1)

    

    entropy_logging_callback = EntropyLoggingCallback(log_interval=1000, verbose=1)

    

    if args.resume is not None and os.path.exists(args.resume):

        console.log(f"[yellow]Resuming training from checkpoint: {args.resume}[/yellow]")

            # Load the model

        model = CustomSAC.load(

            args.resume, 

            env=env,

            device="cuda" if torch.cuda.is_available() else "cpu",

            total_timesteps_for_entropy=args.total_timesteps,

            is_hybrid=args.hybrid

        )   

        

        # Update the logger

        model.set_logger(new_logger)

        

        # Calculate the remaining timesteps

        remaining_timesteps = args.total_timesteps - model.num_timesteps

        

        # Log the current state

        with torch.no_grad():

            current_alpha = torch.exp(model.log_ent_coef).item()

        console.log(f"[cyan]Current entropy coefficient: {current_alpha:.4f}[/cyan]")

        console.log(f"[cyan]Current timesteps: {model.num_timesteps}[/cyan]")

        console.log(f"[cyan]Remaining timesteps: {remaining_timesteps}[/cyan]")

    else:

        console.log("[cyan]Creating a new model.[/cyan]")

        model = CustomSAC(

            "MultiInputPolicy",

            env,

            verbose=1,

            logger=new_logger,

            tensorboard_log="./sac_tensorboard/",

            device="cuda" if torch.cuda.is_available() else "cpu",

            learning_rate=learning_rate,

            buffer_size=get_adaptive_buffer_size(),

            learning_starts=100,

            batch_size=batch_size,

            tau=tau,            policy_kwargs=policy_kwargs,

            total_timesteps_for_entropy=args.total_timesteps,

            is_hybrid=args.hybrid        )

    # Apply memory optimizations for hybrid training
    if args.hybrid:
        console.log("[yellow]Applying memory optimizations for hybrid training...[/yellow]")
        
        # Monitor memory before optimization
        allocated_before, cached_before = get_memory_usage()
        console.log(f"[cyan]Memory before optimization: {allocated_before:.2f}GB allocated, {cached_before:.2f}GB cached[/cyan]")
        
        # Apply quantization if enabled
        model = apply_int4_quantization(model)
        
        # Monitor memory after optimization
        allocated_after, cached_after = get_memory_usage()
        console.log(f"[cyan]Memory after optimization: {allocated_after:.2f}GB allocated, {cached_after:.2f}GB cached[/cyan]")
        
        # Additional hybrid training parameters
        console.log("[green]✓ Hybrid training setup complete[/green]")
        console.log(f"[cyan]• Using pre-trained ViT from: {args.pretrained_vit}[/cyan]")
        console.log(f"[cyan]• Reduced batch size: {batch_size}[/cyan]")
        console.log(f"[cyan]• Feature dimension: {policy_kwargs['features_extractor_kwargs']['features_dim']}[/cyan]")

    console.log("[bold green]Starting training...")

    callbacks = [checkpoint_callback, loss_logging_callback, stuck_detection_callback, entropy_logging_callback]

    # Execute training based on mode
    if args.hybrid:
        # Setup hybrid training components
        progressive_trainer, hybrid_callback, reward_shaper = setup_hybrid_training(args, env, model)
        
        # Add hybrid callback to the list
        callbacks.append(hybrid_callback)
        
        # Execute progressive hybrid training
        console.log("[bold cyan]Executing Hybrid Training Pipeline...[/bold cyan]")
        progressive_trainer.train_progressive(env, callbacks)
        
        # Export hybrid model
        export_hybrid_model(model, "models/hybrid_model_final.pth")
        
    else:
        # Standard RL training
        console.log("[bold blue]Executing Standard RL Training...[/bold blue]")
        model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    # After model creation, update the environment with the model reference
    env.envs[0].model = model

    model_path = "sac_carla_model_enhanced"
    model.save(model_path)
    console.log(f"[bold green]Model saved to {model_path}[/bold green]")