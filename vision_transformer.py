"""
Vision Transformer for CARLA Autonomous Driving
Implements ViT with multi-modal fusion for images, LiDAR, and vehicle state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=84, patch_size=16, in_channels=3, embed_dim=512):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, sqrt(n_patches), sqrt(n_patches))
        x = rearrange(x, 'b e h w -> b (h w) e')  # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, embed_dim * 3)
        qkv = rearrange(qkv, 'b s (three h d) -> three b h s d', three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        attention_output = rearrange(attention_output, 'b h s d -> b s (h d)')
        
        # Final projection
        output = self.projection(attention_output)
        output = self.projection_dropout(output)
        
        return output

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.attention(self.norm1(x))
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class LiDAREncoder(nn.Module):
    """Encode LiDAR point cloud data"""
    def __init__(self, max_points=2048, embed_dim=256):
        super().__init__()
        self.max_points = max_points
        self.embed_dim = embed_dim
        
        # Point-wise feature extraction
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 64),  # [x, y, z, intensity]
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
        # Global feature aggregation
        self.global_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, lidar_points):
        # lidar_points shape: (batch_size, max_points, 4)
        batch_size = lidar_points.shape[0]
        
        # Handle variable number of points by masking
        # Points with all zeros are considered padding
        valid_mask = torch.any(lidar_points != 0, dim=-1)  # (batch_size, max_points)
        
        # Encode each point
        point_features = self.point_encoder(lidar_points)  # (batch_size, max_points, embed_dim)
        
        # Mask invalid points
        point_features = point_features * valid_mask.unsqueeze(-1)
        
        # Global max pooling to get fixed-size representation
        global_features, _ = torch.max(point_features, dim=1)  # (batch_size, embed_dim)
        
        # Final encoding
        encoded_features = self.global_encoder(global_features)
        
        return encoded_features

class StateEncoder(nn.Module):
    """Encode vehicle state information"""
    def __init__(self, state_dim=5, embed_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
        
    def forward(self, state):
        return self.encoder(state)

class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing different modalities"""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        output = self.output_proj(attention_output)
        return output

class VisionTransformerDriving(nn.Module):
    """Vision Transformer for autonomous driving with multi-modal fusion"""
    
    def __init__(self, 
                 img_size=84, 
                 patch_size=16, 
                 in_channels=3,
                 embed_dim=512,
                 depth=6,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 state_dim=5,
                 max_lidar_points=2048,
                 num_actions=3):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Image processing
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.n_patches
        
        # Positional embedding for image patches
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # LiDAR processing
        self.lidar_encoder = LiDAREncoder(max_lidar_points, embed_dim // 2)
        
        # State processing
        self.state_encoder = StateEncoder(state_dim, embed_dim // 2)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Cross-modal fusion
        self.modal_fusion = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2 + embed_dim // 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, num_actions),
            nn.Tanh()  # Actions are in [-1, 1] range
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(self, images, states, lidar_points=None):
        batch_size = images.shape[0]
        
        # Process images
        image_patches = self.patch_embedding(images)  # (batch_size, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        image_tokens = torch.cat([cls_tokens, image_patches], dim=1)
        
        # Add positional embedding
        image_tokens += self.pos_embedding
        
        # Apply transformer blocks to image tokens
        for block in self.transformer_blocks:
            image_tokens = block(image_tokens)
        
        # Extract class token (global image representation)
        image_features = image_tokens[:, 0]  # (batch_size, embed_dim)
        
        # Process other modalities
        state_features = self.state_encoder(states)  # (batch_size, embed_dim//2)
        
        if lidar_points is not None:
            lidar_features = self.lidar_encoder(lidar_points)  # (batch_size, embed_dim//2)
        else:
            lidar_features = torch.zeros(batch_size, self.embed_dim // 2, device=images.device)
        
        # Fuse all modalities
        fused_features = torch.cat([image_features, state_features, lidar_features], dim=1)
        fused_features = self.modal_fusion(fused_features)
        
        # Predict actions
        actions = self.action_head(fused_features)
        
        return actions
    
    def get_attention_maps(self, images, states, lidar_points=None):
        """Get attention maps for visualization"""
        batch_size = images.shape[0]
        
        # Process images
        image_patches = self.patch_embedding(images)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        image_tokens = torch.cat([cls_tokens, image_patches], dim=1)
        image_tokens += self.pos_embedding
        
        attention_maps = []
        
        # Store attention maps from each layer
        for block in self.transformer_blocks:
            # Get attention weights from the block
            attention_weights = block.attention(block.norm1(image_tokens))
            attention_maps.append(attention_weights)
            image_tokens = block(image_tokens)
        
        return attention_maps

# Utility function to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model factory function
def create_vit_driving_model(config=None):
    """Create a Vision Transformer model for driving"""
    if config is None:
        config = {
            'img_size': 84,
            'patch_size': 16,
            'embed_dim': 512,
            'depth': 6,
            'num_heads': 8,
            'state_dim': 5,
            'max_lidar_points': 2048
        }
    
    model = VisionTransformerDriving(**config)
    
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    return model
