"""
Video Transformer Model for Text-to-Video Diffusion
FIXED: Uses Cross-Attention for proper text conditioning
"""
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for text conditioning"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x, text_emb):
        """
        Args:
            x: [B, Seq_Len, Dim] - spatial features
            text_emb: [B, Text_Seq_Len, Dim] - text embeddings
        """
        # Cross-attention: query from x, key/value from text
        x = x + self.cross_attn(self.norm1(x), self.norm2(text_emb), self.norm2(text_emb))[0]
        x = x + self.ff(self.norm1(x))
        return x


class VideoTransformer(ModelMixin, ConfigMixin):
    """
    Transformer-based denoiser with PROPER text conditioning via cross-attention
    """
    @register_to_config
    def __init__(self, num_layers=12, hidden_dim=768, num_heads=12, sequence_length=1024): 
        super().__init__()
        
        # Initialize embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, sequence_length, hidden_dim) * 0.02)
        self.time_embed = nn.Sequential(
            nn.Linear(320, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.input_proj = nn.Linear(4, hidden_dim) 
        self.text_proj = nn.Linear(512, hidden_dim)  # Project text to hidden_dim
        
        # Alternate self-attention and cross-attention blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Self-attention block
            self.blocks.append(TransformerBlock(hidden_dim, num_heads))
            # Cross-attention block for text conditioning
            self.blocks.append(CrossAttentionBlock(hidden_dim, num_heads))
        
        # Output projection with zero initialization
        self.output_proj = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, sample, timestep, encoder_hidden_states):
        """
        Args:
            sample: [Batch, Channels, Height, Width] - single frame latents
            timestep: [Batch] or [Batch, 320] - timestep embeddings
            encoder_hidden_states: [Batch, Seq_Len, Text_Dim] - text embeddings
        """
        sample_dtype = sample.dtype
        
        b, c, h, w = sample.shape
        x = sample.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = self.input_proj(x.to(next(self.input_proj.parameters()).dtype)).to(sample_dtype)
        
        # Add time embedding
        if timestep.dim() == 1:
            timestep_emb = self._get_timestep_embedding(timestep, 320).to(sample_dtype)
            time_embed_dtype = next(self.time_embed.parameters()).dtype
            t_emb = self.time_embed(timestep_emb.to(time_embed_dtype)).unsqueeze(1).to(sample_dtype)
        else:
            time_embed_dtype = next(self.time_embed.parameters()).dtype
            t_emb = self.time_embed(timestep.to(time_embed_dtype)).unsqueeze(1).to(sample_dtype)
        
        # Add positional embedding
        pos_emb = self.pos_embed[:, :x.size(1), :].to(sample_dtype)
        x = x + t_emb + pos_emb
        
        # Project text embeddings to hidden_dim
        text_proj_dtype = next(self.text_proj.parameters()).dtype
        text_emb = self.text_proj(encoder_hidden_states.to(text_proj_dtype)).to(sample_dtype)
        
        # Pass through alternating self-attention and cross-attention blocks
        for i, block in enumerate(self.blocks):
            if isinstance(block, CrossAttentionBlock):
                # Cross-attention: condition on text
                x = block(x, text_emb)
            else:
                # Self-attention
                x = block(x)
        
        # Output projection
        x = self.output_proj(x)
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2)
    
    def _get_timestep_embedding(self, timesteps, dim):
        """Create sinusoidal timestep embeddings"""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb
