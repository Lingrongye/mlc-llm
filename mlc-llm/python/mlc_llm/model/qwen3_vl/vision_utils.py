"""
Vision utilities for Qwen3-VL.

This module provides utilities for:
1. Computing position embeddings (rotary + learned) for vision
2. Preprocessing images for the vision model
3. Converting between formats

These utilities are meant to be called from Python before passing
data to the compiled MLC model.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def compute_vision_rotary_pos_emb(
    grid_thw: List[Tuple[int, int, int]],
    head_dim: int,
    spatial_merge_size: int = 2,
    theta: float = 10000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rotary position embeddings (cos, sin) for vision tokens.
    
    Following HuggingFace Qwen3-VL:
    1. Compute frequency table based on max height/width
    2. For each token, compute (row, col) position
    3. Look up frequencies and compute cos/sin
    
    Args:
        grid_thw: List of (t, h, w) tuples for each image
            t: number of temporal frames
            h: height in patches
            w: width in patches
        head_dim: Head dimension (hidden_size // num_heads)
        spatial_merge_size: Spatial merge size (default 2)
        theta: Base value for frequency computation (default 10000.0)
        
    Returns:
        Tuple of (cos, sin) arrays, each of shape (total_tokens, head_dim)
    """
    # Compute rotary embedding dimension
    # Following HF: rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
    dim = head_dim // 2
    
    # Find max height/width for frequency table
    max_hw = max(max(h, w) for t, h, w in grid_thw)
    
    # Compute inverse frequencies
    # inv_freq = 1.0 / (theta ** (arange(0, dim, 2) / dim))
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    
    # Compute frequency table
    # freqs = outer(arange(max_hw), inv_freq)
    seq = np.arange(max_hw, dtype=np.float32)
    freq_table = np.outer(seq, inv_freq)  # (max_hw, dim // 2)
    
    # Compute position IDs for all tokens
    all_pos_ids = []
    merge_size = spatial_merge_size
    
    for t, h, w in grid_thw:
        merged_h = h // merge_size
        merged_w = w // merge_size
        
        # Following HuggingFace rot_pos_emb implementation:
        # For each patch position, compute (row, col) after spatial merge pattern
        positions = []
        for br in range(merged_h):
            for bc in range(merged_w):
                for ir in range(merge_size):
                    for ic in range(merge_size):
                        row_idx = br * merge_size + ir
                        col_idx = bc * merge_size + ic
                        positions.append([row_idx, col_idx])
        
        pos_ids = np.array(positions, dtype=np.int64)
        
        # Repeat for temporal frames
        if t > 1:
            pos_ids = np.tile(pos_ids, (t, 1))
        
        all_pos_ids.append(pos_ids)
    
    # Concatenate all position IDs
    pos_ids = np.concatenate(all_pos_ids, axis=0)  # (total_tokens, 2)
    
    # Look up frequencies
    # Following HF: embeddings = freq_table[pos_ids]
    row_freqs = freq_table[pos_ids[:, 0]]  # (total, dim // 2)
    col_freqs = freq_table[pos_ids[:, 1]]  # (total, dim // 2)
    
    # Combine row and col frequencies
    # Following HF: embeddings.flatten(1)
    embeddings = np.concatenate([row_freqs, col_freqs], axis=1)  # (total, dim)
    
    # Duplicate for full head_dim
    # Following HF: emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    emb = np.concatenate([embeddings, embeddings], axis=-1)  # (total, head_dim)
    
    # Compute cos and sin
    cos = np.cos(emb).astype(np.float32)
    sin = np.sin(emb).astype(np.float32)
    
    return cos, sin


def compute_vision_learned_pos_emb(
    grid_thw: List[Tuple[int, int, int]],
    pos_embed_weights: np.ndarray,
    hidden_size: int,
    num_grid_per_side: int,
    spatial_merge_size: int = 2,
) -> np.ndarray:
    """
    Compute learned position embeddings using bilinear interpolation.
    
    Following HuggingFace Qwen3-VL fast_pos_embed_interpolate:
    1. Compute bilinear interpolation indices and weights
    2. Interpolate from the learned embedding table
    3. Permute for spatial merge pattern
    
    Args:
        grid_thw: List of (t, h, w) tuples for each image
        pos_embed_weights: Learned position embedding weights of shape (num_pos, hidden_size)
        hidden_size: Hidden size
        num_grid_per_side: Number of grid positions per side (sqrt(num_pos))
        spatial_merge_size: Spatial merge size
        
    Returns:
        Position embeddings of shape (total_tokens, hidden_size)
    """
    pos_embeds_list = []
    merge_size = spatial_merge_size
    
    for t, h, w in grid_thw:
        # Compute bilinear interpolation indices and weights
        h_idxs = np.linspace(0, num_grid_per_side - 1, h, dtype=np.float32)
        w_idxs = np.linspace(0, num_grid_per_side - 1, w, dtype=np.float32)
        
        h_floor = h_idxs.astype(np.int64)
        w_floor = w_idxs.astype(np.int64)
        h_ceil = np.clip(h_floor + 1, 0, num_grid_per_side - 1)
        w_ceil = np.clip(w_floor + 1, 0, num_grid_per_side - 1)
        
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor
        
        # Compute 4-corner indices
        idx_00 = h_floor[:, None] * num_grid_per_side + w_floor[None, :]  # (h, w)
        idx_01 = h_floor[:, None] * num_grid_per_side + w_ceil[None, :]
        idx_10 = h_ceil[:, None] * num_grid_per_side + w_floor[None, :]
        idx_11 = h_ceil[:, None] * num_grid_per_side + w_ceil[None, :]
        
        # Compute interpolation weights
        w_00 = (1 - dh)[:, None] * (1 - dw)[None, :]
        w_01 = (1 - dh)[:, None] * dw[None, :]
        w_10 = dh[:, None] * (1 - dw)[None, :]
        w_11 = dh[:, None] * dw[None, :]
        
        # Interpolate
        # pos_embed shape: (h*w, hidden_size)
        pos_embed = (
            pos_embed_weights[idx_00.flatten()] * w_00.flatten()[:, None] +
            pos_embed_weights[idx_01.flatten()] * w_01.flatten()[:, None] +
            pos_embed_weights[idx_10.flatten()] * w_10.flatten()[:, None] +
            pos_embed_weights[idx_11.flatten()] * w_11.flatten()[:, None]
        )
        
        # Reshape and repeat for temporal frames
        pos_embed = pos_embed.reshape(h, w, hidden_size)
        pos_embed = np.tile(pos_embed[None, :, :, :], (t, 1, 1, 1))  # (t, h, w, hidden_size)
        
        # Permute for spatial merge pattern
        # Following HF: permute(0, 1, 3, 2, 4, 5).flatten(0, 4)
        h_m, w_m = h // merge_size, w // merge_size
        pos_embed = pos_embed.reshape(t, h_m, merge_size, w_m, merge_size, hidden_size)
        pos_embed = pos_embed.transpose(0, 1, 3, 2, 4, 5)
        pos_embed = pos_embed.reshape(-1, hidden_size)
        
        pos_embeds_list.append(pos_embed)
    
    # Concatenate all
    learned_pos_embed = np.concatenate(pos_embeds_list, axis=0).astype(np.float32)
    return learned_pos_embed


def compute_all_vision_pos_embeddings(
    grid_thw: List[Tuple[int, int, int]],
    pos_embed_weights: np.ndarray,
    hidden_size: int,
    head_dim: int,
    num_grid_per_side: int,
    spatial_merge_size: int = 2,
    theta: float = 10000.0,
) -> Dict[str, np.ndarray]:
    """
    Compute all position embeddings needed for vision model.
    
    This is a convenience function that computes both rotary and learned
    position embeddings in one call.
    
    Args:
        grid_thw: List of (t, h, w) tuples for each image
        pos_embed_weights: Learned position embedding weights
        hidden_size: Hidden size
        head_dim: Head dimension
        num_grid_per_side: Number of grid positions per side
        spatial_merge_size: Spatial merge size
        theta: Base value for rotary frequencies
        
    Returns:
        Dictionary with:
            - 'rotary_cos': Rotary cosine of shape (total_tokens, head_dim)
            - 'rotary_sin': Rotary sine of shape (total_tokens, head_dim)
            - 'learned_pos_embed': Learned position embedding of shape (total_tokens, hidden_size)
            - 'total_tokens': Total number of tokens
    """
    # Compute rotary position embeddings
    rotary_cos, rotary_sin = compute_vision_rotary_pos_emb(
        grid_thw=grid_thw,
        head_dim=head_dim,
        spatial_merge_size=spatial_merge_size,
        theta=theta,
    )
    
    # Compute learned position embeddings
    learned_pos_embed = compute_vision_learned_pos_emb(
        grid_thw=grid_thw,
        pos_embed_weights=pos_embed_weights,
        hidden_size=hidden_size,
        num_grid_per_side=num_grid_per_side,
        spatial_merge_size=spatial_merge_size,
    )
    
    return {
        'rotary_cos': rotary_cos,
        'rotary_sin': rotary_sin,
        'learned_pos_embed': learned_pos_embed,
        'total_tokens': rotary_cos.shape[0],
    }


def compute_cu_seqlens(grid_thw: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Compute cumulative sequence lengths for variable-length attention.
    
    Args:
        grid_thw: List of (t, h, w) tuples for each image
        
    Returns:
        Cumulative sequence lengths array starting with 0
    """
    lengths = [t * h * w for t, h, w in grid_thw]
    cu_seqlens = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    return cu_seqlens


def get_grid_thw_from_image_size(
    image_height: int,
    image_width: int,
    patch_size: int = 16,
    temporal_patch_size: int = 2,
    spatial_merge_size: int = 2,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
) -> Tuple[int, int, int]:
    """
    Compute grid_thw for a single image based on its dimensions.
    
    This follows the Qwen3-VL image preprocessing logic.
    
    Args:
        image_height: Original image height
        image_width: Original image width
        patch_size: Size of each patch
        temporal_patch_size: Temporal patch size (for video)
        spatial_merge_size: Spatial merge size
        min_pixels: Minimum number of pixels
        max_pixels: Maximum number of pixels
        
    Returns:
        Tuple of (t, h, w) for grid dimensions
    """
    # For a single image, temporal dimension is 1 (or temporal_patch_size for padding)
    t = temporal_patch_size  # Usually 2 for Qwen3-VL
    
    # Compute spatial grid dimensions
    # The image is resized such that the total pixels are within bounds
    aspect_ratio = image_width / image_height
    
    # Calculate target dimensions maintaining aspect ratio
    total_pixels = image_height * image_width
    
    if total_pixels < min_pixels:
        scale = (min_pixels / total_pixels) ** 0.5
        target_height = int(image_height * scale)
        target_width = int(image_width * scale)
    elif total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
        target_height = int(image_height * scale)
        target_width = int(image_width * scale)
    else:
        target_height = image_height
        target_width = image_width
    
    # Round to nearest multiple of patch_size
    target_height = ((target_height + patch_size - 1) // patch_size) * patch_size
    target_width = ((target_width + patch_size - 1) // patch_size) * patch_size
    
    # Compute grid dimensions (in patches)
    h = target_height // patch_size
    w = target_width // patch_size
    
    return t, h, w


class VisionPositionEmbeddingComputer:
    """
    Helper class for computing vision position embeddings.
    
    This class caches the position embedding weights and provides
    methods for computing position embeddings for different images.
    """
    
    def __init__(
        self,
        pos_embed_weights: np.ndarray,
        hidden_size: int,
        head_dim: int,
        num_position_embeddings: int = 2304,
        spatial_merge_size: int = 2,
        theta: float = 10000.0,
    ):
        """
        Initialize the position embedding computer.
        
        Args:
            pos_embed_weights: Learned position embedding weights
            hidden_size: Hidden size
            head_dim: Head dimension
            num_position_embeddings: Number of position embeddings
            spatial_merge_size: Spatial merge size
            theta: Base value for rotary frequencies
        """
        self.pos_embed_weights = pos_embed_weights
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.theta = theta
    
    def compute(
        self, 
        grid_thw: List[Tuple[int, int, int]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute all position embeddings for given grid dimensions.
        
        Args:
            grid_thw: List of (t, h, w) tuples
            
        Returns:
            Dictionary with rotary_cos, rotary_sin, learned_pos_embed
        """
        return compute_all_vision_pos_embeddings(
            grid_thw=grid_thw,
            pos_embed_weights=self.pos_embed_weights,
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.spatial_merge_size,
            theta=self.theta,
        )
