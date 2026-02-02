"""
Position embedding utilities for Qwen3-VL Vision Model.

This module implements the rotary position embedding for Vision Transformer
following the official Qwen3-VL HuggingFace implementation.

Key components:
1. Qwen3VLVisionRotaryEmbedding - Computes frequency table
2. rot_pos_emb - Computes (cos, sin) position embeddings from grid_thw
3. apply_rotary_pos_emb_vision - Applies RoPE to query and key tensors
"""

from tvm.relax.frontend.nn import Module, Tensor, Parameter, op
from tvm.relax.frontend.nn.op import wrap_nested
from tvm.relax.op import strided_slice as relax_strided_slice
from tvm import relax as rx
from tvm import tir
import math


def op_strided_slice(x, axes, begin, end):
    """Strided slice wrapper for Relax."""
    return wrap_nested(relax_strided_slice(x._expr, axes, begin, end), name="strided_slice")


# ============================================================
# Helper Functions
# ============================================================

def rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input.
    
    Args:
        x: Input tensor of shape (..., head_dim)
        
    Returns:
        Tensor with rotated halves: (-x2, x1)
    """
    # Split into two halves
    x_shape = x.shape
    ndim = len(x_shape)
    last_axis = ndim - 1  # TVM doesn't support negative axis, compute positive
    half_dim = x_shape[-1] // 2
    
    # Get first and second halves
    x1 = op_strided_slice(x, axes=[last_axis], begin=[0], end=[half_dim])
    x2 = op_strided_slice(x, axes=[last_axis], begin=[half_dim], end=[x_shape[-1]])
    
    # Negate x2 and concatenate
    neg_x2 = op.negative(x2)
    return op.concat([neg_x2, x1], dim=-1)


def apply_rotary_pos_emb_vision(
    q: Tensor, 
    k: Tensor, 
    cos: Tensor, 
    sin: Tensor,
) -> tuple:
    """
    Applies Rotary Position Embedding to query and key tensors for Vision.
    
    Following the HuggingFace Qwen3-VL implementation:
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)  # Add head dimension
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    
    Args:
        q: Query tensor of shape (seq_len, num_heads, head_dim)
        k: Key tensor of shape (seq_len, num_heads, head_dim)
        cos: Cosine tensor of shape (seq_len, head_dim)
        sin: Sine tensor of shape (seq_len, head_dim)
        
    Returns:
        Tuple of (q_embed, k_embed) with rotary position encoding applied
    """
    # Cast cos/sin to match q/k dtype for binary operations
    cos = cos.astype(q.dtype)
    sin = sin.astype(q.dtype)
    
    # Add head dimension: (seq_len, head_dim) -> (seq_len, 1, head_dim)
    cos = op.reshape(cos, (cos.shape[0], 1, cos.shape[1]))
    sin = op.reshape(sin, (sin.shape[0], 1, sin.shape[1]))
    
    # Apply rotary embedding
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    q_cos = op.multiply(q, cos)
    q_rot = rotate_half(q)
    q_sin = op.multiply(q_rot, sin)
    q_embed = op.add(q_cos, q_sin)
    
    # k_embed = (k * cos) + (rotate_half(k) * sin)
    k_cos = op.multiply(k, cos)
    k_rot = rotate_half(k)
    k_sin = op.multiply(k_rot, sin)
    k_embed = op.add(k_cos, k_sin)
    
    return q_embed, k_embed


# ============================================================
# Rotary Position Embedding Module
# ============================================================

class Qwen3VLVisionRotaryEmbedding(Module):
    """
    Rotary Position Embedding for Qwen3-VL Vision.
    
    This computes a frequency table that can be indexed by position IDs.
    
    Following HuggingFace:
        inv_freq = 1.0 / (theta ** (arange(0, dim, 2) / dim))
        freqs = outer(seq, inv_freq)
    """
    
    def __init__(self, dim: int, theta: float = 10000.0, max_position: int = 4096):
        """
        Initialize the rotary embedding.
        
        Args:
            dim: Dimension of the embedding (head_dim // 2)
            theta: Base value for frequency computation
            max_position: Maximum position index supported
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_position = max_position
        
        # Precompute inverse frequencies
        # inv_freq = 1.0 / (theta ** (arange(0, dim, 2) / dim))
        # This gives us dim // 2 frequencies
        import numpy as np
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        
        # Precompute frequency table for all positions up to max_position
        # freqs[pos, i] = pos * inv_freq[i]
        positions = np.arange(max_position, dtype=np.float32)
        freqs = np.outer(positions, inv_freq)  # (max_position, dim // 2)
        
        # Store as parameter (will be loaded as constant)
        self.freq_table = Parameter((max_position, dim // 2), dtype="float32", name="freq_table")
        # Note: The actual values will be set during weight loading
        # For now, we just define the shape
    
    def forward(self, max_seq_len: int) -> Tensor:
        """
        Get frequency table up to max_seq_len positions.
        
        Args:
            max_seq_len: Maximum sequence length needed
            
        Returns:
            Frequency table of shape (max_seq_len, dim // 2)
        """
        # Return slice of precomputed table
        return op_strided_slice(self.freq_table, axes=[0], begin=[0], end=[max_seq_len])


# ============================================================
# Position ID Computation
# ============================================================

def compute_vision_pos_ids_static(
    t: int, h: int, w: int, 
    merge_size: int,
    device: str = "cuda",
) -> Tensor:
    """
    Compute position IDs for a single image/video with static shapes.
    
    Following HuggingFace Qwen3-VL:
        For each patch, compute its (row, col) position after considering
        the spatial merge pattern.
    
    Args:
        t: Number of temporal frames
        h: Height in patches  
        w: Width in patches
        merge_size: Spatial merge size (typically 2)
        
    Returns:
        Position IDs tensor of shape (t * h * w, 2) containing (row_idx, col_idx)
    """
    import numpy as np
    
    merged_h = h // merge_size
    merged_w = w // merge_size
    
    # Compute positions for one frame
    # Following HuggingFace:
    # block_rows = arange(merged_h), block_cols = arange(merged_w)
    # intra_row = arange(merge_size), intra_col = arange(merge_size)
    # row_idx = block_rows * merge_size + intra_row
    # col_idx = block_cols * merge_size + intra_col
    
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
    
    return pos_ids  # Shape: (t * h * w, 2)


# ============================================================
# Complete Position Embedding Computation
# ============================================================

def compute_rotary_pos_emb(
    grid_thw: list,  # List of (t, h, w) tuples
    head_dim: int,
    spatial_merge_size: int,
    theta: float = 10000.0,
    dtype: str = "float32",
):
    """
    Compute complete rotary position embeddings for vision.
    
    This is called at runtime to compute position embeddings.
    
    Following HuggingFace Qwen3-VL:
        1. Compute freq_table = rotary_pos_emb(max_hw)
        2. For each token, get (row, col) position
        3. embeddings = freq_table[pos_ids]  # (total, 2, dim//2)
        4. embeddings = embeddings.flatten(1)  # (total, dim)
        5. emb = concat(embeddings, embeddings)  # (total, 2*dim)
        6. return (emb.cos(), emb.sin())
    
    Args:
        grid_thw: List of (t, h, w) tuples for each image
        head_dim: Head dimension for rotary embedding
        spatial_merge_size: Spatial merge size
        theta: Base value for frequencies
        dtype: Output dtype
        
    Returns:
        Tuple of (cos, sin) tensors, each of shape (total_tokens, head_dim)
    """
    import numpy as np
    
    # Compute frequency table dimension
    dim = head_dim // 2  # Following HF: rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
    
    # Find max height/width for frequency table
    max_hw = max(max(h, w) for t, h, w in grid_thw)
    
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    
    # Compute frequency table
    seq = np.arange(max_hw, dtype=np.float32)
    freq_table = np.outer(seq, inv_freq)  # (max_hw, dim // 2)
    
    # Compute position IDs for all tokens
    all_pos_ids = []
    for t, h, w in grid_thw:
        pos_ids = compute_vision_pos_ids_static(t, h, w, spatial_merge_size)
        all_pos_ids.append(pos_ids)
    
    pos_ids = np.concatenate(all_pos_ids, axis=0)  # (total_tokens, 2)
    
    # Lookup frequencies
    # embeddings[i] = [freq_table[pos_ids[i, 0]], freq_table[pos_ids[i, 1]]]
    row_freqs = freq_table[pos_ids[:, 0]]  # (total, dim // 2)
    col_freqs = freq_table[pos_ids[:, 1]]  # (total, dim // 2)
    
    # Combine row and col frequencies
    # embeddings = stack([row_freqs, col_freqs], axis=1).flatten(1)
    embeddings = np.concatenate([row_freqs, col_freqs], axis=1)  # (total, dim)
    
    # Duplicate for full head_dim
    emb = np.concatenate([embeddings, embeddings], axis=-1)  # (total, head_dim)
    
    # Compute cos and sin
    cos = np.cos(emb).astype(dtype)
    sin = np.sin(emb).astype(dtype)
    
    return cos, sin


# ============================================================
# Relax-based Position Embedding (for compilation)
# ============================================================

class VisionPositionEmbedding(Module):
    """
    Vision Position Embedding module that can be compiled with MLC.
    
    This provides learned position embeddings with bilinear interpolation
    plus rotary position embeddings for attention.
    """
    
    def __init__(
        self, 
        hidden_size: int,
        head_dim: int,
        num_position_embeddings: int = 2304,
        spatial_merge_size: int = 2,
        theta: float = 10000.0,
        max_position: int = 4096,
    ):
        """
        Initialize position embedding.
        
        Args:
            hidden_size: Hidden size for learned embeddings
            head_dim: Head dimension for rotary embeddings
            num_position_embeddings: Number of learned position embeddings
            spatial_merge_size: Spatial merge size
            theta: Base value for rotary frequencies
            max_position: Maximum position for rotary
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_position_embeddings = num_position_embeddings
        self.num_grid_per_side = int(num_position_embeddings ** 0.5)
        self.spatial_merge_size = spatial_merge_size
        self.theta = theta
        
        # Learned position embeddings
        self.pos_embed = Parameter(
            (num_position_embeddings, hidden_size), 
            dtype="float32",
            name="pos_embed"
        )
        
        # Precomputed rotary frequency table
        # Shape: (max_position, head_dim // 4)  
        # Note: dim = head_dim // 2, but inv_freq has dim // 2 elements
        self.rotary_dim = head_dim // 2
        self.inv_freq_dim = self.rotary_dim // 2
        
        import numpy as np
        inv_freq = 1.0 / (theta ** (np.arange(0, self.rotary_dim, 2, dtype=np.float32) / self.rotary_dim))
        positions = np.arange(max_position, dtype=np.float32)
        freq_table = np.outer(positions, inv_freq)  # (max_position, rotary_dim // 2)
        
        self.freq_table = Parameter(
            (max_position, self.inv_freq_dim),
            dtype="float32", 
            name="freq_table"
        )
    
    def get_rotary_cos_sin(self, pos_ids: Tensor, total_tokens: int) -> tuple:
        """
        Get cos and sin for rotary embedding given position IDs.
        
        Args:
            pos_ids: Position IDs of shape (total_tokens, 2) for (row, col)
            total_tokens: Total number of tokens
            
        Returns:
            Tuple of (cos, sin), each of shape (total_tokens, head_dim)
        """
        # This would need dynamic indexing which is complex in Relax
        # For now, we use a simplified approach with static shapes
        pass


def compute_position_embeddings_numpy(
    grid_thw: list,
    hidden_size: int,
    head_dim: int,
    pos_embed_weights: 'np.ndarray',
    num_grid_per_side: int,
    spatial_merge_size: int,
    theta: float = 10000.0,
):
    """
    Compute both learned position embeddings and rotary embeddings using NumPy.
    
    This can be called from Python and the results passed to the model.
    
    Args:
        grid_thw: List of (t, h, w) tuples
        hidden_size: Hidden size
        head_dim: Head dimension
        pos_embed_weights: Learned position embedding weights (num_pos, hidden_size)
        num_grid_per_side: Number of grid positions per side
        spatial_merge_size: Spatial merge size
        theta: Base value for rotary
        
    Returns:
        Dictionary with:
            - 'pos_embed': Learned position embeddings (total, hidden_size)
            - 'rotary_cos': Rotary cosine (total, head_dim)  
            - 'rotary_sin': Rotary sine (total, head_dim)
    """
    import numpy as np
    
    # 1. Compute learned position embeddings with bilinear interpolation
    pos_embeds_list = []
    
    for t, h, w in grid_thw:
        # Bilinear interpolation indices and weights
        h_idxs = np.linspace(0, num_grid_per_side - 1, h)
        w_idxs = np.linspace(0, num_grid_per_side - 1, w)
        
        h_floor = h_idxs.astype(np.int64)
        w_floor = w_idxs.astype(np.int64)
        h_ceil = np.clip(h_floor + 1, 0, num_grid_per_side - 1)
        w_ceil = np.clip(w_floor + 1, 0, num_grid_per_side - 1)
        
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor
        
        # 4 corners
        idx_00 = h_floor[:, None] * num_grid_per_side + w_floor[None, :]
        idx_01 = h_floor[:, None] * num_grid_per_side + w_ceil[None, :]
        idx_10 = h_ceil[:, None] * num_grid_per_side + w_floor[None, :]
        idx_11 = h_ceil[:, None] * num_grid_per_side + w_ceil[None, :]
        
        # Weights
        w_00 = (1 - dh)[:, None] * (1 - dw)[None, :]
        w_01 = (1 - dh)[:, None] * dw[None, :]
        w_10 = dh[:, None] * (1 - dw)[None, :]
        w_11 = dh[:, None] * dw[None, :]
        
        # Interpolate
        pos_embed = (
            pos_embed_weights[idx_00.flatten()] * w_00.flatten()[:, None] +
            pos_embed_weights[idx_01.flatten()] * w_01.flatten()[:, None] +
            pos_embed_weights[idx_10.flatten()] * w_10.flatten()[:, None] +
            pos_embed_weights[idx_11.flatten()] * w_11.flatten()[:, None]
        )  # (h * w, hidden_size)
        
        # Repeat for temporal frames and permute for spatial merge
        pos_embed = pos_embed.reshape(h, w, hidden_size)
        pos_embed = np.tile(pos_embed[None, :, :, :], (t, 1, 1, 1))
        
        # Permute for spatial merge pattern
        merge = spatial_merge_size
        h_m, w_m = h // merge, w // merge
        pos_embed = pos_embed.reshape(t, h_m, merge, w_m, merge, hidden_size)
        pos_embed = pos_embed.transpose(0, 1, 3, 2, 4, 5)
        pos_embed = pos_embed.reshape(-1, hidden_size)
        
        pos_embeds_list.append(pos_embed)
    
    learned_pos_embed = np.concatenate(pos_embeds_list, axis=0)
    
    # 2. Compute rotary position embeddings
    rotary_cos, rotary_sin = compute_rotary_pos_emb(
        grid_thw, head_dim, spatial_merge_size, theta
    )
    
    return {
        'pos_embed': learned_pos_embed.astype(np.float32),
        'rotary_cos': rotary_cos.astype(np.float32),
        'rotary_sin': rotary_sin.astype(np.float32),
    }
