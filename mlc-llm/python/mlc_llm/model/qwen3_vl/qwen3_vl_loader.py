"""
Minimal loader for Qwen3-VL.
"""
import functools
from typing import Callable, List

import numpy as np
from mlc_llm.loader import ExternMapping, QuantizeMapping
from mlc_llm.quantization import BlockScaleQuantize, Quantization
from mlc_llm.model.qwen3_vl.qwen3_vl_config import Qwen3VLConfig
from mlc_llm.model.qwen3_vl.qwen3_vl_model import Qwen3VLForConditionalGeneration


def _compute_vision_rotary_embedding(
    head_dim: int,
    max_position: int = 4096,
    theta: float = 10000.0,
    dtype: str = "float32",
):
    """
    Compute precomputed cos/sin for rotary position embeddings.
    
    This follows the Qwen3-VL VisionRotaryEmbedding implementation:
    - inv_freq = 1.0 / (theta ** (arange(0, dim, 2) / dim))
    - freqs = outer(positions, inv_freq)
    - cos/sin = cos/sin(concat(freqs, freqs))
    
    Args:
        head_dim: Head dimension
        max_position: Maximum position supported
        theta: Base value for frequencies
        dtype: Output dtype
        
    Returns:
        Tuple of (cos, sin) arrays, each of shape (max_position, head_dim)
    """
    # Following HF: dim = head_dim // 2 for VisionRotaryEmbedding
    dim = head_dim // 2
    
    # Compute inverse frequencies
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    
    # Compute frequency table for all positions
    positions = np.arange(max_position, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # (max_position, dim // 2)
    
    # Duplicate frequencies: (max_position, dim // 2) -> (max_position, dim)
    freqs = np.concatenate([freqs, freqs], axis=-1)
    
    # Duplicate again for full head_dim: (max_position, dim) -> (max_position, head_dim)
    emb = np.concatenate([freqs, freqs], axis=-1)
    
    # Compute cos and sin
    cos = np.cos(emb).astype(dtype)
    sin = np.sin(emb).astype(dtype)
    
    return cos, sin


def huggingface(model_config: Qwen3VLConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of MLC LLM parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : Qwen3VLConfig
        The configuration of the Qwen3-VL model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from MLC to HuggingFace PyTorch.
    """
    model = Qwen3VLForConditionalGeneration(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    
    if isinstance(quantization, BlockScaleQuantize):
        # Convert the model to block-scale quantized model before loading parameters
        model = quantization.quantize_model(model, QuantizeMapping({}, {}), "")
        if model_config.text_config.weight_block_size is None:
             raise ValueError(
                "The input Qwen3-VL model is not fp8 block quantized. "
                "Thus BlockScaleQuantize is not supported."
            )

    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    if (
        not isinstance(quantization, BlockScaleQuantize)
        and model_config.text_config.weight_block_size is not None
    ):
         raise ValueError(
            "The input Qwen3-VL model is fp8 block quantized. "
            "Please use BlockScaleQuantize for the model."
        )

    # Helper function to add both weight and scale mappings
    def add_weight_and_scale_mapping(
        weight_mlc_name: str,
        weight_hf_names: List[str],
        weight_transform_func: Callable,
    ):
        if weight_mlc_name not in named_parameters:
            return

        mlc_param = named_parameters[weight_mlc_name]
        mapping.add_mapping(
            weight_mlc_name,
            weight_hf_names,
            functools.partial(weight_transform_func, dtype=mlc_param.dtype),
        )

        if isinstance(quantization, BlockScaleQuantize):
            scale_mlc_name = f"{weight_mlc_name}_scale_inv"
            if scale_mlc_name in named_parameters:
                scale_hf_names = [f"{name}_scale_inv" for name in weight_hf_names]
                scale_param = named_parameters[scale_mlc_name]
                mapping.add_mapping(
                    scale_mlc_name,
                    scale_hf_names,
                    functools.partial(weight_transform_func, dtype=scale_param.dtype),
                )
    
    # ==========================
    # Text Model Mapping
    # ==========================
    prefix = "model.language_model"
    
    for i in range(model_config.text_config.num_hidden_layers):
        # map attention weight
        attn_mlc = f"{prefix}.layers.{i}.self_attn"
        attn_hf = f"{prefix}.layers.{i}.self_attn"
        
        # Merge Q, K, V
        add_weight_and_scale_mapping(
            f"{attn_mlc}.c_attn.weight",
            [
                f"{attn_hf}.q_proj.weight",
                f"{attn_hf}.k_proj.weight",
                f"{attn_hf}.v_proj.weight",
            ],
            lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
        )
        
        if model_config.text_config.attention_bias:
            mlc_name = f"{attn_mlc}.c_attn.bias"
            if mlc_name in named_parameters:
                mlc_param = named_parameters[mlc_name]
                mapping.add_mapping(
                    mlc_name,
                    [
                        f"{attn_hf}.q_proj.bias",
                        f"{attn_hf}.k_proj.bias",
                        f"{attn_hf}.v_proj.bias",
                    ],
                    functools.partial(
                        lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )
        
        # map mlp weight
        mlp_mlc = f"{prefix}.layers.{i}.mlp"
        mlp_hf = f"{prefix}.layers.{i}.mlp"
        
        # Merge Gate, Up
        add_weight_and_scale_mapping(
            f"{mlp_mlc}.gate_up_proj.weight",
            [
                f"{mlp_hf}.gate_proj.weight",
                f"{mlp_hf}.up_proj.weight",
            ],
            lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
        )

    # ==========================
    # Vision Model Mapping
    # ==========================
    
    # Precompute rotary embeddings for vision model
    vision_config = model_config.vision_config
    head_dim = vision_config.hidden_size // vision_config.num_heads
    max_position = 4096  # Same as in vision model definition
    
    freq_cos, freq_sin = _compute_vision_rotary_embedding(
        head_dim=head_dim,
        max_position=max_position,
        theta=10000.0,
        dtype="float32",
    )
    
    # We need to use an existing HF parameter as a "trigger" for loading self-generated params
    # Use pos_embed which exists in HF weights
    trigger_param = "model.visual.pos_embed.weight"

    for mlc_name, mlc_param in named_parameters.items():
        if mlc_name not in mapping.param_map:
            # Special handling for precomputed rotary embeddings
            # Map to an existing HF param but ignore it in the transform function
            if mlc_name == "model.visual.freq_cos.weight":
                mapping.add_mapping(
                    mlc_name,
                    [trigger_param],  # Use existing param as trigger
                    functools.partial(
                        lambda _dummy, dtype, freq=freq_cos: freq.astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )
            elif mlc_name == "model.visual.freq_sin.weight":
                mapping.add_mapping(
                    mlc_name,
                    [trigger_param],  # Use existing param as trigger
                    functools.partial(
                        lambda _dummy, dtype, freq=freq_sin: freq.astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )
            else:
                # Check if this is a vision parameter or text parameter that maps 1:1
                mapping.add_mapping(
                    mlc_name,
                    [mlc_name],
                    functools.partial(
                        lambda x, dtype: x.astype(dtype),
                        dtype=mlc_param.dtype,
                    ),
                )
            
    return mapping