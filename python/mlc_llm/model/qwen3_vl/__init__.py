"""Qwen3-VL model implementation for MLC-LLM."""

from .qwen3_vl_config import Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig
from .qwen3_vl_model import Qwen3VLModel, Qwen3VLForConditionalGeneration
from .qwen3_vl_vision import Qwen3VLVisionModel
from .qwen3_vl_text import Qwen3VLTextModel
from .vision_utils import (
    compute_vision_rotary_pos_emb,
    compute_vision_learned_pos_emb,
    compute_all_vision_pos_embeddings,
    compute_cu_seqlens,
    get_grid_thw_from_image_size,
    VisionPositionEmbeddingComputer,
)

# 推理引擎 (延迟导入以避免循环依赖)
def Qwen3VLEngine(*args, **kwargs):
    """Qwen3-VL 多模态推理引擎"""
    from .qwen3_vl_engine import Qwen3VLEngine as _Engine
    return _Engine(*args, **kwargs)

def create_engine(*args, **kwargs):
    """创建 Qwen3-VL 引擎的便捷函数"""
    from .qwen3_vl_engine import create_engine as _create
    return _create(*args, **kwargs)

__all__ = [
    # Config
    "Qwen3VLConfig",
    "Qwen3VLTextConfig", 
    "Qwen3VLVisionConfig",
    # Model
    "Qwen3VLModel",
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLVisionModel",
    "Qwen3VLTextModel",
    # Vision utils
    "compute_vision_rotary_pos_emb",
    "compute_vision_learned_pos_emb",
    "compute_all_vision_pos_embeddings",
    "compute_cu_seqlens",
    "get_grid_thw_from_image_size",
    "VisionPositionEmbeddingComputer",
    # Engine (推理 API)
    "Qwen3VLEngine",
    "create_engine",
]
