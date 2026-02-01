"""
Qwen3-VL Engine - 简洁的多模态推理 API

使用方式:
    from mlc_llm.model.qwen3_vl import Qwen3VLEngine
    
    engine = Qwen3VLEngine("path/to/Qwen3-VL-4B-mlc")
    
    # 简单对话
    response = engine.chat("path/to/image.jpg", "描述这张图片")
    print(response)
    
    # 或使用 OpenAI 风格 API
    response = engine.chat_completion(
        messages=[
            {"role": "user", "content": [
                {"type": "image", "image_url": "path/to/image.jpg"},
                {"type": "text", "text": "这是什么？"}
            ]}
        ]
    )
"""

import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Optional

import tvm
from tvm import relax
from tvm.runtime import ShapeTuple


class Qwen3VLEngine:
    """Qwen3-VL 多模态推理引擎 - 提供简洁的 API"""
    
    # 特殊 token IDs
    IMAGE_TOKEN_ID = 151655
    EOS_TOKENS = {151643, 151645}
    
    # 模型参数
    PATCH_SIZE = 16
    TEMPORAL_PATCH_SIZE = 2
    MERGE_SIZE = 2
    HEAD_DIM = 64
    NUM_GRID_PER_SIDE = 48
    
    # 图像处理参数
    MIN_PIXELS = 56 * 56
    MAX_PIXELS = 28 * 28 * 1280
    IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
    IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711])
    
    def __init__(
        self, 
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda:0",
    ):
        """
        初始化 Qwen3-VL 引擎
        
        Args:
            model_path: MLC 编译后的模型路径
            tokenizer_path: Tokenizer 路径，默认使用 model_path 同级的原始模型
            device: 设备，如 "cuda:0"
        """
        self.model_path = model_path
        self.device = tvm.device(device)
        
        # 加载模型
        self._load_model()
        
        # 加载 tokenizer
        tokenizer_path = tokenizer_path or self._find_tokenizer_path()
        self._load_tokenizer(tokenizer_path)
        
        print(f"Qwen3-VL Engine 已就绪")
    
    def _find_tokenizer_path(self) -> str:
        """自动查找 tokenizer 路径"""
        # 尝试常见路径
        candidates = [
            self.model_path.replace("-mlc", "-Instruct"),
            self.model_path.replace("-mlc", ""),
            os.path.join(os.path.dirname(self.model_path), "Qwen3-VL-4B-Instruct"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise ValueError(f"找不到 tokenizer，请手动指定 tokenizer_path")
    
    def _load_model(self):
        """加载 MLC 模型"""
        lib_path = os.path.join(self.model_path, "lib.so")
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"模型库不存在: {lib_path}")
        
        self.lib = tvm.runtime.load_module(lib_path)
        self.vm = relax.VirtualMachine(self.lib, self.device)
        
        # 加载元数据和参数
        cpu_vm = relax.VirtualMachine(self.lib, tvm.runtime.device("cpu"))
        self.metadata = json.loads(cpu_vm["_metadata"]())
        
        from tvm.contrib import tvmjs
        params, _ = tvmjs.load_tensor_cache(self.model_path, self.device)
        param_names = [p["name"] for p in self.metadata["params"]]
        self.params = [params[name] for name in param_names]
        
        # 加载配置
        config_path = os.path.join(self.model_path, "mlc-chat-config.json")
        with open(config_path) as f:
            self.config = json.load(f)
        
        # 获取函数
        self._embed = self.vm["embed"]
        self._image_embed = self.vm["image_embed"]
        self._prefill = self.vm["prefill"]
        self._decode = self.vm["decode"]
        self._create_kv_cache = self.vm["create_tir_paged_kv_cache"]
        
        # KV cache 管理
        self._add_seq = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self._begin_fwd = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self._end_fwd = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    
    def _load_tokenizer(self, tokenizer_path: str):
        """加载 tokenizer"""
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )
    
    # ==================== 图像处理 ====================
    
    def _preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]):
        """预处理图像"""
        # 加载图像
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        else:
            raise TypeError(f"不支持的图像类型: {type(image_input)}")
        
        width, height = image.size
        
        # 智能调整尺寸
        num_pixels = height * width
        if num_pixels < self.MIN_PIXELS:
            scale = np.sqrt(self.MIN_PIXELS / num_pixels)
            new_h, new_w = int(height * scale), int(width * scale)
        elif num_pixels > self.MAX_PIXELS:
            scale = np.sqrt(self.MAX_PIXELS / num_pixels)
            new_h, new_w = int(height * scale), int(width * scale)
        else:
            new_h, new_w = height, width
        
        # 对齐到 patch_size * merge_size
        factor = self.PATCH_SIZE * self.MERGE_SIZE
        new_h = max((new_h // factor) * factor, factor)
        new_w = max((new_w // factor) * factor, factor)
        
        image = image.resize((new_w, new_h), Image.BICUBIC)
        
        # 归一化
        img = np.array(image, dtype=np.float32) / 255.0
        img = (img - self.IMAGE_MEAN) / self.IMAGE_STD
        img = img.transpose(2, 0, 1)
        
        # 创建时间维度
        patches = np.stack([img] * self.TEMPORAL_PATCH_SIZE, axis=0)
        
        # 计算 grid
        grid_t = patches.shape[0] // self.TEMPORAL_PATCH_SIZE
        grid_h = new_h // self.PATCH_SIZE
        grid_w = new_w // self.PATCH_SIZE
        channel = patches.shape[1]
        
        # Reshape
        patches = patches.reshape(
            grid_t, self.TEMPORAL_PATCH_SIZE, channel,
            grid_h // self.MERGE_SIZE, self.MERGE_SIZE, self.PATCH_SIZE,
            grid_w // self.MERGE_SIZE, self.MERGE_SIZE, self.PATCH_SIZE,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        
        # Flatten
        flatten = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.TEMPORAL_PATCH_SIZE * self.PATCH_SIZE * self.PATCH_SIZE
        )
        
        return flatten.astype(np.float16), (grid_t, grid_h, grid_w)
    
    def _compute_2d_rope(self, h_patches: int, w_patches: int):
        """计算 2D 旋转位置编码"""
        half_dim = self.HEAD_DIM // 2
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, half_dim, 2, dtype=np.float32) / half_dim))
        
        max_hw = max(h_patches, w_patches)
        freq_table = np.outer(np.arange(max_hw, dtype=np.float32), inv_freq)
        
        merged_h = h_patches // self.MERGE_SIZE
        merged_w = w_patches // self.MERGE_SIZE
        
        row_idx = (np.arange(merged_h)[:, None, None, None] * self.MERGE_SIZE + 
                   np.arange(self.MERGE_SIZE)[None, None, :, None])
        col_idx = (np.arange(merged_w)[None, :, None, None] * self.MERGE_SIZE + 
                   np.arange(self.MERGE_SIZE)[None, None, None, :])
        
        row_idx = np.broadcast_to(row_idx, (merged_h, merged_w, self.MERGE_SIZE, self.MERGE_SIZE)).flatten()
        col_idx = np.broadcast_to(col_idx, (merged_h, merged_w, self.MERGE_SIZE, self.MERGE_SIZE)).flatten()
        
        emb = np.concatenate([freq_table[row_idx], freq_table[col_idx]], axis=-1)
        emb = np.concatenate([emb, emb], axis=-1)
        
        return np.cos(emb).astype(np.float16), np.sin(emb).astype(np.float16)
    
    def _compute_pos_ids(self, h_patches: int, w_patches: int):
        """计算位置 IDs"""
        merged_h = h_patches // self.MERGE_SIZE
        merged_w = w_patches // self.MERGE_SIZE
        
        h_idxs = np.linspace(0, self.NUM_GRID_PER_SIDE - 1, h_patches)
        w_idxs = np.linspace(0, self.NUM_GRID_PER_SIDE - 1, w_patches)
        
        h_grid, w_grid = np.meshgrid(np.floor(h_idxs).astype(np.int32), 
                                      np.floor(w_idxs).astype(np.int32), indexing='ij')
        pos_ids = h_grid * self.NUM_GRID_PER_SIDE + w_grid
        pos_ids = pos_ids.reshape(merged_h, self.MERGE_SIZE, merged_w, self.MERGE_SIZE)
        pos_ids = pos_ids.transpose(0, 2, 1, 3).flatten().astype(np.int32)
        
        return pos_ids
    
    def _embed_image(self, image_input):
        """获取图像嵌入"""
        pixel_values, (t, h, w) = self._preprocess_image(image_input)
        
        cos, sin = self._compute_2d_rope(h, w)
        pos_ids = self._compute_pos_ids(h, w)
        
        pixel_tensor = tvm.runtime.tensor(pixel_values, device=self.device)
        cos_tensor = tvm.runtime.tensor(cos, device=self.device)
        sin_tensor = tvm.runtime.tensor(sin, device=self.device)
        pos_tensor = tvm.runtime.tensor(pos_ids, device=self.device)
        
        embeds = self._image_embed(pixel_tensor, cos_tensor, sin_tensor, pos_tensor, self.params)
        num_tokens = t * (h // self.MERGE_SIZE) * (w // self.MERGE_SIZE)
        
        return embeds.numpy(), num_tokens
    
    def _embed_tokens(self, token_ids: List[int]):
        """获取文本嵌入"""
        tensor = tvm.runtime.tensor(np.array(token_ids, dtype=np.int32), device=self.device)
        return self._embed(tensor, self.params).numpy()
    
    # ==================== 生成 ====================
    
    def _generate(self, embeds: np.ndarray, max_tokens: int = 512) -> str:
        """生成文本"""
        # 创建 KV cache
        kv_cache = self._create_kv_cache(
            ShapeTuple([1]),
            ShapeTuple([min(self.config.get("context_window_size", 8192), 4096)]),
            ShapeTuple([self.config.get("prefill_chunk_size", 2048)]),
            ShapeTuple([16]),
            ShapeTuple([0]),
        )
        
        # Prefill
        seq_len = embeds.shape[0]
        combined = embeds.reshape(1, seq_len, -1).astype(np.float16)
        tensor = tvm.runtime.tensor(combined, device=self.device)
        
        self._add_seq(kv_cache, 0)
        self._begin_fwd(kv_cache, ShapeTuple([0]), ShapeTuple([seq_len]))
        logits, kv_cache = self._prefill(tensor, kv_cache, self.params)
        self._end_fwd(kv_cache)
        
        # Decode
        generated = []
        for _ in range(max_tokens):
            next_token = int(np.argmax(logits.numpy()[0, -1, :]))
            if next_token in self.EOS_TOKENS:
                break
            generated.append(next_token)
            
            next_emb = self._embed_tokens([next_token]).reshape(1, 1, -1).astype(np.float16)
            next_tensor = tvm.runtime.tensor(next_emb, device=self.device)
            
            self._begin_fwd(kv_cache, ShapeTuple([0]), ShapeTuple([1]))
            logits, kv_cache = self._decode(next_tensor, kv_cache, self.params)
            self._end_fwd(kv_cache)
        
        return self.tokenizer.decode(generated, skip_special_tokens=True)
    
    # ==================== 公开 API ====================
    
    def chat(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 512,
    ) -> str:
        """
        简单的图文对话接口
        
        Args:
            image: 图像路径、PIL Image 或 numpy 数组
            prompt: 用户问题
            system_prompt: 系统提示词
            max_tokens: 最大生成 token 数
        
        Returns:
            模型回复文本
        
        Example:
            >>> engine = Qwen3VLEngine("path/to/model")
            >>> response = engine.chat("cat.jpg", "这是什么动物？")
            >>> print(response)
        """
        # 获取图像嵌入
        image_embeds, num_image_tokens = self._embed_image(image)
        
        # 构建 prompt
        text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>"
            f"{'<|image_pad|>' * num_image_tokens}"
            f"<|vision_end|>\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        text_embeds = self._embed_tokens(token_ids)
        
        # 替换图像 token
        positions = [i for i, t in enumerate(token_ids) if t == self.IMAGE_TOKEN_ID]
        for i, pos in enumerate(positions[:image_embeds.shape[0]]):
            text_embeds[pos] = image_embeds[i]
        
        return self._generate(text_embeds, max_tokens)
    
    def chat_completion(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        stream: bool = False,
    ) -> Union[str, Dict]:
        """
        OpenAI 风格的聊天完成接口
        
        Args:
            messages: 消息列表，支持多模态内容
            max_tokens: 最大生成 token 数
            stream: 是否流式输出（暂不支持）
        
        Returns:
            模型回复
        
        Example:
            >>> response = engine.chat_completion([
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": [
            ...         {"type": "image", "image_url": "cat.jpg"},
            ...         {"type": "text", "text": "这是什么？"}
            ...     ]}
            ... ])
        """
        if stream:
            raise NotImplementedError("流式输出暂不支持")
        
        # 解析消息
        system_prompt = "You are a helpful assistant."
        image = None
        user_text = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content if isinstance(content, str) else ""
            
            elif role == "user":
                if isinstance(content, str):
                    user_text = content
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "image":
                            image = item.get("image_url") or item.get("image")
                        elif item.get("type") == "text":
                            user_text = item.get("text", "")
        
        if image is None:
            raise ValueError("未找到图像输入")
        
        response_text = self.chat(image, user_text, system_prompt, max_tokens)
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": -1,
                "total_tokens": -1
            }
        }
    
    def __call__(
        self,
        image: Union[str, Image.Image, np.ndarray],
        prompt: str,
        **kwargs
    ) -> str:
        """允许直接调用实例"""
        return self.chat(image, prompt, **kwargs)


# 便捷函数
def create_engine(model_path: str, **kwargs) -> Qwen3VLEngine:
    """创建 Qwen3-VL 引擎的便捷函数"""
    return Qwen3VLEngine(model_path, **kwargs)
