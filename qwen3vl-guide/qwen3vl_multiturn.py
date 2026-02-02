#!/usr/bin/env python3
"""
Qwen3-VL 多轮对话引擎
支持：图片+文本混合多轮对话，KV Cache 复用
"""

import os
import sys
import numpy as np
from PIL import Image
from typing import List, Dict, Union, Optional

# 环境设置
os.environ['TVM_HOME'] = '/root/autodl-tmp/mlc-llm/build/tvm'
os.environ['TVM_LIBRARY_PATH'] = '/root/autodl-tmp/mlc-llm/build/tvm'
os.environ['LD_LIBRARY_PATH'] = '/root/autodl-tmp/mlc-llm/build/lib:/root/autodl-tmp/mlc-llm/build/tvm:' + os.environ.get('LD_LIBRARY_PATH', '')

sys.path.insert(0, '/root/autodl-tmp/mlc-llm/python')

import tvm
from tvm import relax
from tvm.runtime import ShapeTuple
import json


class Qwen3VLMultiturnEngine:
    """支持多轮对话的 Qwen3-VL 引擎"""
    
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
    
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = tvm.device(device)
        
        # 加载模型
        self._load_model()
        
        # 加载 tokenizer
        tokenizer_path = tokenizer_path or self._find_tokenizer_path()
        self._load_tokenizer(tokenizer_path)
        
        # 多轮对话状态
        self._kv_cache = None
        self._seq_len = 0
        self._conversation = []  # 对话历史
        self._has_image = False  # 当前会话是否包含图片
        
        print("Qwen3-VL 多轮对话引擎已就绪")
    
    def _find_tokenizer_path(self) -> str:
        candidates = [
            self.model_path.replace("-mlc", "-Instruct"),
            self.model_path.replace("-mlc", ""),
            os.path.join(os.path.dirname(self.model_path), "Qwen3-VL-4B-Instruct"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise ValueError("找不到 tokenizer")
    
    def _load_model(self):
        lib_path = os.path.join(self.model_path, "lib.so")
        self.lib = tvm.runtime.load_module(lib_path)
        self.vm = relax.VirtualMachine(self.lib, self.device)
        
        cpu_vm = relax.VirtualMachine(self.lib, tvm.runtime.device("cpu"))
        self.metadata = json.loads(cpu_vm["_metadata"]())
        
        from tvm.contrib import tvmjs
        params, _ = tvmjs.load_tensor_cache(self.model_path, self.device)
        param_names = [p["name"] for p in self.metadata["params"]]
        self.params = [params[name] for name in param_names]
        
        config_path = os.path.join(self.model_path, "mlc-chat-config.json")
        with open(config_path) as f:
            self.config = json.load(f)
        
        self._embed = self.vm["embed"]
        self._image_embed = self.vm["image_embed"]
        self._prefill = self.vm["prefill"]
        self._decode = self.vm["decode"]
        self._create_kv_cache = self.vm["create_tir_paged_kv_cache"]
        
        self._add_seq = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
        self._begin_fwd = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
        self._end_fwd = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    
    def _load_tokenizer(self, tokenizer_path: str):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    def _preprocess_image(self, image_input):
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert('RGB')
        else:
            raise TypeError(f"不支持的图像类型: {type(image_input)}")
        
        width, height = image.size
        num_pixels = height * width
        
        if num_pixels < self.MIN_PIXELS:
            scale = np.sqrt(self.MIN_PIXELS / num_pixels)
            new_h, new_w = int(height * scale), int(width * scale)
        elif num_pixels > self.MAX_PIXELS:
            scale = np.sqrt(self.MAX_PIXELS / num_pixels)
            new_h, new_w = int(height * scale), int(width * scale)
        else:
            new_h, new_w = height, width
        
        factor = self.PATCH_SIZE * self.MERGE_SIZE
        new_h = max((new_h // factor) * factor, factor)
        new_w = max((new_w // factor) * factor, factor)
        
        image = image.resize((new_w, new_h), Image.BICUBIC)
        
        img = np.array(image, dtype=np.float32) / 255.0
        img = (img - self.IMAGE_MEAN) / self.IMAGE_STD
        img = img.transpose(2, 0, 1)
        
        patches = np.stack([img] * self.TEMPORAL_PATCH_SIZE, axis=0)
        
        grid_t = patches.shape[0] // self.TEMPORAL_PATCH_SIZE
        grid_h = new_h // self.PATCH_SIZE
        grid_w = new_w // self.PATCH_SIZE
        channel = patches.shape[1]
        
        patches = patches.reshape(
            grid_t, self.TEMPORAL_PATCH_SIZE, channel,
            grid_h // self.MERGE_SIZE, self.MERGE_SIZE, self.PATCH_SIZE,
            grid_w // self.MERGE_SIZE, self.MERGE_SIZE, self.PATCH_SIZE,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        
        flatten = patches.reshape(
            grid_t * grid_h * grid_w,
            channel * self.TEMPORAL_PATCH_SIZE * self.PATCH_SIZE * self.PATCH_SIZE
        )
        
        return flatten.astype(np.float16), (grid_t, grid_h, grid_w)
    
    def _compute_2d_rope(self, h_patches: int, w_patches: int):
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
        tensor = tvm.runtime.tensor(np.array(token_ids, dtype=np.int32), device=self.device)
        return self._embed(tensor, self.params).numpy()
    
    def _ensure_kv_cache(self):
        """确保 KV cache 存在"""
        if self._kv_cache is None:
            self._kv_cache = self._create_kv_cache(
                ShapeTuple([1]),
                ShapeTuple([min(self.config.get("context_window_size", 8192), 4096)]),
                ShapeTuple([self.config.get("prefill_chunk_size", 2048)]),
                ShapeTuple([16]),
                ShapeTuple([0]),
            )
            self._add_seq(self._kv_cache, 0)
            self._seq_len = 0
    
    def _generate_continue(self, embeds: np.ndarray, max_tokens: int = 512) -> str:
        """在现有 KV cache 基础上继续生成"""
        self._ensure_kv_cache()
        
        seq_len = embeds.shape[0]
        combined = embeds.reshape(1, seq_len, -1).astype(np.float16)
        tensor = tvm.runtime.tensor(combined, device=self.device)
        
        self._begin_fwd(self._kv_cache, ShapeTuple([0]), ShapeTuple([seq_len]))
        logits, self._kv_cache = self._prefill(tensor, self._kv_cache, self.params)
        self._end_fwd(self._kv_cache)
        
        self._seq_len += seq_len
        
        # Decode
        generated = []
        for _ in range(max_tokens):
            next_token = int(np.argmax(logits.numpy()[0, -1, :]))
            if next_token in self.EOS_TOKENS:
                break
            generated.append(next_token)
            
            next_emb = self._embed_tokens([next_token]).reshape(1, 1, -1).astype(np.float16)
            next_tensor = tvm.runtime.tensor(next_emb, device=self.device)
            
            self._begin_fwd(self._kv_cache, ShapeTuple([0]), ShapeTuple([1]))
            logits, self._kv_cache = self._decode(next_tensor, self._kv_cache, self.params)
            self._end_fwd(self._kv_cache)
            self._seq_len += 1
        
        return self.tokenizer.decode(generated, skip_special_tokens=True)
    
    def reset(self):
        """重置对话，清空历史和 KV cache"""
        self._kv_cache = None
        self._seq_len = 0
        self._conversation = []
        self._has_image = False
        print("✅ 对话已重置")
    
    def chat(
        self,
        message: str,
        image: Optional[Union[str, Image.Image, np.ndarray]] = None,
        max_tokens: int = 512,
        system_prompt: str = "You are a helpful assistant."
    ) -> str:
        """
        多轮对话接口
        
        Args:
            message: 用户消息
            image: 可选的图片（仅在对话开始时需要）
            max_tokens: 最大生成 token 数
            system_prompt: 系统提示词（仅首轮生效）
        
        Returns:
            模型回复
        """
        is_first_turn = len(self._conversation) == 0
        
        if is_first_turn:
            # 首轮：构建完整的开始 prompt
            if image is not None:
                # 带图片的首轮
                image_embeds, num_image_tokens = self._embed_image(image)
                self._has_image = True
                
                text = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n<|vision_start|>"
                    f"{'<|image_pad|>' * num_image_tokens}"
                    f"<|vision_end|>\n{message}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                text_embeds = self._embed_tokens(token_ids)
                
                # 替换图像 token
                positions = [i for i, t in enumerate(token_ids) if t == self.IMAGE_TOKEN_ID]
                for i, pos in enumerate(positions[:image_embeds.shape[0]]):
                    text_embeds[pos] = image_embeds[i]
                
                embeds = text_embeds
            else:
                # 纯文本首轮
                text = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\n{message}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                embeds = self._embed_tokens(token_ids)
        else:
            # 后续轮次：只需要添加用户消息和助手开始标记
            # 先添加上一轮助手回复的结束标记
            prev_reply = self._conversation[-1]["assistant"]
            end_text = f"{prev_reply}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
            token_ids = self.tokenizer.encode(end_text, add_special_tokens=False)
            embeds = self._embed_tokens(token_ids)
        
        # 生成回复
        response = self._generate_continue(embeds, max_tokens)
        
        # 记录对话
        self._conversation.append({
            "user": message,
            "assistant": response,
            "has_image": image is not None
        })
        
        return response
    
    def get_history(self) -> List[Dict]:
        """获取对话历史"""
        return self._conversation.copy()


class ChatDemo:
    """交互式对话演示"""
    
    def __init__(self, model_path: str):
        print("=" * 60)
        print("🤖 Qwen3-VL 多轮对话 Demo (KV Cache 复用版)")
        print("=" * 60)
        
        self.engine = Qwen3VLMultiturnEngine(model_path)
        self.current_image = None
        self.print_help()
    
    def print_help(self):
        print("\n" + "=" * 60)
        print("📖 使用说明:")
        print("-" * 60)
        print("  直接输入文字    - 对话（首轮可带图片）")
        print("  /image <路径>   - 设置图片（下次对话使用）")
        print("  /reset          - 重置对话（清空上下文）")
        print("  /history        - 查看对话历史")
        print("  /help           - 显示帮助")
        print("  /quit           - 退出")
        print("-" * 60)
        print("💡 图片只需在首轮设置，后续可继续对图片提问！")
        print("=" * 60 + "\n")
    
    def run(self):
        while True:
            try:
                # 提示符
                if self.current_image and len(self.engine._conversation) == 0:
                    prompt = f"📷 [{os.path.basename(self.current_image)}] 你: "
                else:
                    prompt = "你: "
                
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else ""
                    
                    if cmd in ["/quit", "/exit", "/q"]:
                        print("👋 再见！")
                        break
                    elif cmd == "/help":
                        self.print_help()
                    elif cmd == "/reset":
                        self.engine.reset()
                        self.current_image = None
                    elif cmd == "/history":
                        history = self.engine.get_history()
                        if not history:
                            print("📝 对话历史为空")
                        else:
                            print(f"\n📝 对话历史 ({len(history)} 轮):")
                            for i, turn in enumerate(history):
                                img_mark = "📷 " if turn.get("has_image") else ""
                                user_msg = turn["user"][:40] + "..." if len(turn["user"]) > 40 else turn["user"]
                                asst_msg = turn["assistant"][:40] + "..." if len(turn["assistant"]) > 40 else turn["assistant"]
                                print(f"  {i+1}. 👤 {img_mark}{user_msg}")
                                print(f"     🤖 {asst_msg}")
                    elif cmd == "/image":
                        if arg and os.path.exists(arg):
                            if len(self.engine._conversation) > 0:
                                print("⚠️ 图片只能在对话开始时设置，请先 /reset")
                            else:
                                self.current_image = arg
                                print(f"✅ 图片已设置: {arg}")
                                print("💡 现在输入问题开始对话")
                        else:
                            print(f"❌ 图片不存在: {arg}")
                    else:
                        print(f"❌ 未知命令，输入 /help 查看帮助")
                else:
                    # 对话
                    print("\n🤖 助手: ", end="", flush=True)
                    
                    # 首轮带图片
                    if len(self.engine._conversation) == 0 and self.current_image:
                        response = self.engine.chat(user_input, image=self.current_image)
                    else:
                        response = self.engine.chat(user_input)
                    
                    print(response + "\n")
                    
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break


def main():
    model_path = "/root/autodl-tmp/Qwen3-VL-4B-mlc"
    demo = ChatDemo(model_path)
    demo.run()


if __name__ == "__main__":
    main()
