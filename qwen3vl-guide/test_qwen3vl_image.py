#!/usr/bin/env python3
"""
测试 Qwen3-VL 多模态推理
"""

import os
import sys

# 环境设置
os.environ['TVM_HOME'] = '/root/autodl-tmp/mlc-llm/build/tvm'
os.environ['TVM_LIBRARY_PATH'] = '/root/autodl-tmp/mlc-llm/build/tvm'
os.environ['LD_LIBRARY_PATH'] = '/root/autodl-tmp/mlc-llm/build/lib:/root/autodl-tmp/mlc-llm/build/tvm:' + os.environ.get('LD_LIBRARY_PATH', '')

sys.path.insert(0, '/root/autodl-tmp/mlc-llm/python')

# 导入引擎
from mlc_llm.model.qwen3_vl import Qwen3VLEngine

def main():
    model_path = "/root/autodl-tmp/Qwen3-VL-4B-mlc"
    image_path = "/root/OCR_BENCH.jpg"
    
    print("=" * 60)
    print("Qwen3-VL 多模态推理测试")
    print("=" * 60)
    print(f"模型路径: {model_path}")
    print(f"图片路径: {image_path}")
    print("=" * 60)
    
    # 创建引擎
    print("\n正在加载模型...")
    engine = Qwen3VLEngine(model_path)
    print("模型加载完成!")
    
    # 测试图片理解
    print("\n正在进行图片理解...")
    question = "请仔细观察这张图片，描述图片中的所有文字内容。"
    
    response = engine.chat(image_path, question)
    
    print("\n" + "=" * 60)
    print("问题:", question)
    print("=" * 60)
    print("模型回复:")
    print("=" * 60)
    print(response)
    print("=" * 60)

if __name__ == "__main__":
    main()
