# Qwen3-VL MLC-LLM 完整迁移指南

> 本文档记录了在 CUDA 环境下从零开始编译 MLC-LLM 并部署 Qwen3-VL 多模态模型的完整流程。
> 测试环境：RTX 4090 (Compute Capability 8.9), CUDA 11.8, Ubuntu

---

## 目录

1. [环境要求](#1-环境要求)
2. [第一步：基础环境准备](#2-第一步基础环境准备)
3. [第二步：克隆代码仓库](#3-第二步克隆代码仓库)
4. [第三步：关键文件修改](#4-第三步关键文件修改)
5. [第四步：编译 MLC-LLM](#5-第四步编译-mlc-llm)
6. [第五步：安装 Python 包](#6-第五步安装-python-包)
7. [第六步：下载原始模型](#7-第六步下载原始模型)
8. [第七步：模型转换与编译](#8-第七步模型转换与编译)
9. [第八步：修改模型配置文件](#9-第八步修改模型配置文件)
10. [第九步：测试推理](#10-第九步测试推理)
11. [常见问题与解决方案](#11-常见问题与解决方案)

---

## 1. 环境要求

### 1.1 硬件要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA GPU (Compute Capability ≥ 7.0) | RTX 4090 (CC 8.9) |
| 显存 | 8GB | 24GB |
| 内存 | 32GB | 64GB |
| 磁盘 | 50GB 可用空间 | 100GB SSD |

### 1.2 软件版本要求

| 软件 | 最低版本 | 测试通过版本 |
|------|---------|-------------|
| Python | 3.10 | 3.11.14 |
| CMake | **3.24** | 4.2.1 |
| Rust | 1.70 | 1.93.0 |
| CUDA | 11.8 | 11.8 |
| LLVM | 10 | 10.0.0 |
| GCC | 9 | 系统默认 |

---

## 2. 第一步：基础环境准备

### 2.1 创建 Conda 环境

```bash
# 创建新的 conda 环境
conda create -n mlc python=3.11 -y
conda activate mlc
```

### 2.2 安装 CMake (≥ 3.24)

```bash
# 使用 pip 安装最新版 CMake
pip install cmake>=3.24

# 验证版本
cmake --version
```

### 2.3 安装/更新 Rust

```bash
# 安装 Rust (如未安装)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 更新到最新稳定版
rustup update stable

# 验证版本
rustc --version
```

### 2.4 安装 LLVM-10

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y llvm-10 llvm-10-dev

# 验证安装
llvm-config-10 --version
```

### 2.5 安装 Python 依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate safetensors pillow numpy
```

---

## 3. 第二步：克隆代码仓库

```bash
cd /root/autodl-tmp

# 克隆支持 Qwen3-VL 的 MLC-LLM 仓库
git clone --recursive git@github.com:Lingrongye/mlc-llm.git
cd mlc-llm

# 切换到 Qwen3-VL 支持分支
git checkout qwen3-vl-clean

# 初始化并更新所有子模块
git submodule update --init --recursive

# 验证子模块状态
git submodule status | head -5
```

---

## 4. 第三步：关键文件修改

### ⚠️ 重要：以下修改是编译成功的关键！

### 4.1 修改 `CMakeLists.txt`

**文件位置**: `/root/autodl-tmp/mlc-llm/CMakeLists.txt`

**修改内容**: 找到 `BUILD_DUMMY_LIBTVM` 设置，将 `ON` 改为 `OFF`

```cmake
# 修改前
set(BUILD_DUMMY_LIBTVM ON)

# 修改后 (启用完整 TVM 编译器功能)
set(BUILD_DUMMY_LIBTVM OFF)  # Changed from ON to enable full TVM compiler features
```

**为什么需要修改**: `BUILD_DUMMY_LIBTVM=ON` 只会构建 TVM 运行时版本，缺少编译器功能（如 `tvm.codegen.llvm.GetDefaultTargetTriple`），导致模型编译失败。

### 4.2 修改第三方库 CMake 版本要求

**文件 1**: `/root/autodl-tmp/mlc-llm/3rdparty/tokenizers-cpp/msgpack/CMakeLists.txt`

```cmake
# 修改前
CMAKE_MINIMUM_REQUIRED (VERSION 3.1 FATAL_ERROR)

# 修改后
CMAKE_MINIMUM_REQUIRED (VERSION 3.5 FATAL_ERROR)
```

**文件 2**: `/root/autodl-tmp/mlc-llm/3rdparty/tokenizers-cpp/sentencepiece/CMakeLists.txt`

```cmake
# 修改前
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# 修改后
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
```

### 4.3 创建编译配置文件

**文件位置**: `/root/autodl-tmp/mlc-llm/build/config.cmake`

**一条命令创建文件**（复制整块命令执行）:

```bash
mkdir -p /root/autodl-tmp/mlc-llm/build && cat > /root/autodl-tmp/mlc-llm/build/config.cmake << 'EOF'
set(USE_CUDA ON)
set(USE_CUTLASS OFF)
set(USE_CUBLAS ON)
set(USE_THRUST OFF)
set(USE_LLVM "/usr/lib/llvm-10/bin/llvm-config")
set(CMAKE_CUDA_ARCHITECTURES "89")
EOF
```

> ⚠️ **注意**: `CMAKE_CUDA_ARCHITECTURES` 的值 "89" 是针对 RTX 4090 的，请根据你的 GPU 修改（参见下方对照表）

### 4.4 配置说明

| 配置项 | 值 | 说明 |
|--------|------|------|
| `USE_CUDA` | ON | 启用 CUDA 支持 |
| `USE_CUTLASS` | **OFF** | 禁用 CUTLASS（编译时间过长/可能超时） |
| `USE_CUBLAS` | ON | 使用 cuBLAS 替代 CUTLASS |
| `USE_THRUST` | OFF | CUDA 11.8 下可能有兼容性问题 |
| `USE_LLVM` | 路径 | **必须启用**，否则缺少编译器函数 |
| `CMAKE_CUDA_ARCHITECTURES` | "89" | 根据你的 GPU 设置（见下表） |

### 4.5 GPU 架构参数对照表

| GPU 型号 | Compute Capability | CMAKE_CUDA_ARCHITECTURES |
|----------|-------------------|-------------------------|
| RTX 4090/4080/4070 | 8.9 | 89 |
| RTX 3090/3080/3070 | 8.6 | 86 |
| A100 | 8.0 | 80 |
| V100 | 7.0 | 70 |
| RTX 2080/2070 | 7.5 | 75 |

查看你的 GPU 架构：
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

---

## 5. 第四步：编译 MLC-LLM

```bash
cd /root/autodl-tmp/mlc-llm/build

# 配置 CMake
cmake ..

# 编译 (使用所有 CPU 核心)
make -j$(nproc)
```

**预计编译时间**: 30-60 分钟（取决于 CPU 性能）

**编译成功标志**: 无错误信息，生成 `libtvm.so` 等库文件

---

## 6. 第五步：安装 Python 包

```bash
# 安装 TVM FFI
cd /root/autodl-tmp/mlc-llm/3rdparty/tvm/3rdparty/ffi
pip install -e .

# 安装 TVM
cd /root/autodl-tmp/mlc-llm/3rdparty/tvm/python
pip install -e .

# 安装 MLC-LLM
cd /root/autodl-tmp/mlc-llm/python
pip install -e .

# 验证安装
python -c "import mlc_llm; print('MLC-LLM 安装成功')"
python -c "import tvm; print('TVM 版本:', tvm.__version__)"
```

**⚠️ 重要**: 如果之前安装过 `apache-tvm-ffi` 或 `tvm` 包，需要先卸载：
```bash
pip uninstall apache-tvm-ffi tvm -y
```

---

## 7. 第六步：下载原始模型

```bash
cd /root/autodl-tmp

# 使用 huggingface-cli 下载
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir ./Qwen3-VL-4B-Instruct

# 或使用 modelscope (国内更快)
pip install modelscope
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir ./Qwen3-VL-4B-Instruct
```

---

## 8. 第七步：模型转换与编译

### 8.1 设置环境变量

```bash
export TVM_LIBRARY_PATH=/root/autodl-tmp/mlc-llm/build/tvm
export LD_LIBRARY_PATH=/root/autodl-tmp/mlc-llm/build/lib:/root/autodl-tmp/mlc-llm/build/tvm:$LD_LIBRARY_PATH
```

### 8.2 转换模型权重

```bash
cd /root/autodl-tmp

python -m mlc_llm convert_weight \
    ./Qwen3-VL-4B-Instruct \
    --model-type qwen3_vl \
    --quantization q4f16_1 \
    -o ./Qwen3-VL-4B-mlc
```

### 8.3 生成配置文件

```bash
python -m mlc_llm gen_config \
    ./Qwen3-VL-4B-Instruct \
    --model-type qwen3_vl \
    --quantization q4f16_1 \
    --conv-template qwen3_vl \
    -o ./Qwen3-VL-4B-mlc
```

### 8.4 编译模型库

```bash
python -m mlc_llm compile \
    ./Qwen3-VL-4B-mlc \
    --model-type qwen3_vl \
    --quantization q4f16_1 \
    --device cuda \
    -o ./Qwen3-VL-4B-mlc/lib.so
```

**编译成功**: 生成 `lib.so` 文件（约 30MB）

---

## 9. 第八步：修改模型配置文件

### ⚠️ 关键步骤！

**文件位置**: `/root/autodl-tmp/Qwen3-VL-4B-mlc/mlc-chat-config.json`

需要在 `model_config` 中添加顶层参数，否则推理时会报 `vocab_size not found` 错误。

**添加以下字段到 `model_config` 的开头**:

```json
{
  "version": "0.1.0",
  "model_type": "qwen3_vl",
  "quantization": "q4f16_1",
  "model_config": {
    "vocab_size": 151936,
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "context_window_size": 262144,
    "prefill_chunk_size": 2048,
    "tensor_parallel_shards": 1,
    "text_config": {
      // ... 原有内容保持不变 ...
    },
    "vision_config": {
      // ... 原有内容保持不变 ...
    }
  }
}
```

**完整的 mlc-chat-config.json 示例**:

参见项目中的 `/root/autodl-tmp/Qwen3-VL-4B-mlc/mlc-chat-config.json`

---

## 10. 第九步：测试推理

### 10.1 创建测试脚本

**文件**: `/root/autodl-tmp/test_qwen3vl_image.py`

```python
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
    image_path = "/path/to/your/image.jpg"  # 替换为你的图片路径
    
    print("=" * 60)
    print("Qwen3-VL 多模态推理测试")
    print("=" * 60)
    
    # 创建引擎
    print("正在加载模型...")
    engine = Qwen3VLEngine(model_path)
    print("模型加载完成!")
    
    # 测试图片理解
    question = "请描述这张图片的内容。"
    response = engine.chat(image_path, question)
    
    print("\n问题:", question)
    print("回复:", response)

if __name__ == "__main__":
    main()
```

### 10.2 运行测试

```bash
cd /root/autodl-tmp
conda activate mlc
python test_qwen3vl_image.py
```

### 10.3 Python API 使用方式

```python
from mlc_llm.model.qwen3_vl import Qwen3VLEngine

# 创建引擎
engine = Qwen3VLEngine("/root/autodl-tmp/Qwen3-VL-4B-mlc")

# 方式 1: 最简单
response = engine.chat("image.jpg", "描述这张图片")

# 方式 2: 使用 PIL Image
from PIL import Image
img = Image.open("image.jpg")
response = engine.chat(img, "这张图片里有什么？")

# 方式 3: 使用 numpy 数组
import numpy as np
img_array = np.array(Image.open("image.jpg"))
response = engine.chat(img_array, "描述图片")
```

---

## 11. 常见问题与解决方案

### 问题 1: `error: identifier "__hfma2" is undefined`

**原因**: CUDA 架构设置不正确

**解决**: 检查 GPU 架构并正确设置 `CMAKE_CUDA_ARCHITECTURES`
```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

### 问题 2: `could not compile macro_rules_attribute`

**原因**: Rust 版本过低

**解决**: 
```bash
rustup update stable
rustc --version  # 确保 >= 1.70
```

### 问题 3: `CMake 3.18 or higher is required`

**原因**: CMake 版本过低

**解决**: 
```bash
pip install cmake>=3.24
```

### 问题 4: `ValueError: Cannot find object type index for script.PrinterConfig`

**原因**: TVM Python 包冲突

**解决**: 
```bash
pip uninstall apache-tvm-ffi tvm -y
cd /root/autodl-tmp/mlc-llm/3rdparty/tvm/3rdparty/ffi && pip install -e .
cd /root/autodl-tmp/mlc-llm/3rdparty/tvm/python && pip install -e .
```

### 问题 5: `nvcc: Terminated` (编译超时)

**原因**: CUTLASS 编译时间过长

**解决**: 在 `config.cmake` 中禁用 CUTLASS
```cmake
set(USE_CUTLASS OFF)
set(USE_CUBLAS ON)
```

### 问题 6: `ValueError: Unknown model type: qwen3_vl`

**原因**: 当前分支不支持 Qwen3-VL

**解决**: 切换到正确的分支
```bash
git checkout qwen3-vl-clean
git pull
```

### 问题 7: `Cannot find global function tvm.codegen.llvm.GetDefaultTargetTriple`

**原因**: 未启用 LLVM 或 `BUILD_DUMMY_LIBTVM=ON`

**解决**: 
1. 修改 `CMakeLists.txt` 中 `BUILD_DUMMY_LIBTVM` 为 `OFF`
2. 在 `config.cmake` 中启用 LLVM:
```cmake
set(USE_LLVM "/usr/lib/llvm-10/bin/llvm-config")
```
3. 重新编译

### 问题 8: `key 'vocab_size' not found in the JSON object`

**原因**: `mlc-chat-config.json` 缺少顶层模型参数

**解决**: 参见第 9 步，在 `model_config` 中添加必要的顶层参数

### 问题 9: `fatal error: tvm/target/target.h: No such file or directory`

**原因**: 子模块未正确初始化或分支缺少文件

**解决**: 
```bash
git submodule update --init --recursive
git pull  # 拉取最新更新
```

---

## 目录结构参考

```
/root/autodl-tmp/
├── mlc-llm/                          # MLC-LLM 源代码
│   ├── build/                        # 编译输出目录
│   │   ├── config.cmake              # 编译配置文件 ⚠️ 需创建
│   │   ├── lib/
│   │   └── tvm/
│   ├── python/                       # Python 包
│   ├── 3rdparty/
│   │   ├── tvm/                      # TVM 子模块
│   │   └── tokenizers-cpp/
│   │       ├── msgpack/CMakeLists.txt    # ⚠️ 需修改
│   │       └── sentencepiece/CMakeLists.txt  # ⚠️ 需修改
│   └── CMakeLists.txt                # ⚠️ 需修改 BUILD_DUMMY_LIBTVM
│
├── Qwen3-VL-4B-Instruct/             # 原始模型
│   ├── config.json
│   ├── model-*.safetensors
│   └── tokenizer.json
│
└── Qwen3-VL-4B-mlc/                  # 转换后的模型
    ├── lib.so                        # 编译后的模型库
    ├── mlc-chat-config.json          # ⚠️ 需修改
    ├── params_shard_*.bin            # 量化后的权重
    ├── tokenizer.json
    └── vocab.json
```

---

## 快速检查清单

在开始编译前，确保完成以下检查：

- [ ] CMake 版本 ≥ 3.24
- [ ] Rust 版本 ≥ 1.70
- [ ] LLVM-10 已安装
- [ ] `CMakeLists.txt` 中 `BUILD_DUMMY_LIBTVM` 设为 `OFF`
- [ ] `config.cmake` 已创建并配置正确的 GPU 架构
- [ ] 第三方库的 CMake 版本要求已修改 (3.1 → 3.5)

编译完成后，确保完成以下检查：

- [ ] TVM FFI 已安装 (`pip install -e .`)
- [ ] TVM 已安装 (`pip install -e .`)
- [ ] MLC-LLM 已安装 (`pip install -e .`)
- [ ] 模型权重已转换
- [ ] 模型配置已生成
- [ ] 模型库 `lib.so` 已编译
- [ ] `mlc-chat-config.json` 已添加顶层参数

---

## 联系与反馈

如果遇到问题，请检查：
1. 错误日志的完整输出
2. 当前的环境版本 (`cmake --version`, `rustc --version`, `nvcc --version`)
3. GPU 架构是否正确配置

---

*文档版本: 1.0*  
*最后更新: 2026-02-03*  
*测试环境: RTX 4090, CUDA 11.8, Ubuntu, Python 3.11*
