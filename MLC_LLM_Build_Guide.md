# MLC LLM 从源码构建指南（qwen3vl 分支）

## 环境信息

- **系统**: Linux (AutoDL)
- **GPU**: NVIDIA GeForce RTX 4090 (compute capability 8.9)
- **CUDA**: 11.8
- **Conda 环境**: mlc

---

## 一、构建步骤

### Step 1: 开启代理并激活环境

```bash
clashon
conda activate mlc
```

### Step 2: 安装构建依赖

```bash
conda install -c conda-forge "cmake>=3.24" rust git -y
```

### Step 3: 克隆仓库并切换到 qwen3vl 分支

```bash
cd /root/autodl-tmp
git clone --recursive https://github.com/mlc-ai/mlc-llm.git
cd mlc-llm
git checkout qwen3vl
git submodule update --init --recursive
```

### Step 4: 配置构建（交互式）

```bash
mkdir -p build && cd build
python ../cmake/gen_cmake_config.py
```

**配置选项参考：**

| 选项 | 推荐值 | 说明 |
|------|--------|------|
| TVM_SOURCE_DIR | 直接回车 | 使用默认 3rdparty/tvm |
| USE_CUDA | **y** | NVIDIA GPU 必须开启 |
| USE_CUTLASS | **y** | CUDA 优化库 |
| USE_CUBLAS | **y** | 矩阵运算加速 |
| USE_ROCM | **n** | AMD GPU 专用，关闭 |
| USE_OPENCL | **n** | 不需要 |
| USE_VULKAN | **n** | 不需要 |
| USE_METAL | **n** | Mac 专用，关闭 |

### Step 5: 修复配置问题（重要！）

配置完成后，需要手动修改 `config.cmake`：

```bash
# 1. 禁用 THRUST（与 CUDA 11.8 不兼容）
sed -i 's/set(USE_THRUST ON)/set(USE_THRUST OFF)/' config.cmake

# 2. 设置正确的 CUDA 架构（RTX 4090 = 89）
echo 'set(CMAKE_CUDA_ARCHITECTURES "89")' >> config.cmake
```

**常见 GPU 架构对照：**

| GPU | 架构 |
|-----|------|
| RTX 4090/4080/4070 | 89 |
| RTX 3090/3080/3070 | 86 |
| RTX 2080/2070 | 75 |
| V100 | 70 |
| A100 | 80 |

### Step 6: 编译

```bash
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j $(nproc)
```

### Step 7: 安装 Python 包

```bash
cd /root/autodl-tmp/mlc-llm/python
pip install -e .
```

### Step 8: 验证安装

```bash
python -c "import mlc_llm; print(mlc_llm)"
```

---

## 二、遇到的问题及解决方案

### 问题 1: CMake 版本兼容性错误

**错误信息：**
```
CMake Error at 3rdparty/tokenizers-cpp/msgpack/CMakeLists.txt:1
Compatibility with CMake < 3.5 has been removed from CMake.
```

**原因：** 新版 CMake (4.x) 不再兼容旧的 CMakeLists.txt 文件

**解决方案：** 添加参数绕过兼容性检查
```bash
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

---

### 问题 2: Thrust 编译错误

**错误信息：**
```
error: no instance of overloaded function "thrust::sort_by_key" matches the argument list
```

**原因：** CUDA 11.8 的 Thrust 库与 TVM 代码中的 API 调用不兼容

**解决方案：** 在 `config.cmake` 中禁用 Thrust
```bash
sed -i 's/set(USE_THRUST ON)/set(USE_THRUST OFF)/' config.cmake
```

---

### 问题 3: CUDA 架构错误（__hfma2 未定义）

**错误信息：**
```
error: identifier "__hfma2" is undefined
```

**原因：** CMake 自动检测的 CUDA 架构为 sm_52（Maxwell），但 `__hfma2` 是半精度浮点运算指令，需要 sm_53 及以上架构才支持

**解决方案：** 手动指定正确的 GPU 架构
```bash
# RTX 4090 对应架构 89
echo 'set(CMAKE_CUDA_ARCHITECTURES "89")' >> config.cmake
```

---

### 问题 4: CMakeCache.txt 损坏

**错误信息：**
```
CMakeCache.txt:17: *** missing separator. Stop.
```

**原因：** CMake 缓存文件损坏

**解决方案：** 清理缓存后重新配置
```bash
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
```

---

## 三、完整的一键构建脚本

将以下内容保存为 `build_mlc.sh`：

```bash
#!/bin/bash
set -e

# 开启代理
clashon

# 激活环境
conda activate mlc

# 进入构建目录
cd /root/autodl-tmp/mlc-llm/build

# 修复配置
sed -i 's/set(USE_THRUST ON)/set(USE_THRUST OFF)/' config.cmake
grep -q "CMAKE_CUDA_ARCHITECTURES" config.cmake || echo 'set(CMAKE_CUDA_ARCHITECTURES "89")' >> config.cmake

# 清理并构建
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j $(nproc)

# 安装 Python 包
cd ../python
pip install -e .

# 验证安装
echo "验证安装..."
python -c "import mlc_llm; print('✅ MLC LLM 安装成功:', mlc_llm)"

echo "🎉 构建完成！"
```

运行脚本：
```bash
chmod +x build_mlc.sh
./build_mlc.sh
```

---

## 四、目录结构

```
/root/autodl-tmp/
├── mlc-llm/                    # MLC LLM 源码
│   ├── 3rdparty/               # 第三方依赖
│   │   ├── tvm/                # TVM 编译器
│   │   ├── tokenizers-cpp/     # Tokenizer
│   │   └── ...
│   ├── build/                  # 构建目录
│   │   ├── config.cmake        # 构建配置
│   │   ├── libmlc_llm.so       # 编译产物
│   │   └── libtvm_runtime.so   # TVM 运行时
│   ├── python/                 # Python 包
│   └── ...
└── Qwen3-VL-4B-Instruct/       # 模型权重（如果下载了）
```

---

## 五、参考链接

- [MLC LLM 官方文档](https://llm.mlc.ai/docs/install/mlc_llm.html)
- [MLC LLM GitHub](https://github.com/mlc-ai/mlc-llm)
- [TVM 官方文档](https://tvm.apache.org/docs/)

---

## 六、常见问题 FAQ

**Q: 编译需要多长时间？**
A: 取决于 CPU 核心数，通常 10-30 分钟。

**Q: 可以更换 TVM_SOURCE_DIR 吗？**
A: 可以，修改 `config.cmake` 中的 `TVM_SOURCE_DIR` 值，然后重新编译。

**Q: 如何更新到最新代码？**
A: 
```bash
cd /root/autodl-tmp/mlc-llm
git pull
git submodule update --init --recursive
# 然后重新编译
```

**Q: 编译后的库文件在哪里？**
A: 在 `build/` 目录下，主要是 `libmlc_llm.so` 和 `libtvm_runtime.so`
