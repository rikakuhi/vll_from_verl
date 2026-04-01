# 来自 VERL 的 vLLM 服务器

这是一个从 [VERL](https://github.com/volcengine/verl) 代码库中提取的独立 vLLM 服务器实现。它完全遵循 VERL 中 vLLM 集成的相同方法和配置。

## 功能特性

- **完全兼容 VERL**：使用与 VERL 相同的配置参数、参数解析和服务器初始化方式
- **完整的 vLLM 功能支持**：支持所有 vLLM 功能，包括：
  - 张量并行（Tensor parallelism）
  - LoRA 微调
  - 量化（FP8, torchao）
  - MoE 模型的专家并行（Expert parallel）
  - 前缀缓存（Prefix caching）和分块预填充（Chunked prefill）
  - Prometheus 监控指标
- **独立运行**：不依赖 VERL 或 Ray（可选分布式功能除外）
- **OpenAI API 兼容**：提供标准的 vLLM OpenAI 兼容 API 端点

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 注意：vLLM 安装可能需要额外的系统依赖
# 请参考：https://docs.vllm.ai/en/latest/getting_started/installation.html
```

## 使用方法

> **重要提示**：为了准确复现 VERL 中的 vLLM 启动行为并调试问题，**强烈推荐使用命令行方式（`python launch_vllm.py`）**，因为这种方式与 VERL 原生启动 vLLM 服务的方法完全一致，包括相同的参数解析、CLI 调用和服务器初始化流程。

### 基本用法（推荐 ✅）

```bash
python launch_vllm.py <模型路径或HuggingFace模型ID>
```

示例：
```bash
python launch_vllm.py meta-llama/Llama-3.1-8B-Instruct
```

### 高级配置（推荐 ✅）

服务器支持 VERL `rollout.yaml` 中的所有配置选项：

```bash
python launch_vllm.py meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --enable-lora
```

### 编程式使用（不推荐用于调试 ❌）

您也可以通过编程方式使用服务器，但**这种方式与 VERL 的启动流程不完全一致**，因为绕过了 vLLM 的官方 CLI 参数解析：

```python
from vllm_server import StandaloneVLLMServer, VLLMServerConfig

# 创建配置
config = VLLMServerConfig(
    dtype="bfloat16",
    tensor_model_parallel_size=2,
    max_model_len=8192,
    gpu_memory_utilization=0.8,
    enable_lora=True,
    lora_rank=16
)

# 启动服务器
server = await launch_vllm_server("meta-llama/Llama-3.1-8B-Instruct", config)
```

## 配置选项

所有配置选项都与 VERL 的 `RolloutConfig` 保持一致：

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `dtype` | `bfloat16` | 模型数据类型 |
| `tensor_model_parallel_size` | `1` | 张量并行大小 |
| `max_model_len` | `None` | 最大上下文长度 |
| `gpu_memory_utilization` | `0.5` | GPU 内存用于 KV 缓存的比例 |
| `enable_lora` | `False` | 启用 LoRA 支持 |
| `lora_rank` | `0` | LoRA 秩（如果 enable_lora=True，则必须 > 0） |
| `quantization` | `None` | 量化方法（`fp8`, `torchao`） |
| `max_num_seqs` | `1024` | 最大序列数 |
| `max_num_batched_tokens` | `8192` | 最大批处理 token 数 |
| `enable_chunked_prefill` | `True` | 启用分块预填充 |
| `enable_prefix_caching` | `True` | 启用前缀缓存 |

## 实现细节

此实现在 VERL 代码库中提取了核心的 vLLM 服务器逻辑：

1. **参数构建**：完全按照 VERL 的方式构建 vLLM CLI 参数
2. **配置解析**：使用 VERL 的配置结构和默认值
3. **服务器初始化**：遵循 VERL 的异步服务器初始化模式
4. **LoRA 集成**：使用 VERL 的魔术常量（`VLLM_LORA_INT_ID=123`）
5. **量化支持**：包含 FP8 和 torchao 量化处理
6. **MoE 支持**：处理专家并行和路由回放功能

## 与原始 VERL 的差异

- **无 Ray 依赖**：移除了用于独立运行的 Ray actor 包装
- **简化的网络**：默认使用 localhost 而不是多节点设置
- **降低复杂性**：移除了混合训练/rollout 模式切换
- **独立专注**：针对纯推理场景进行了优化

## 测试

要测试服务器：

```bash
# 启动服务器
python launch_vllm.py meta-llama/Llama-3.1-8B-Instruct

# 测试 curl 请求（服务器运行在端口 8000 上时）
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "Hello, how are you?",
        "max_tokens": 50
    }'
```

## 许可证

此代码源自 VERL（Apache License 2.0）。完整的许可证详情请参阅原始的 [VERL 仓库](https://github.com/volcengine/verl)。