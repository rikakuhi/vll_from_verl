# vLLM 与 mini-SWE-agent 集成

本目录包含将 mini-SWE-agent 与您的独立 vLLM 服务器连接的集成代码。

## 这是什么任务？

这是一个**持续测试任务**，专门设计用来帮助您调试在 VERL 中使用 Qwen3-Coder-30B-A3B-Instruct 模型时遇到的"全是感叹号"问题。

### 任务目的
- **复现问题**：向您的独立 vLLM 服务器发送重复的推理请求，查看是否会出现相同的"全是感叹号 + NaN logprobs"问题
- **隔离问题根源**：确定问题出在：
  - 您的 vLLM 服务器配置/模型本身（如果在这里能复现问题）
  - VERL 的权重同步或训练循环集成逻辑（如果在这里无法复现问题）
- **收集证据**：收集详细的日志和响应，以理解故障模式

### 任务工作流程
1. **启动您的 vLLM 服务器**：使用与 VERL 中完全相同的配置
2. **运行持续测试**：向您的模型发送代码生成提示
3. **自动检测问题**：识别只包含感叹号的响应
4. **保存完整结果**：用于分析和调试

## 任务轨迹保存在哪里？

### 连续测试脚本 (`continuous_task.py`)
- **默认保存位置**：`continuous_test_results/` 目录
- **文件格式**：JSON 文件
- **保存内容**：
  - `final_results.json`：最终完整结果
  - `results_batch_X.json`：每10次迭代的批次结果
- **自定义输出目录**：使用 `--output-dir` 参数指定

### mini-SWE-agent CLI
- **默认保存位置**：`~/.config/minisweagent/last_mini_run.traj.json`
- **自定义保存位置**：使用 `-o` 或 `--output` 参数指定
- **保存内容**：完整的任务轨迹，包括所有模型调用、环境交互和最终结果

### 轨迹文件内容结构
每个轨迹文件包含以下信息：
```json
{
  "iteration": 1,
  "task": "Write a Python function to implement binary search.",
  "success": true,
  "response": "def binary_search(arr, target):...",
  "error": "",
  "duration": 2.34,
  "has_exclamation_marks_only": false,
  "has_nan_logprobs": false
}
```

## 文件概览

- `vllm_integration.py`：自定义模型类，包装您的 vLLM 服务器的 OpenAI 兼容 API
- `vllm_config.yaml`：mini-SWE-agent 配置文件，用于连接您的 vLLM 服务器
- `continuous_task.py`：持续测试脚本，用于重现感叹号问题
- `test_tasks.json`：示例测试任务，专注于代码生成（适用于 Qwen3-Coder）

## 使用说明

### 1. 安装依赖

```bash
cd E:\code\vllm_from_verl
pip install -r requirements.txt
```

### 2. 启动您的 vLLM 服务器

```bash
# 在一个终端中
python launch_vllm.py Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --tensor-parallel-size <您的TP大小> \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.5 \
    --enable-chunked-prefill \
    --enable-prefix-caching
```

### 3. 运行持续测试

```bash
# 在另一个终端中
python integration/continuous_task.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --vllm-url "http://localhost:8000" \
    --iterations 100 \
    --delay 1.0 \
    --output-dir "test_results_qwen3"
```

### 4. 使用 mini-SWE-agent CLI

您也可以使用标准的 mini-SWE-agent CLI 配合自定义配置：

```bash
# 以开发模式安装 mini-swe-agent
pip install -e mini-swe-agent

# 使用自定义 vLLM 配置运行
mini -c integration/vllm_config.yaml -t "写一个Python函数实现二分查找。"
```

## 调试感叹号问题

`continuous_task.py` 脚本专门设计用来帮助您复现和调试在 VERL 中遇到的感叹号问题：

- **自动检测只包含感叹号的响应**
- **记录详细结果，包括响应内容、错误和计时**
- **运行多次迭代以增加复现间歇性问题的机会**
- **将结果保存到 JSON 文件供分析**

如果持续测试在您的独立 vLLM 服务器上发现了感叹号问题，这表明问题很可能在 vLLM 配置或模型本身，而不是 VERL 的集成逻辑。

如果在独立服务器上没有发现问题，那么问题很可能在 VERL 的权重同步或训练循环集成中。

## 配置选项

### VLLMModel 参数

- `model_name`：要使用的模型名称（必需）
- `vllm_server_url`：您的 vLLM 服务器 URL（默认：http://localhost:8000）
- `api_key`：API 密钥（本地 vLLM 通常不需要）
- `timeout`：请求超时时间（秒，默认：300）
- `max_retries`：最大重试次数（默认：3）

### 连续任务运行器选项

- `--model`：要测试的模型名称
- `--vllm-url`：vLLM 服务器 URL
- `--tasks-file`：包含自定义测试任务的 JSON 文件
- `--iterations`：要运行的测试迭代次数
- `--delay`：任务之间的延迟（秒）
- `--output-dir`：保存测试结果的目录

## 故障排除

### 常见问题

1. **连接被拒绝**：确保您的 vLLM 服务器正在运行且可访问
2. **模型未找到**：验证模型名称是否与 vLLM 服务器中加载的模型匹配
3. **超时错误**：增加超时参数或检查 GPU 内存使用情况

### 调试技巧

- 检查 vLLM 服务器日志中的错误消息
- 监控测试期间的 GPU 内存使用情况
- 如果内存不足，尝试使用更小的批处理大小或减少上下文长度
- 尝试不同的 temperature 和 top_p 值，查看问题是否与采样参数相关

## 许可证

此集成代码遵循与主仓库相同的 Apache License 2.0。