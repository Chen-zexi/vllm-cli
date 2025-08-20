# Built-in Profiles Guide

Seven carefully designed profiles cover most common use cases and hardware configurations. All profiles include multi-GPU detection that automatically sets tensor parallelism to utilize all available GPUs.

## General Purpose Profiles

### `standard` - Minimal configuration with smart defaults
Uses vLLM's defaults configuration. Perfect for most models and hardware setups.

**Use Case:** Starting point for any model, general inference tasks
**Configuration:** No additional arguments - uses vLLM defaults
**Environment Variables:** None

### `moe_optimized` - Optimized for Mixture of Experts models
Enables expert parallelism for MoE models with optimized environment variables.

**Use Case:** Qwen-MoE, Mixtral, DeepSeek-MoE, and other MoE architectures
**Configuration:**
- `--enable-expert-parallel`

**Environment Variables:**
- `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1`

### `high_throughput` - Maximum performance configuration
Aggressive settings for maximum request throughput with Triton flash attention.

**Use Case:** High-traffic scenarios
**Configuration:**
- `--max-model-len 8192`
- `--gpu-memory-utilization 0.95`
- `--enable-chunked-prefill`
- `--max-num-batched-tokens 8192`
- `--trust-remote-code`
- `--enable-prefix-caching`

**Environment Variables:**
- `VLLM_USE_TRITON_FLASH_ATTN=1`

### `low_memory` - Memory-constrained environments
Reduces memory usage through FP8 quantization and conservative settings.

**Use Case:** Limited GPU memory, running larger models on smaller GPUs
**Configuration:**
- `--max-model-len 4096`
- `--gpu-memory-utilization 0.70`
- `--trust-remote-code`
- `--quantization fp8`

**Environment Variables:**
- `VLLM_CPU_KVCACHE_SPACE=4`

## Hardware-Specific Profiles for GPT-OSS Models

> **Note:** For the latest GPU-specific optimizations and model requirements, please refer to the [vLLM GPT recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html). The built-in profiles provide a starting point that can be customized based on your specific hardware and model requirements.

### `gpt_oss_ampere` - GPT-OSS on NVIDIA A100 (Ampere)

Optimized for GPT-OSS models on A100 GPUs.

**Target Hardware:** NVIDIA A100 40GB/80GB
**Optimal Models:** GPT-OSS models up to 70B parameters
**Configuration:**
- `--enable-expert-parallel`
- `--trust-remote-code`
- `--gpu-memory-utilization 0.90`
- `--enable-chunked-prefill`
- `--enable-prefix-caching`
- `--async-scheduling`

**Environment Variables:**
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1`
- `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1`

### `gpt_oss_hopper` - GPT-OSS on NVIDIA H100/H200 (Hopper)

Optimized for GPT-OSS models on H100/H200 GPUs.

**Target Hardware:** NVIDIA H100 80GB, H200
**Optimal Models:** GPT-OSS models including 175B+ parameters
**Configuration:**
- `--enable-expert-parallel`
- `--trust-remote-code`
- `--gpu-memory-utilization 0.90`
- `--enable-chunked-prefill`
- `--enable-prefix-caching`
- `--async-scheduling`

**Environment Variables:**
- `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1`

### `gpt_oss_blackwell` - GPT-OSS on NVIDIA Blackwell (B100/B200)

Optimized for GPT-OSS models on latest Blackwell GPUs.

**Target Hardware:** NVIDIA B100, B200 (Future GPUs)
**Optimal Models:** Next-generation models with BF16 precision
**Configuration:**
- `--enable-expert-parallel`
- `--trust-remote-code`
- `--gpu-memory-utilization 0.90`
- `--enable-chunked-prefill`
- `--enable-prefix-caching`
- `--async-scheduling`

**Environment Variables:**
- `VLLM_USE_TRTLLM_ATTENTION=1`
- `VLLM_USE_TRTLLM_DECODE_ATTENTION=1`
- `VLLM_USE_TRTLLM_CONTEXT_ATTENTION=1`
- `VLLM_USE_FLASHINFER_MXFP4_BF16_MOE=1`
- `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING=1`

## Profile Selection Guide

### Quick Decision Tree

1. **What's your primary goal?**
   - Testing/Development → `standard`
   - Maximum speed → `high_throughput`
   - Limited memory → `low_memory`

2. **What model architecture?**
   - Mixture of Experts → `moe_optimized`
   - GPT-OSS on specific GPU → `gpt_oss_*` profiles

3. **What's your hardware?**
   - NVIDIA A100 → `gpt_oss_ampere`
   - NVIDIA H100/H200 → `gpt_oss_hopper`
   - NVIDIA Blackwell → `gpt_oss_blackwell`
   - Other → `standard` or `high_throughput`

## Customizing Profiles

### Creating Custom Profiles

You can create custom profiles based on these built-in ones:

1. **Via Interactive Mode:**
   ```
   Settings → Profile Management → Create Profile
   ```

2. **Via Configuration File:**
   Edit `~/.config/vllm-cli/user_profiles.json`:
   ```json
   {
     "my_custom_profile": {
       "tensor_parallel_size": 4,
       "max_model_len": 8192,
       "quantization": "awq",
       "gpu_memory_utilization": 0.95,
       "environment_variables": {
         "VLLM_ATTENTION_BACKEND": "FLASH_ATTN"
       }
     }
   }
   ```

### Modifying Built-in Profiles

Built-in profiles can be overridden by creating a user profile with the same name. The user profile will take precedence.

## Performance Tuning Tips

### Memory Optimization
- Start with `gpu_memory_utilization: 0.9` and increase gradually
- Use quantization (`awq`, `gptq`, `fp8`) for larger models
- Reduce `max_model_len` if you don't need long context

### Throughput Optimization
- Increase `max_num_seqs` for more concurrent requests
- Use larger `max_num_batched_tokens` for better batching
- Enable continuous batching with async output processing

### Latency Optimization
- Reduce `max_num_seqs` for lower latency
- Use flash attention backends
- Disable prefix caching if not needed

## Environment Variables Reference

Common environment variables used in profiles:

| Variable | Purpose | Values |
|----------|---------|---------|
| `VLLM_ATTENTION_BACKEND` | Attention computation backend | `FLASH_ATTN`, `XFORMERS`, `TRITON` |
| `VLLM_USE_TRITON_FLASH_ATTN` | Enable Triton flash attention | `0`, `1` |
| `VLLM_ENABLE_FUSED_MOE_ACTIVATION_CHUNKING` | MoE activation chunking | `0`, `1` |
| `VLLM_USE_FLASHINFER_MXFP4_BF16_MOE` | BF16 precision for MoE | `0`, `1` |
| `VLLM_USE_TRTLLM_ATTENTION` | TensorRT-LLM attention | `0`, `1` |
| `CUDA_VISIBLE_DEVICES` | GPU device selection | `0`, `0,1`, etc. |

## See Also

- [Usage Guide](usage-guide.md) - Complete usage instructions
- [vLLM Documentation](https://docs.vllm.ai/) - Official vLLM docs
- [vLLM GPT Recipes](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) - GPU-specific optimizations
