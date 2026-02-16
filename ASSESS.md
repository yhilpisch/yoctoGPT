# Assessment and Proposed Improvements for yoctoGPT

This document outlines concise, actionable, and beneficial improvements for the `yoctoGPT` codebase to enhance performance, scalability, and educational value.

### 1. Core Model & Inference
- **KV Caching**: Implement Key-Value (KV) caching in the `generate` method. Currently, the model recomputes all past keys and values at every step, leading to $O(N^2)$ inference. KV caching reduces this to $O(N)$.
- **Pre-compute RoPE**: In `AdvancedGPT`, pre-compute the rotary frequency buffers once and register them as buffers to avoid repeated trigonometry and tensor allocations during the forward pass.
- **`torch.compile` Support**: Add a simple flag or logic to wrap the model in `torch.compile()`. For PyTorch 2.0+, this can provide a 2x-3x speedup on compatible hardware with minimal code changes.

### 2. Training Infrastructure
- **Weight Decay Separation**: Centralize the logic to separate parameters that should receive weight decay (weights) from those that should not (biases, Norm parameters, embeddings) into a utility. Currently, this is only explicitly handled in `train.py`.
- **Distributed Training (DDP)**: Add basic support for `DistributedDataParallel`. Even for a "yocto" project, showing how to scale to multiple GPUs is a high-value educational addition.
- **Mixed Precision (AMP)**: Integrate `torch.cuda.amp` (or `torch.amp` for newer versions) to allow training in `float16` or `bfloat16`. This significantly reduces memory usage and speeds up training on modern GPUs (T4, A100).

### 3. Data & Tokenization
- **Dataset Streaming**: For larger corpora, implement a streaming dataset that doesn't require loading the entire `.bin` file into memory (using `np.memmap`).
- **BOS/EOS Token Handling**: Explicitly handle Beginning-of-Sentence and End-of-Sentence tokens in the `TokenDataset` and `sampler.py` to improve the quality of generated completions.

### 4. Code Consistency
- **Unified Forward API**: Standardize the `forward` method signature across `GPT`, `AdvancedGPT`, and `PerformanceGPT`. Currently, `AdvancedGPT` supports a `labels` argument for in-module loss calculation, while the others do not.
- **Configuration Parity**: Ensure all configuration classes (`GPTConfig`, `AdvancedGPTConfig`, etc.) share a common base or consistent field names to simplify model switching in CLI tools.
