# 2B Parameter Local AI Model Development Plan (Apple Silicon / MLX)

This document outlines a comprehensive, phase-by-phase architecture and execution plan for building, training, and experimenting with a custom 2-Billion parameter language model natively on Apple Silicon using the MLX framework.

## Phase 1: Laboratory Setup & Infrastructure
Before writing the training loop, the environment and observability infrastructure must be established to monitor the "black box" of the training process.

* **MLX Environment Architecture:** Set up a dedicated virtual environment (`uv` or `conda`) strictly tuned for Apple Silicon. Ensure MLX and `mlx-lm` are compiled to leverage the unified memory and GPU matrix coprocessors.
* **Hardware Profiling:** Integrate Apple-specific profiling tools (like `powermetrics` or `asitop`) to monitor GPU utilization, memory bandwidth saturation, and thermal throttling in real-time. 
* **Observability Stack:** Integrate Weights & Biases (W&B) or a similar telemetry tool to track real-time loss curves, gradient norms, learning rates, and hardware metrics.
* **The Baseline "Toy" Model:** Implement a 50M–100M parameter micro-model. This serves as an end-to-end unit test to verify that the training loop, data pipeline, and checkpointing logic work flawlessly before scaling up.

## Phase 2: Data Engineering & Tokenization
A model's intelligence is strictly bounded by its data quality. For a 2B parameter model, a high-quality pre-training corpus of 100B to 500B tokens is required.

* **The Dataset Mix:** Curate a balanced blend of high-quality web data (e.g., FineWeb-Edu), code (e.g., Stack v2), and specialized knowledge (textbooks, ArXiv papers).
* **Data Deduplication & De-contamination:** Implement MinHash or exact substring matching to remove duplicate documents and ensure benchmark questions are not accidentally included in the training data.
* **The Tokenizer:** Train a custom byte-pair encoding (BPE) tokenizer using `tiktoken` or `SentencePiece`. Optimize the vocabulary size (e.g., 32k to 128k) to balance compression rate versus embedding layer memory footprint.
* **High-Throughput Data Pipeline:** Write a highly optimized data loader that utilizes memory-mapped files (`mmap`). The goal is to pre-tokenize the entire dataset into binary shards so the SSD streams directly into the unified memory buffer without CPU bottlenecking.

## Phase 3: Architecture & Mathematical Design
Defining the mathematical "shape" of the model. This phase establishes the core components before weights are initialized.

* **Core Specifications:** Define the hyperparameter grid: number of transformer blocks, hidden layer dimensions, and the number of attention heads required to hit the 2B parameter target.
* **Positional Embeddings:** Implement Rotary Positional Embeddings (RoPE) to allow the model to understand the relative distance between tokens.
* **Modern Normalization & Activations:** Swap legacy components for modern standards—use RMSNorm instead of LayerNorm, and SwiGLU instead of GELU/ReLU for the feed-forward networks.
* **Attention Mechanisms:** Implement Grouped-Query Attention (GQA) to drastically reduce the memory bandwidth required during inference. This is also the stage to code any custom experimental attention mechanisms.

## Phase 4: Pre-Training (The Long Burn)
The foundational training phase where the model learns syntax, facts, and logic from the raw data tokens.

* **Optimizer & Learning Rate Schedule:** Implement the AdamW optimizer paired with a Cosine Annealing learning rate schedule, ensuring a proper "warm-up" phase to prevent early gradient spikes.
* **Hyperparameter Tuning:** Run short, focused sweeps to find the optimal learning rate, weight decay, and batch size for the specific MLX/Hardware configuration.
* **Robust Checkpointing:** Design an interruptible checkpoint system that saves the Model Weights, Optimizer State, and Random Number Generator (RNG) state to disk at regular intervals. 
* **Validation & Loss Monitoring:** Run the model against a held-out validation dataset continuously. Monitor the "Perplexity" metric to ensure the model is actually generalizing rather than just memorizing the training data.

## Phase 5: Post-Training & "Special Sauce" Experiments
Once the base model has learned language and logic, it can be steered, optimized, and structurally modified.

* **Supervised Fine-Tuning (SFT):** Transition the base model into an instruction-following assistant using a curated dataset of high-quality prompt-response pairs.
* **Alignment (DPO/RLHF):** Implement Direct Preference Optimization (DPO) to teach the model how to choose better, safer, or more accurate responses based on human preference data.
* **Multi-Token Prediction (MTP):** Modify the architecture to add additional output heads, training the model to predict 2-4 future tokens simultaneously to improve reasoning and speculative decoding speed.
* **Architectural Ablation Studies:** Train identical but smaller versions of the model—one with standard attention and one with your custom experimental mechanisms—to empirically compare their loss curves.

## Phase 6: Evaluation, Optimization & Deployment
Proving the model's capabilities and optimizing it for real-world execution.

* **Standardized Benchmarking:** Evaluate the model against industry-standard benchmarks such as MMLU (knowledge/reasoning), HumanEval (coding), and GSM8K (math).
* **Quantization:** Utilize MLX-LM tools to quantize the model weights from FP16 down to 8-bit or 4-bit precision, drastically reducing memory footprint while maintaining performance.
* **Inference Optimization:** Write a custom, stripped-down inference script that maximizes token-per-second generation using MLX primitives, caching strategies, and speculative decoding.
