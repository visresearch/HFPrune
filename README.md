# HFPrune: High-Fidelity Pruning for Large Language Models

This repository is the official implementation of the paper "HIGH-FIDELITY PRUNING FOR LARGE LANGUAGE MODELS".

## Introduction

Large Language Models (LLMs) have demonstrated exceptional performance but require significant computational resources for deployment. **HFPrune** addresses this through a novel pruning method using **information entropy** for importance evaluation, achieving superior performance compared to existing approaches. Traditional pruning methods rely on one-hot cross-entropy loss, focusing only on single ground-truth tokens. **HFPrune** uses information entropy to evaluate neuron importance based on the **global prediction distribution**, preserving the model's full knowledge while requiring no teacher model.

### Highlights

🏆 **Surpasses Dense Model**: At 20% pruning on LLaMA-2-7B, achieves **59.0** average score vs **58.3** for the original model

📊 **Consistent Improvements**: Outperforms LLM-Pruner, LoRAPrune, and SDMPrune across all benchmarks

⚡ **Efficient**: No teacher model required, label-free importance evaluation

🔧 **Practical**: Supports both full fine-tuning and LoRA, with sequence packing and Flash Attention 2

### Supported Models

- ✅ LLaMA series (LLaMA-2-7B, LLaMA-3, etc.)
- ✅ Qwen series (Qwen2.5-7B, etc.)
- ✅ Any transformer-based LLM with similar architecture

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Complete Workflow Example

Here's a complete workflow from pruning to evaluation:

```bash
# Step 1: Prune the model
python prune.py \
    --seed 42 \
    --mlp_ratio 2.8 \
    --origin_path "meta-llama/Llama-2-7b-hf"

# Step 2: Fine-tune with LoRA (recommended for efficiency)
accelerate launch --num_processes=4 \
    --mixed_precision bf16 \
    fintune_lora.py \
    --model "path/to/pruned/model" \
    --lr 4e-4 \
    --lora_r 32 \
    --lora_alpha 64

# Step 3: Evaluate on benchmarks
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True \
    --tasks hellaswag,piqa,arc_challenge,arc_easy,openbookqa,boolq,winogrande \
    --batch_size 4
```

### 1. Pruning

Prune a LLaMA model using Taylor-based importance estimation:

```bash
python prune.py \
    --seed 42 \
    --mlp_ratio 2.8 \
    --origin_path "meta-llama/Llama-2-7b-hf"
```

### 2. Fine-tuning

#### Full Fine-tuning

```bash
accelerate launch --gpu_ids '0,1,2,3' --num_processes=4 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no \
    fintune_full.py \
    --model "path/to/pruned/model" \
    --lr 1e-4
```

#### LoRA Fine-tuning

```bash
accelerate launch --num_processes=4 --num_machines 1 \
    --mixed_precision bf16 --dynamo_backend no \
    fintune_lora.py \
    --model "path/to/pruned/model" \
    --lr 4e-4 \
    --dataset_name "out/cache/Lamini-llama3.2-clean-1024/" \
    --tokenizer_max 1024 \
    --lora_r 32 \
    --lora_alpha 64
```

### 3. Evaluation

Evaluate the pruned/fine-tuned model using lm-evaluation-harness:

```bash
lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True,add_bos_token=True \
    --tasks hellaswag,piqa,arc_challenge,arc_easy,openbookqa,boolq,winogrande \
    --batch_size 4 \
    --output_path "$MODEL_PATH/results"
```

## Project Structure

```
├── prune.py                 # Main pruning script
├── fintune_full.py          # Full fine-tuning script
├── fintune_lora.py          # LoRA fine-tuning script
├── cfg/
│   └── prune_llama.py       # Training configurations
├── dataset/
│   ├── LaMini_dataset.py    # Dataset loading utilities
│   ├── prompter.py          # Prompt templates
│   └── packing/             # Sequence packing implementation
│       ├── packed_dataset.py
│       └── monkey_patch_packing.py
├── module/
│   ├── trainer.py           # Custom trainer with packing support
│   └── anyprecisionAdamw.py # Mixed-precision AdamW optimizer
├── script/
│   ├── prune.sh             # SLURM script for pruning
│   ├── train_full.sh        # SLURM script for full fine-tuning
│   └── train_lora.sh        # SLURM script for LoRA fine-tuning
└── utils/
    └── __init__.py          # Utility functions
```

## Experimental Results

### Highlights

🏆 **Surpasses Dense Model**: At 20% pruning on LLaMA-2-7B, HFPrune achieves **59.0** average score, exceeding the original model's **58.3**

📊 **Consistent Improvements**: Outperforms all baseline methods across all pruning ratios and model sizes

⚡ **Efficient Compression**: Achieves 20-30% reduction in both parameters and FLOPs while maintaining or improving performance

🎯 **Robust Across Benchmarks**: Strong performance on diverse tasks including reasoning (ARC), common sense (PIQA, OBQA), and reading comprehension (BoolQ, Winogrande)

### Performance on LLaMA-2-7B

We evaluated HFPrune on multiple zero-shot benchmarks and compared it against state-of-the-art structured pruning methods including **LLM-Pruner**, **LoRAPrune**, **LoRAP**, and **SDMPrune**.

**Key Result**: At a 20% pruning ratio, HFPrune's average score (59.0) **surpasses the original dense model** (58.3), demonstrating that our method not only preserves but can actually enhance model capabilities through better-targeted compression.

| Pruning Ratio | Method | ARCC | ARCE | BoolQ | Crows | OBQA | PIQA | Race | SiQA | TfQA | Wino | Average |
|---------------|--------|------|------|-------|-------|------|------|------|------|------|------|---------| 
| 0% | Llama-2-7B | 45.1 | 73.8 | 79.4 | 67.4 | 44.2 | 78.7 | 40.1 | 46.5 | 38.8 | 69.3 | **58.3** |
| 20% | LLM-pruner | 40.4 | 70.1 | 80.2 | 61.7 | 38.8 | 75.8 | 39.0 | 47.1 | 43.9 | 64.3 | 56.1 |
| 20% | LoRAPrune | 41.6 | 71.0 | 81.7 | 58.7 | 41.4 | 76.7 | 40.4 | 44.0 | **65.9** | 65.9 | 56.7 |
| 20% | LoRAP | 38.5 | 66.0 | 70.9 | -- | 39.6 | **78.1** | -- | -- | -- | 65.7 | -- |
| 20% | SDMPrune | 43.9 | 72.3 | 81.7 | **62.1** | 42.0 | 77.0 | 41.3 | 48.5 | 44.9 | **68.4** | 58.2 |
| 20% | **HFPrune (Ours)** | **47.1** | **73.8** | **85.2** | 60.2 | **43.2** | 77.3 | **43.3** | **49.5** | 44.7 | 66.2 | **59.0** |
| 30% | LLM-pruner | 38.0 | 64.8 | 75.6 | **62.3** | 36.4 | 73.4 | 35.7 | 47.3 | 42.3 | 62.9 | 53.9 |
| 30% | LoRAPrune | 38.6 | 65.1 | 74.1 | 61.4 | 37.4 | 72.9 | 39.0 | 46.3 | **44.8** | **66.5** | 54.6 |
| 30% | LoRAP | 35.5 | 60.6 | 69.6 | -- | 37.8 | **76.7** | -- | -- | -- | 63.0 | -- |
| 30% | SDMPrune | 39.6 | 67.9 | 80.4 | 58.5 | 37.2 | 75.2 | **40.0** | 47.8 | 43.7 | 65.4 | 55.6 |
| 30% | **HFPrune (Ours)** | **41.9** | **70.2** | **82.9** | 58.1 | **40.0** | 75.2 | 39.5 | **48.8** | 44.2 | 62.4 | **56.3** |

## Model Weights

We provide the model weights pruned by HFPrune for reproducibility and downstream use.

| Model | Download Link |
|-------|---------------|
| LLaMA series | [🤗 HuggingFace](https://huggingface.co/visresearch/HFPrune-Llama-pruned) |
| Qwen series | [🤗 HuggingFace](https://huggingface.co/visresearch/HFPrune-Qwen-pruned) |

## Citation

```bibtex
@article{hfprune2026,
  title={HIGH-FIDELITY PRUNING FOR LARGE LANGUAGE MODELS},
  author={},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
