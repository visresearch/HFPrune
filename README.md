# HFPrune: High-Fidelity Pruning for Large Language Models

This repository is the official implementation of the paper "HIGH-FIDELITY PRUNING FOR LARGE LANGUAGE MODELS".



## Main Experimental Results

We evaluated HFPrune on multiple zero-shot benchmarks and compared it against state-of-the-art methods, including LLM-pruner, LoRAPrune, and SDMPrune.

### 1. Performance on LLaMA-2-7B

HFPrune shows excellent performance on LLaMA-2-7B. At a 20% pruning ratio, our model's average score (59.0) even **surpasses the original dense model** (58.3).

| **Pruning Ratio** | **Method**         | **ARCC** | **ARCE** | **BoolQ** | **OBQA** | **PIQA** | **Wino** | **Average** |
| ----------------- | ------------------ | -------- | -------- | --------- | -------- | -------- | -------- | ----------- |
| 0%                | Llama-2-7B         | 45.1     | 73.8     | 79.4      | 44.2     | 78.7     | 69.3     | **58.3**    |
| 20%               | LLM-pruner         | 40.4     | 70.1     | 80.2      | 38.8     | 75.8     | 64.3     | 56.1        |
| 20%               | LoRAPrune          | 41.6     | 71.0     | 81.7      | 41.4     | 76.7     | 65.9     | 56.7        |
| 20%               | SDMPrune           | 43.9     | 72.3     | 81.7      | 42.0     | 77.0     | 68.4     | 58.2        |
| 20%               | **HFPrune (Ours)** | **47.1** | **73.8** | **85.2**  | **43.2** | **77.3** | **66.2** | **59.0**    |
| 30%               | LLM-pruner         | 38.0     | 64.8     | 75.6      | 36.4     | 73.4     | 62.9     | 53.9        |
| 30%               | LoRAPrune          | 38.6     | 65.1     | 74.1      | 37.4     | 72.9     | 66.5     | 54.6        |
| 30%               | SDMPrune           | 39.6     | 67.9     | 80.4      | 37.2     | 75.2     | 65.4     | 55.6        |
| 30%               | **HFPrune (Ours)** | **41.9** | **70.2** | **82.9**  | **40.0** | **75.2** | **62.4** | **56.3**    |



## Installation and Usage

```
pip install -r requirements.txt
```

### Pruning

```
python prune.py 
	--seed 42 
	--mlp_ratio 2.8 
	--origin_path "meta-llama/Llama-2-7b-hf"
```

### Evaluation

*(You can add command examples for evaluating the pruned model here)*

```
lm_eval --model hf \
        --model_args pretrained=$ckpt,trust_remote_code=True,add_bos_token=True \
        --tasks hellaswag,piqa,arc_challenge,arc_easy,openbookqa,boolq,winogrande,truthfulqa_mc2,crows_pairs_english,race \
        --batch_size 4 \
        --output_path "$ckpt/results"
```

## Pretrained Weights

We provide the model weights pruned by HFPrune for reproducibility and downstream use.

| **Model**  | **Pruning Ratio** | **Download Link** |
| ---------- | ----------------- | ----------------- |
| LLaMA-2-7B | 20%               | [Coming Soon]     |
| LLaMA-2-7B | 30%               | [Coming Soon]     |
| Qwen2.5-7B | 20%               | [Coming Soon]     |
| Qwen2.5-7B | 30%               | [Coming Soon]     |
| ...        | ...               | [Coming Soon]     |

## Citation

```
@article{hfprune2026,
  title={HIGH-FIDELITY PRUNING FOR LARGE LANGUAGE MODELS},
  author={Anonymous},
  journal={},
  year={2026}
}
```