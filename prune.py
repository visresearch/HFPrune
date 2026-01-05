import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random
import torch

import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, LlamaTokenizer, AutoConfig
import torch.nn as nn
import sys
from tqdm import tqdm
import time
import torch.nn.functional as F
def save_importance_data(result_dict, file_path):
        torch.save(result_dict, file_path)
        print(f"✅ saved importance data to{file_path}")
        
def count_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1_000_000:.2f}M")
    return total_params
def prune_given_indices(in_linear: nn.Linear, out_linear: nn.Linear, gate_linear: nn.Linear, idxs_select: torch.Tensor=None):
    in_weight = in_linear.weight.data
    in_bias = in_linear.bias.data if in_linear.bias is not None else None
    out_weight = out_linear.weight.data
    out_bias = out_linear.bias.data if out_linear.bias is not None else None
    gate_weight = gate_linear.weight.data
    gate_bias = gate_linear.bias.data if gate_linear.bias is not None else None
    
    # --------------- input linear ---------------
    in_weight_prune = in_weight[idxs_select]
    in_bias_prune = in_bias[idxs_select] \
        if in_bias is not None else None
    in_linear_prune = nn.Linear(
        in_features=in_weight_prune.shape[1], 
        out_features=in_weight_prune.shape[0],
        bias=in_bias_prune is not None
    )
    in_linear_prune.weight.data = in_weight_prune
    if in_bias is not None:
        in_linear_prune.bias.data = in_bias_prune
    # --------------- gate linear -----------------
    gate_weight_prune = gate_weight[idxs_select]
    gate_bias_prune = gate_bias[idxs_select] \
        if gate_bias is not None else None
    gate_linear_prune = nn.Linear(
        in_features=gate_weight_prune.shape[1], 
        out_features=gate_weight_prune.shape[0],
        bias=gate_bias_prune is not None
    )
    gate_linear_prune.weight.data = gate_weight_prune
    if gate_bias is not None:
        gate_linear_prune.bias.data = gate_bias_prune
    # --------------- output linear ---------------
    out_weight_prune = out_weight[:, idxs_select]
    out_linear_prune = nn.Linear(
        in_features=out_weight_prune.shape[1], 
        out_features=out_weight_prune.shape[0],
        bias=out_bias is not None
    )
    out_linear_prune.weight.data = out_weight_prune
    if out_bias is not None:
        out_linear_prune.bias.data = out_bias
    return in_linear_prune, out_linear_prune, gate_linear_prune

def get_importance_taylor(path, seed=None):
    seq_len = 1024
    dataset_path = ''
    dataset = load_dataset(dataset_path, data_files={'train': ''}, split='train')

    if seed:
        dataset = dataset.shuffle(seed=seed)
    else:
        dataset = dataset.shuffle()
    tokenizer = AutoTokenizer.from_pretrained(path)
    filtered_samples = []
    for example in dataset:
        inputs = tokenizer(example['text'], truncation=False, padding=False)
        if len(inputs['input_ids']) > seq_len:
            i = random.randint(0, len(inputs['input_ids']) - seq_len - 1)
            j = i + seq_len
            inputs['input_ids'] = inputs['input_ids'][i:j]
            inputs['attention_mask'] = inputs['attention_mask'][i:j]
            filtered_samples.append(inputs)
        # if len(filtered_samples) >= 1024:
        #     break
    dataset = Dataset.from_list(filtered_samples)
    # print(dataset)
    model = AutoModelForCausalLM.from_pretrained(path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16,device_map="auto")
    for p in model.parameters():
        p.requires_grad = False
    for idx in range(len(model.model.layers)):
        for p in model.model.layers[idx].mlp.down_proj.parameters():
            p.requires_grad = True
        for p in model.model.layers[idx].mlp.up_proj.parameters():
            p.requires_grad = True
        for p in model.model.layers[idx].mlp.gate_proj.parameters():
            p.requires_grad = True
    activations = {}
    gradients = {}

    def get_activation_hook(name):
        def hook(model, input, output):
            activations[name] = input[0].detach()
        return hook

    def get_gradient_hook(name):
        def hook(model, grad_input, grad_output):
            gradients[name] = grad_input[0].detach()
        return hook

    for idx, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_hook(get_activation_hook(idx))
        layer.mlp.down_proj.register_full_backward_hook(get_gradient_hook(idx))
        
    print("forward!")
    
    importance = [
        torch.zeros(layer.mlp.down_proj.in_features, device=model.device, dtype=torch.bfloat16)
        for layer in model.model.layers
    ]

    for example in tqdm(dataset):
        input_ids = torch.tensor(example["input_ids"], dtype=torch.int).unsqueeze(0).cuda()

        model.zero_grad()
        
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        
        
        entropy_loss = -torch.sum(probs * log_probs, dim=-1).mean()
        
        entropy_loss.requires_grad_(True)

        entropy_loss.backward()


        for idx in range(len(model.model.layers)):

            if idx in activations and idx in gradients:

                act = activations[idx]
                grad = gradients[idx]
                down_proj_importance = torch.abs(act * grad).sum(dim=(0, 1))
                importance[idx] += down_proj_importance.to(importance[idx].device)
        
        activations.clear()
        gradients.clear()
    
    res = {"importance": importance}
    return res
def prune(importance, origin_path, mlp_ratio):
    model = AutoModelForCausalLM.from_pretrained(origin_path,attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16)
    
    model = model.to(torch.bfloat16).cuda()
    importance = importance["importance"]
    
    unprune=count_parameters(model)
    for idx in range(len(model.model.layers)):
        importance[idx] = importance[idx].cuda()
        dim = model.model.layers[idx].mlp.down_proj.out_features

        _, indices = torch.topk(importance[idx], k=int(dim * mlp_ratio), dim=0, largest=True)
        indices, _ = torch.sort(indices)
        model.model.layers[idx].mlp.up_proj, model.model.layers[idx].mlp.down_proj, \
            model.model.layers[idx].mlp.gate_proj = \
        prune_given_indices(model.model.layers[idx].mlp.up_proj, model.model.layers[idx].mlp.down_proj, model.model.layers[idx].mlp.gate_proj, indices)
    pruned=count_parameters(model)
    ra=int(pruned/unprune*100)
    print(f"{ra}%")
    print(f"unprune:{unprune},pruned:{pruned}")
    save_path = f""
    model.config.intermediate_size = int(mlp_ratio * model.config.hidden_size)
    model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(origin_path)
    tokenizer.save_pretrained(save_path)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prune")
    parser.add_argument("--seed", type=int, required=True, help="seed")
    parser.add_argument("--mlp_ratio", type=float, required=True, help="mlp_ratio")
    parser.add_argument("--origin_path", type=str, required=True, help="origin_path")
    args = parser.parse_args()

    origin_path = args.origin_path
    seed = args.seed
    mlp_ratio = args.mlp_ratio

    taylor_importance = get_importance_taylor(origin_path, seed=seed)
    output_filename = "./importance/taylor_importance.pt"
    save_importance_data(taylor_importance, output_filename)
    
    taylor_importance = torch.load(output_filename, map_location=torch.device('cpu'))
    prune(taylor_importance, origin_path, mlp_ratio)

