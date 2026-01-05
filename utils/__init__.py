import os
import re
import time
import peft
import torch.distributed as dist
def get_output_dir(config):
    if dist.get_rank() == 0:
        if isinstance(config["dataset"]["name"], list):
            dataset = ""
            for d in config["dataset"]["name"]:
                if 'lamini' in d.lower():
                    dataset += 'lamini_'
                elif 'alpaca' in d.lower():
                    dataset += 'alpaca'
                elif "smoltalk" in d.lower():
                    dataset += "smoltalk"
        elif 'lamini' in config["dataset"]["name"].lower():
            dataset = 'lamini'
        elif 'alpaca' in config["dataset"]["name"].lower():
            dataset = 'alpaca'
        elif "smoltalk" in config["dataset"]["name"].lower():
            dataset = "smoltalk"
        elif "smollm-corpus" in config["dataset"]["name"].lower():
            dataset = "smollm-corpus"
        elif "c4" in config["dataset"]["name"].lower():
            dataset = "c4"
        round = config["models"]
        dataset += f"_r{round}"
        if "distillation" in config.keys():
            t = config["distillation"]["temperature"]
            alpha = config["distillation"]["alpha"]
            learning_rate = config["training"]["learning_rate"]
            date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            r = get_round_number(config["models"])
            dir_name = date + f"_r{r}_lr{learning_rate}_t{t}_alpha{alpha}_{dataset}"
            output_dir = os.path.join(config["training"]["output_dir"], config["project_name"], dir_name)
        else:
            learning_rate = config["training"]["learning_rate"]
            gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
            date = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            r = get_round_number(config["models"])
            dir_name = date + f"_r{r}_lr{learning_rate}_{dataset}_{gradient_accumulation_steps}"
            output_dir = os.path.join(config["training"]["output_dir"], config["project_name"], dir_name)
    else:
        output_dir = None
    
    output_dir = [output_dir]
    dist.broadcast_object_list(output_dir, src=0)
    return output_dir[0]
def get_round_number(path):
    match = re.search(r"diversity(\d+)", path)
    if match:
        diversity_number = int(match.group(1))
        return diversity_number
    else:
        return None