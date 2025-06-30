from transformers import AutoTokenizer, AutoModelForCausalLM, BloomForCausalLM, LlamaForCausalLM
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
import argparse
import random
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import os
from typing import Union, List, Dict, Tuple, Optional, Any
from functools import reduce
import pickle


translate_instruction = 'You are a translation specialist who specializes in translating texts from {src} into {tgt} in {domain} domain. Please translate the following content into {tgt} and only reply the translated sentence starting with "{tgt}:" without line breaks or any special tokens.'

domain_translation_patterns_with_domain_tgt = [
    ("{lang1}: {sent1}", "{lang2}: {sent2}")
]

def find_all_target_modules(model):
    target_module_name_list = []
    for name, module in model.named_modules():
        if len(module._parameters) > 0:
            if "mlp" in name:
                target_module_name_list.append(name)
    print(target_module_name_list)
    return target_module_name_list


def read_data(domain, src, tgt):
    file_path = f"data/high_quality/shuf.{domain}.train.{src}-{tgt}.jsonl"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    sent_list = []
    for line in lines:
        line = json.loads(line)
        sent_list.append((line["translation"][src].replace("\n", ""), line["translation"][tgt].replace("\n", "")))
    
    return sent_list


def construct_translation_prompt(
    train_example: Dict[str, Union[str, int]],
    prompt_template: str,
    instruction: str = "",
) -> Tuple[str, str]:
    messages = []
    if instruction != "":
        instruction = instruction.format(**train_example)
        messages.append({"role": "system", "content": instruction})

    prompt_template = random.choice(prompt_template)
    prompt_input = (
        prompt_template[0].format(lang1= train_example["src"], lang2= train_example["tgt"], sent1= train_example["sent1"].strip("\n"), domain=train_example["domain"])
        )
    messages.append({"role": "user", "content": prompt_input})
    
    prompt_input = messages
    tgt_lang = train_example["tgt"]
    label = f"{tgt_lang}: " + train_example["sent2"].strip("\n")

    return prompt_input, label
        

def find_specific_target_module(model, target_module_name):
    for name, module in model.named_modules():
        if name == target_module_name:
            return module

def forward_hook(module, input, output, module_name):
    forward_cache.append(output)

def backward_hook(module, grad_input, grad_output, module_name):
    backward_cache.append(grad_output[0])

def add_hooks(model, target_module_names):
    hook_forwards, hook_backwards = [], []
    for name, module in model.named_modules():
        if name in target_module_names:
            print(name, module)
            handle_forward = module.register_forward_hook(lambda m, i, o, module_name=name: forward_hook(m, i, o, name))
            handle_backward = module.register_backward_hook(lambda m, gi, go, module_name=name: backward_hook(m, gi, go, name))
            hook_forwards.append(handle_forward)
            hook_backwards.append(handle_backward)

    return hook_forwards, hook_backwards

def remove_hook(hook_forwards, hook_backwards):
    for hook_forward, hook_backward in zip(hook_forwards, hook_backwards):
        hook_forward.remove()
        hook_backward.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='model path', default='Qwen2.5-7B-Instruct')
    parser.add_argument('--src', type=str, help='source language', default='de')
    parser.add_argument('--tgt', type=str, help='target language', default='en')
    
    args = parser.parse_args()

    lang_code_dict = {"en": "English", "de": "German", "zh": "Chinese"}

    domain_code_list = ['it', 'law', 'med', 'sub']
    domain_code_list = ['edu', 'spok', 'thes']
    domain_code_dict = {"it": "IT", "med": "medical", "koran": "Koran", "law": "law", "thes": "thesis", "sub": "subtitles", "spok": "spoken", "edu": "education"}

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
   
    target_module_names = find_all_target_modules(model)

    forward_cache = []
    backward_cache = []
    importance_matrix_dict = {key: {key: [] for key in target_module_names} for key in domain_code_list}

  
    hook_forwards, hook_backwards = add_hooks(model, target_module_names)

    domain_data_count = []

    for i in range(len(domain_code_list)):
        cur_domain_code = domain_code_list[i]
        sent_list = read_data(cur_domain_code, args.src, args.tgt)
        
        domain_data_count.append(len(sent_list))

        new_sent_list = []

        for line in sent_list:
            example = {"src": lang_code_dict[args.src], "tgt": lang_code_dict[args.tgt], "sent1": line[0], "sent2": line[1], "domain": domain_code_dict[cur_domain_code]}
            prompt_input, label = construct_translation_prompt(train_example=example, prompt_template=domain_translation_patterns_with_domain_tgt, instruction=translate_instruction)
            result = tokenizer.apply_chat_template(prompt_input, tokenize=False, add_generation_prompt=True)
            bos_token = ""
            eos_token = ""
            new_sent_list.append({"input": result, "label": label})
        
        for j in tqdm(range(len(new_sent_list))):
            inputs = tokenizer(new_sent_list[j]["input"], return_tensors="pt").to("cuda")
            label = tokenizer(new_sent_list[j]["label"], return_tensors="pt").to("cuda")

            input_ids = inputs["input_ids"]
            label_ids = label["input_ids"]
            label = torch.cat([torch.full_like(input_ids, -100), label_ids], dim=1)
            input_ids = torch.cat((input_ids, label_ids), dim=1)
            
            outputs = model(input_ids, labels=label)
            loss = outputs.loss
            loss.backward()


            for k in range(len(target_module_names)):
                cur_module_name = target_module_names[k]
                importance_matrix = (forward_cache[k] * backward_cache[len(backward_cache) - k - 1]).abs()
                importance_matrix = importance_matrix.view(-1, importance_matrix.size(-1)).mean(0)
                importance_matrix_dict[cur_domain_code][cur_module_name].append(importance_matrix.detach().cpu())
            
            forward_cache = []
            backward_cache = []
            model.zero_grad()
            
    remove_hook(hook_forwards, hook_backwards)


    _reshape_importance_matrix = {key: [] for key in target_module_names}
    reshape_importance_matrix = {key: [] for key in target_module_names}
    mi_scores_multi_domain = {key: [] for key in target_module_names}


    domain_labels = np.zeros(sum(domain_data_count), dtype=int)
    for i in range(len(domain_data_count)):
        start = sum(domain_data_count[:i])
        end = sum(domain_data_count[:i+1])
        domain_labels[start:end] = i + 1


    for cur_module_name in target_module_names:
        for cur_domain_code in domain_code_list:
            temp_domain_matrix = torch.stack(importance_matrix_dict[cur_domain_code][cur_module_name])
            _reshape_importance_matrix[cur_module_name].append(temp_domain_matrix)
        temp_importance_matrix = torch.stack(_reshape_importance_matrix[cur_module_name])
        temp_importance_matrix = temp_importance_matrix.flatten(0, 1)
        temp_importance_matrix = torch.split(temp_importance_matrix, 1, dim=1)
        reshape_importance_matrix[cur_module_name] = [t.squeeze(1) for t in temp_importance_matrix]

        mi = []
        for tensor in reshape_importance_matrix[cur_module_name]:
            tensor_np = tensor.to(torch.float).numpy()
            mi.append(mutual_info_score(tensor_np, domain_labels))
        
        print(f"{cur_module_name}_mi")
        mi_scores_multi_domain[cur_module_name] = mi
    
    mi_scores_file = f"neurons/mi_scores_multi_domain_qwen2_5.pkl"
    with open(mi_scores_file, 'wb') as f:
        pickle.dump({"mutual_info_dict": mi_scores_multi_domain}, f)
