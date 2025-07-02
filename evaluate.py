import os
import torch
from transformers import AutoTokenizer, BloomForCausalLM, AutoModelForCausalLM, LlamaForCausalLM
from tqdm import tqdm
import json
import random
import argparse
from jinja2 import Template
from typing import Union, List, Dict, Tuple, Optional, Any
from StopAtSpecificTokenCriteria import StopAtSpecificTokenCriteria
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
import pickle
import numpy as np

def read_data(domain, src, tgt):
    file_path = f"data/{domain}.test.{src}-{tgt}.instruct.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        sent_list = json.load(f)

    return sent_list

def construct_translation_prompt_with_template(
    input: str,
    output: str,
    instruction: str = "",
) -> Tuple[str, str]:
    messages = []
    if instruction != "":
        messages.append({"role": "system", "content": instruction})

    messages.append({"role": "user", "content": input.strip("\n")})

    prompt_input = messages
    test_prompt_label = output.strip("\n")

    return prompt_input, test_prompt_label

def write_data(output_list, data_path):
    with open(data_path, "w", encoding="utf-8") as f:
        for line in output_list:
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, help='source language', default='de')
    parser.add_argument('--tgt', type=str, help='target language', default='en')
    parser.add_argument('--model_path', type=str, help='model path', default=None)
    parser.add_argument('--model_type', type=str, help='model type: llama2, llama3, qwen2.5', default="qwen2.5")
    
    args = parser.parse_args()

    output_path_prefix = "trans_results/"

    model_path = args.model_path

    model_type = "qwen2.5"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)
 
    lang_code_dict = {"en": "English", "de": "German", "zh": "Chinese"}

    domain_code_list = ['it', 'koran', 'law', 'med', 'the']
    # domain_code_list = ['edu', 'spok', 'thes', 'sci', 'blog']
    domain_code_dict = {"it": "IT", "med": "medical", "koran": "Koran", "law": "law", "the": "thesis", "sub": "subtitles", "sci": "science", "blog": "microblog"}

    batch_size = 1

    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[151645])) ##qwen2 <|im_end|>
    # stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[128009])) #llama3 <|eot_id|>
    # stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[29889])) #llama2



    for i in range(len(domain_code_list)):
        cur_domain_code = domain_code_list[i]
        
        output_list = []
        sent_list = read_data(cur_domain_code, args.src, args.tgt)

        new_sent_list = []
        original_input = []
        for line in sent_list:
            prompt_input, test_prompt_label = construct_translation_prompt_with_template(input=line["input"], output=line["output"], instruction=line["instruction"])
           
            result = tokenizer.apply_chat_template(prompt_input, tokenize=False, add_generation_prompt=True)
           
            bos_token = ""
            eos_token = ""
            
            new_sent_list.append(result)

            if model_type == "llama3":
                original_input.append(prompt_input[0]['role'] + "\n\n" + prompt_input[0]['content'] + prompt_input[1]['role'] + "\n\n" + prompt_input[1]['content'])
            elif model_type == "qwen2.5":
                original_input.append(prompt_input[0]['role'] + "\n" + prompt_input[0]['content'] + "\n" + prompt_input[1]['role'] + "\n" + prompt_input[1]['content'] + "\n")
            


        for j in tqdm(range(0, len(new_sent_list), batch_size), desc=domain_code_list[i]):
            inputs = tokenizer(new_sent_list[j], return_tensors="pt").to("cuda")
            outputs = model.generate(inputs.input_ids, stopping_criteria=stopping_criteria, temperature=0.0001)

            for k in range(outputs.shape[0]):
                cur_output = tokenizer.decode(outputs[k], skip_special_tokens=True)
                if model_type == "llama2":
                    cur_output = cur_output.replace(new_sent_list[j + k][3:], "", 1)
                elif model_type == "llama3" or model_type == "qwen2.5":
                    cur_output = cur_output.replace(original_input[j], "", 1)
                cur_output = cur_output.replace("\n", "").replace("\r", "").strip()
                output_list.append(cur_output)

        write_data(output_list, output_path_prefix + domain_code_list[i] + "_mlp_few_data_qwen25.txt")
