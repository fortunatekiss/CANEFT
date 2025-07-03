import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
import random
from sklearn.metrics import mutual_info_score
from typing import Union, Dict, Tuple
import pickle
import os

# Global caches for forward and backward hooks
forward_cache = []
backward_cache = []

# Translation instruction template
translate_instruction = 'You are a translation specialist who specializes in translating texts from {src} into {tgt} in {domain} domain. Please translate the following content into {tgt} and only reply the translated sentence starting with "{tgt}:" without line breaks or any special tokens.'

# Translation prompt patterns
domain_translation_patterns_with_domain_tgt = [
    ("{lang1}: {sent1}", "{lang2}: {sent2}")
]

def find_all_target_modules(model: torch.nn.Module) -> list[str]:
    """
    Identifies all MLP module names within the model that have parameters.
    """
    target_module_name_list = []
    for name, module in model.named_modules():
        if len(module._parameters) > 0: # Check if the module has parameters directly
            if "mlp" in name:
                target_module_name_list.append(name)
    print(f"Identified target MLP modules: {target_module_name_list}")
    return target_module_name_list

def read_data(domain: str, src: str, tgt: str, data_path_prefix: str) -> list[Tuple[str, str]]:
    """
    Reads translation data for a given domain and language pair.
    """
    file_path = os.path.join(data_path_prefix, f"shuf.{domain}.train.{src}-{tgt}.jsonl")
    
    sent_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line)
                # Replace newlines in sentences for cleaner processing
                sent_list.append((line["translation"][src].replace("\n", ""), line["translation"][tgt].replace("\n", "")))
    except FileNotFoundError:
        print(f"Warning: Data file not found for domain '{domain}' at {file_path}. Skipping this domain.")
        return []
    except KeyError:
        print(f"Warning: Missing '{src}' or '{tgt}' key in a line from {file_path}. Skipping line.")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Failed to decode JSON from a line in {file_path}. Skipping line.")
        return []
    
    return sent_list

def construct_translation_prompt(
    train_example: Dict[str, Union[str, int]],
    prompt_templates: list[Tuple[str, str]],
    instruction: str = "",
    tokenizer=None # Added tokenizer to construct chat template
) -> Tuple[str, str]:
    """
    Constructs the prompt and label for translation using a chat template.
    """
    messages = []
    if instruction != "":
        instruction = instruction.format(**train_example)
        messages.append({"role": "system", "content": instruction})

    prompt_template = random.choice(prompt_templates)
    prompt_input_text = prompt_template[0].format(
        lang1=train_example["src"],
        lang2=train_example["tgt"],
        sent1=train_example["sent1"].strip("\n"),
        domain=train_example["domain"]
    )
    messages.append({"role": "user", "content": prompt_input_text})
    
    # Apply chat template
    if tokenizer:
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages]) + "\n"

    tgt_lang = train_example["tgt"]
    label = f"{tgt_lang}: " + train_example["sent2"].strip("\n")

    return full_prompt, label
        
def forward_hook(module, input, output, module_name):
    """Stores the output of the forward pass."""
    global forward_cache
    forward_cache.append(output)

def backward_hook(module, grad_input, grad_output, module_name):
    """Stores the gradient of the output from the backward pass."""
    global backward_cache
    backward_cache.append(grad_output[0]) # grad_output is a tuple, we usually want the first element

def add_hooks(model: torch.nn.Module, target_module_names: list[str]):
    """Registers forward and backward hooks on specified modules."""
    hook_forwards, hook_backwards = [], []
    for name, module in model.named_modules():
        if name in target_module_names:
            # print(f"Adding hooks to: {name}") # Debugging
            handle_forward = module.register_forward_hook(lambda m, i, o, module_name=name: forward_hook(m, i, o, name))
            handle_backward = module.register_backward_hook(lambda m, gi, go, module_name=name: backward_hook(m, gi, go, name))
            hook_forwards.append(handle_forward)
            hook_backwards.append(handle_backward)
    return hook_forwards, hook_backwards

def remove_hook(hook_forwards: list, hook_backwards: list):
    """Removes all registered hooks."""
    for hook_forward in hook_forwards:
        hook_forward.remove()
    for hook_backward in hook_backwards:
        hook_backward.remove()
    print("All hooks removed.")


def main():
    parser = argparse.ArgumentParser(description="Calculate Mutual Information scores for MLP neurons across domains.")
    parser.add_argument('--model_path', type=str, default='llms/Qwen2.5-7B-Instruct',
                        help='Hugging Face model ID or local path to the model.')
    parser.add_argument('--model_type', type=str, default="qwen2.5",
                        choices=["llama2", "llama3", "qwen2.5"],
                        help='Model type: llama2, llama3, qwen2.5 (influences output file name).')
    parser.add_argument('--src_lang', type=str, default='de',
                        help='Source language code (e.g., "de", "en").')
    parser.add_argument('--tgt_lang', type=str, default='en',
                        help='Target language code (e.g., "en", "de").')
    parser.add_argument('--data_prefix', type=str, default='data/high_quality',
                        help='Path prefix to the data directory (e.g., "data/high_quality").')
    parser.add_argument('--output_dir', type=str, default='neurons',
                        help='Directory to save the mutual information scores pickle file.')
    parser.add_argument('--domains', nargs='+', type=str, 
                        default=['it', 'law', 'med', 'sub', 'edu', 'spok', 'thes'], # Combined common domains
                        help='List of domains to process (e.g., it law med sub).')
    parser.add_argument('--max_samples_per_domain', type=int, default=1000,
                        help='Maximum number of samples to process per domain to limit computation time.')

    args = parser.parse_args()

    # --- Configuration from Args ---
    MODEL_PATH = args.model_path
    MODEL_TYPE = args.model_type
    SRC_LANG = args.src_lang
    TGT_LANG = args.tgt_lang
    DATA_PATH_PREFIX = args.data_prefix
    OUTPUT_DIR = args.output_dir
    DOMAIN_CODE_LIST = args.domains
    MAX_SAMPLES_PER_DOMAIN = args.max_samples_per_domain

    # Language and domain code mappings (can be extended)
    lang_code_dict = {"en": "English", "de": "German", "zh": "Chinese"}
    domain_code_dict = {
        "it": "IT", "med": "medical", "koran": "Koran", "law": "law",
        "thes": "thesis", "sub": "subtitles", "spok": "spoken",
        "edu": "education", "sci": "science", "blog": "microblog"
    }

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory for MI scores: {OUTPUT_DIR}")

    # --- Load Tokenizer and Model ---
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Ensure pad token is set

    print(f"Loading model from {MODEL_PATH} with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map='auto', torch_dtype=torch.bfloat16)
    model.eval() # Set model to evaluation mode; we only need forward pass for activations

    # --- Identify Target MLP Modules ---
    target_module_names = find_all_target_modules(model)
    if not target_module_names:
        print("No MLP modules found. Exiting.")
        return

    # --- Prepare for Hooking and Data Collection ---
    global forward_cache, backward_cache
    
    # Initialize importance_matrix_dict to store (activation * gradient) for each neuron
    importance_matrix_dict = {
        domain: {module_name: [] for module_name in target_module_names}
        for domain in DOMAIN_CODE_LIST
    }

    hook_forwards, hook_backwards = add_hooks(model, target_module_names)

    domain_data_count = []

    # --- Data Processing and Activation/Gradient Collection ---
    for i, cur_domain_code in enumerate(DOMAIN_CODE_LIST):
        print(f"\nProcessing domain: {cur_domain_code} ({domain_code_dict.get(cur_domain_code, cur_domain_code)})")
        sent_list = read_data(cur_domain_code, SRC_LANG, TGT_LANG, DATA_PATH_PREFIX)
        
        # Limit the number of samples processed per domain
        if MAX_SAMPLES_PER_DOMAIN > 0 and len(sent_list) > MAX_SAMPLES_PER_DOMAIN:
            print(f"Limiting {len(sent_list)} samples to {MAX_SAMPLES_PER_DOMAIN} for domain {cur_domain_code}.")
            sent_list = random.sample(sent_list, MAX_SAMPLES_PER_DOMAIN)
        
        domain_data_count.append(len(sent_list))

        if not sent_list:
            print(f"No data to process for domain: {cur_domain_code}. Skipping.")
            continue

        processed_samples = []
        for line in sent_list:
            # Construct example for prompt
            example = {
                "src": lang_code_dict.get(SRC_LANG, SRC_LANG),
                "tgt": lang_code_dict.get(TGT_LANG, TGT_LANG),
                "sent1": line[0],
                "sent2": line[1],
                "domain": domain_code_dict.get(cur_domain_code, cur_domain_code)
            }
            # Construct prompt and label using the tokenizer for chat template
            prompt_input_text, label = construct_translation_prompt(
                train_example=example,
                prompt_templates=domain_translation_patterns_with_domain_tgt,
                instruction=translate_instruction,
                tokenizer=tokenizer
            )
            processed_samples.append({"input": prompt_input_text, "label": label})

        print(f"Collecting activations and gradients for {len(processed_samples)} samples in {cur_domain_code}...")
        for j in tqdm(range(len(processed_samples))):
            inputs = tokenizer(processed_samples[j]["input"], return_tensors="pt").to(model.device)
            label_tokens = tokenizer(processed_samples[j]["label"], return_tensors="pt").to(model.device)

            input_ids = inputs["input_ids"]
            label_ids = label_tokens["input_ids"]
            
            labels_for_loss = torch.cat([torch.full_like(input_ids, -100), label_ids], dim=1)
            full_input_ids = torch.cat((input_ids, label_ids), dim=1)

            forward_cache = []
            backward_cache = []

            # Forward pass
            outputs = model(input_ids=full_input_ids, labels=labels_for_loss)
            loss = outputs.loss

            # Backward pass
            if loss is not None:
                loss.backward()
            else:
                print(f"Warning: Loss is None for sample {j} in domain {cur_domain_code}. Skipping backward pass.")
                continue

            # Calculate and store (activation * gradient) for each target module
            # Note: backward_cache accumulates in reverse order of forward_cache
            for k, cur_module_name in enumerate(target_module_names):
                # Ensure caches are not empty and indices are valid
                if k < len(forward_cache) and (len(backward_cache) - 1 - k) >= 0:
                    activation = forward_cache[k]
                    gradient = backward_cache[len(backward_cache) - 1 - k] # Get corresponding gradient

                    # Element-wise product of activation and gradient
                    importance_matrix = (activation * gradient).abs()
                    # Reshape and average: Assuming importance_matrix is (batch_size, sequence_length, neuron_dim)
                    # We want to average across batch and sequence length to get per-neuron scores
                    if importance_matrix.dim() >= 2:
                        importance_matrix = importance_matrix.view(-1, importance_matrix.size(-1)).mean(0)
                    else: # Handle cases where it might be 1D or other unexpected shapes
                        importance_matrix = importance_matrix.mean() 

                    importance_matrix_dict[cur_domain_code][cur_module_name].append(importance_matrix.detach().cpu())
                else:
                    print(f"Warning: Cache indexing issue for module {cur_module_name} (k={k}). forward_cache len: {len(forward_cache)}, backward_cache len: {len(backward_cache)}. Skipping.")

            model.zero_grad() # Clear gradients for the next sample
            
        print(f"Finished collecting for domain: {cur_domain_code}. Total samples: {len(sent_list)}")

    # Remove hooks after all data has been processed
    remove_hook(hook_forwards, hook_backwards)
    del forward_cache, backward_cache # Clear global caches

    # --- Mutual Information Calculation ---
    reshape_importance_matrix = {key: [] for key in target_module_names}
    mi_scores_multi_domain = {key: [] for key in target_module_names}

    # Construct domain labels: 0 for first domain, 1 for second, etc.
    # The domain_data_count must align with the order of DOMAIN_CODE_LIST
    domain_labels = np.zeros(sum(domain_data_count), dtype=int)
    current_idx = 0
    for i, count in enumerate(domain_data_count):
        domain_labels[current_idx : current_idx + count] = i # 0-indexed domain labels
        current_idx += count

    if len(np.unique(domain_labels)) < 2:
        print("Warning: Mutual Information requires at least two classes (domains). Check your data or selected domains.")
        print("Skipping MI calculation as only one domain or no data was processed effectively.")
        return # Exit if MI calculation is not possible

    print("\nCalculating Mutual Information scores...")
    for cur_module_name in target_module_names:
        stacked_domain_matrices = []
        for cur_domain_code in DOMAIN_CODE_LIST:
            # Only append if there's data for this domain and module
            if importance_matrix_dict[cur_domain_code][cur_module_name]:
                stacked_domain_matrices.append(torch.stack(importance_matrix_dict[cur_domain_code][cur_module_name]))
            else:
                pass 

        if not stacked_domain_matrices:
            print(f"No importance data collected for module: {cur_module_name}. Skipping MI calculation for this module.")
            mi_scores_multi_domain[cur_module_name] = []
            continue

        temp_importance_matrix = torch.cat(stacked_domain_matrices, dim=0)

        temp_importance_matrix_split = torch.split(temp_importance_matrix, 1, dim=1)
        reshape_importance_matrix[cur_module_name] = [t.squeeze(1) for t in temp_importance_matrix_split]

        mi = []
        for neuron_importance_tensor in reshape_importance_matrix[cur_module_name]:
            neuron_importance_np = neuron_importance_tensor.to(torch.float).numpy()
            
            if len(np.unique(neuron_importance_np)) > 1:
                n_bins = min(10, len(np.unique(neuron_importance_np)) - 1)
                if n_bins > 0:
                    bins = np.quantile(neuron_importance_np, np.linspace(0, 1, n_bins + 1))
                    binned_importance = np.digitize(neuron_importance_np, bins)
                    if len(np.unique(binned_importance)) > 1:
                        mi.append(mutual_info_score(binned_importance, domain_labels))
                    else:
                        mi.append(0.0)
                else:
                    mi.append(0.0)
            else:
                mi.append(0.0)

        print(f"MI scores for module {cur_module_name}: Max={np.max(mi):.4f}, Mean={np.mean(mi):.4f}")
        mi_scores_multi_domain[cur_module_name] = mi
    
    # --- Save MI Scores ---
    mi_scores_file_path = os.path.join(OUTPUT_DIR, f"mi_scores_multi_domain_{MODEL_TYPE}.pkl")
    with open(mi_scores_file_path, 'wb') as f:
        pickle.dump({"mutual_info_dict": mi_scores_multi_domain}, f)
    print(f"\nMutual Information scores saved to {mi_scores_file_path}")

if __name__ == "__main__":
    main()