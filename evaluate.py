import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList
from tqdm import tqdm
import json
import random
import argparse
from typing import List, Dict, Tuple, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from StopAtSpecificTokenCriteria import StopAtSpecificTokenCriteria
except ImportError:
    logging.warning("StopAtSpecificTokenCriteria not found. Using a dummy class. "
                    "Please ensure StopAtSpecificTokenCriteria.py is in your PYTHONPATH or directory.")
    from transformers.generation.stopping_criteria import StoppingCriteria
    class StopAtSpecificTokenCriteria(StoppingCriteria):
        def __init__(self, token_id_list: List[int]):
            self.token_id_list = token_id_list

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            return any(input_ids[0, -1].item() == token_id for token_id in self.token_id_list)


# Global constants for language and domain mappings
LANG_CODE_DICT = {"en": "English", "de": "German", "zh": "Chinese"}
DOMAIN_CODE_DICT = {
    "it": "IT", "med": "medical", "koran": "Koran", "law": "law",
    "the": "thesis", "sub": "subtitles", "sci": "science", "blog": "microblog",
    "edu": "education", "spok": "spoken" # Add other domains if they exist
}

def read_data(domain: str, src: str, tgt: str, data_dir: str) -> List[Dict[str, str]]:
    """
    Reads translation test data from a JSON file.
    Expected format: List of dicts, each with "input", "output", "instruction".
    """
    file_path = os.path.join(data_dir, f"{domain}.test.{src}-{tgt}.instruct.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sent_list = json.load(f)
        logging.info(f"Loaded {len(sent_list)} samples for domain '{domain}' from {file_path}")
        return sent_list
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}. Skipping domain '{domain}'.")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}. Skipping domain '{domain}'.")
        return []
    except Exception as e:
        logging.error(f"An error occurred while reading {file_path}: {e}. Skipping domain '{domain}'.")
        return []

def construct_translation_prompt_with_template(
    input_text: str,
    instruction_text: str,
    tokenizer: AutoTokenizer,
    add_generation_prompt: bool = True
) -> Tuple[str, str]:
    messages = []
    if instruction_text:
        messages.append({"role": "system", "content": instruction_text.strip("\n")})

    messages.append({"role": "user", "content": input_text.strip("\n")})

    # Apply tokenizer's chat template to get the final prompt string
    full_prompt_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=add_generation_prompt
    )
    return full_prompt_text

def write_data(output_lines: List[str], data_path: str):
    try:
        with open(data_path, "w", encoding="utf-8") as f:
            for line in output_lines:
                f.write(line + "\n")
        logging.info(f"Results saved to {data_path}")
    except Exception as e:
        logging.error(f"Failed to write data to {data_path}: {e}")

def get_stop_criteria_for_model(model_type: str) -> List[int]:
    if model_type == "qwen2.5":
        return [151645] # <|im_end|>
    elif model_type == "llama3":
        return [128009] # <|eot_id|>
    elif model_type == "llama2":
        return [29889]
    else:
        logging.warning(f"Unknown model type '{model_type}'. Using default EOS token ID (if tokenizer has one).")
        return []

def main():
    parser = argparse.ArgumentParser(description="Run inference on a language model for translation tasks across domains.")
    parser.add_argument('--src_lang', type=str, default='de',
                        help='Source language code (e.g., "de", "en").')
    parser.add_argument('--tgt_lang', type=str, default='en',
                        help='Target language code (e.g., "en", "de").')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the fine-tuned Hugging Face model or Hugging Face model ID.')
    parser.add_argument('--model_type', type=str, default="qwen2.5",
                        choices=["llama2", "llama3", "qwen2.5"],
                        help='Model type: llama2, llama3, qwen2.5 (influences stopping criteria and output parsing).')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing domain-specific test JSON files (e.g., data/it.test.de-en.instruct.json).')
    parser.add_argument('--output_prefix', type=str, default='trans_results',
                        help='Prefix directory for saving translation results.')
    parser.add_argument('--domains', nargs='+', type=str,
                        default=['it', 'koran', 'law', 'med', 'the', 'edu', 'spok'],
                        help='List of domains to evaluate (e.g., it law med).')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference. Note: currently optimized for batch_size=1 due to prompt handling.')
    parser.add_argument('--temperature', type=float, default=0.0001,
                        help='Sampling temperature for generation. Use 0 for greedy decoding (or very small for consistency).')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='Maximum number of new tokens to generate.')

    args = parser.parse_args()

    # --- Setup Paths and Model ---
    os.makedirs(args.output_prefix, exist_ok=True)
    logging.info(f"Results will be saved in: {args.output_prefix}")

    logging.info(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"Loading model from {args.model_path} with device_map='auto'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map='auto', 
        torch_dtype=torch.bfloat16
    )
    model.eval()

    # --- Setup Stopping Criteria ---
    stop_token_ids = get_stop_criteria_for_model(args.model_type)
    stopping_criteria_list = StoppingCriteriaList([StopAtSpecificTokenCriteria(token_id_list=stop_token_ids)])
    if not stop_token_ids:
        logging.warning("No specific stopping token IDs configured. Generation might continue until max_new_tokens or EOS token.")
    else:
        logging.info(f"Using stopping token IDs: {stop_token_ids} for model type: {args.model_type}")

    # --- Inference Loop ---
    for domain_code in args.domains:
        logging.info(f"\nProcessing domain: {domain_code} ({DOMAIN_CODE_DICT.get(domain_code, domain_code)})")
        
        # Read data for current domain
        sent_list = read_data(domain_code, args.src_lang, args.tgt_lang, args.data_dir)
        if not sent_list:
            continue

        processed_prompts: List[str] = []
        original_inputs_for_parsing: List[str] = []

        # Pre-process all prompts
        for i, item in enumerate(sent_list):
            try:
                full_prompt_text = construct_translation_prompt_with_template(
                    input_text=item["input"],
                    instruction_text=item["instruction"],
                    tokenizer=tokenizer
                )
                processed_prompts.append(full_prompt_text)

                base_prompt_for_comparison = construct_translation_prompt_with_template(
                    input_text=item["input"],
                    instruction_text=item["instruction"],
                    tokenizer=tokenizer,
                    add_generation_prompt=False
                )
                original_inputs_for_parsing.append(base_prompt_for_comparison)

            except Exception as e:
                logging.error(f"Error processing prompt for sample {i} in domain {domain_code}: {e}. Skipping.")
                processed_prompts.append("") 
                original_inputs_for_parsing.append("")


        generated_outputs_raw: List[str] = []
        for j in tqdm(range(0, len(processed_prompts), args.batch_size), desc=f"Generating ({domain_code})"):
            batch_prompts = processed_prompts[j:j + args.batch_size]
            
            batch_prompts = [p for p in batch_prompts if p]
            if not batch_prompts:
                continue

            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    stopping_criteria=stopping_criteria_list,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=args.temperature > 0.0001,
                )

            for k in range(outputs.shape[0]):
                input_length = inputs.input_ids[k].shape[0]
                generated_tokens_only = outputs[k, input_length:]
                clean_output = tokenizer.decode(generated_tokens_only, skip_special_tokens=True)
                
                # Further cleanup
                clean_output = clean_output.replace("\n", " ").replace("\r", " ").strip()
                
                for stop_id in stop_token_ids:
                    stop_str = tokenizer.decode([stop_id], skip_special_tokens=True)
                    if stop_str:
                         clean_output = clean_output.replace(stop_str, "").strip()
             
                clean_output = clean_output.lstrip(':').lstrip().strip()
                
                generated_outputs_raw.append(clean_output)

        output_file_name = f"{domain_code}_{args.src_lang}-{args.tgt_lang}_mlp_fine_tuned_{args.model_type}_temp{args.temperature}.txt"
        if args.model_path and "epoch" in args.model_path:
            epoch_info = "_".join([part for part in args.model_path.split(os.sep) if "epoch" in part])
            output_file_name = output_file_name.replace(".txt", f"_{epoch_info}.txt")
        
        domain_output_path = os.path.join(args.output_prefix, output_file_name)
        write_data(generated_outputs_raw, domain_output_path)

    logging.info("Inference complete for all domains.")

if __name__ == "__main__":
    main()