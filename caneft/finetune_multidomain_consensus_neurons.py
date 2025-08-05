import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
from multidomain_dataset import MultiDomainDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Fine-tune a language model with selective multi-domian consensus-algined neuron training.")

    parser.add_argument('--model_name', type=str, default="llms/Qwen2.5-7B-Instruct",
                        help='Hugging Face model ID or local path to the model.')
    parser.add_argument('--src_lang', type=str, default="de",
                        help='Source language code (e.g., "de", "en").')
    parser.add_argument('--tgt_lang', type=str, default="en",
                        help='Target language code (e.g., "en", "de").')
    parser.add_argument('--mi_scores_file', type=str, default="neurons/mi_scores_multi_domain_qwen2_5.pkl",
                        help='Path to the pickle file containing mutual information scores.')
    parser.add_argument('--save_dir', type=str, default="saves",
                        help='Directory to save trained models and TensorBoard logs.')
    parser.add_argument('--update_frequency', type=int, default=5,
                        help='Number of batches after which to perform an optimizer step.')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of training epochs.')
    parser.add_argument('--important_neuron_percentage', type=float, default=0.01,
                        help='Percentage of top MLP neurons to fine-tune (e.g., 0.01 for 1%).')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for data loaders. Note: gradient accumulation is used when batch_size=1 and update_frequency > 1.')

    args = parser.parse_args()

    # Access arguments via args.<argument_name>
    MODEL_NAME = args.model_name
    SRC_LANG = args.src_lang
    TGT_LANG = args.tgt_lang
    MI_SCORES_FILE = args.mi_scores_file
    SAVE_DIR = args.save_dir
    UPDATE_FREQUENCY = args.update_frequency
    NUM_EPOCHS = args.num_epochs
    IMPORTANT_NEURON_PERCENTAGE = args.important_neuron_percentage
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size


    # --- Device Setup ---
    global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    if not global_devices:
        raise RuntimeError("No CUDA devices found. Please ensure you have a compatible GPU and drivers installed.")

    max_memory = {k: '40GB' for k in global_devices}

    # --- Load Tokenizer and Model ---
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # Set pad token

    print(f"Loading model from {MODEL_NAME} with device_map='balanced'...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map='balanced',
        torch_dtype=torch.bfloat16,
        max_memory=max_memory
    )
    model.train()

    # --- Load Mutual Information Scores ---
    print(f"Loading mutual information scores from {MI_SCORES_FILE}...")
    try:
        with open(MI_SCORES_FILE, 'rb') as f:
            data = pickle.load(f)
            importance_matrix_dict = data["mutual_info_dict"]
    except FileNotFoundError:
        print(f"Error: MI scores file not found at {MI_SCORES_FILE}. Please ensure it exists.")
        return
    except Exception as e:
        print(f"Error loading MI scores: {e}")
        return

    # --- Identify Important Neurons ---
    all_values = []
    all_keys = []

    for key, value in importance_matrix_dict.items():
        if "mlp" in key:
            all_values.extend(value)
            all_keys.extend([(key, i) for i in range(len(value))])

    total_neurons = len(all_values)
    num_selected = int(total_neurons * IMPORTANT_NEURON_PERCENTAGE)

    if num_selected == 0:
        print(f"Warning: No neurons selected for fine-tuning. Consider adjusting IMPORTANT_NEURON_PERCENTAGE ({IMPORTANT_NEURON_PERCENTAGE}).")
        print("Exiting as no parameters will be updated.")
        return

    sorted_indices = np.argsort(all_values)[::-1][:num_selected]

    layer_indices = {}
    for idx in sorted_indices:
        layer, neuron_idx = all_keys[idx]
        if layer not in layer_indices:
            layer_indices[layer] = []
        layer_indices[layer].append(neuron_idx)

    print(f"Selected {num_selected} (Top {IMPORTANT_NEURON_PERCENTAGE*100:.2f}%) MLP neurons for fine-tuning across {len(layer_indices)} layers.")

    # --- Optimizer Setup ---
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # --- Dataset and DataLoader Setup ---
    print("Loading training and validation datasets...")
    train_dataset = MultiDomainDataset(
        tokenizer=tokenizer,
        src=SRC_LANG,
        tgt=TGT_LANG,
        split="train"
    )
    valid_dataset = MultiDomainDataset(
        tokenizer=tokenizer,
        src=SRC_LANG,
        tgt=TGT_LANG,
        split="dev"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

    # --- Training Loop ---
    update_counter = 0
    log_dir = os.path.join(SAVE_DIR, f"{MODEL_NAME}_logs", "logs")
    writer = SummaryWriter(log_dir)
    print(f"Logging to TensorBoard directory: {log_dir}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        dataloader_len = len(train_dataloader)
        pbar = tqdm(total=dataloader_len, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} (Train)')

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids[0], attention_mask=attention_mask[0], labels=labels[0])
            loss = outputs.loss

            # Scale loss if accumulating gradients (though for batch_size=1, this is effectively 1)
            loss = loss / UPDATE_FREQUENCY
            loss.backward()

            if (batch_idx + 1) % UPDATE_FREQUENCY == 0 or (batch_idx + 1) == dataloader_len:
                for name, param in model.named_parameters():
                    module_name = ('.').join(name.split('.')[:-1])
                    if module_name in layer_indices:
                        important_neuron_idxs = layer_indices[module_name]
                        if 'weight' in name and param.grad is not None:
                            mask = torch.zeros_like(param.grad)
                            if param.dim() == 2:
                                if param.shape[0] == len(value):
                                    for idx in important_neuron_idxs:
                                        mask[idx, :] = 1
                                elif param.shape[1] == len(value):
                                     for idx in important_neuron_idxs:
                                        mask[:, idx] = 1
                                else:
                                    pass 
                            elif param.dim() == 1:
                                if param.shape[0] == len(value):
                                    for idx in important_neuron_idxs:
                                        mask[idx] = 1
                            param.grad *= mask
                        elif 'bias' in name and param.grad is not None:
                            mask = torch.zeros_like(param.grad)
                            for idx in important_neuron_idxs:
                                if param.dim() == 1 and param.shape[0] == len(value):
                                    mask[idx] = 1
                            param.grad *= mask

                optimizer.step()
                optimizer.zero_grad()

                total_train_loss += (loss * UPDATE_FREQUENCY).item()
                update_counter += 1

                writer.add_scalar("Loss/Train", (loss * UPDATE_FREQUENCY).item(), update_counter)
                pbar.set_postfix({"Loss": f"{(loss * UPDATE_FREQUENCY).item():.4f}"})

            pbar.update(1)

        pbar.close()

        avg_train_loss = total_train_loss / update_counter if update_counter > 0 else 0
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} Training complete. Average Loss: {avg_train_loss:.4f}")

        # --- Validation Loop ---
        model.eval()
        total_val_loss = 0
        pbar_val = tqdm(total=len(valid_dataloader), desc=f'Epoch {epoch+1}/{NUM_EPOCHS} (Validation)')
        with torch.no_grad():
            for batch_idx_val, batch_val in enumerate(valid_dataloader):
                input_ids_val = batch_val['input_ids'].to(model.device)
                attention_mask_val = batch_val['attention_mask'].to(model.device)
                labels_val = batch_val['labels'].to(model.device)

                outputs_val = model(input_ids=input_ids_val[0], attention_mask=attention_mask_val[0], labels=labels_val[0])
                total_val_loss += outputs_val.loss.item()
                pbar_val.update(1)
        pbar_val.close()

        avg_val_loss = total_val_loss / len(valid_dataloader)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Validation complete.")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- Save Model ---
        model_save_path = os.path.join(SAVE_DIR, f"{MODEL_NAME}_all_domain_mlp_epoch_{epoch+1}")
        os.makedirs(model_save_path, exist_ok=True)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Model and tokenizer saved to {model_save_path}")

    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    main()