# CANEFT (Consensus-Aligned Neuron Efficient Fine-Tuning)

## Overview

This repository implements the method proposed in the paper "Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation". The approach identifies and fine-tunes the most important neurons for multi-domain translation tasks, achieving efficient model adaptation.

## Key Features

- **Multi-Domain Support**: Handles translations across IT, medical, legal, subtitles, and other specialized domains
- **Efficient Fine-Tuning**: Only fine-tunes the most important neurons, reducing computational costs
- **Dual Language Pair Support**: Supports German-English and Chinese-English translation
- **Consensus Alignment**: Identifies consensus neurons important for multi-domain translation through mutual information analysis

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── caneft/                  # Main package directory
│   ├── __init__.py         # Package initialization
│   ├── evaluate.py         # Evaluation script
│   ├── find_multidomain_consensus_neurons.py    # Neuron identification
│   ├── finetune_multidomain_consensus_neurons.py # Neuron fine-tuning
│   └── multidomain_dataset.py  # Dataset handling
├── data/                   # Preprocessed multi-domain translation data
├── llms/                   # Pre-trained language models directory
├── neurons/               # Neuron analysis results directory
└── requirements.txt       # Project dependencies
```

## Dataset

We use two public datasets:

1. **German-English Translation Dataset**: From Aharoni and Goldberg (2020)
   - Download: https://github.com/roeeaharoni/unsupervised-domain-clusters

2. **Chinese-English Translation Dataset**: From UM-Corpus (Tian et al., 2014)
   - Download: http://nlp2ct.cis.umac.mo/um-corpus/index.html

Dataset preprocessing:
- Training set: 10k samples per domain
- Evaluation set: 2k samples per domain
- Preprocessed data available in the `data` directory

## Usage

### 1. Identify Multi-Domain Consensus-Aligned Neurons

```bash
python -m caneft.find_multidomain_consensus_neurons \
    --model_path llms/Qwen2.5-7B-Instruct \
    --model_type qwen2.5 \
    --src_lang de \
    --tgt_lang en \
    --data_prefix data \
    --output_dir neurons
```

### 2. Neuron-Efficient Fine-Tuning

```bash
python -m caneft.finetune_multidomain_consensus_neurons \
    --model_name llms/Qwen2.5-7B-Instruct \
    --src_lang de \
    --tgt_lang en \
    --mi_scores_file neurons/mi_scores_multi_domain_qwen2_5.pkl \
    --save_dir saves \
    --num_epochs 2 \
    --important_neuron_percentage 0.01
```

### 3. Model Evaluation

```bash
python -m caneft.evaluate \
    --model_path saves/Qwen2.5-7B-Instruct_all_domain_mlp_epoch_2 \
    --model_type qwen2.5 \
    --src_lang de \
    --tgt_lang en \
    --data_dir data \
    --output_prefix trans_results
```

## Key Parameters

- `--model_type`: Supported model types (llama2, llama3, qwen2.5)
- `--src_lang`/`--tgt_lang`: Source/target language codes
- `--important_neuron_percentage`: Percentage of important neurons to fine-tune
- `--num_epochs`: Number of training epochs
- `--temperature`: Sampling temperature for generation

## Citation

If you use this code, please cite our paper:

```bibtex
@article{CANEFT2024,
  title={Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation},
  author={},
  journal={},
  year={2025}
}
```