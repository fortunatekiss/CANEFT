# README

## Overview

This rep for paper "Consensus-Aligned Neuron Efficient Fine-Tuning Large Language Models for Multi-Domain Machine Translation".

## Dataset

We use two public datasets, Aharoni and Goldberg (2020) for German-English translation and UM-Corpus (Tian et al., 2014) for Chinese-English translation. We sampled 10k from the training set and 2k for evaluation set. We provide the extracted training set and validation set in data. 

You can also download the full dataset from the following link:

German-English: https://github.com/roeeaharoni/unsupervised-domain-clusters

Chinese-English: http://nlp2ct.cis.umac.mo/um-corpus/index.html

## Scripts

The following scripts are provided to perform various tasks:

### 1. Multi-Domain Consensus-Aligned Neuron Selection

To select neurons, run the following command:

```bash
python find_multidomain_consensus_neurons.py
```

### 2. Neuron-Efficient Fine-Tuning

To fine-tuning the selected multi-domain consensus-aligned neurons, run the following command:

```bash
python finetune_multidomain_consensus_neurons.py
```

### 3. Inference

To inference on multi-domain translation task, run the following command:

```bash
python evaluate.py
```

## Usage

1. **Download the dataset**: Follow the link provided to download the Natural Instructions dataset.
2. **Run the scripts**: Use the provided commands to select neurons, fine-tuning and inference.

Ensure you have the necessary dependencies installed before running the scripts. You can install the dependencies using the following command:

```bash
pip install -r requirements.txt
```