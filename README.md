# Self-supervised Explanatory Transcriptomics Encoder(SETE) for Downstream and Transfer Learning

## Overview
SETE-SSL extends the original SETE framework by integrating self-supervised learning (SSL), classification modules, and transfer learning modules.  
The model learns biologically interpretable representations that generalize across datasets and tasks.

The model preserves the **visible neural network hierarchy** defined by Gene Ontology (GO) terms—each neuron corresponds to a biological subsystem—and extends SETE through three major stages:

1. **Self-Supervised Pretraining (Masked Input Reconstruction):**  
   A subset of input gene expression features is randomly masked, and the model learns to reconstruct the original input from the remaining visible genes.  
   This reconstruction-based self-supervised learning enables the network to capture biologically meaningful and robust representations without relying on labeled data, forming a pretrained encoder.

2. **Downstream Classification:**  
   The pretrained encoder is evaluated on classification tasks under three different configurations:  
   - **Benchmark Model:** trained from scratch without any pretraining.  
   - **Fine-Tuning Model:** initialized with the pretrained encoder, with all parameters jointly optimized.  
   - **Linear Probing Model:** the pretrained encoder is frozen, and only the final classifier layers are trained.  
   These experiments evaluate the transferability and discriminative power of the hierarchical embeddings learned during the SSL stage.

3. **Cross-Dataset Transfer Learning:**  
   To assess the generalization capability of the pretrained model, the encoder trained on TCGA data is directly transferred to the GTEx dataset.  
   The model demonstrates strong cross-dataset performance, showing that representations learned from self-supervised input reconstruction can effectively generalize to unseen biological domains.

---

## Key Features

| Feature | Description |
|----------|--------------|
| **Visible Hierarchical Encoder** | Preserves the original SETE architecture in which each neuron corresponds to a biological subsystem defined by Gene Ontology (GO) terms, maintaining interpretability throughout the network. |
| **Masked Input Reconstruction (SSL Pretraining)** | Implements a self-supervised learning strategy by randomly masking a portion of input gene expression features and reconstructing the original input. This enables the model to learn biologically meaningful and noise-tolerant representations without labeled supervision. |
| **Downstream Multi-Strategy Evaluation** | Evaluates pretrained embeddings on classification tasks under three training schemes: (1) benchmark model trained from scratch, (2) fine-tuning model with the pretrained encoder jointly optimized, and (3) linear probing model with frozen encoder parameters. |
| **Cross-Dataset Transfer Learning** | Applies the pretrained model to the GTEx dataset for cross-domain validation, demonstrating strong generalization of the learned gene-expression representations beyond TCGA data. |
| **Interpretability and Biological Insight** | Provides GO-term–level attention weights and reconstruction relevance maps, allowing visualization of pathway contributions and subsystem dependencies involved in specific predictions. *Note: this interpretability module is planned to be implemented using a Graph Attention Network (GAT) and is not yet included in this version of the rep*


---

## Environment Installation
Please install all required packages according to the versions listed in **`requirements.txt`**.  
To set up your environment:

```bash
python3 -m venv py38
source py38/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Structure and File Description

After setting up the environment, the project directory should contain the following key folders and scripts:

### 1. Gene Expression Data and Ontology Files (`data/`)

This project focuses exclusively on **gene expression modeling** (no drug or mutation inputs are used).  
The `data/` directory contains all necessary expression matrices, gene index mappings, and ontology definitions used throughout SSL pretraining, classification, and transfer learning.

Key files include:

- **gene2ind.txt / gene_dict.csv** — mapping from gene symbols to numerical indices used to construct model input tensors.  
- **gtex_data.csv / gtex_df_filtered.csv** — preprocessed GTEx expression matrices (samples × genes), used for transfer learning and cross-domain evaluation.  
- **tcga_mad_genes.tsv / tcga_sample_counts.tsv / tcga_sample_identifiers.tsv** — TCGA RNA-seq processed data and metadata for SSL pretraining and downstream classification.  
- **gtex_mad_genes.tsv / gtex_sample_counts.tsv** — normalized GTEx RNA-seq data for downstream and transfer tasks.  
- **train_tcga_expression_matrix_processed.tsv.gz / test_tcga_expression_matrix_processed.tsv.gz** — compressed training and test gene expression matrices for TCGA.  
- **train_gtex_expression_matrix_processed.tsv.gz / test_gtex_expression_matrix_processed.tsv.gz** — compressed matrices for GTEx.  
- **drugcell_ont.txt** — Gene Ontology hierarchical definition file (used to build the visible neural network structure).  

> Note:  
> - No drug fingerprint or mutation files are required for this version.  
> - Ensure all file names and directory paths remain consistent with the provided examples, as they are directly referenced in the scripts.
> - Due to the large file sizes, the files in `data` directories are stored in the project’s Release section instead of being included directly in the repository.

---

### 2. Core Functional Scripts (`codes/`)

The `codes/` directory contains all base modules for the **SETE** implementation:

- **drugcell_NN.py** — defines the hierarchical visible neural network encoder based on GO-term structure.  
- **train_drugcell.py** — trains the encoder using the provided gene expression inputs and GO structure.  
- **predict_drugcell.py / predict_drugcell_cpu.py** — perform forward inference with the trained encoder (GPU/CPU versions).  
- **tf_test.py** — optional testing script for verifying model consistency and tensor outputs.  
- **utils/** — helper utilities for data loading, normalization, and evaluation metric computation.


---

### 3. SSL Pretraining, Classification Models, and Random Graph Generation
At the top level of the repository:

- **`train_SSL.ipynb`** — scripts for self-supervised pretraining.  
  These mask a subset of gene expression features and train the model to reconstruct the original input, generating the pretrained model **`model_032_updated.pt`**.  

- **`Classification_model.py` / `Classification_model_GTEx.py`** — downstream classification modules for TCGA and GTEx datasets.  
  Support three experimental setups:
  1. **Benchmark** (trained from scratch)  
  2. **Fine-Tuning** (initialized from pretrained encoder)  
  3. **Linear Probing** (frozen encoder, train only classifier head)

- **`Generate_random_graph.ipynb`** — constructs random DAG structures preserving degree distributions to serve as control experiments.  
  Generated random graphs are stored in the `random_graphs/` directory.

> **Note:**  
> All automatic model-saving operations within training scripts are currently **commented out** (marked with `#`) to avoid accidental overwriting.  
> To enable saving new models, simply uncomment the corresponding `torch.save()` or equivalent lines in the training scripts.

---

### 4. Transfer Learning to GTEx
Transfer learning experiments evaluate cross-domain generalization from cancer transcriptomics (TCGA) to normal tissue expression (GTEx):

- **`transfer_tcga_gtex_fine_tuning.ipynb`** — fine-tunes the pretrained TCGA encoder on GTEx classification.  
- **`transfer_tcga_gtex_linear_probing.ipynb`** — evaluates frozen encoder transferability.  
- **`transfer_tcga_gtex_benchmark.ipynb`** — trains a GTEx model from scratch as baseline.  

These experiments demonstrate how representations learned through self-supervised input reconstruction transfer effectively to unseen biological domains.

> **Note:**  
> All automatic model-saving operations within training scripts are currently **commented out** (marked with `#`) to avoid accidental overwriting.  
> To enable saving new models, simply uncomment the corresponding `torch.save()` or equivalent lines in the training scripts.

---

### 5. Saved Models and Results
The **`saved_models_and_results/`** directory contains all stored models and visualizations generated during the training and testing process.

Key contents include:

- **Trained Models:**  
  Models of benchmark, fine-tuning, and linear probing experiments, saved during or after training.  
  These trained models can be reused for further evaluation, transfer learning, or downstream analyses.

- **Visualization Utilities (`tcga_draw.py`):**  
  A dedicated script for plotting training curves, accuracy trajectories, and comparative visualizations between different models.
  
> Note:  
> - No drug fingerprint or mutation files are required for this version.  
> - Ensure all file names and directory paths remain consistent with the provided examples, as they are directly referenced in the scripts.
> - Due to the large file sizes, the saved models of `Saved Models and Results` directories are stored in the project’s Release section instead of being included directly in the repository.

---

### 6. Model Evaluation and Visualization (`tcga_classification_test.ipynb`)
The **`tcga_classification_test.ipynb`** script provides a unified evaluation framework for analyzing stored models and visualizing learned representations.

**Key functionalities include:**

- **Model Evaluation:**  
  Automatically loads pretrained or fine-tuned models from the `saved_models_and_results/` directory and computes classification metrics such as accuracy, recall, and F1-score.

- **t-SNE Embedding Visualization:**  
  Generates two-dimensional t-SNE projections of **GO-term–level embeddings**, illustrating how different cancer or tissue types cluster in latent space.

- **Flexible Model Selection:**  
  Users can directly modify the variable **`model_paths`** in the script to specify which trained models to evaluate or visualize.  
  Multiple model checkpoints can be listed for comparative analysis between benchmark, fine-tuning, and distillation settings.

> **Tip:**  
> This script is particularly useful for validating how hierarchical representations transfer across datasets (e.g., TCGA → GTEx) and for visually assessing the separability of biological categories in the learned embedding space.


---

### 7. Common Utilities and Analysis Notebooks
- **`tcga_classification_*` notebooks** — run benchmark, fine-tuning, and linear probing tasks on TCGA.  
- **`random_tcga_*` notebooks** — evaluate performance on random graph variants for structural ablation.  
- **`Data_preparation_TCGA.ipynb` / `Data_preparation_GTEx.ipynb`** — preprocess datasets, normalize expression matrices, and prepare input tensors for SSL or classification.

---

>  **Summary:**  
> The repository provides a modular pipeline integrating self-supervised input reconstruction, downstream supervised learning, random structural controls, and transfer learning between TCGA and GTEx.  
> The pretrained model (`model_032_updated.pt`) can be reused for other biological datasets following the same gene indexing scheme.
