# OmicReasonNet

## Overview

OmicReasonNet is a computational framework for cancer subtyping using single-omics data by fusing topological and semantic priors. It integrates statistical priors from biological networks and semantic priors from large language models to enhance classification accuracy and interpretability.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Pandas
- SciPy
- scikit-learn
- openpyxl
- xlrd

Install dependencies using:
```
pip install torch numpy pandas scipy scikit-learn openpyxl xlrd
```

## Directory Structure

- `models.py`: Defines the neural network models including GraphConvolution, View_Encoder, View_specific_Classifier, and VCDN.
- `utils.py`: Utility functions for data processing, graph construction, and model handling.
- `opt.py`: Optimizer class for training.
- `layers.py`: InnerProductDecoder layer.
- `clac_metric.py`: Functions for calculating evaluation metrics.
- `clr.py`: Cyclic learning rate scheduler.
- `disease_subtypes_classification.py`: Script for training and testing the model on disease subtypes.
- `feat_importance.py`: Script for calculating feature importance.
- `important_biomarker_identification.py`: Script for identifying important biomarkers.

## Data Preparation

- Place association data in `./Association Data/` (e.g., `mRNAnumbers.xlsx`, `Association matrix.xlsx`).
- Place representation data in `./Representation Data/{dataset}/` (e.g., `samples_mRNA.xlsx`, `labels.xlsx`, `feature_name.xlsx`).

## Usage

### Disease Subtypes Classification

Run `disease_subtypes_classification.py` to train and evaluate the model.

Adjust parameters like `data_folders`, `view_list`, epochs, learning rates in the script.

### Feature Importance

Run `feat_importance.py` to compute feature importance.

Specify `data_folder`, `model_folder`, `view_list`, `num_class`.

### Important Biomarker Identification

Run `important_biomarker_identification.py` to identify biomarkers.

Adjust `data_folder`, file paths as needed.

## Training and Testing

The framework uses 5-fold cross-validation. Models are pretrained and then fine-tuned.

Evaluation metrics: Accuracy, F1-score, AUC.

## License

MIT License
