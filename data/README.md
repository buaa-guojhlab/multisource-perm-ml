# Prediction of Permeability for Porous TPMs via Multi-Source Fusion

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

This repository contains the official implementation of the paper: **"[Permeability Prediction Method for Ablative Porous Materials by Integrating Multi-Source Data]"**.

We propose a deep learning framework to predict the permeability of **porous thermal protection materials (TPMs)** by integrating microstructure images and macroscopic descriptors. The model leverages **DSMC simulations** as ground truth and explores three multi-source fusion strategies:

1.  **Decision-level Fusion**: Fusing predictions from separate image and scalar branches.
2.  **Feature Concatenation**: Concatenating latent features from CNN and MLP encoders.
3.  **Cross-modal Attention**: Using attention mechanisms to capture interactions between microstructure and macro descriptors.

## 📂 Repository Structure

```bash
multisource-perm-ml/
├── models/                     # Neural network definitions
│   ├── base_cnn.py            # 3D CNN for image analysis
│   ├── feature_concat.py      # Feature concatenation model
│   ├── cross_attention.py     # Cross-modal attention model
│   └── decision_fusion.py     # Decision-level fusion models
├── utils/                     # Helper functions and utilities
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── optimization.py        # Hyperparameter optimization with Optuna
│   ├── metrics.py             # Evaluation metrics
│   ├── visualization.py       # Visualization tools
│   ├── logger.py              # Training logger
│   └── model_config.py        # Model configuration management
├── config/                    # Configuration files
├── data/                      # Data directory (to be populated by user)
├── checkpoints/               # Model checkpoints (created during training)
├── physics_models/            # Physics branch models (created by train_physics_branch.py)
├── logs/                      # Training logs (created during training)
├── results/                   # Evaluation results (created during testing)
├── predictions/               # Prediction results (created during testing)
├── optimization_results/      # Hyperparameter optimization results
├── train.py                   # Main training script (image branch)
├── train_physics_branch.py    # Physics branch training script
├── test.py                    # Main testing/evaluation script
├── optimize_models.py         # Hyperparameter optimization script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### 1. Installation

Clone the repository and set up the environment:

```bash
git clone [https://github.com/buaa-guojhlab/multisource-perm-ml.git](https://github.com/buaa-guojhlab/multisource-perm-ml.git)
cd multisource-perm-ml

# We recommend using a Conda environment
conda create -n perm-pred python=3.8
conda activate perm-pred

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

#### Data Availability
The full DSMC simulation dataset used in our paper is **not publicly available** at this time due to ongoing research and intellectual property restrictions.

However, we provide a **minimal sample dataset** in the `data` folder. This allows users to run the code immediately to verify the pipeline and model architecture.

#### Using Your Own Data
To train the model on your own dataset, please organize your files as follows:

1.  **Microstructure Images:** Place images (e.g., `.png`, `.tif`) in a specific directory.
    * Recommended input size: `256x256` (grayscale or RGB).
2.  **Macro Descriptors:** Provide a CSV file containing the descriptors and permeability labels.

**Expected CSV Format:**
```csv
filename,       porosity,  tortuosity,  permeability_label
sample_01.png,  0.45,      1.2,         1.5e-10
sample_02.png,  0.50,      1.1,         2.1e-10
...
```

#### Run Demo
You can run a quick training session using the provided sample data to perform a sanity check:

```bash
python train.py --strategy base --epochs 1 --batch_size 2
```

### 3. Usage

#### Training
You can train the model using different fusion strategies by specifying the `--strategy` argument.

**Option 1: Base physics model**
```bash
# Train physics branch model
python train_physics_branch.py \
    --data_path data/Train_set.xlsx \
    --output_dir physics_models \
    --model_type random_forest

# Generate predictions from physics branch
python train_physics_branch.py --mode predict \
    --model_path physics_models/best_physics_model.pkl \
    --train_data data/Train_set.xlsx \
    --val_data data/Valid_set.xlsx \
    --test_data data/Test_set.xlsx \
    --output_file predictions/physics_predictions.csv
```

**Option 2: Base CNN model**
```bash
# Base CNN model (image-only)
python train.py --strategy base \
    --train_data data/Train_set --train_excel data/Train_set.xlsx \
    --val_data data/Valid_set --val_excel data/Valid_set.xlsx
```

**Option 3: Decision-level Fusion**
```bash
# Train decision-level fusion (requires both branches)
python train.py --strategy decision \
    --fusion_method all \
    --image_model_path checkpoints/best_model_base.pth \
    --physics_model_path predictions/physics_predictions.csv
```

**Option 4: Feature Concat Fusion**
```bash
python train.py --strategy concat \
    --train_data data/Train_set --train_excel data/Train_set.xlsx \
    --val_data data/Valid_set --val_excel data/Valid_set.xlsx
```

**Option 5: Cross-modal Attention Fusion**
```bash
python train.py --strategy attention \
    --train_data data/Train_set --train_excel data/Train_set.xlsx \
    --val_data data/Valid_set --val_excel data/Valid_set.xlsx
```

#### Hyperparameter Optimization
To optimize the hyperparameters of the model, run the following command:
```bash
# Optimize all models
python optimize_models.py \
    --models all \
    --train_data data/Train_set \
    --train_excel data/Train_set.xlsx \
    --val_data data/Valid_set \
    --val_excel data/Valid_set.xlsx \
    --trials 50 \
    --visualize

# Optimize specific model
python optimize_models.py --models attention --trials 30 --visualize

# Use optimized parameters for training
python train.py --strategy attention --use_best_params --params_dir optimization_results/latest_folder
```


#### Evaluation
To evaluate the model on the test set using a trained checkpoint:

```bash
# Test base CNN model
python test.py --strategy base \
    --checkpoint checkpoints/best_model_base.pth \
    --test_data data/Test_set \
    --test_excel data/Test_set.xlsx \
    --save_predictions \
    --plot_results

# Test attention model
python test.py --strategy attention \
    --checkpoint checkpoints/best_model_attention.pth \
    --test_data data/Test_set \
    --test_excel data/Test_set.xlsx \
    --save_predictions \
    --plot_results

# Test decision-level fusion and compare all methods
python test.py --strategy decision \
    --checkpoint fusion_checkpoints/best_fusion_svm.pkl \
    --image_model_path checkpoints/best_model_base.pth \
    --physics_model_path predictions/physics_predictions.csv \
    --compare_all_fusion \
    --save_predictions \
    --plot_results
```

## 📊 Results

| Fusion Strategy | RMSE | MAE | R² Score |
| :--- | :---: | :---: | :---: |
| Decision-level | 0.XXX | 0.XXX | 0.XX |
| Feature Concat | 0.XXX | 0.XXX | 0.XX |
| **Cross-modal Attention** | **0.XXX** | **0.XXX** | **0.XX** |

*(Detailed experimental results and analysis can be found in our paper.)*

## 📜 Citation

If you find this code or our research useful, please cite our paper:

```bibtex
@article{YourName2025Permeability,
  title={Prediction of Permeability for Porous TPMs via Multi-Source Fusion},
  author={Guo, Jinghui and [Student Name] and [Other Authors]},
  journal={[Journal Name]},
  year={2025},
  note={To be published / Preprint available at [Link]}
}
```

## 🛡️ License

This project is released under the **GNU General Public License v3.0 (GPLv3)**.

  * ✅ **Academic & Educational Use:** Highly encouraged.
  * ❌ **Commercial Use:** Commercial usage without open-sourcing your specific application is **NOT** allowed under GPL-3.0.
  * For commercial licensing options, please contact the authors.

## ✉️ Contact

This project is maintained by **Prof. Jinghui Guo's Research Group** at Beihang University (BUAA).

  * **Issues:** For technical questions, please open an [Issue](https://www.google.com/search?q=https://github.com/buaa-guojhlab/multisource-perm-ml/issues).
  * **Lab Website:** []
