# Prediction of Permeability for Porous TPMs via Multi-Source Fusion

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

This repository contains the official implementation of the paper: **"[Permeability Prediction Method for Ablative Porous Materials by Integrat-ing Multi-Source Data]"**.

We propose a deep learning framework to predict the permeability of **porous thermal protection materials (TPMs)** by integrating microstructure images and macroscopic descriptors. The model leverages **DSMC simulations** as ground truth and explores three multi-source fusion strategies:

1.  **Decision-level Fusion**: Fusing predictions from separate image and scalar branches.
2.  **Feature Concatenation**: Concatenating latent features from CNN and MLP encoders.
3.  **Cross-modal Attention**: Using attention mechanisms to capture interactions between microstructure and macro descriptors.

## 📂 Repository Structure

```bash
multisource-perm-ml/
├── assets/             # Images for README (e.g., model architecture)
├── data/
│   └── samples/        # Minimal sample data for sanity check
├── models/             # Neural network definitions (Fusion modules)
├── utils/              # Helper functions for preprocessing and metrics
├── train.py            # Main training script
├── test.py             # Evaluation script
├── requirements.txt    # Python dependencies
└── README.md
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

However, we provide a **minimal sample dataset** in the `data/samples/` folder. This allows users to run the code immediately to verify the pipeline and model architecture.

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
python train.py --data_dir data/samples --batch_size 2 --epochs 1
```

### 3. Usage

#### Training
You can train the model using different fusion strategies by specifying the `--strategy` argument.

**Option 1: Cross-modal Attention (Recommended)**
```bash
python train.py --strategy attention --epochs 100 --batch_size 32
```

**Option 2: Feature Concatenation**
```bash
python train.py --strategy concat --epochs 100 --batch_size 32
```

**Option 3: Decision-level Fusion**
```bash
python train.py --strategy decision --epochs 100 --batch_size 32
```

#### Evaluation
To evaluate the model on the test set using a trained checkpoint:

```bash
python test.py --checkpoint checkpoints/best_model_attention.pth
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

## 🛡️ License

This project is released under the **GNU General Public License v3.0 (GPLv3)**.

  * ✅ **Academic & Educational Use:** Highly encouraged.
  * ❌ **Commercial Use:** Commercial usage without open-sourcing your specific application is **NOT** allowed under GPL-3.0.
  * For commercial licensing options, please contact the authors.

## ✉️ Contact

This project is maintained by **Prof. Jinghui Guo's Research Group** at Beihang University (BUAA).

  * **Issues:** For technical questions, please open an [Issue](https://www.google.com/search?q=https://github.com/buaa-guojhlab/multisource-perm-ml/issues).
  * **Lab Website:** []
