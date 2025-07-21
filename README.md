# 🚀 TabPFN Benchmark Suite

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Required-76b900.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)

*A comprehensive benchmarking framework for evaluating TabPFN against traditional gradient boosting methods on tabular data*

[🔬 Research Paper](https://www.nature.com/articles/s41586-024-08328-6) • [📊 Results](#results) • [🛠️ Installation](#installation) • [🚀 Quick Start](#quick-start)

</div>

---

## 🎯 Overview

This project provides a robust benchmarking framework to evaluate the performance of **TabPFN** (Tabular Prior-Data Fitted Networks) against traditional gradient boosting methods on a variety of tabular datasets. TabPFN represents a paradigm shift in tabular data modeling by using transformer-based neural networks that can make predictions without traditional training.

### 🔍 What is TabPFN?

TabPFN is a revolutionary approach to tabular data prediction that:
- **No training required**: Makes predictions directly using a pre-trained transformer
- **Few-shot learning**: Works effectively with small datasets
- **Automatic feature engineering**: Handles mixed data types automatically
- **Fast inference**: No hyperparameter tuning needed for basic usage

### 🏆 Benchmarked Models

| Model | Type | Strengths |
|-------|------|-----------|
| **TabPFN** | Neural Network | Zero-shot learning, handles small datasets well |
| **XGBoost** | Gradient Boosting | Robust, well-established, excellent performance |
| **LightGBM** | Gradient Boosting | Fast training, memory efficient |
| **CatBoost** | Gradient Boosting | Handles categorical features automatically |

## 🛠️ Installation

### Prerequisites

- **Python 3.10+**
- **CUDA-compatible GPU** (required for optimal performance)
- **CUDA toolkit** installed

### 📦 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/tabpfn-bench.git
   cd tabpfn-bench
   ```

2. **Install PyTorch with CUDA support**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install remaining dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify CUDA availability**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")
   ```

## 🚀 Quick Start

### 🔄 Full Benchmark Suite

Run both classification and regression benchmarks:

```bash
python main.py
```

### 📊 Classification Only

```bash
python clf_main.py
```

### 📈 Regression Only

```bash
python reg_main.py
```


## 📊 Results

Results are automatically saved to:
- `results/classification_results.json` - Classification benchmark results
- `results/regression_results.json` - Regression benchmark results

### 📋 Metrics Evaluated

#### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Informedness**: Sensitivity + Specificity - 1
- **Markedness**: Precision + NPV - 1
- **Matthews Correlation Coefficient**: Balanced measure for imbalanced datasets

#### Regression Metrics
- **MAE**: Mean Absolute Error
- **NMAE**: Normalized Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **NRMSE**: Normalized Root Mean Square Error
- **R²**: Coefficient of determination
- **Adjusted R²**: R² adjusted for number of features

### 📈 Example Results Structure

```json
{
  "1": {
    "TabPFNClassifier_default": {
      "accuracy": 0.9234,
      "precision": 0.9145,
      "recall": 0.9198,
      "f1": 0.9171,
      "roc_auc": 0.9756,
      "informedness": 0.8432,
      "markedness": 0.8421,
      "matthews_corrcoef": 0.8456
    },
    "TabPFNClassifier_tuned": {
      "accuracy": 0.9267,
      "precision": 0.9189,
      "recall": 0.9234,
      "f1": 0.9211,
      "roc_auc": 0.9789,
      "informedness": 0.8523,
      "markedness": 0.8512,
      "matthews_corrcoef": 0.8534
    }
  }
}
```

## 🗂️ Project Structure

```
📁 tabpfn-bench/
├── 📁 results/                    # Benchmark results
│   ├── classification_results.json
│   └── regression_results.json
├── 📁 tools/                      # Core utilities
│   ├── 📁 tabular_metrics/        # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── classification.py      # Classification metrics
│   │   └── regression.py          # Regression metrics
│   ├── benchmark_classification.py # Classification benchmarking
│   ├── benchmark_regression.py    # Regression benchmarking
│   ├── constants.py               # Hyperparameter spaces
│   ├── dataset.py                 # Dataset loading utilities
│   ├── hyperparameter_tuning.py   # Hyperopt integration
│   └── preprocess.py              # Data preprocessing
├── clf_main.py                    # Classification entry point
├── reg_main.py                    # Regression entry point
├── main.py                        # Full benchmark suite
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## ⚙️ Configuration

### 🎛️ Hyperparameter Tuning

The framework uses **Hyperopt** with Tree-structured Parzen Estimator (TPE) for hyperparameter optimization:

- **Default tuning time**: 4 hours per model
- **Search space**: Defined in `tools/constants.py`
- **Optimization metric**: Log-loss for classification, MSE for regression
- **Cross-validation**: 80/20 train/validation split

### 🗃️ Datasets

The benchmark uses curated datasets from **OpenML**:

#### Classification Datasets (29 datasets)
- Binary and multi-class problems
- Various sizes and feature types
- Real-world and synthetic datasets

#### Regression Datasets (28 datasets)
- Continuous target variables
- Mixed feature types
- Different complexity levels

### 🔧 Preprocessing Pipeline

1. **Imputation**: Mean for numerical, mode for categorical
2. **Encoding**: One-hot encoding for categorical features
3. **Scaling**: Min-Max normalization
4. **Feature handling**: Automatic detection of categorical features


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this benchmarking framework in your research, please cite:

```bibtex
@article{hollmann2025tabpfn,
    title={Accurate predictions on small data with a tabular foundation model},
    author={Hollmann, Noah and M{\"u}ller, Samuel and Purucker, Lennart and Krishnakumar, Arjun and K{\"o}rfer, Max and Hoo, Shi Bin and Schirrmeister, Robin Tibor and Hutter, Frank},
    journal={Nature},
    volume={637},
    pages={319--326},
    year={2025},
    month={01},
    day={09},
    doi={10.1038/s41586-024-08328-6},
    publisher={Springer Nature},
    url={https://www.nature.com/articles/s41586-024-08328-6}
}
```

## 🙏 Acknowledgments

- **TabPFN Team** for the revolutionary approach to tabular data
- **OpenML** community for providing diverse datasets
- **Scikit-learn**, **XGBoost**, **LightGBM**, and **CatBoost** teams for excellent ML libraries
- **PyTorch** team for the deep learning framework

---

<div align="center">

**⭐ Star this repository if you find it useful!**


</div>