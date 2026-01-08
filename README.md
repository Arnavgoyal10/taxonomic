# Taxonomic Classification ML Pipeline

A production-ready Machine Learning pipeline for predicting taxonomic **Order** and **Class** of organisms based on their protein sequences.

## 🎯 Overview

This project implements **six different classification approaches**:

### Traditional Machine Learning:
1. **Naive Bayes** - K-mer count vectorization (fast baseline)
2. **Logistic Regression** - TF-IDF k-mer features (scalable linear model)
3. **Random Forest** - Advanced feature engineering (robust ensemble)
4. **XGBoost** - Gradient boosting (state-of-the-art traditional ML)

### Deep Learning:
5. **1D-CNN** - Multi-scale convolutions (pattern recognition)
6. **BiLSTM** - Bidirectional LSTM (sequence dependencies)

## 📁 Project Structure

```
taxinomic/
├── config.py                       # Central configuration file
├── taxon.csv                       # Input metadata (tab-separated)
│
├── 01_data_prep.py                 # Data preparation & sequence fetching
│
├── 02_train_naive_bayes.py        # Naive Bayes (k-mer counts)
├── 03_train_random_forest.py      # Random Forest (feature engineering)
├── 04_train_cnn.py                 # 1D-CNN (deep learning)
├── 05_train_logistic_regression.py # Logistic Regression (TF-IDF)
├── 06_train_xgboost.py            # XGBoost (gradient boosting)
├── 07_train_bilstm.py             # BiLSTM (sequence modeling)
│
├── requirements.txt               # Python dependencies
├── INSTALL.md                     # Installation guide
├── README.md                      # This file
├── .gitignore                     # Git ignore rules
│
├── models/                        # Saved models (generated)
├── results/                       # Evaluation results (generated)
├── plots/                         # Confusion matrices & plots (generated)
└── logs/                          # TensorBoard logs (generated)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

**IMPORTANT:** Before running any scripts, edit `config.py` and set your NCBI credentials:

```python
# In config.py
NCBI_EMAIL = "your.email@example.com"  # REQUIRED
NCBI_API_KEY = "your_api_key_here"     # OPTIONAL but recommended
```

- **Email**: Required by NCBI's usage policy
- **API Key**: Optional but increases rate limit from 3 to 10 requests/sec
  - Get one at: https://www.ncbi.nlm.nih.gov/account/

### 3. Data Preparation

Run the data preparation script to:
- Clean the metadata file
- Filter low-frequency categories
- Fetch protein sequences from NCBI

```bash
python 01_data_prep.py
```

**Output:** `master_dataset.csv` containing sequences and labels

**Note:** This step may take several hours for large datasets due to NCBI rate limits.

### 4. Train Models

Train all models (each trains both Order and Class classifiers):

```bash
# Traditional ML (CPU-friendly)
python 02_train_naive_bayes.py       # Fastest, good baseline
python 05_train_logistic_regression.py  # Fast, scalable
python 03_train_random_forest.py     # Slower, robust
python 06_train_xgboost.py          # Slower, best traditional ML

# Deep Learning (GPU recommended)
python 04_train_cnn.py              # 1D Convolutions
python 07_train_bilstm.py          # Bidirectional LSTM
```

## 🎛️ Hyperparameter Tuning

All hyperparameters are centralized in `config.py`. Key parameters:

### Data Preparation
- `MIN_SAMPLES_THRESHOLD`: Minimum samples per Order (default: 10)
- `MAX_SEQS_PER_ORGANISM`: Max sequences per organism (default: 50)

### K-mer Settings (NB, RF, XGB)
- `K_MER_SIZE`: K-mer size for tokenization (default: 4)
- `MAX_VOCAB_SIZE`: Maximum vocabulary size (default: 5000)

### Logistic Regression Settings
- `TFIDF_MAX_FEATURES`: TF-IDF vocabulary size (default: 10000)
- `TFIDF_NGRAM_RANGE`: K-mer range for TF-IDF (default: (3, 5))
- `LR_C`: Regularization strength (default: 1.0)
- `LR_PENALTY`: Regularization type (default: "l2")

### XGBoost Settings
- `XGB_N_ESTIMATORS`: Number of boosting rounds (default: 300)
- `XGB_MAX_DEPTH`: Maximum tree depth (default: 10)
- `XGB_LEARNING_RATE`: Learning rate (default: 0.1)
- `XGB_SUBSAMPLE`: Row subsample ratio (default: 0.8)

### CNN Settings
- `SEQUENCE_LENGTH`: Fixed length for padding/truncation (default: 200)
- `EMBEDDING_DIM`: Embedding dimension (default: 128)
- `CNN_FILTERS`: Filters per conv layer (default: [128, 256, 512])
- `KERNEL_SIZES`: Kernel sizes (default: [3, 5, 7])

### BiLSTM Settings
- `LSTM_UNITS`: Units per LSTM layer (default: [256, 128])
- `LSTM_DROPOUT`: Dropout rate (default: 0.3)
- `LSTM_RECURRENT_DROPOUT`: Recurrent dropout (default: 0.2)

### Training Settings
- `BATCH_SIZE`: Batch size (default: 32)
- `EPOCHS`: Training epochs (default: 50)
- `LEARNING_RATE`: Learning rate for CNN (default: 0.001)

## 📊 Output Files

### Models
- `models/naive_bayes_{target}_{timestamp}.pkl`
- `models/logistic_regression_{target}_{timestamp}.pkl`
- `models/random_forest_{target}_{timestamp}.pkl`
- `models/xgboost_{target}_{timestamp}.pkl`
- `models/cnn_{target}_{timestamp}.h5` + metadata
- `models/bilstm_{target}_{timestamp}.h5` + metadata

### Results
- `results/results_{model}_{target}_{timestamp}.txt`
- Confusion matrices: `plots/confusion_matrix_{model}_{target}_{timestamp}.png`
- Training history (CNN): `plots/training_history_cnn_{target}_{timestamp}.png`
- Feature importance (RF): `plots/feature_importance_rf_{target}_{timestamp}.png`

### Logs
- `data_prep.log`
- `train_naive_bayes.log`
- `train_random_forest.log`
- `train_cnn.log`
- TensorBoard logs: `logs/cnn_{target}_{timestamp}/`

## 🔬 Model Details

### 1. Naive Bayes
- **Features**: K-mer count vectors
- **Algorithm**: Multinomial Naive Bayes
- **Training Time**: ⚡ Very Fast (seconds to minutes)
- **Advantages**: Fast, probabilistic, interpretable
- **Best for**: Quick baseline, large datasets

### 2. Logistic Regression
- **Features**: TF-IDF weighted k-mers (range 3-5)
- **Algorithm**: Logistic Regression with L2 regularization
- **Training Time**: ⚡ Fast (minutes)
- **Advantages**: Scalable, linear decision boundaries, interpretable coefficients
- **Best for**: High-dimensional sparse features, multi-class problems

### 3. Random Forest
- **Features**: 
  - Amino acid composition (20 AAs)
  - K-mer frequencies
  - Physicochemical properties (8 categories)
  - Sequence statistics
- **Algorithm**: Ensemble of 200 decision trees
- **Training Time**: ⏱️ Medium (minutes to hours)
- **Advantages**: Handles non-linearity, robust to overfitting, feature importance
- **Best for**: Interpretable non-linear models

### 4. XGBoost
- **Features**: Same as Random Forest
- **Algorithm**: Gradient boosting with regularization
- **Training Time**: ⏱️ Medium-Slow (minutes to hours)
- **Advantages**: State-of-the-art traditional ML, handles imbalance, early stopping
- **Best for**: Best traditional ML accuracy, competitions

### 5. 1D-CNN
- **Architecture**:
  - Embedding layer (22 amino acids → 128-dim vectors)
  - Multi-scale 1D convolutions (kernels: 3, 5, 7)
  - Global max pooling
  - Dense layers with batch norm & dropout
- **Training Time**: 🐌 Slow (hours with GPU, days with CPU)
- **Advantages**: Learns local patterns automatically
- **Best for**: Pattern recognition in sequences

### 6. BiLSTM
- **Architecture**:
  - Embedding layer
  - Stacked Bidirectional LSTM layers [256, 128 units]
  - Dense layers with dropout
- **Training Time**: 🐌 Very Slow (hours with GPU)
- **Advantages**: Captures long-range dependencies, bidirectional context
- **Best for**: Sequential patterns, order matters

## 📈 Model Evaluation

All models report:
- **Accuracy**: Overall classification accuracy
- **MCC**: Matthews Correlation Coefficient (balanced metric)
- **Precision/Recall/F1**: Per-class metrics
- **Confusion Matrix**: Visual representation of predictions
- **Top-3 Accuracy** (CNN only): Whether true label is in top 3 predictions

## 💡 Tips for Best Results

1. **Model Selection**:
   - **Start with**: Logistic Regression or Naive Bayes (fast baseline)
   - **CPU-only machines**: Use XGBoost or Random Forest
   - **Have GPU**: Try CNN and BiLSTM for best accuracy
   - **Large dataset (>100k sequences)**: Logistic Regression scales well
   - **Small dataset (<10k sequences)**: Random Forest or XGBoost

2. **Data Quality**:
   - Ensure `taxon.csv` has clean, consistent labels
   - The script handles whitespace, but check for typos
   - Balance your dataset if possible (stratified sampling is automatic)

3. **Hyperparameter Tuning**:
   - Start with `K_MER_SIZE=4` for balanced performance
   - Increase `MAX_VOCAB_SIZE` or `TFIDF_MAX_FEATURES` for large datasets
   - For deep learning, adjust `SEQUENCE_LENGTH` based on your data
   - XGBoost: tune `XGB_MAX_DEPTH` and `XGB_LEARNING_RATE` for your data

4. **Computational Resources**:
   - **Naive Bayes/Logistic Regression**: CPU is fine (fastest)
   - **Random Forest/XGBoost**: Benefits from multi-core CPU
   - **CNN/BiLSTM**: **GPU highly recommended** (10-100x faster)

5. **Training Strategy**:
   - Train all models and compare results
   - Use ensemble predictions (majority voting)
   - Check confusion matrices to identify problematic classes
   - Monitor training with TensorBoard for deep learning models

## 🐛 Troubleshooting

### "NCBI Email not configured"
- Edit `config.py` and set `NCBI_EMAIL`

### "master_dataset.csv not found"
- Run `01_data_prep.py` first

### "No sequences found for organism X"
- Some assemblies may not have protein data in NCBI
- This is normal; the script continues with other organisms

### CNN training is slow
- Use a GPU (NVIDIA with CUDA support)
- Reduce `BATCH_SIZE`, `SEQUENCE_LENGTH`, or `EMBEDDING_DIM`
- Reduce dataset size for testing

### Out of memory (CNN)
- Reduce `BATCH_SIZE` in `config.py`
- Reduce `SEQUENCE_LENGTH` or `EMBEDDING_DIM`
- Use a machine with more RAM/VRAM

## 📚 Input Data Format

`taxon.csv` should be tab-separated with these columns:
- `Assembly Accession`: NCBI assembly accession (e.g., GCF_027475565.1)
- `Organism Name`: Scientific name
- `Order`: Taxonomic order (may have whitespace)
- `Class`: Taxonomic class (may have whitespace)

Example:
```
Assembly Name	Assembly Accession	...	Organism Name	...	 Order          	 Class    
VMU_Ajub_v1.0	GCF_027475565.1	...	Acinonyx jubatus	...	 Carnivora      	 Mammalia 
```

The script automatically:
- Strips whitespace from labels
- Filters low-frequency categories
- Handles missing data

## 🔄 Workflow Summary

```
taxon.csv
    ↓
01_data_prep.py (fetch sequences from NCBI)
    ↓
master_dataset.csv
    ↓
    ├─→ 02_train_naive_bayes.py ────────┐
    ├─→ 05_train_logistic_regression.py ┤
    ├─→ 03_train_random_forest.py ──────┤→ Models & Results
    ├─→ 06_train_xgboost.py ────────────┤   (6 models × 2 targets)
    ├─→ 04_train_cnn.py ────────────────┤
    └─→ 07_train_bilstm.py ─────────────┘
```

## 🏆 Model Comparison Summary

| Model | Speed | Accuracy* | GPU Required | Interpretable | Best Use Case |
|-------|-------|-----------|--------------|---------------|---------------|
| Naive Bayes | ⚡⚡⚡ | ⭐⭐⭐ | No | Yes | Quick baseline |
| Logistic Reg | ⚡⚡⚡ | ⭐⭐⭐ | No | Yes | Large datasets |
| Random Forest | ⚡⚡ | ⭐⭐⭐⭐ | No | Partial | Robust model |
| XGBoost | ⚡ | ⭐⭐⭐⭐⭐ | No | Partial | Best traditional ML |
| CNN | 🐌 | ⭐⭐⭐⭐⭐ | Recommended | No | Pattern recognition |
| BiLSTM | 🐢 | ⭐⭐⭐⭐⭐ | Recommended | No | Sequence modeling |

*Accuracy will vary based on your specific dataset

## 📝 Citation

If you use this pipeline in your research, please cite:

```
Taxonomic Classification ML Pipeline
Author: [Your Name]
Year: 2026
GitHub: [Repository URL]
```

## 📧 Support

For issues or questions:
1. Check the logs in `*.log` files
2. Review `config.py` settings
3. Ensure all dependencies are installed
4. Check NCBI API status if data fetching fails

## 📜 License

This project is provided as-is for research and educational purposes.

## 🙏 Acknowledgments

- NCBI for providing genomic data via Entrez API
- BioPython for sequence handling tools
- scikit-learn and TensorFlow communities

---

**Happy Classifying! 🧬🔬**

