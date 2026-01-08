"""
Configuration File for Taxonomic Classification ML Pipeline
============================================================
Central configuration for hyperparameters and model settings.
Modify values here to experiment with different configurations.
"""

# =============================================================================
# DATA PREPARATION SETTINGS
# =============================================================================

# Minimum number of samples required per Order category
# Orders with fewer samples will be filtered out
MIN_SAMPLES_THRESHOLD = 10

# Maximum number of protein sequences to fetch per organism
# This prevents a single organism from dominating the dataset
MAX_SEQS_PER_ORGANISM = 50

# NCBI Entrez API Configuration
# IMPORTANT: Set your email and API key for NCBI Entrez access
NCBI_EMAIL = "your.email@example.com"  # REQUIRED: Replace with your email
NCBI_API_KEY = None  # OPTIONAL: Replace with your API key for faster access (10 requests/sec vs 3 requests/sec)

# Batch size for fetching sequences from NCBI
FETCH_BATCH_SIZE = 100

# Network timeout settings (seconds)
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


# =============================================================================
# FEATURE ENGINEERING SETTINGS
# =============================================================================

# K-mer size for sequence tokenization (used in NB and RF)
# Common values: 3, 4, 5
K_MER_SIZE = 4

# Maximum vocabulary size for k-mer vectorization
# Limits the feature space to top N most frequent k-mers
MAX_VOCAB_SIZE = 5000


# =============================================================================
# LOGISTIC REGRESSION SETTINGS
# =============================================================================

# TF-IDF settings for Logistic Regression
TFIDF_MAX_FEATURES = 10000  # Maximum features for TF-IDF
TFIDF_NGRAM_RANGE = (3, 5)  # N-gram range for TF-IDF (k-mer range)

# Logistic Regression hyperparameters
LR_C = 1.0  # Inverse of regularization strength
LR_MAX_ITER = 1000  # Maximum iterations
LR_SOLVER = "saga"  # Solver (saga supports L1/L2/elasticnet)
LR_PENALTY = "l2"  # Regularization penalty


# =============================================================================
# XGBOOST SETTINGS
# =============================================================================

# XGBoost hyperparameters
XGB_N_ESTIMATORS = 300  # Number of boosting rounds
XGB_MAX_DEPTH = 10  # Maximum tree depth
XGB_LEARNING_RATE = 0.1  # Learning rate (eta)
XGB_SUBSAMPLE = 0.8  # Subsample ratio
XGB_COLSAMPLE_BYTREE = 0.8  # Feature sampling ratio
XGB_MIN_CHILD_WEIGHT = 3  # Minimum sum of instance weight
XGB_GAMMA = 0.1  # Minimum loss reduction
XGB_REG_ALPHA = 0.1  # L1 regularization
XGB_REG_LAMBDA = 1.0  # L2 regularization


# =============================================================================
# DEEP LEARNING SETTINGS (CNN)
# =============================================================================

# Fixed sequence length for padding/truncation
# Sequences longer than this will be truncated
# Sequences shorter than this will be padded
SEQUENCE_LENGTH = 200

# Embedding dimension for amino acid representation
EMBEDDING_DIM = 128

# Convolutional layer settings
CNN_FILTERS = [128, 256, 512]  # Number of filters in each conv layer
KERNEL_SIZES = [3, 5, 7]  # Kernel sizes for different conv layers
POOL_SIZE = 2

# Dropout rate for regularization
DROPOUT_RATE = 0.5


# =============================================================================
# BILSTM SETTINGS
# =============================================================================

# BiLSTM architecture settings
LSTM_UNITS = [256, 128]  # Units in each LSTM layer
LSTM_DROPOUT = 0.3  # Dropout between LSTM layers
LSTM_RECURRENT_DROPOUT = 0.2  # Recurrent dropout within LSTM


# =============================================================================
# TRAINING SETTINGS
# =============================================================================

# Batch size for model training
BATCH_SIZE = 32

# Number of training epochs
EPOCHS = 50

# Learning rate
LEARNING_RATE = 0.001

# Early stopping patience (epochs)
EARLY_STOPPING_PATIENCE = 5

# Validation split ratio
VALIDATION_SPLIT = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

# Number of cross-validation folds
CV_FOLDS = 5


# =============================================================================
# MODEL OUTPUT PATHS
# =============================================================================

# Directory to save trained models
MODEL_OUTPUT_DIR = "models"

# Directory to save evaluation results
RESULTS_OUTPUT_DIR = "results"

# Directory to save plots and visualizations
PLOTS_OUTPUT_DIR = "plots"


# =============================================================================
# AMINO ACID VOCABULARY
# =============================================================================

# Standard 20 amino acids + special characters
AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "X",  # Unknown
    "-",  # Gap
]

# Amino acid to integer mapping
AA_TO_INT = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# Integer to amino acid mapping
INT_TO_AA = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}
