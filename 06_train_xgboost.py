"""
XGBoost Classifier for Taxonomic Classification
===============================================
Uses gradient boosting with advanced feature engineering.
Often outperforms Random Forest with better handling of class imbalance.

Author: ML Pipeline
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import logging
from datetime import datetime
from collections import Counter

# Import configuration
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_xgboost.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ProteinFeatureEngineering:
    """
    Advanced feature engineering for protein sequences.
    (Reusing from Random Forest with minor optimizations for XGBoost)
    """

    # Amino acid physicochemical properties
    AA_PROPERTIES = {
        "hydrophobic": ["A", "V", "I", "L", "M", "F", "W", "P"],
        "polar": ["S", "T", "C", "Y", "N", "Q"],
        "charged_positive": ["K", "R", "H"],
        "charged_negative": ["D", "E"],
        "aromatic": ["F", "W", "Y"],
        "aliphatic": ["A", "V", "I", "L"],
        "tiny": ["A", "C", "G", "S", "T"],
        "small": ["A", "B", "C", "D", "G", "N", "P", "S", "T", "V"],
    }

    def __init__(self, k=4):
        """
        Initialize feature engineering.

        Args:
            k (int): K-mer size
        """
        self.k = k
        self.kmer_vocab = None

    def calculate_aa_composition(self, sequence):
        """Calculate amino acid composition."""
        sequence = sequence.upper()
        total = len(sequence)

        if total == 0:
            return {aa: 0.0 for aa in config.AMINO_ACIDS}

        counts = Counter(sequence)
        composition = {aa: counts.get(aa, 0) / total for aa in config.AMINO_ACIDS}

        return composition

    def calculate_kmer_composition(self, sequence, k=None):
        """Calculate k-mer composition."""
        if k is None:
            k = self.k

        sequence = sequence.upper()
        if len(sequence) < k:
            return {}

        kmers = [sequence[i : i + k] for i in range(len(sequence) - k + 1)]
        total = len(kmers)

        counts = Counter(kmers)
        composition = {kmer: count / total for kmer, count in counts.items()}

        return composition

    def calculate_physicochemical_properties(self, sequence):
        """Calculate physicochemical property-based features."""
        sequence = sequence.upper()
        total = len(sequence)

        if total == 0:
            return {prop: 0.0 for prop in self.AA_PROPERTIES.keys()}

        features = {}
        for prop_name, aa_list in self.AA_PROPERTIES.items():
            count = sum(1 for aa in sequence if aa in aa_list)
            features[f"prop_{prop_name}"] = count / total

        return features

    def calculate_sequence_statistics(self, sequence):
        """Calculate basic sequence statistics."""
        features = {
            "seq_length": len(sequence),
            "seq_length_log": np.log1p(len(sequence)),
        }

        return features

    def extract_features(self, sequences, fit=False):
        """
        Extract all features from sequences.

        Args:
            sequences (list): List of protein sequences
            fit (bool): Whether to fit k-mer vocabulary

        Returns:
            pd.DataFrame: Feature matrix
        """
        logger.info(f"Extracting features from {len(sequences)} sequences...")

        all_features = []

        for seq in sequences:
            features = {}

            # Amino acid composition
            aa_comp = self.calculate_aa_composition(seq)
            features.update({f"aa_{aa}": freq for aa, freq in aa_comp.items()})

            # Physicochemical properties
            phys_props = self.calculate_physicochemical_properties(seq)
            features.update(phys_props)

            # Sequence statistics
            seq_stats = self.calculate_sequence_statistics(seq)
            features.update(seq_stats)

            # K-mer composition
            kmer_comp = self.calculate_kmer_composition(seq, self.k)

            # If fitting, collect all k-mers
            if fit:
                features.update(
                    {f"kmer_{kmer}": freq for kmer, freq in kmer_comp.items()}
                )
            else:
                # Use only vocabulary k-mers
                if self.kmer_vocab:
                    for kmer in self.kmer_vocab:
                        features[f"kmer_{kmer}"] = kmer_comp.get(kmer, 0.0)

            all_features.append(features)

        # Convert to DataFrame
        df = pd.DataFrame(all_features)

        # Handle k-mer vocabulary
        if fit:
            # Select top k-mers by variance
            kmer_cols = [col for col in df.columns if col.startswith("kmer_")]
            if len(kmer_cols) > config.MAX_VOCAB_SIZE:
                logger.info(
                    f"Selecting top {config.MAX_VOCAB_SIZE} k-mers by variance..."
                )
                kmer_df = df[kmer_cols]
                variances = kmer_df.var()
                top_kmers = variances.nlargest(config.MAX_VOCAB_SIZE).index
                self.kmer_vocab = [col.replace("kmer_", "") for col in top_kmers]

                # Keep only top k-mers
                keep_cols = [col for col in df.columns if not col.startswith("kmer_")]
                keep_cols.extend(top_kmers)
                df = df[keep_cols]
            else:
                self.kmer_vocab = [col.replace("kmer_", "") for col in kmer_cols]

        # Fill NaN with 0
        df = df.fillna(0)

        logger.info(f"Feature matrix shape: {df.shape}")
        logger.info(f"Number of features: {df.shape[1]}")

        return df


class XGBoostTaxonomyClassifier:
    """
    XGBoost classifier for taxonomic prediction.
    """

    def __init__(self, target="order"):
        """
        Initialize classifier.

        Args:
            target (str): Target label ('order' or 'class')
        """
        self.target = target
        self.k_mer_size = config.K_MER_SIZE
        self.random_seed = config.RANDOM_SEED

        self.feature_engineer = ProteinFeatureEngineering(k=self.k_mer_size)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # XGBoost model will be initialized after we know num_classes
        self.model = None

        # Create output directories
        os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.PLOTS_OUTPUT_DIR, exist_ok=True)

    def load_data(self, filepath="master_dataset.csv"):
        """
        Load the master dataset.

        Args:
            filepath (str): Path to master dataset

        Returns:
            tuple: (sequences, labels)
        """
        logger.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)

        logger.info(f"Loaded {len(df)} sequences")

        # Get target column
        target_col = f"label_{self.target}"
        sequences = df["sequence"].values
        labels = df[target_col].values

        logger.info(f"Target: {self.target}")
        logger.info(f"Unique classes: {len(np.unique(labels))}")
        logger.info(f"\nClass distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
            logger.info(f"  {label}: {count}")

        return sequences, labels

    def prepare_features(self, sequences, labels, test_size=0.2):
        """
        Prepare features and split data.

        Args:
            sequences (np.ndarray): Protein sequences
            labels (np.ndarray): Target labels
            test_size (float): Test set proportion

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        logger.info("\nPreparing features...")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        num_classes = len(self.label_encoder.classes_)

        # Split data BEFORE feature extraction
        sequences_train, sequences_test, y_train, y_test = train_test_split(
            sequences,
            y_encoded,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y_encoded,
        )

        logger.info(f"Train set: {len(sequences_train)} sequences")
        logger.info(f"Test set: {len(sequences_test)} sequences")

        # Extract features
        X_train = self.feature_engineer.extract_features(sequences_train, fit=True)
        X_test = self.feature_engineer.extract_features(sequences_test, fit=False)

        # Ensure same columns
        missing_cols = set(X_train.columns) - set(X_test.columns)
        for col in missing_cols:
            X_test[col] = 0
        X_test = X_test[X_train.columns]

        # Scale features (XGBoost doesn't strictly need scaling, but can help)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize XGBoost model now that we know num_classes
        self.model = xgb.XGBClassifier(
            n_estimators=config.XGB_N_ESTIMATORS,
            max_depth=config.XGB_MAX_DEPTH,
            learning_rate=config.XGB_LEARNING_RATE,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
            min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
            gamma=config.XGB_GAMMA,
            reg_alpha=config.XGB_REG_ALPHA,
            reg_lambda=config.XGB_REG_LAMBDA,
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            num_class=num_classes if num_classes > 2 else None,
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
            early_stopping_rounds=10,
            verbosity=1,
        )

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train(self, X_train, y_train):
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("\nTraining XGBoost classifier...")
        logger.info(f"Model parameters:")
        logger.info(f"  n_estimators: {config.XGB_N_ESTIMATORS}")
        logger.info(f"  max_depth: {config.XGB_MAX_DEPTH}")
        logger.info(f"  learning_rate: {config.XGB_LEARNING_RATE}")

        # Split a small validation set for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=self.random_seed
        )

        self.model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            verbose=True,
        )

        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        logger.info(f"Training accuracy: {train_acc:.4f}")

        # Feature importance
        self.analyze_feature_importance()

    def analyze_feature_importance(self):
        """Analyze and plot feature importance."""
        logger.info("\nAnalyzing feature importance...")

        # Get feature importance
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Get feature names
        feature_names = list(
            self.feature_engineer.extract_features(["A" * 10], fit=False).columns
        )

        # Log top 20 features
        logger.info("\nTop 20 most important features:")
        for i, idx in enumerate(indices[:20]):
            if idx < len(feature_names):
                logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.6f}")

        # Plot feature importance
        self.plot_feature_importance(importances, feature_names)

    def plot_feature_importance(self, importances, feature_names, top_n=30):
        """Plot top N feature importances."""
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(
            range(top_n),
            [
                feature_names[i] if i < len(feature_names) else f"Feature {i}"
                for i in indices
            ],
        )
        plt.xlabel("Feature Importance (Gain)")
        plt.title(
            f"Top {top_n} Feature Importances - XGBoost ({self.target.capitalize()})"
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(
            config.PLOTS_OUTPUT_DIR,
            f"feature_importance_xgb_{self.target}_{timestamp}.png",
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to: {plot_path}")
        plt.close()

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test set.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            dict: Evaluation metrics
        """
        logger.info("\n" + "=" * 60)
        logger.info("Model Evaluation")
        logger.info("=" * 60)

        # Predictions
        y_pred = self.model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"\nTest Accuracy: {accuracy:.4f}")

        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_test, y_pred)
        logger.info(f"Matthews Correlation Coefficient: {mcc:.4f}")

        # Classification report
        target_names = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Save confusion matrix plot
        self.plot_confusion_matrix(cm, target_names)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "mcc": mcc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "confusion_matrix": cm,
        }

        return metrics

    def plot_confusion_matrix(self, cm, class_names):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(12, 10))

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Proportion"},
        )

        plt.title(
            f"Confusion Matrix - XGBoost ({self.target.capitalize()})\nK-mer size: {self.k_mer_size}"
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(
            config.PLOTS_OUTPUT_DIR,
            f"confusion_matrix_xgb_{self.target}_{timestamp}.png",
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to: {plot_path}")
        plt.close()

    def save_model(self):
        """Save the trained model and components."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_data = {
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "target": self.target,
            "k_mer_size": self.k_mer_size,
            "timestamp": timestamp,
        }

        model_path = os.path.join(
            config.MODEL_OUTPUT_DIR, f"xgboost_{self.target}_{timestamp}.pkl"
        )

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to: {model_path}")

    def save_results(self, metrics):
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            config.RESULTS_OUTPUT_DIR, f"results_xgb_{self.target}_{timestamp}.txt"
        )

        with open(results_path, "w") as f:
            f.write(f"XGBoost Classifier - {self.target.capitalize()} Prediction\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  K-mer size: {self.k_mer_size}\n")
            f.write(f"  Max vocabulary: {config.MAX_VOCAB_SIZE}\n")
            f.write(f"  N estimators: {config.XGB_N_ESTIMATORS}\n")
            f.write(f"  Max depth: {config.XGB_MAX_DEPTH}\n")
            f.write(f"  Learning rate: {config.XGB_LEARNING_RATE}\n")
            f.write(f"  Random seed: {self.random_seed}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  MCC: {metrics['mcc']:.4f}\n\n")
            f.write(f"Timestamp: {timestamp}\n")

        logger.info(f"Results saved to: {results_path}")

    def run_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("=" * 60)
        logger.info(f"XGBoost Training Pipeline - {self.target.upper()}")
        logger.info("=" * 60)

        # Load data
        sequences, labels = self.load_data()

        # Prepare features
        X_train, X_test, y_train, y_test = self.prepare_features(sequences, labels)

        # Train model
        self.train(X_train, y_train)

        # Evaluate on test set
        metrics = self.evaluate(X_test, y_test)

        # Save model
        self.save_model()

        # Save results
        self.save_results(metrics)

        logger.info("\n" + "=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)


def main():
    """Main execution function."""

    # Check if master dataset exists
    if not os.path.exists("master_dataset.csv"):
        print("\n" + "!" * 60)
        print("ERROR: master_dataset.csv not found!")
        print("!" * 60)
        print("\nPlease run 01_data_prep.py first to generate the dataset.")
        print("\n" + "!" * 60)
        sys.exit(1)

    # Train Order classifier
    logger.info("\n\n" + "#" * 60)
    logger.info("# TRAINING ORDER CLASSIFIER")
    logger.info("#" * 60 + "\n")

    order_classifier = XGBoostTaxonomyClassifier(target="order")
    order_classifier.run_pipeline()

    # Train Class classifier
    logger.info("\n\n" + "#" * 60)
    logger.info("# TRAINING CLASS CLASSIFIER")
    logger.info("#" * 60 + "\n")

    class_classifier = XGBoostTaxonomyClassifier(target="class")
    class_classifier.run_pipeline()


if __name__ == "__main__":
    main()
