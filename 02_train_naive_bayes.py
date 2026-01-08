"""
Naive Bayes Classifier for Taxonomic Classification
===================================================
Uses K-mer based Count Vectorization for sequence feature extraction.
Trains separate models for Order and Class prediction.

Author: ML Pipeline
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import sys
import logging
from datetime import datetime

# Import configuration
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_naive_bayes.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class KMerFeatureExtractor:
    """
    Extract K-mer features from protein sequences.
    """

    def __init__(self, k=4, max_features=5000):
        """
        Initialize K-mer feature extractor.

        Args:
            k (int): K-mer size
            max_features (int): Maximum vocabulary size
        """
        self.k = k
        self.max_features = max_features
        self.vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            max_features=max_features,
            lowercase=False,
            token_pattern=None,
        )

    def generate_kmers(self, sequence):
        """
        Generate k-mers from a sequence.

        Args:
            sequence (str): Protein sequence

        Returns:
            str: Space-separated k-mers
        """
        kmers = [sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1)]
        return " ".join(kmers)

    def fit_transform(self, sequences):
        """
        Fit vectorizer and transform sequences.

        Args:
            sequences (list): List of protein sequences

        Returns:
            np.ndarray: Feature matrix
        """
        logger.info(f"Extracting {self.k}-mers from {len(sequences)} sequences...")
        kmer_sequences = [self.generate_kmers(seq) for seq in sequences]
        features = self.vectorizer.fit_transform(kmer_sequences)
        logger.info(f"Feature matrix shape: {features.shape}")
        return features

    def transform(self, sequences):
        """
        Transform sequences using fitted vectorizer.

        Args:
            sequences (list): List of protein sequences

        Returns:
            np.ndarray: Feature matrix
        """
        kmer_sequences = [self.generate_kmers(seq) for seq in sequences]
        return self.vectorizer.transform(kmer_sequences)


class NaiveBayesTaxonomyClassifier:
    """
    Naive Bayes classifier for taxonomic prediction.
    """

    def __init__(self, target="order"):
        """
        Initialize classifier.

        Args:
            target (str): Target label ('order' or 'class')
        """
        self.target = target
        self.k_mer_size = config.K_MER_SIZE
        self.max_vocab = config.MAX_VOCAB_SIZE
        self.random_seed = config.RANDOM_SEED

        self.feature_extractor = KMerFeatureExtractor(
            k=self.k_mer_size, max_features=self.max_vocab
        )
        self.model = MultinomialNB(alpha=1.0)
        self.label_encoder = LabelEncoder()

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

        # Split data BEFORE feature extraction (important for avoiding data leakage)
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
        X_train = self.feature_extractor.fit_transform(sequences_train)
        X_test = self.feature_extractor.transform(sequences_test)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """
        Train the Naive Bayes model.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        logger.info("\nTraining Naive Bayes classifier...")
        self.model.fit(X_train, y_train)

        # Training accuracy
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        logger.info(f"Training accuracy: {train_acc:.4f}")

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

    def cross_validate(self, X, y):
        """
        Perform cross-validation.

        Args:
            X: Feature matrix
            y: Labels

        Returns:
            dict: Cross-validation scores
        """
        logger.info(f"\nPerforming {config.CV_FOLDS}-fold cross-validation...")

        cv = StratifiedKFold(
            n_splits=config.CV_FOLDS, shuffle=True, random_state=self.random_seed
        )
        scores = cross_val_score(self.model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)

        logger.info(f"Cross-validation scores: {scores}")
        logger.info(
            f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})"
        )

        return {"scores": scores, "mean": scores.mean(), "std": scores.std()}

    def plot_confusion_matrix(self, cm, class_names):
        """
        Plot and save confusion matrix.

        Args:
            cm (np.ndarray): Confusion matrix
            class_names (list): Class names
        """
        plt.figure(figsize=(12, 10))

        # Normalize confusion matrix
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Proportion"},
        )

        plt.title(
            f"Confusion Matrix - Naive Bayes ({self.target.capitalize()})\nK-mer size: {self.k_mer_size}"
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
            f"confusion_matrix_nb_{self.target}_{timestamp}.png",
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to: {plot_path}")
        plt.close()

    def save_model(self):
        """Save the trained model and components."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model_data = {
            "model": self.model,
            "feature_extractor": self.feature_extractor,
            "label_encoder": self.label_encoder,
            "target": self.target,
            "k_mer_size": self.k_mer_size,
            "max_vocab": self.max_vocab,
            "timestamp": timestamp,
        }

        model_path = os.path.join(
            config.MODEL_OUTPUT_DIR, f"naive_bayes_{self.target}_{timestamp}.pkl"
        )

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to: {model_path}")

    def save_results(self, metrics):
        """
        Save evaluation results to file.

        Args:
            metrics (dict): Evaluation metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            config.RESULTS_OUTPUT_DIR, f"results_nb_{self.target}_{timestamp}.txt"
        )

        with open(results_path, "w") as f:
            f.write(f"Naive Bayes Classifier - {self.target.capitalize()} Prediction\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  K-mer size: {self.k_mer_size}\n")
            f.write(f"  Max vocabulary: {self.max_vocab}\n")
            f.write(f"  Random seed: {self.random_seed}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  MCC: {metrics['mcc']:.4f}\n\n")
            f.write(f"Timestamp: {timestamp}\n")

        logger.info(f"Results saved to: {results_path}")

    def run_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("=" * 60)
        logger.info(f"Naive Bayes Training Pipeline - {self.target.upper()}")
        logger.info("=" * 60)

        # Load data
        sequences, labels = self.load_data()

        # Prepare features
        X_train, X_test, y_train, y_test = self.prepare_features(sequences, labels)

        # Train model
        self.train(X_train, y_train)

        # Cross-validation
        # Combine train and test for CV (on full dataset)
        X_full = np.vstack([X_train.toarray(), X_test.toarray()])
        y_full = np.concatenate([y_train, y_test])
        cv_results = self.cross_validate(X_full, y_full)

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

    order_classifier = NaiveBayesTaxonomyClassifier(target="order")
    order_classifier.run_pipeline()

    # Train Class classifier
    logger.info("\n\n" + "#" * 60)
    logger.info("# TRAINING CLASS CLASSIFIER")
    logger.info("#" * 60 + "\n")

    class_classifier = NaiveBayesTaxonomyClassifier(target="class")
    class_classifier.run_pipeline()


if __name__ == "__main__":
    main()
