"""
Bidirectional LSTM for Taxonomic Classification
===============================================
Uses BiLSTM architecture to capture long-range dependencies in protein sequences.
Excellent for sequential patterns in biological data.

Author: ML Pipeline
Date: 2026
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
)
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
        logging.FileHandler("train_bilstm.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)


class SequencePreprocessor:
    """
    Preprocess protein sequences for LSTM input.
    """

    def __init__(self, max_length=200):
        """
        Initialize preprocessor.

        Args:
            max_length (int): Maximum sequence length for padding/truncation
        """
        self.max_length = max_length
        self.aa_to_int = config.AA_TO_INT
        self.vocab_size = len(config.AMINO_ACIDS)

    def encode_sequence(self, sequence):
        """
        Encode a protein sequence to integers.

        Args:
            sequence (str): Protein sequence

        Returns:
            list: Integer-encoded sequence
        """
        sequence = sequence.upper()
        # Map unknown amino acids to 'X'
        encoded = [self.aa_to_int.get(aa, self.aa_to_int["X"]) for aa in sequence]
        return encoded

    def pad_or_truncate(self, encoded_seq):
        """
        Pad or truncate sequence to fixed length.

        Args:
            encoded_seq (list): Integer-encoded sequence

        Returns:
            np.ndarray: Padded/truncated sequence
        """
        if len(encoded_seq) >= self.max_length:
            return np.array(encoded_seq[: self.max_length])
        else:
            # Pad with zeros
            padded = np.zeros(self.max_length, dtype=np.int32)
            padded[: len(encoded_seq)] = encoded_seq
            return padded

    def preprocess_sequences(self, sequences):
        """
        Preprocess a list of sequences.

        Args:
            sequences (list): List of protein sequences

        Returns:
            np.ndarray: Preprocessed sequence array
        """
        logger.info(f"Preprocessing {len(sequences)} sequences...")
        logger.info(f"Max sequence length: {self.max_length}")

        processed = []
        for seq in sequences:
            encoded = self.encode_sequence(seq)
            padded = self.pad_or_truncate(encoded)
            processed.append(padded)

        processed_array = np.array(processed)
        logger.info(f"Processed array shape: {processed_array.shape}")

        return processed_array


class BiLSTMTaxonomyClassifier:
    """
    Bidirectional LSTM classifier for taxonomic prediction.
    """

    def __init__(self, target="order"):
        """
        Initialize BiLSTM classifier.

        Args:
            target (str): Target label ('order' or 'class')
        """
        self.target = target
        self.sequence_length = config.SEQUENCE_LENGTH
        self.embedding_dim = config.EMBEDDING_DIM
        self.lstm_units = config.LSTM_UNITS
        self.lstm_dropout = config.LSTM_DROPOUT
        self.lstm_recurrent_dropout = config.LSTM_RECURRENT_DROPOUT
        self.batch_size = config.BATCH_SIZE
        self.epochs = config.EPOCHS
        self.learning_rate = config.LEARNING_RATE
        self.random_seed = config.RANDOM_SEED

        self.preprocessor = SequencePreprocessor(max_length=self.sequence_length)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        self.num_classes = None

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

    def prepare_data(self, sequences, labels, test_size=0.2):
        """
        Prepare data for training.

        Args:
            sequences (np.ndarray): Protein sequences
            labels (np.ndarray): Target labels
            test_size (float): Test set proportion

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("\nPreparing data...")

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        logger.info(f"Number of classes: {self.num_classes}")

        # Split data
        sequences_train, sequences_test, y_train, y_test = train_test_split(
            sequences,
            y_encoded,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y_encoded,
        )

        # Further split train into train and validation
        sequences_train, sequences_val, y_train, y_val = train_test_split(
            sequences_train,
            y_train,
            test_size=0.2,  # 20% of train for validation
            random_state=self.random_seed,
            stratify=y_train,
        )

        logger.info(f"Train set: {len(sequences_train)} sequences")
        logger.info(f"Validation set: {len(sequences_val)} sequences")
        logger.info(f"Test set: {len(sequences_test)} sequences")

        # Preprocess sequences
        X_train = self.preprocessor.preprocess_sequences(sequences_train)
        X_val = self.preprocessor.preprocess_sequences(sequences_val)
        X_test = self.preprocessor.preprocess_sequences(sequences_test)

        # Convert labels to categorical (one-hot encoding)
        y_train_cat = keras.utils.to_categorical(y_train, self.num_classes)
        y_val_cat = keras.utils.to_categorical(y_val, self.num_classes)
        y_test_cat = keras.utils.to_categorical(y_test, self.num_classes)

        return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_test

    def build_model(self):
        """
        Build the BiLSTM model architecture.

        Returns:
            keras.Model: Compiled BiLSTM model
        """
        logger.info("\nBuilding BiLSTM model...")
        logger.info(f"Architecture:")
        logger.info(f"  Sequence length: {self.sequence_length}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  LSTM units: {self.lstm_units}")
        logger.info(f"  LSTM dropout: {self.lstm_dropout}")
        logger.info(f"  LSTM recurrent dropout: {self.lstm_recurrent_dropout}")

        # Input layer
        input_layer = layers.Input(shape=(self.sequence_length,), name="input")

        # Embedding layer
        embedding = layers.Embedding(
            input_dim=self.preprocessor.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.sequence_length,
            mask_zero=True,  # Mask padding
            name="embedding",
        )(input_layer)

        # First BiLSTM layer
        x = layers.Bidirectional(
            layers.LSTM(
                self.lstm_units[0],
                return_sequences=True if len(self.lstm_units) > 1 else False,
                dropout=self.lstm_dropout,
                recurrent_dropout=self.lstm_recurrent_dropout,
            ),
            name="bilstm_1",
        )(embedding)

        x = layers.BatchNormalization(name="bn_lstm_1")(x)

        # Additional BiLSTM layers
        for i, units in enumerate(self.lstm_units[1:], start=2):
            x = layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=False,
                    dropout=self.lstm_dropout,
                    recurrent_dropout=self.lstm_recurrent_dropout,
                ),
                name=f"bilstm_{i}",
            )(x)
            x = layers.BatchNormalization(name=f"bn_lstm_{i}")(x)

        # Dense layers
        x = layers.Dense(256, activation="relu", name="dense_1")(x)
        x = layers.BatchNormalization(name="bn_dense_1")(x)
        x = layers.Dropout(config.DROPOUT_RATE, name="dropout_1")(x)

        x = layers.Dense(128, activation="relu", name="dense_2")(x)
        x = layers.BatchNormalization(name="bn_dense_2")(x)
        x = layers.Dropout(config.DROPOUT_RATE / 2, name="dropout_2")(x)

        # Output layer
        output = layers.Dense(self.num_classes, activation="softmax", name="output")(x)

        # Create model
        model = models.Model(
            inputs=input_layer, outputs=output, name=f"bilstm_{self.target}"
        )

        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
            ],
        )

        # Model summary
        model.summary(print_fn=logger.info)

        return model

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the BiLSTM model.

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
        """
        logger.info("\n" + "=" * 60)
        logger.info("Training BiLSTM Model")
        logger.info("=" * 60)

        # Build model
        self.model = self.build_model()

        # Callbacks
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_checkpoint_path = os.path.join(
            config.MODEL_OUTPUT_DIR, f"bilstm_{self.target}_{timestamp}_best.h5"
        )

        callback_list = [
            callbacks.ModelCheckpoint(
                model_checkpoint_path,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=os.path.join("logs", f"bilstm_{self.target}_{timestamp}"),
                histogram_freq=1,
            ),
        ]

        # Train model
        logger.info(f"\nStarting training for {self.epochs} epochs...")
        logger.info(f"Batch size: {self.batch_size}")

        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callback_list,
            verbose=1,
        )

        logger.info("\nTraining complete!")

        # Plot training history
        self.plot_training_history()

    def plot_training_history(self):
        """Plot and save training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        axes[0].plot(self.history.history["accuracy"], label="Train Accuracy")
        axes[0].plot(self.history.history["val_accuracy"], label="Val Accuracy")
        axes[0].set_title("Model Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss plot
        axes[1].plot(self.history.history["loss"], label="Train Loss")
        axes[1].plot(self.history.history["val_loss"], label="Val Loss")
        axes[1].set_title("Model Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(
            config.PLOTS_OUTPUT_DIR,
            f"training_history_bilstm_{self.target}_{timestamp}.png",
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training history plot saved to: {plot_path}")
        plt.close()

    def evaluate(self, X_test, y_test_cat, y_test):
        """
        Evaluate the model on test set.

        Args:
            X_test: Test data
            y_test_cat: Test labels (one-hot)
            y_test: Test labels (integer encoded)

        Returns:
            dict: Evaluation metrics
        """
        logger.info("\n" + "=" * 60)
        logger.info("Model Evaluation")
        logger.info("=" * 60)

        # Evaluate
        test_loss, test_accuracy, test_top3 = self.model.evaluate(
            X_test, y_test_cat, verbose=1
        )
        logger.info(f"\nTest Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Top-3 Accuracy: {test_top3:.4f}")

        # Predictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

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
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "top3_accuracy": test_top3,
            "mcc": mcc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "confusion_matrix": cm,
        }

        return metrics

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
            cmap="RdPu",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Proportion"},
        )

        plt.title(f"Confusion Matrix - BiLSTM ({self.target.capitalize()})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(
            config.PLOTS_OUTPUT_DIR,
            f"confusion_matrix_bilstm_{self.target}_{timestamp}.png",
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to: {plot_path}")
        plt.close()

    def save_model(self):
        """Save the trained model and components."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save Keras model
        model_path = os.path.join(
            config.MODEL_OUTPUT_DIR, f"bilstm_{self.target}_{timestamp}.h5"
        )
        self.model.save(model_path)
        logger.info(f"Keras model saved to: {model_path}")

        # Save preprocessor and label encoder
        metadata = {
            "preprocessor": self.preprocessor,
            "label_encoder": self.label_encoder,
            "target": self.target,
            "sequence_length": self.sequence_length,
            "embedding_dim": self.embedding_dim,
            "lstm_units": self.lstm_units,
            "num_classes": self.num_classes,
            "timestamp": timestamp,
        }

        metadata_path = os.path.join(
            config.MODEL_OUTPUT_DIR, f"bilstm_{self.target}_{timestamp}_metadata.pkl"
        )

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Model metadata saved to: {metadata_path}")

    def save_results(self, metrics):
        """
        Save evaluation results to file.

        Args:
            metrics (dict): Evaluation metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(
            config.RESULTS_OUTPUT_DIR,
            f"results_bilstm_{self.target}_{timestamp}.txt",
        )

        with open(results_path, "w") as f:
            f.write(f"BiLSTM Classifier - {self.target.capitalize()} Prediction\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Configuration:\n")
            f.write(f"  Sequence length: {self.sequence_length}\n")
            f.write(f"  Embedding dim: {self.embedding_dim}\n")
            f.write(f"  LSTM units: {self.lstm_units}\n")
            f.write(f"  LSTM dropout: {self.lstm_dropout}\n")
            f.write(f"  LSTM recurrent dropout: {self.lstm_recurrent_dropout}\n")
            f.write(f"  Batch size: {self.batch_size}\n")
            f.write(f"  Epochs: {self.epochs}\n")
            f.write(f"  Learning rate: {self.learning_rate}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Test Loss: {metrics['test_loss']:.4f}\n")
            f.write(f"  Test Accuracy: {metrics['test_accuracy']:.4f}\n")
            f.write(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}\n")
            f.write(f"  MCC: {metrics['mcc']:.4f}\n\n")
            f.write(f"Timestamp: {timestamp}\n")

        logger.info(f"Results saved to: {results_path}")

    def run_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("=" * 60)
        logger.info(f"BiLSTM Training Pipeline - {self.target.upper()}")
        logger.info("=" * 60)

        # Load data
        sequences, labels = self.load_data()

        # Prepare data
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test_cat,
            y_test,
        ) = self.prepare_data(sequences, labels)

        # Train model
        self.train(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        metrics = self.evaluate(X_test, y_test_cat, y_test)

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

    # Check GPU availability
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU(s) available: {len(gpus)}")
        for gpu in gpus:
            logger.info(f"  {gpu}")
    else:
        logger.info("No GPU available. Training will use CPU.")

    # Train Order classifier
    logger.info("\n\n" + "#" * 60)
    logger.info("# TRAINING ORDER CLASSIFIER")
    logger.info("#" * 60 + "\n")

    order_classifier = BiLSTMTaxonomyClassifier(target="order")
    order_classifier.run_pipeline()

    # Train Class classifier
    logger.info("\n\n" + "#" * 60)
    logger.info("# TRAINING CLASS CLASSIFIER")
    logger.info("#" * 60 + "\n")

    class_classifier = BiLSTMTaxonomyClassifier(target="class")
    class_classifier.run_pipeline()


if __name__ == "__main__":
    main()
