"""
Data Preparation Script for Taxonomic Classification
=====================================================
This script:
1. Loads and cleans the taxon.csv metadata
2. Filters out low-frequency Order categories
3. Fetches protein sequences from NCBI using BioPython
4. Saves the final dataset as master_dataset.csv

Author: ML Pipeline
Date: 2026
"""

import pandas as pd
import numpy as np
from Bio import Entrez, SeqIO
from tqdm import tqdm
import time
import sys
import warnings
from collections import Counter
import logging

# Import configuration
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("data_prep.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class DataPreparation:
    """
    Data preparation pipeline for taxonomic classification.
    Handles data cleaning, filtering, and sequence fetching.
    """

    def __init__(self):
        """Initialize the data preparation pipeline."""
        # IMPORTANT: Configure your NCBI email and API key in config.py
        Entrez.email = config.NCBI_EMAIL
        if config.NCBI_API_KEY:
            Entrez.api_key = config.NCBI_API_KEY
            logger.info("Using NCBI API key for faster access (10 req/sec)")
        else:
            logger.warning("No NCBI API key provided. Limited to 3 requests/sec")
            logger.warning("Get an API key at: https://www.ncbi.nlm.nih.gov/account/")

        self.min_samples = config.MIN_SAMPLES_THRESHOLD
        self.max_seqs_per_organism = config.MAX_SEQS_PER_ORGANISM
        self.batch_size = config.FETCH_BATCH_SIZE
        self.timeout = config.REQUEST_TIMEOUT
        self.max_retries = config.MAX_RETRIES

    def load_and_clean_metadata(self, filepath="taxon.csv"):
        """
        Load and clean the taxonomy metadata file.

        Args:
            filepath (str): Path to the taxon.csv file

        Returns:
            pd.DataFrame: Cleaned metadata
        """
        logger.info(f"Loading metadata from {filepath}...")

        # Load CSV (tab-separated)
        df = pd.read_csv(filepath, sep="\t")

        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Strip whitespace from all string columns
        # This is CRUCIAL for handling dirty data like ' Rodentia ' -> 'Rodentia'
        logger.info("Stripping whitespace from all string columns...")
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].str.strip()

        # Keep only relevant columns
        required_cols = ["Assembly Accession", "Organism Name", " Order", " Class"]
        # Handle potential column name variations (with/without spaces)
        actual_cols = []
        for col in required_cols:
            if col in df.columns:
                actual_cols.append(col)
            elif col.strip() in df.columns:
                actual_cols.append(col.strip())

        # Rename columns to standard names
        rename_map = {}
        for col in df.columns:
            clean_col = col.strip()
            if "Assembly Accession" in col:
                rename_map[col] = "assembly_accession"
            elif "Organism Name" in col:
                rename_map[col] = "organism_name"
            elif "Order" in col:
                rename_map[col] = "order"
            elif "Class" in col:
                rename_map[col] = "class"

        df = df.rename(columns=rename_map)

        # Keep only relevant columns
        keep_cols = ["assembly_accession", "organism_name", "order", "class"]
        df = df[keep_cols]

        # Remove rows with missing values
        df = df.dropna()

        logger.info(f"After cleaning: {len(df)} records")

        return df

    def filter_low_frequency_orders(self, df):
        """
        Filter out Order categories with fewer than MIN_SAMPLES_THRESHOLD samples.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        logger.info(f"Filtering Orders with < {self.min_samples} samples...")

        # Count samples per Order
        order_counts = df["order"].value_counts()
        logger.info(f"\nOrder distribution BEFORE filtering:")
        logger.info(f"\n{order_counts}")

        # Identify Orders to keep
        valid_orders = order_counts[order_counts >= self.min_samples].index

        # Filter dataframe
        df_filtered = df[df["order"].isin(valid_orders)].copy()

        logger.info(f"\nKept {len(valid_orders)} Orders (out of {len(order_counts)})")
        logger.info(
            f"Filtered dataset: {len(df_filtered)} records (down from {len(df)})"
        )

        return df_filtered

    def check_class_diversity(self, df):
        """
        Check if dataset contains multiple Classes or only Mammalia.
        Warn user if Class column should be dropped.

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            bool: True if multiple classes exist, False otherwise
        """
        unique_classes = df["class"].unique()
        logger.info(f"\nUnique Classes in dataset: {unique_classes}")

        if len(unique_classes) == 1:
            logger.warning(f"Dataset contains only '{unique_classes[0]}'!")
            logger.warning(
                "Consider dropping the 'Class' column or adding non-mammalian data."
            )
            return False
        else:
            logger.info(
                f"Dataset contains {len(unique_classes)} classes. Keeping 'Class' column."
            )
            return True

    def fetch_protein_sequences(self, assembly_accession, max_seqs=None):
        """
        Fetch protein sequences for a given assembly accession using NCBI Entrez.

        Args:
            assembly_accession (str): NCBI assembly accession
            max_seqs (int): Maximum number of sequences to fetch

        Returns:
            list: List of protein sequence strings
        """
        if max_seqs is None:
            max_seqs = self.max_seqs_per_organism

        sequences = []

        for attempt in range(self.max_retries):
            try:
                # Search for protein sequences associated with assembly
                search_handle = Entrez.esearch(
                    db="protein",
                    term=f"{assembly_accession}[Assembly]",
                    retmax=max_seqs,
                    usehistory="y",
                )
                search_results = Entrez.read(search_handle)
                search_handle.close()

                # Get the ID list
                id_list = search_results["IdList"]

                if not id_list:
                    logger.debug(f"No proteins found for {assembly_accession}")
                    return []

                # Limit to max_seqs
                id_list = id_list[:max_seqs]

                # Fetch sequences in batch
                fetch_handle = Entrez.efetch(
                    db="protein", id=id_list, rettype="fasta", retmode="text"
                )

                # Parse FASTA records
                records = SeqIO.parse(fetch_handle, "fasta")
                for record in records:
                    sequences.append(str(record.seq))

                fetch_handle.close()

                # Rate limiting (be nice to NCBI servers)
                if config.NCBI_API_KEY:
                    time.sleep(0.11)  # ~10 requests per second with API key
                else:
                    time.sleep(0.35)  # ~3 requests per second without API key

                return sequences

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed for {assembly_accession}: {str(e)}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"Failed to fetch sequences for {assembly_accession} after {self.max_retries} attempts"
                    )
                    return []

        return sequences

    def build_master_dataset(self, df):
        """
        Build the master dataset by fetching sequences for all organisms.

        Args:
            df (pd.DataFrame): Cleaned and filtered metadata

        Returns:
            pd.DataFrame: Master dataset with sequences and labels
        """
        logger.info("\nStarting sequence fetching from NCBI...")
        logger.info(f"This may take a while for {len(df)} organisms...")
        logger.info(f"Max {self.max_seqs_per_organism} sequences per organism")

        master_data = {
            "sequence": [],
            "label_order": [],
            "label_class": [],
            "source_accession": [],
        }

        successful_fetches = 0
        failed_fetches = 0
        total_sequences = 0

        # Progress bar
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching sequences"):
            assembly_acc = row["assembly_accession"]
            organism = row["organism_name"]
            order = row["order"]
            cls = row["class"]

            # Fetch sequences
            sequences = self.fetch_protein_sequences(assembly_acc)

            if sequences:
                successful_fetches += 1
                total_sequences += len(sequences)

                # Add to master data
                for seq in sequences:
                    master_data["sequence"].append(seq)
                    master_data["label_order"].append(order)
                    master_data["label_class"].append(cls)
                    master_data["source_accession"].append(assembly_acc)
            else:
                failed_fetches += 1
                logger.debug(f"No sequences for {organism} ({assembly_acc})")

        logger.info(f"\n{'='*60}")
        logger.info(f"Sequence Fetching Summary:")
        logger.info(f"{'='*60}")
        logger.info(f"Successful fetches: {successful_fetches}/{len(df)}")
        logger.info(f"Failed fetches: {failed_fetches}/{len(df)}")
        logger.info(f"Total sequences collected: {total_sequences}")
        logger.info(
            f"Average sequences per organism: {total_sequences/successful_fetches:.2f}"
        )

        # Convert to DataFrame
        master_df = pd.DataFrame(master_data)

        # Check sequence length distribution
        seq_lengths = master_df["sequence"].str.len()
        logger.info(f"\nSequence Length Statistics:")
        logger.info(f"  Mean: {seq_lengths.mean():.2f}")
        logger.info(f"  Median: {seq_lengths.median():.2f}")
        logger.info(f"  Min: {seq_lengths.min()}")
        logger.info(f"  Max: {seq_lengths.max()}")

        # Check label distribution
        logger.info(f"\nFinal Order distribution:")
        logger.info(f"\n{master_df['label_order'].value_counts()}")

        return master_df

    def save_master_dataset(self, df, output_path="master_dataset.csv"):
        """
        Save the master dataset to CSV.

        Args:
            df (pd.DataFrame): Master dataset
            output_path (str): Output file path
        """
        logger.info(f"\nSaving master dataset to {output_path}...")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} sequences")
        logger.info(f"File size: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    def run(self, input_csv="taxon.csv", output_csv="master_dataset.csv"):
        """
        Run the complete data preparation pipeline.

        Args:
            input_csv (str): Input metadata file
            output_csv (str): Output master dataset file
        """
        logger.info("=" * 60)
        logger.info("Starting Data Preparation Pipeline")
        logger.info("=" * 60)

        # Step 1: Load and clean metadata
        df = self.load_and_clean_metadata(input_csv)

        # Step 2: Filter low-frequency Orders
        df = self.filter_low_frequency_orders(df)

        # Step 3: Check Class diversity
        self.check_class_diversity(df)

        # Step 4: Build master dataset with sequences
        master_df = self.build_master_dataset(df)

        # Step 5: Save master dataset
        self.save_master_dataset(master_df, output_csv)

        logger.info("\n" + "=" * 60)
        logger.info("Data Preparation Complete!")
        logger.info("=" * 60)
        logger.info(f"Output saved to: {output_csv}")
        logger.info(f"Ready for model training.")


def main():
    """Main execution function."""

    # Check if email is configured
    if config.NCBI_EMAIL == "your.email@example.com":
        print("\n" + "!" * 60)
        print("ERROR: NCBI Email not configured!")
        print("!" * 60)
        print("\nPlease edit config.py and set your NCBI_EMAIL.")
        print("This is REQUIRED by NCBI's usage policy.")
        print("\nExample:")
        print("  NCBI_EMAIL = 'your.email@example.com'")
        print("\nOptionally, also set NCBI_API_KEY for faster access:")
        print("  NCBI_API_KEY = 'your_api_key_here'")
        print("\nGet an API key at: https://www.ncbi.nlm.nih.gov/account/")
        print("\n" + "!" * 60)
        sys.exit(1)

    # Run pipeline
    pipeline = DataPreparation()
    pipeline.run(input_csv="taxon.csv", output_csv="master_dataset.csv")


if __name__ == "__main__":
    main()
