"""
Convert csv file to huggingface dataset set up for nli task.

Usage: python csv_to_hf.py 
        --csv_path <path_to_csv> 
        --text_column <name of column containing text to cluster> 
        --hf_name <name_of_hf_dataset>
"""
import argparse
from datasets import Dataset
import pandas as pd


def convert_csv_to_hf(csv_path, text_column, hf_name):
    """
    Convert csv file to huggingface dataset set up for nli task.
    """
    # read csv file
    df = pd.read_csv(csv_path)
    # create huggingface dataset
    dataset = Dataset.from_pandas(df)
    # rename text column
    dataset = dataset.rename_column(text_column, "text")
    # save dataset
    dataset.save_to_disk(hf_name)


def main():
    """
    Convert csv file to huggingface dataset set up for nli task.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to csv file containing text to cluster",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
        help="Name of column containing text to cluster",
    )
    parser.add_argument(
        "--hf_name",
        type=str,
        required=True,
        help="Name of huggingface dataset to create",
    )
    args = parser.parse_args()
    # check args.  If any are missing, display usage and exit
    if not args.csv_path or not args.text_column or not args.hf_name:
        parser.print_usage()
        exit(1)
    convert_csv_to_hf(args.csv_path, args.text_column, args.hf_name)
