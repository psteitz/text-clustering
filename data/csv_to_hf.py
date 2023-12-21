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


def _get_df(csv_path: str) -> pd.DataFrame:
    """
    Get pandas dataframe from csv file, handling decoding errors.

    Args:
        csv_path (str): Path to csv file containing text to cluster.

    Returns:
        pd.DataFrame: Dataframe containing text to cluster.
    """
    # read csv file
    # if a decoding error occurs, try adding encoding="cp1252" (Windows encoding)
    try:
        df = pd.read_csv(csv_path)
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp1252")
    return df


def convert_csv_to_hf(csv_path, text_column, hf_name):
    """
    Convert csv file to huggingface dataset set up for nli task.
    Rename text column to "text", drope other columns and rows with empty text.

    Args:
        csv_path (str): Path to csv file containing text to cluster.
        text_column (str): Name of column containing text to cluster.
        hf_name (str): Name of huggingface dataset to create.
    """
    # read csv file
    # if a decoding error occurs, try adding encoding="cp1252" (Windows encoding)
    df = _get_df(csv_path)
    # drop columns other than text column
    df = df.drop(df.columns.difference([text_column]), axis=1)
    # drop rows with empty text
    df = df.dropna(subset=[text_column])
    # create huggingface dataset
    dataset = Dataset.from_pandas(df, preserve_index=False)
    # rename text column
    dataset = dataset.rename_column(text_column, "text")
    # save dataset
    dataset.save_to_disk(hf_name)


def get_column_names(csv_path: str) -> list:
    """
    Get column names from csv file.

    Args:
        csv_path (str): Path to csv file containing text to cluster.

    Returns:
        list: List of column names.
    """
    # read csv file
    df = _get_df(csv_path)
    # get column names
    return list(df.columns)


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
