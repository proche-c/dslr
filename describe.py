"""
Script: describe.py
-------------------

This script serves as the command-line interface for the custom Describe class,
which manually computes descriptive statistics for a dataset without using any
built-in statistical functions. It mimics the behavior of the pandas `describe()`
method while complying with the project constraint of computing all metrics
manually.

Functionality:
    - Reads a CSV dataset.
    - Uses the Describe class to compute count, mean, std, min, max,
      and percentile statistics for each numerical column.
    - Prints the computed summary table.
    - Optionally compares the manual implementation with pandas' built-in
      describe() for debugging purposes (via the --test flag).

Usage:
    python describe.py dataset_train.csv
    python describe.py dataset_train.csv --test

Arguments:
    filename : Path to the CSV dataset.
    --test / -t : Optional flag to validate results against pandas.describe().

This script is part of the Data Science × Logistic Regression project.
"""

from utils.utils import read_file
from describe.describe_class import Describe
from test.describe import test_describe
import pandas as pd
import argparse
import os.path
import sys

def arguments_configuration():
    """
    Configures and parses command-line arguments for the describe tool.

    Returns:
        argparse.Namespace: An object containing parsed arguments:
            - filename (str): The path to the input CSV dataset.
            - test (bool): Whether to run the optional comparison between the
              manual Describe output and pandas' describe() method.

    The function defines:
        -- A required positional argument: 'filename'.
        -- An optional '--test' / '-t' flag to trigger result comparison.

    This parser is used to control how the script behaves when executed from
    the command line.
    """
    parser = argparse.ArgumentParser(
        prog='Describe',
        description='Given a dataset, this program mimic the behaviour of describe method, and display the results'
    )

    parser.add_argument('filename')
    parser.add_argument('--test', '-t', action='store_true', help='Compare the results of the manual describe to the results of describe method of pandas')

    return parser.parse_args()


if __name__ == "__main__":
    args = arguments_configuration()
    file_path = os.path.abspath(args.filename)
    data = read_file(file_path)
    d = Describe(data)
    d.print()
    if args.test:
        test_describe(data)
