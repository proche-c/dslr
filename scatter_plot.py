"""
Scatter Plot Script

This script reads a dataset and automatically identifies the pair of numeric features 
that are most strongly correlated (closest to ±1 in Pearson correlation). It then 
plots a scatter plot comparing these two features, showing both the raw values and 
standardized values (z-score normalization performed manually).

Functionality:
- Reads a CSV (or supported format) dataset from the given filename.
- Selects only numeric columns for analysis.
- Computes Pearson correlation manually for all unique feature pairs.
- Determines the pair of features with the strongest correlation.
- Plots two scatter plots:
    1. Raw feature values.
    2. Standardized feature values (mean = 0, std = 1).
- Handles missing values by dropping them.
- Handles datasets with no numeric features gracefully, printing an error message.

Usage:
    python3 scatter_plot.py dataset.csv

Arguments:
    filename : str
        Path to the dataset file to analyze.

Notes:
- The Pearson correlation is computed manually without using pandas or numpy built-ins.
- If the dataset contains no numeric columns or all pairs have zero variance, 
  no scatter plot is generated.
- The script gracefully handles SIGINT and SIGTERM signals for safe termination.
- For the subject dataset, the pair with the highest correlation is:
      Astronomy and Defense Against the Dark Arts --> Pearson: -1
"""

import argparse
import signal
from plots.scatter_plot import *
from utils.utils import read_file, termination_handler

def arguments_configuration():
    parser = argparse.ArgumentParser(
        prog='Scatter plot',
        description='Given a dataset path and a feature, this script plots a scatter plot of the two features that are similar'
    )

    parser.add_argument('filename')

    return parser.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, termination_handler)
    signal.signal(signal.SIGTERM, termination_handler)
    args = arguments_configuration()
    df = read_file(args.filename)
    scatter_plot(df)