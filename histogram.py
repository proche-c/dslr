
"""
Histogram Script
================

This script loads a dataset and displays the histogram of the most
homogeneous numeric feature across Hogwarts houses.

The homogeneity is computed by comparing the inter-group variance
between the mean values of each feature for the different houses.
The feature with the lowest variance is considered the most
homogeneous one.

The script performs the following steps:

1. Parse command-line arguments to obtain the dataset file path.
2. Load the dataset into a pandas DataFrame.
3. Compute homogeneity and determine the most homogeneous feature.
4. Plot two histograms (Matplotlib + Seaborn) for that feature.

The script also handles graceful termination when receiving SIGINT
(Ctrl+C) or SIGTERM.

Usage
-----
    python3 histogram.py <dataset.csv>

Parameters
----------
filename : str
    Path to the dataset CSV file.

Notes
-----
- This script assumes the dataset includes a target label column
  (e.g., Hogwarts House) as defined in `utils.config.target_label`.
- If the dataset contains no numeric features, the script will notify
  the user and exit safely.
- For the official `dslr` subject dataset, the feature with the lowest
  variance between house means — and therefore the most homogeneous one —
  is **Care of Magical Creatures**.
"""

import argparse
import signal
from plots.histogram import *
from utils.utils import read_file, termination_handler

def arguments_configuration():
    parser = argparse.ArgumentParser(
        prog='Histogram',
        description='Given a dataset path and a feature, this script plots an histogram with the scores of said feature for each Hogwarts House'
    )

    parser.add_argument('filename')

    return parser.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, termination_handler)
    signal.signal(signal.SIGTERM, termination_handler)
    args = arguments_configuration()
    df = read_file(args.filename)
    histogram(df)