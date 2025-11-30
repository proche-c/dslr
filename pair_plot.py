"""
Pair Plot Script

This script generates a Seaborn pairplot for numeric features in a dataset,
analyzing correlations and variances between features before plotting. It
also prints information about strongly correlated feature pairs and features
with low variance between group means.

The main steps performed are:
1. Read a dataset from the given file path.
2. Compute feature homogeneity using `calculate_homogeneity`.
3. Compute Pearson correlation for all numeric feature pairs using `calculate_pearson`.
4. Print:
   - Strongly correlated features (|Pearson correlation| >= 0.8)
   - Features with very low variance between group means (<= 50)
5. Generate a Seaborn pairplot of all numeric features, colored by the target label.

Usage
-----
python3 pair_plot.py <dataset_file>

Parameters
----------
filename : str
    Path to the CSV dataset file.

Notes
-----
- Strong correlation is defined as |Pearson correlation| >= 0.8.
- Low variance features are those with variance between group means <= 50.
- The pairplot visualizes relationships between numeric features by the target label (house).

Dataset Subject Example
-----------------------
When using the subject dataset, the features selected for the pairplot are:
  - Astronomy
  - Ancient Runes
  - Charms
"""

import argparse
import signal
from plots.pair_plot import *
from utils.utils import read_file, termination_handler

def arguments_configuration():
    parser = argparse.ArgumentParser(
        prog='Pair plot',
        description='Given a dataset path and a feature, this script plots a pair plot matrix'
    )

    parser.add_argument('filename')

    return parser.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, termination_handler)
    signal.signal(signal.SIGTERM, termination_handler)
    args = arguments_configuration()
    df = read_file(args.filename)
    pair_plot(df)