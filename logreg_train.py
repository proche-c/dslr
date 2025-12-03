"""
Logistic Regression Training Script
===================================

This script trains a multi-class logistic regression model using one-vs-all
classification. It loads a dataset, preprocesses the selected features
(standardization included), trains a separate binary classifier for each class,
stores the learned parameters, and plots the cost function evolution.

Workflow
--------
1. Parse command-line arguments (dataset file, learning rate, max steps, etc.).
2. Load the dataset from the provided path.
3. Identify the unique class labels in the target column.
4. For each class:
   - Initialize a `Model` instance configured for that class.
   - Train a logistic regression classifier with gradient descent.
   - Store the resulting parameters (weights, means, standard deviations).
   - Record cost evolution across training steps.
5. Save all learned model parameters to `weights.json`.
6. Display training cost plots for all trained models.

Command-line Arguments
----------------------
filename : str
    Path to the dataset to load.

--max_steps, -ms : int, optional (default=15000)
    Maximum number of gradient descent iterations allowed during training.

--min_step_size, -mss : float, optional (default=0.00005)
    Minimum update magnitude required to continue training. If all parameter
    updates fall below this threshold, training stops early.

-lr : float, optional (default=0.01)
    Learning rate controlling the magnitude of gradient descent updates.

Output
------
weights.json
    JSON file containing learned model parameters for each class, including:
    - theta values (θ0, θ1, θ2, θ3)
    - means and standard deviations of the features used for normalization

Plots
-----
A matplotlib window displaying the cost function evolution for each class
during training.

"""

import argparse
import signal
import json
import matplotlib.pyplot as ptl
from logistic_regression.train import Model
from utils.utils import read_file, termination_handler
from utils import config

def arguments_configuration():
    parser = argparse.ArgumentParser(
        prog='Logistic regression train',
        description='Given a dataset path, the program will train a multi-classifier model'
    )

    parser.add_argument('filename')

    # Hiperparameters of logistic regression model
    parser.add_argument('--max_steps', '-ms', default=15000, type=int, help='Maximum number of gradient-descent iterations allowed during training')
    parser.add_argument('--min_step_size', '-mss', default=0.00005, type=int, help='Minimum allowable gradient-descent step size')
    parser.add_argument('-lr', default=0.01, type=float, help='Controls the magnitude of each gradient-descent update')

    return parser.parse_args()

def plot_costs(costs:list):
    ptl.rcParams["figure.figsize"] = (25, 10)

    num_models = len(costs)


    for i, (house, df_cost) in enumerate(costs):
        ptl.subplot(1,num_models, i + 1)
        ptl.plot(df_cost['step'], df_cost['cost'])
        ptl.title(house)
        ptl.xlabel('step')
        ptl.ylabel('cost')

    ptl.tight_layout()
    ptl.show()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, termination_handler)
    signal.signal(signal.SIGTERM, termination_handler)
    args = arguments_configuration()
    df = read_file(args.filename)
    houses = df[config.target_label].unique().tolist()
    weights = {}
    costs = []
    for house in houses:
        a = Model(df, config.feature_1, config.feature_2, config.feature_3, config.target_label, house)
        theta = a.train(args)
        costs.append((house, a.df_cost))
        means = [a.mean_1, a.mean_2, a.mean_3]
        stds = [a.std_1, a.std_2, a.std_3]
        weights[house] = {"theta_0": theta[0],
                          "theta_1": theta[1],
                          "theta_2": theta[2],
                          "theta_3": theta[3],
                          "means": [a.mean_1, a.mean_2, a.mean_3],
                          "stds": [a.std_1, a.std_2, a.std_3],
                        #   "features": [FEATURE_1, FEATURE_2, FEATURE_3]
                        }

    with open('weights.json', 'w') as jsonfile:
        json.dump(weights, jsonfile)

    plot_costs(costs)