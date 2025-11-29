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
        theta = a.train()
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