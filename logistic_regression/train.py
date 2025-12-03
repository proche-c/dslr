import pandas as pd
import sys
import numpy as np
import math
from utils.utils import get_mean, get_std, standarize

# Small constant added to sigmoid outputs when computing log-loss
# to avoid evaluating log(0), which tends to −∞ and breaks training.
eps = 1e-15

class Model():
    """
    Logistic regression model using three numerical features.

    This class extracts three selected features from a DataFrame, computes
    their mean and standard deviation, standardizes the data, encodes the
    target class as a binary variable, and trains logistic regression
    parameters using gradient descent.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing the classifier column and the 3 selected features.
    feature_1, feature_2, feature_3 : str
        Names of the features to be used as predictors.
    classifier : str
        Column name representing the categorical target (houses).
    house : str
        The specific house to classify (positive class = 1).

    Notes
    -----
    - All rows containing NaN values in the selected columns are removed.
    - Means and standard deviations are precomputed for later standardization.
    - If any feature has standard deviation 0, training cannot proceed.
    """
    def __init__(self, df:pd.DataFrame, feature_1:str, feature_2:str, feature_3:str, classifier:str, house:str):
        """
        Initialize the model by extracting and validating the required columns,
        removing missing values, computing statistics (mean and std), and creating
        the binary outcome column for the selected target house.

        Raises
        ------
        Exception
            If any of the requested columns does not exist or cannot be processed.
        SystemExit
            If initialization fails due to invalid input data.
        """
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.feature_3 = feature_3
        try:
            self.df = df[[classifier, feature_1, feature_2, feature_3]]
            df[self.feature_1] = pd.to_numeric(df[self.feature_1], errors="coerce")
            df[self.feature_2] = pd.to_numeric(df[self.feature_2], errors="coerce")
            df[self.feature_3] = pd.to_numeric(df[self.feature_3], errors="coerce")
            self.df = self.df.dropna()
            if self.df.empty:
                print("No numeric values in data")
                exit(1)
            if self.df[self.feature_1].empty or self.df[self.feature_2].empty or self.df[self.feature_3].empty:
                print("Empty column")
                exit(1)
            self.mean_1 = get_mean(self.df[self.feature_1])
            self.mean_2 = get_mean(self.df[self.feature_2])
            self.mean_3 = get_mean(self.df[self.feature_3])
            self.std_1 = get_std(self.df[self.feature_1], self.mean_1)
            self.std_2 = get_std(self.df[self.feature_2], self.mean_2)
            self.std_3 = get_std(self.df[self.feature_3], self.mean_3)
        except Exception as e:
            print(f"An exception type {type(e).__name__} has ocurred, please check the input file and the features")
            sys.exit(1)
        if math.isnan(self.mean_1) or math.isnan(self.mean_2) or math.isnan(self.mean_3) \
        or math.isnan(self.std_1) or math.isnan(self.std_2) or math.isnan(self.std_3):
            print("Error: mean or std could not be computed. Column may contain non-numeric values.")
            sys.exit(1)
        self.df['outcome'] = (self.df[classifier] == house).astype(int)


# Creo un dataframe con steps y costo porque a la mejor hace falta luego

    def train(self, args):
        """
        Train the logistic regression model using gradient descent.

        The method:
        - standardizes all three features,
        - computes the sigmoid prediction for each row,
        - evaluates the log-loss cost function,
        - computes partial derivatives,
        - updates model parameters using the learning rate,
        - stops early when all parameter updates fall below `min_step_size`.

        Parameters
        ----------
        args : Namespace
            Arguments namespace containing:
            - lr : float
                Learning rate for gradient descent.
            - max_steps : int
                Maximum number of gradient descent iterations.
            - min_step_size : float
                Minimum absolute update size for early stopping.

        Returns
        -------
        list[float]
            A list `[θ0, θ1, θ2, θ3]` representing the trained parameters.
            If standard deviation is zero for any feature, returns the initial
            parameters without training.

        Notes
        -----
        - If any feature has `std = 0`, standardization is impossible and
        training is aborted with a warning.
        - Uses `eps` to avoid evaluating log(0) in the cost computation.
        - Stores a DataFrame `df_cost` with (step, cost) to inspect convergence.
        """
        theta = [0, 0, 0, 0]
        if args.lr <= 0 or args.max_steps <= 0 or args.min_step_size <= 0:
            print("Learning rate, max_steps and min_step_size must be positive numbers")
            exit(1)
        if self.std_1 == 0 or self.std_2 == 0 or self.std_3 == 0:
            print("Invalid data, standard desviation in a column is zero, cannot standardize")
            sys.exit(1)
        try:
            self.df[f'{self.feature_1}_standarized'] = standarize(self.df[self.feature_1], self.mean_1, self.std_1)
            self.df[f'{self.feature_2}_standarized'] = standarize(self.df[self.feature_2], self.mean_2, self.std_2)
            self.df[f'{self.feature_3}_standarized'] = standarize(self.df[self.feature_3], self.mean_3, self.std_3)
        except Exception as e:
            print(f"An exception type {type(e).__name__} has ocurred, please check the input file and the features")
            sys.exit(1)
        cost = {}
        for step in range(args.max_steps):
            z = (theta[0]
                + theta[1] * self.df[f'{self.feature_1}_standarized']
                + theta[2] * self.df[f'{self.feature_2}_standarized']
                + theta[3] * self.df[f'{self.feature_3}_standarized'])
            # if z < 0 --> np.exp(-z) = np.exp(1000) -> INF -> overflow, np.exp(z) / (1 + np.exp(z) avoids the overflow
            self.df['sigmoid'] = np.where(
                z >= 0,
                1 / (1 + np.exp(-z)),
                np.exp(z) / (1 + np.exp(z))
            )
            
            self.df['cost'] =(- self.df['outcome'] * np.log(self.df['sigmoid'] + eps)
                            - (1 - self.df['outcome']) * np.log(1 - self.df['sigmoid'] + eps))
            cost[step] = (self.df['cost'].sum() / self.df.__len__())
            self.df['partial_lost_0'] = self.df['sigmoid'] - self.df['outcome']
            self.df['partial_lost_1'] = (self.df['sigmoid'] - self.df['outcome']) * self.df[f'{self.feature_1}_standarized']
            self.df['partial_lost_2'] = (self.df['sigmoid'] - self.df['outcome']) * self.df[f'{self.feature_2}_standarized']
            self.df['partial_lost_3'] = (self.df['sigmoid'] - self.df['outcome']) * self.df[f'{self.feature_3}_standarized']
            delta_0 = self.df['partial_lost_0'].sum() / self.df.__len__()
            delta_1 = self.df['partial_lost_1'].sum() / self.df.__len__()
            delta_2 = self.df['partial_lost_2'].sum() / self.df.__len__()
            delta_3 = self.df['partial_lost_3'].sum() / self.df.__len__()
            theta_new_0 = theta[0] - args.lr * delta_0
            theta_new_1 = theta[1] - args.lr * delta_1
            theta_new_2 = theta[2] - args.lr * delta_2
            theta_new_3 = theta[3] - args.lr * delta_3
            theta_new = [theta_new_0, theta_new_1, theta_new_2, theta_new_3]
            if all(abs(theta_new[i] - theta[i]) < args.min_step_size for i in range(4)):
                    break
            theta = [theta_new[0], theta_new[1], theta_new[2], theta_new[3]]
        self.df_cost = pd.DataFrame(cost.items(), columns=["step", "cost"])
        return theta