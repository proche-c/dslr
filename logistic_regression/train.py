import pandas as pd
import sys
import numpy as np
from utils.utils import get_mean, get_std, standarize

# Hiperparametros. Estos seguramente habra que ponerlos como argumentos
# Con estos valores de hiperparametros yo creo que el modelo esta mas o menos bien ajustado
MAX_STEPS = 15000
MIN_STEP_SIZE = 0.00005
LR = 0.01

# constante para que nunca se haga log(0), matematicamente expresado, ln(0), que tiende a - inf
eps = 1e-15

class Model():
    def __init__(self, df:pd.DataFrame, feature_1:str, feature_2:str, feature_3:str, classifier:str, house:str):
        try:
            self.feature_1 = feature_1
            self.feature_2 = feature_2
            self.feature_3 = feature_3
            self.df = df[[classifier, feature_1, feature_2, feature_3]]
            self.df = self.df.dropna()
            self.mean_1 = get_mean(self.df[self.feature_1])
            self.mean_2 = get_mean(self.df[self.feature_2])
            self.mean_3 = get_mean(self.df[self.feature_3])
            self.std_1 = get_std(self.df[self.feature_1], self.mean_1)
            self.std_2 = get_std(self.df[self.feature_2], self.mean_2)
            self.std_3 = get_std(self.df[self.feature_3], self.mean_3)
        except Exception as e:
            print(f"An exception type {type(e).__name__} has ocurred, please check the input file and the features")
            sys.exit(1)
        self.df['outcome'] = (self.df[classifier] == house).astype(int)


# Creo un dataframe con steps y costo porque a la mejor hace falta luego

    def train(self):
        theta = [0, 0, 0, 0]
        self.df[f'{self.feature_1}_standarized'] = standarize(self.df[self.feature_1], self.mean_1, self.std_1)
        self.df[f'{self.feature_2}_standarized'] = standarize(self.df[self.feature_2], self.mean_2, self.std_2)
        self.df[f'{self.feature_3}_standarized'] = standarize(self.df[self.feature_3], self.mean_3, self.std_3)
        cost = {}
        for step in range(MAX_STEPS):
            self.df['sigmoid'] = 1 / (1 + np.exp(-(theta[0] + theta[1] * self.df[f'{self.feature_1}_standarized'] + theta[2] * self.df[f'{self.feature_2}_standarized'] + theta[3] * self.df[f'{self.feature_3}_standarized'])))    
            self.df['cost'] = - self.df['outcome'] * np.log(self.df['sigmoid'] + eps) - (1 - self.df['outcome']) * np.log(1 - self.df['sigmoid'] + eps)
            cost[step] = (self.df['cost'].sum() / self.df.__len__())
            self.df['partial_lost_0'] = self.df['sigmoid'] - self.df['outcome']
            self.df['partial_lost_1'] = (self.df['sigmoid'] - self.df['outcome']) * self.df[f'{self.feature_1}_standarized']
            self.df['partial_lost_2'] = (self.df['sigmoid'] - self.df['outcome']) * self.df[f'{self.feature_2}_standarized']
            self.df['partial_lost_3'] = (self.df['sigmoid'] - self.df['outcome']) * self.df[f'{self.feature_3}_standarized']
            delta_0 = self.df['partial_lost_0'].sum() / self.df.__len__()
            delta_1 = self.df['partial_lost_1'].sum() / self.df.__len__()
            delta_2 = self.df['partial_lost_2'].sum() / self.df.__len__()
            delta_3 = self.df['partial_lost_3'].sum() / self.df.__len__()
            theta_new_0 = theta[0] - LR * delta_0
            theta_new_1 = theta[1] - LR * delta_1
            theta_new_2 = theta[2] - LR * delta_2
            theta_new_3 = theta[3] - LR * delta_3
            theta_new = [theta_new_0, theta_new_1, theta_new_2, theta_new_3]
            if all(abs(theta_new[i] - theta[i]) < MIN_STEP_SIZE for i in range(4)):
                    break
            theta = [theta_new[0], theta_new[1], theta_new[2], theta_new[3]]
        self.df_cost = pd.DataFrame(cost.items(), columns=["step", "cost"])
        # print(self.df_cost.head())
        # print(self.df_cost.tail())
        return theta