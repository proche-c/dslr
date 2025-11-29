import json
import pandas as pd
import numpy as np
from utils.utils import standarize
from utils import config

def   Predict(df_predict, jsonpath):
    try:
        with open(jsonpath, 'r') as jsonfile:
            weights = json.load(jsonfile)
    except Exception as e:
        print(f'An exception type {type(e).__name__} has ocurred, please check the json file {jsonpath}')

    try:
        df_weights = pd.DataFrame.from_dict(weights, orient='index')
        # features = df_weights.iloc[0]['features']
        features = [config.feature_1, config.feature_2, config.feature_3]
        means = df_weights.iloc[0]['means']
        stds = df_weights.iloc[0]['stds']
        
    except Exception as e:
        print(f'An exception type {type(e).__name__} has ocurred, the data in {jsonpath} could not generate a DataFrame')

    try:
        original_index = df_predict.index.copy()
        df_predict = df_predict[features]
        df_clean = df_predict.dropna().copy()
        df_clean[f'{features[0]}_standarized'] = standarize(df_clean[features[0]], means[0], stds[0])
        df_clean[f'{features[1]}_standarized'] = standarize(df_clean[features[1]], means[1], stds[1])
        df_clean[f'{features[2]}_standarized'] = standarize(df_clean[features[2]], means[2], stds[2])
    except Exception as e:
        print(f'An exception type {type(e).__name__} has ocurred standarizing DataFrame')            

    for house in df_weights.index:
        df_clean[f'z_{house}'] = (
            df_weights.loc[house]['theta_0']
            + df_weights.loc[house]['theta_1'] * df_clean[f'{features[0]}_standarized']
            + df_weights.loc[house]['theta_2'] * df_clean[f'{features[1]}_standarized']
            + df_weights.loc[house]['theta_3'] * df_clean[f'{features[2]}_standarized']
        )
        df_clean[f'p_{house}'] = 1 / (1 + np.exp(- df_clean[f'z_{house}']))

    p_cols = [f'p_{house}' for house in df_weights.index]
    df_clean[config.target_label] = df_clean[p_cols].idxmax(axis=1).str.replace('p_', '')
    df_final = pd.DataFrame(index=original_index)
    df_final[config.target_label] = None
    df_final.loc[df_clean.index, config.target_label] = df_clean[config.target_label]
    return df_final


