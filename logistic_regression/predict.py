import json
import pandas as pd
import numpy as np
from utils.utils import standarize
from utils import config

def   Predict(df_predict, jsonpath):
    """
    Predicts the Hogwarts house for each row in a dataset using a trained
    multiclass logistic regression model and weights stored in a JSON file.

    This function:
    - Loads model parameters (θ, means, stds) from a JSON file.
    - Validates the JSON structure and the numeric integrity of the data.
    - Extracts and converts the required features to numeric form.
    - Drops rows with missing or invalid values.
    - Standardizes the three selected features using the stored means and stds.
    - Computes the linear term z for each house.
    - Applies a numerically stable sigmoid function to obtain probabilities.
    - Selects the most probable class for each row.
    - Returns a DataFrame aligned to the original index with the predictions.

    Parameters
    ----------
    df_predict : pandas.DataFrame
        Input DataFrame containing at least the three feature columns defined in
        `config.feature_1`, `config.feature_2`, and `config.feature_3`.
        Non-numeric values are coerced to NaN and subsequently dropped.

    jsonpath : str
        Path to the JSON file generated during training. The file must contain,
        for each house:
            - "theta_0", "theta_1", "theta_2", "theta_3"
            - "means": list of 3 numerical means
            - "stds": list of 3 numerical std deviations (positive numbers)

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the same index as `df_predict` and a single column
        `config.target_label`, containing the predicted class per row.
        Rows with invalid or non-numeric inputs appear in the output but with
        a `None` prediction.

    Error Handling
    --------------
    The function exits with an explanatory message in the following cases:
    - JSON file not found, unreadable, or malformed.
    - Missing required fields (θ values, means, stds).
    - Invalid numeric values in θ parameters.
    - `means` or `stds` not being lists/tuples of length 3.
    - Non-numeric or negative standard deviations.
    - Input DataFrame lacking required feature columns.
    - All rows becoming NaN after numeric conversion.
    - Overflow when computing the exponential part of the sigmoid function.
    - Invalid probability values (NaN or ±inf).
    - Incorrect or missing target_label during final classification.

    Notes
    -----
    - The sigmoid function uses a numerically stable formulation:
        - For z >= 0:  1 / (1 + exp(-z))
        - For z < 0:   exp(z) / (1 + exp(z))
      to prevent overflow when z is large in magnitude.

    - The output probabilities are compared across houses using `idxmax` to
      determine the final prediction.

    - The returned DataFrame preserves the original indexing, ensuring
      alignment with the input dataset.

    """
    try:
        with open(jsonpath, 'r') as jsonfile:
            weights = json.load(jsonfile)
    except Exception as e:
        print(f'An exception type {type(e).__name__} has ocurred, please check the json file {jsonpath}')
        exit(1)

    try:
        required = {"theta_0","theta_1","theta_2","theta_3","means","stds"}
        for house, data in weights.items():
            if not required.issubset(data):
                print(f"Missing fields in JSON for {house}")
                exit(1)
        df_weights = pd.DataFrame.from_dict(weights, orient='index')
        if df_weights[['theta_0','theta_1','theta_2','theta_3']].isna().any().any():
            print("Invalid numeric values in theta fields")
            exit(1)
        if df_weights.empty:
            print("JSON contains no houses")
            exit(1)
        means = df_weights.iloc[0]['means']
        stds = df_weights.iloc[0]['stds']

    except Exception as e:
        print(f'An exception type {type(e).__name__} has ocurred, the data in {jsonpath} could not generate a DataFrame')
        exit(1)

    try:
        original_index = df_predict.index.copy()
        features = [config.feature_1, config.feature_2, config.feature_3]
        df_predict = df_predict[features]
        df_predict = df_predict.apply(pd.to_numeric, errors="coerce")
        df_clean = df_predict.dropna().copy()
        if df_clean.empty:
            print("No numeric values in data")
            exit(1)
        if df_clean[f'{features[0]}'].empty or df_clean[f'{features[1]}'].empty or df_clean[f'{features[2]}'].empty:
            print("Empty column")
            exit(1)
        if any(not isinstance(x, (int, float)) for x in means):
            print(f"Invalid value for mean: {means}")
            exit(1)
        if any(not isinstance(x, (int, float)) or x <= 0 for x in stds):
            print(f"Invalid value for std: {stds}")
            exit(1)
        if len(means) != 3 or len(stds) != 3:
            print("Means/stds must contain 3 elements")
            exit(1)
        df_clean[f'{features[0]}_standarized'] = standarize(df_clean[features[0]], means[0], stds[0])
        df_clean[f'{features[1]}_standarized'] = standarize(df_clean[features[1]], means[1], stds[1])
        df_clean[f'{features[2]}_standarized'] = standarize(df_clean[features[2]], means[2], stds[2])
    except Exception as e:
        print(f'An exception type {type(e).__name__} has ocurred standarizing DataFrame')
        exit(1)            

    np.seterr(over='raise')
    for house in df_weights.index:
        try:
            df_clean[f'z_{house}'] = (
                df_weights.loc[house]['theta_0']
                + df_weights.loc[house]['theta_1'] * df_clean[f'{features[0]}_standarized']
                + df_weights.loc[house]['theta_2'] * df_clean[f'{features[1]}_standarized']
                + df_weights.loc[house]['theta_3'] * df_clean[f'{features[2]}_standarized']
            )
            df_clean[f'p_{house}'] = np.where(
                df_clean[f'z_{house}'] >= 0,
                1 / (1 + np.exp(- df_clean[f'z_{house}'])),
                np.exp(df_clean[f'z_{house}']) / ( 1 + np.exp(df_clean[f'z_{house}']))
                )
        except FloatingPointError:
            print("Overflow computing exp(z). Values too large. Check your weights or inputs.")
            exit(1)
        except Exception as e:
            print(f'An exception type {type(e).__name__} has ocurred')
            exit(1)
    p_cols = [f'p_{house}' for house in df_weights.index]
    try:
        df_clean[config.target_label] = df_clean[p_cols].idxmax(axis=1).str.replace('p_', '')
    except Exception as e:
        print(f'An exception type {type(e).__name__} has ocurred, check target_label')
        exit(1)
    if df_clean[p_cols].isna().any().any() or np.isinf(df_clean[p_cols]).any().any():
        print("Invalid probability values, check your weights or inputs.")
        exit(1)
    df_final = pd.DataFrame(index=original_index)
    df_final[config.target_label] = None
    df_final.loc[df_clean.index, config.target_label] = df_clean[config.target_label]
    return df_final


