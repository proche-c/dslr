import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import standarize, get_mean, get_std

def plot_scatter(df:pd.DataFrame, feature_1, feature_2):
    try:
        df_to_show = df[[feature_1, feature_2]].dropna().astype(float)
        # print(df_to_show)
    except Exception as e:
        print(f"An exception type {type(e).__name__} has ocurred, please check the input file")
        exit(1)

    plt.rcParams["figure.figsize"] = (15,5)

    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_to_show)

    plt.subplot(1, 2, 2)
    str_1 = f'{feature_1} standarized'
    str_2 = f'{feature_2} standarized'
    mean_1 = get_mean(df_to_show[feature_1])
    mean_2 = get_mean(df_to_show[feature_2])
    std_1 = get_std(df_to_show[feature_1], mean_1)
    std_2 = get_std(df_to_show[feature_2], mean_2)
    df_to_show[str_1] = standarize(df_to_show[feature_1], mean_1, std_1)
    df_to_show[str_2] = standarize(df_to_show[feature_2], mean_2, std_2)
    df_standarized = df_to_show[[str_1, str_2]]
    sns.scatterplot(data=df_standarized)

    plt.show()

def calculate_correlation(df_pair:pd.DataFrame, feature_1, feature_2):
    df_pair =df_pair.dropna().astype(float)
    mean_1 = df_pair[feature_1].sum() / df_pair.__len__()
    mean_2 = df_pair[feature_2].sum() / df_pair.__len__()
    df_pair['diff_product'] = (df_pair[feature_1] - mean_1) * (df_pair[feature_2] - mean_2)
    df_pair['diff1**2'] = (df_pair[feature_1] - mean_1) ** 2
    df_pair['diff2**2'] = (df_pair[feature_2] - mean_2) ** 2
    return df_pair['diff_product'].sum() / ((df_pair['diff1**2'].sum() * df_pair['diff2**2'].sum()) ** 0.5)

def calculate_pearson(df:pd.DataFrame):
    mask = 1
    result_1 = ""
    result_2 = ""
    df = df.select_dtypes(include=['float', 'int'])
    analised = []
    pearsons = []
    for feature_1 in df.columns:
        for feature_2 in df.columns:
            if feature_1 != feature_2 and feature_2 not in analised:
                try:
                    df_pair = df[[feature_1, feature_2]]
                except Exception as e:
                    print(f"An exception type {type(e).__name__} has ocurred, please check the input file")
                    exit(1)
                pearson = calculate_correlation(df_pair, feature_1, feature_2)
                t = (feature_1, feature_2, pearson)
                pearsons.append(t)
                if pearson >= 0 and (1 - pearson) <= mask:
                    mask = 1 - pearson
                    result_1 = feature_1
                    result_2 = feature_2
                if pearson < 0 and (1 + pearson) <= mask:
                    mask = 1 + pearson
                    result_1 = feature_1
                    result_2 = feature_2
        analised.append(feature_1)
    return pearsons, result_1, result_2

def scatter_plot(df:pd.DataFrame, args):
    pearsons, feature_1, feature_2 = calculate_pearson(df)
    plot_scatter(df, feature_1, feature_2)

# Astronomy and defense against the dark arts --> pearson: -1