import pandas as pd
from utils import config
import matplotlib.pyplot as plt
import seaborn as sns
from describe.describe_class import Describe
from utils.utils import print_variances_between_means

def plot_histogram(df:pd.DataFrame, feature):
    """
    Plot histograms of a given feature separated by Hogwarts houses.

    This function generates two subplots:
    - A Matplotlib histogram overlaying distributions of the given feature
      for each house.
    - A Seaborn histogram with hue separation for houses.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the feature and the target label (houses).
    feature : str
        The name of the column to plot.
    """
    try:
        data = df[feature]

    except Exception as e:
        print(f"An exception type {type(e).__name__} has ocurred, please check the input file")
        exit(1)

    house_colors = {
        "Gryffindor": "red",
        "Slytherin": "green",
        "Ravenclaw": "blue",
        "Hufflepuff": "gold"
    }

    plt.rcParams["figure.figsize"] = (15,5)
    plt.subplot(1, 2, 1)
    for house, color in house_colors.items():
        subset = df[df[config.target_label] == house][feature]
        plt.hist(subset, alpha=0.4, label=house, color=color)
    plt.title(f'{feature} by {config.target_label}')
    plt.legend()


    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=feature, hue=config.target_label, palette=house_colors)
    plt.title(f'{feature} by {config.target_label}')
    plt.show()


def calculate_homogeneity(df:pd.DataFrame):
    """
    Calculate inter-group variance for all features and determine the most homogeneous one.

    This function:
    - Computes the mean of each feature for each Hogwarts house (using Describe class).
    - Builds a matrix of means for all groups.
    - Computes the variance of those means for each feature manually (no `.mean()`).
    - Returns the variance per feature and the feature with the lowest variance.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing numeric features and the target label.

    Returns
    -------
    variances : dict
        Mapping {feature_name: inter_group_variance}.
    feature_homogeneity : str
        The feature with the lowest inter-group variance.

    Notes
    -----
    The most homogeneous feature is the one whose means differ the least
    between houses. Lower variance = higher homogeneity.
    """
    group_means = {}
    variance = 0
    variances = {}
    feature_homogeneity = ''

    try:
        for house, group in df.groupby(config.target_label):
            d = Describe(group)
            group_means[house] = d.result.loc['mean']
    except KeyError:
        print(f"Target label '{config.target_label}' not found in dataset. Check input file and config.")
        exit(1)
    except Exception as e:
        print(f"An exception type {type(e).__name__} has ocurred at plots.histogram trying to obtain the df_means ")
        exit(1)

    df_means = pd.DataFrame(data=group_means).T
    if df_means.empty:
        return {}, None

    for feature in df_means.columns:
        global_mean = df_means[feature].sum() / len(group_means)
        df_means[f'{feature}_diff'] = (df_means[feature] - global_mean) ** 2
        current_variance = df_means[f'{feature}_diff'].sum() / (len(group_means) - 1)
        variances[feature] = current_variance
        if feature == df_means.columns[0]:
            variance = current_variance
            feature_homogeneity = feature
        else:
            if current_variance < variance:
                variance = current_variance
                feature_homogeneity = feature
    
    return variances, feature_homogeneity


def histogram(df: pd.DataFrame):
    """
    Compute feature homogeneity, display variances, and plot histogram
    of the most homogeneous feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing features and target label.
    """
    variances, feature = calculate_homogeneity(df)
    if not variances or not feature:
        print("No numeric values in data")
    else:
        print_variances_between_means(variances)
        plot_histogram(df, feature)

