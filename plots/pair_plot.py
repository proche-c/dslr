import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plots.histogram import calculate_homogeneity
from plots.scatter_plot import calculate_pearson
from utils.utils import print_variances_between_means, print_pearsons
from utils import config

def get_strong_correlation(pearsons:list):
    """
    Filter Pearson correlation results to return only strongly correlated feature pairs.

    Parameters
    ----------
    pearsons : list of tuples
        List of Pearson correlation results. Each tuple is
        (feature_1: str, feature_2: str, pearson_value: float).

    Returns
    -------
    list of tuples
        Feature pairs with strong correlation, defined as
        |pearson_value| >= 0.8.

    Notes
    -----
    - Strong positive correlation: pearson_value >= 0.8
    - Strong negative correlation: pearson_value <= -0.8
    """
    strong_correlated_features = []
    for item in pearsons:
        if item[2] > 0.8 or item[2] < -0.8:
            strong_correlated_features.append(item)
    return strong_correlated_features

def get_low_variance_between_means(variances_between_means:dict):
    """
    Filter features to identify those with low variance between group means.

    Parameters
    ----------
    variances_between_means : dict
        Dictionary mapping feature names to variance of means between groups.

    Returns
    -------
    dict
        Features whose variance between group means is <= 50.

    Notes
    -----
    - Low variance indicates higher homogeneity across groups.
    - Threshold of 50 is used as an arbitrary cutoff for “very low” variance.
    """
    low_variance_between_means = {}
    for house, variance in variances_between_means.items():
        if variance <= 50:
            low_variance_between_means[house] = variance
    return low_variance_between_means

def pair_plot(df:pd.DataFrame):
    """
    Analyze correlations and variances, print results, and generate a pairplot for numeric features.

    This function:
    - Computes feature homogeneity (variance between group means) using `calculate_homogeneity`.
    - Computes Pearson correlation for all numeric feature pairs using `calculate_pearson`.
    - Prints strongly correlated feature pairs.
    - Prints features with very low variance between group means.
    - Generates a Seaborn pairplot of all numeric features colored by the target label.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing numeric features and the target label.

    Notes
    -----
    - Strong correlation is defined as |Pearson correlation| >= 0.8.
    - Low variance features are those with variance between group means <= 50.
    - The pairplot visualizes relationships between numeric features by house.
    - With the dataset for the subject:
        - Most homogeneous feature (lowest variance between means) is
          "Care of Magical Creatures".
        - Strongest negative correlation occurs between
          "Astronomy" and "Defense Against the Dark Arts" (Pearson = -1).
    """
    variances_between_means, feature = calculate_homogeneity(df)
    pearsons, feature_1, feature_2 = calculate_pearson(df)

    strong_correlated_features = get_strong_correlation(pearsons)
    if len(strong_correlated_features) != 0:
        print("The following features are strongly correlated")
        print_pearsons(strong_correlated_features)

    low_variance_between_means = get_low_variance_between_means(variances_between_means)
    if len(low_variance_between_means) != 0:
        print("The following features have very low variance between means")
        print_variances_between_means(low_variance_between_means)

    numeric_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    if len(numeric_columns) < 2:
       print("No enough numeric columns in data")
       return
    
    try:
        features_and_house = [config.target_label] + [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
        df = df[features_and_house].dropna().astype(float, errors='ignore')
    except KeyError:
        print(f"Target label '{config.target_label}' not found in dataset. Check input file and config.")
        exit(1)
    except Exception as e:
        print(f"An exception type {type(e).__name__} has ocurred at plots.histogram trying to obtain the df_means ")
        exit(1)

    if df.empty:
        print("No numeric values in data")
        return
    
    plt.rcParams.update({
        'axes.labelsize': 6,   
        'xtick.labelsize': 6,    
        'ytick.labelsize': 6    
    })
    sns.pairplot(
        data=df,
        hue=config.target_label,
        palette='Set2',
        plot_kws={'alpha': 0.6, 's': 40}
    )
    plt.show()

