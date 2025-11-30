import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import standarize, get_mean, get_std, print_pearsons

def plot_scatter(df:pd.DataFrame, feature_1, feature_2):
    """
    Plot two scatter plots comparing two numeric features.

    This function creates a figure with two subplots:

    1. A raw scatter plot showing the distribution of the two features.
    2. A scatter plot of the same features after standardization
       (z-score normalization performed manually using get_mean/get_std).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset containing the numeric features.
    feature_1 : str
        Name of the first feature to plot.
    feature_2 : str
        Name of the second feature to plot.

    Notes
    -----
    - Rows with missing values are removed.
    - All values are cast to float.
    - If the input does not contain valid numeric columns,
      the program prints an error message and exits.
    """
    try:
        df_to_show = df[[feature_1, feature_2]].dropna().astype(float)
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
    """
    Manually compute the Pearson correlation coefficient between two numeric features.

    This function performs the full Pearson calculation without using
    pandas or numpy built-ins, following the mathematical formula:

        r = Σ[(x - mean_x)(y - mean_y)] /
            sqrt( Σ(x - mean_x)^2 * Σ(y - mean_y)^2 )

    Parameters
    ----------
    df_pair : pandas.DataFrame
        DataFrame containing exactly the two features being compared.
    feature_1 : str
        First feature name.
    feature_2 : str
        Second feature name.

    Returns
    -------
    float
        Pearson correlation coefficient for the feature pair.

    Notes
    -----
    - Missing values are removed.
    - All values are cast to float.
    - Computation is fully manual as required by the project.
    """
    df_pair =df_pair.dropna().astype(float)
    mean_1 = df_pair[feature_1].sum() / df_pair.__len__()
    mean_2 = df_pair[feature_2].sum() / df_pair.__len__()
    df_pair['diff_product'] = (df_pair[feature_1] - mean_1) * (df_pair[feature_2] - mean_2)
    df_pair['diff1**2'] = (df_pair[feature_1] - mean_1) ** 2
    df_pair['diff2**2'] = (df_pair[feature_2] - mean_2) ** 2

    if df_pair['diff1**2'].sum() * df_pair['diff2**2'].sum() == 0:
        return 0
    return df_pair['diff_product'].sum() / ((df_pair['diff1**2'].sum() * df_pair['diff2**2'].sum()) ** 0.5)

def calculate_pearson(df:pd.DataFrame):
    """
    Compute Pearson correlation for all pairs of numeric features
    and determine the most strongly correlated pair.

    This function:
    - Keeps only numeric columns.
    - Computes Pearson manually for each unique pair (no repetition).
    - Tracks the pair with highest absolute correlation (closest to ±1).
    - Returns a list of all correlations plus the most correlated pair.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataset possibly containing mixed types.

    Returns
    -------
    pearsons : list of tuples
        Each tuple contains (feature_1, feature_2, pearson_value).
    result_1 : str
        Name of the first feature of the strongest correlation pair.
    result_2 : str
        Name of the second feature of the strongest correlation pair.

    Notes
    -----
    - If the dataset contains no numeric columns, empty results are returned.
    - Pearson computations are manual and use the helper function
      calculate_correlation().
    """
    mask = 1
    result_1 = ""
    result_2 = ""
    analised = []
    pearsons = []

    df = df.select_dtypes(include=['float', 'int'])
    if df.empty:
        return pearsons, result_1, result_2
    
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

def scatter_plot(df:pd.DataFrame):
    """
    Determine the strongest correlated pair of numeric features and plot them.

    This function:
    - Computes all Pearson correlations via calculate_pearson()
    - Identifies the feature pair with the strongest absolute correlation
    - Plots both raw and standardized scatter plots using plot_scatter()

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset possibly containing numeric and non-numeric columns.

    Notes
    -----
    - If the dataset contains no numeric features, a message is printed.
    - Standardization is performed manually using helper functions.
    """
    pearsons, feature_1, feature_2 = calculate_pearson(df)
    if not pearsons or not feature_1 or not feature_2:
        print("No numeric values in data")
    else:
        print_pearsons(pearsons)
        plot_scatter(df, feature_1, feature_2)

# Astronomy and defense against the dark arts --> pearson: -1