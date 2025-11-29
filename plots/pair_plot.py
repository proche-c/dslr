import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plots.histogram import calculate_homogeneity
from plots.scatter_plot import calculate_pearson
from utils.utils import print_variances_between_means, print_pearsons
from utils import config

def get_strong_correlation(pearsons:list):
    strong_correlated_features = []
    for item in pearsons:
        if item[2] > 0.8 or item[2] < -0.8:
            strong_correlated_features.append(item)
    return strong_correlated_features

# def discard_strong_correlated(variances_between_means:dict, pearsons:list):
#     strong_correlated_features = get_strong_correlation(pearsons)
#     print("***The following features are strongly correlated:****")
#     print_pearsons(strong_correlated_features)
#     print("For each pair of strongly correlated features, the one with the lowest variance between means will be discarded from the pair representation")
#     discarded_features = []
#     for item in strong_correlated_features:

#         if variances_between_means[item[0]] > variances_between_means[item[1]]:
#             discarded_features.append(item[1])
#         else:
#             discarded_features.append(item[0])
#     return discarded_features

# def get_higher_variance_features(variances_between_means: dict):

#     sorted_features = sorted(
#         variances_between_means.items(),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     higher_variance_features = [feature for feature, _ in sorted_features[:5]]
#     return higher_variance_features


def pair_plot(df:pd.DataFrame):
    variances_between_means, feature = calculate_homogeneity(df)
    pearsons, feature_1, feature_2 = calculate_pearson(df)
    print_variances_between_means(variances_between_means)
    strong_correlated_features = get_strong_correlation(pearsons)
    print("The following features are strongly correlated")
    print_pearsons(strong_correlated_features)

    features_and_house = [config.target_label] + [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    df = df[features_and_house].dropna().astype(float, errors='ignore')
 
    plt.rcParams.update({
        'axes.labelsize': 6,    # tamaño de los nombres de los ejes
        'xtick.labelsize': 6,    # tamaño de los números en el eje X
        'ytick.labelsize': 6     # tamaño de los números en el eje Y
    })
    sns.pairplot(
        data=df,
        hue=config.target_label,
        palette='Set2',
        plot_kws={'alpha': 0.6, 's': 40}
    )
    plt.show()


# Choosen features:
#   - Astronomy
#   - Ancient runes
#   - Charms
