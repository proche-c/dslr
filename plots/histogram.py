import pandas as pd
from utils import config
import matplotlib.pyplot as plt
import seaborn as sns
from describe.describe_class import Describe

def plot_histogram(df:pd.DataFrame, feature):
    try:
        data = df[feature]
        # print(data.head())
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
        # df_feat = df.groupby(config.target_label)[args.feature]
        subset = df[df[config.target_label] == house][feature]
        plt.hist(subset, alpha=0.4, label=house, color=color)
    plt.title(f'{feature} by {config.target_label}')
    plt.legend()


    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x=feature, hue=config.target_label, palette=house_colors)
    plt.title(f'{feature} by {config.target_label}')
    plt.show()


def calculate_homogeneity(df:pd.DataFrame):
    dfs = []
    for house, group in df.groupby(config.target_label):
        d = Describe(group)
        dfs.append(d.result)

    features = dfs[0].columns

    variance = 0
    feature_homogeneity = features[0]
    homogeneities = {}
    for feature in features:
        means = []
        for df in dfs:
            means.append(df[feature]['mean'])
        mean_global = sum(means) / len(dfs) 
        diffs = []
        for df in dfs:
            diffs.append((df[feature]['mean'] - mean_global) ** 2)
        current_variance = sum(diffs) / (len(dfs) - 1)
        homogeneities[feature] = current_variance
        if feature == features[0]:
            variance = current_variance
        else:
            if current_variance < variance:
                variance = current_variance
                feature_homogeneity = feature
    
    return homogeneities, feature_homogeneity
    # print(feature_homogeneity)



    

def histogram(df: pd.DataFrame, args):
    homogeneities, feature = calculate_homogeneity(df)
    plot_histogram(df, feature)


# Care of magical creatures --> variance between means: 0.004022052715270744



# df_grouped = df.groupby("Hogwarts House")  # agrupamos por casa
# gryffindor_df = df_grouped.get_group("Gryffindor")  # dataframe solo con Gryffindor


            # by_house = df.groupby('Hogwarts House')
            # houses = list(by_house.groups.keys())
            # # by_house.groups.keys() esto devuelve un objeto tipo dict_keys, que es iterable pero no indexable,
            # # por lo que hay que convertirlo en una lista