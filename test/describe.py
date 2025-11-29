import pandas as pd

def test_describe(data: pd.DataFrame):
    """
    Display the statistical summary of a dataset using pandas' built-in
    `describe()` method.

    This function is intended for testing and validation purposes. It allows
    comparison between the manually implemented Describe class and pandas'
    automatic statistical summary.

    Parameters:
        data (pd.DataFrame):
            The dataset to analyze. All numerical columns will be summarized
            using pandas' `describe()` method.

    Output:
        Prints a formatted table containing count, mean, std, min, quartiles,
        and max values computed by pandas.
    """
    try:
        describe_build_in = data.describe()
    except Exception as e:
        print(f"An exception type {type(e).__name__} has ocurred, please check the input file")
        exit(1)
    print("*** Result using the panda's method describe:****")
    print(describe_build_in)