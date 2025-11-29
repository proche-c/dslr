import pandas as pd
import sys

def read_file(file_path):
    try:
        df = pd.read_csv(file_path, index_col=0)
        return df
    except FileNotFoundError:
        print(f'Error: couldn´t find file {file_path}')
        sys.exit(1)
    except PermissionError:
        print(f"Permission denied to access {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An exception type {type(e).__name__} has ocurred, please check the input file")
        sys.exit(1)

def termination_handler(signum, frame):
    print("Termination requested...")
    sys.exit(0)
     
def get_mean(ds:pd.Series):
    valid = ds.dropna()
    return valid.sum() / valid.__len__()

def get_std(ds:pd.Series, mean):
    valid = ds.dropna()
    diff_squared = (valid - mean) ** 2
    return (diff_squared.sum() / valid.__len__()) ** 0.5

def standarize(ds:pd.Series, mean, std):
    return (ds - mean) / std

def print_variances_between_means(variances_between_means:dict):
    print("Variances between means:")
    for key, value in variances_between_means.items():
        print(f'{key}: {value}')

def print_pearsons(pearsons:list):
    print("Pearsons")
    print(pearsons.sort())
    for item in pearsons:
        print(f'{item[0]} and {item[1]}: {item[2]}')