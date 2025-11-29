from describe.describe_class import Describe
import pandas as pd
import argparse
import os.path
import sys

def arguments_configuration():
    parser = argparse.ArgumentParser(
        prog='Describe',
        description='Given a dataset, this program mimic the behaviour of describe method, and display the results'
    )

    parser.add_argument('filename')

    return parser.parse_args()

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

if __name__ == "__main__":
    args = arguments_configuration()
    file_path = os.path.abspath(args.filename)
    data = read_file(file_path)
    d = Describe(data)
    print(d.result)
