
import argparse
import signal
from plots.histogram import *
from utils.utils import read_file, termination_handler

def arguments_configuration():
    parser = argparse.ArgumentParser(
        prog='Histogram',
        description='Given a dataset path and a feature, this script plots an histogram with the scores of said feature for each Hogwarts House'
    )

    parser.add_argument('filename')

    return parser.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, termination_handler)
    signal.signal(signal.SIGTERM, termination_handler)
    args = arguments_configuration()
    df = read_file(args.filename)
    histogram(df, args)