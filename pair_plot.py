import argparse
import signal
from plots.pair_plot import *
from utils.utils import read_file, termination_handler

def arguments_configuration():
    parser = argparse.ArgumentParser(
        prog='Pair plot',
        description='Given a dataset path and a feature, this script plots a pair plot matrix'
    )

    parser.add_argument('filename')

    return parser.parse_args()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, termination_handler)
    signal.signal(signal.SIGTERM, termination_handler)
    args = arguments_configuration()
    df = read_file(args.filename)
    pair_plot(df)