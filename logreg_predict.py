import argparse
import signal
import os
from utils.utils import termination_handler, read_file
from logistic_regression.predict import Predict
from test.test import compare_models

def arguments_configuration():
    parser = argparse.ArgumentParser(
        prog='Logistic regression predict',
        description='Given a dataset path, the program will predict the Howarts House students belong throught a multi-classifier model'
    )

    parser.add_argument('test_file')
    parser.add_argument('--jsonpath', default='weights.json', help='Path to json file gene3rated by trainer programm')
    parser.add_argument('--train_file', default='datasets/dataset_train.csv', help='Path to the train dataset')
    parser.add_argument('--test', '-t', action='store_true', help='Compare the results of the manual model to the scikit-learn mkodel')


    return parser.parse_args()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, termination_handler)
    signal.signal(signal.SIGTERM, termination_handler)
    args = arguments_configuration()

    try:
        jsonpath = os.path.abspath(args.jsonpath)
    except Exception as e: 
        print(f'An exception type {type(e).__name__} has ocurred, please check the json file {args.jsonpath}')

    df_predict = read_file(args.test_file)
    prediction = Predict(df_predict, jsonpath)
    prediction.to_csv('houses.csv', index=True)

    if args.test:
        df_train = read_file(args.train_file)
        compare_models(df_predict, df_train, prediction)