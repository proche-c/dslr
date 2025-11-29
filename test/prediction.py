from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from utils import config

def percentage_outcomes(y, y_train, y_val):
    print("Percentages in y:")
    print(f"Gryffindor: {(y[config.target_label] == 'Gryffindor').sum() * 100 / len(y)}%")
    print(f"Hufflepuff: {(y[config.target_label] == 'Hufflepuff').sum() * 100 / len(y)}%")
    print(f"Ravenclaw: {(y[config.target_label] == 'Ravenclaw').sum() * 100 / len(y)}%") 
    print(f"Slytherin: {(y[config.target_label] == 'Slytherin').sum() * 100 / len(y)}%")
    print("Percentages in y_train:")
    print(f"Gryffindor: {(y_train[config.target_label] == 'Gryffindor').sum() * 100 / len(y_train)}%")
    print(f"Hufflepuff: {(y_train[config.target_label] == 'Hufflepuff').sum() * 100 / len(y_train)}%")
    print(f"Ravenclaw: {(y_train[config.target_label] == 'Ravenclaw').sum() * 100 / len(y_train)}%") 
    print(f"Slytherin: {(y_train[config.target_label] == 'Slytherin').sum() * 100 / len(y_train)}%")
    print("Percentages in y_val:")
    print(f"Gryffindor: {(y_val[config.target_label] == 'Gryffindor').sum() * 100 / len(y_val)}%")
    print(f"Hufflepuff: {(y_val[config.target_label] == 'Hufflepuff').sum() * 100 / len(y_val)}%")
    print(f"Ravenclaw: {(y_val[config.target_label] == 'Ravenclaw').sum() * 100 / len(y_val)}%") 
    print(f"Slytherin: {(y_val[config.target_label] == 'Slytherin').sum() * 100 / len(y_val)}%")

def features_range(x_train, x_val, x_test):
    print(f"Range {config.feature_1}:")
    print(f"In x_train --> min: {x_train[config.feature_1].min()} - max: {x_train[config.feature_1].max()}")
    print(f"In x_val --> min: {x_val[config.feature_1].min()} - max: {x_val[config.feature_1].max()}")
    print(f"In x_test --> min: {x_test[config.feature_1].min()} - max: {x_test[config.feature_1].max()}")
    print(f"Range {config.feature_2}:")
    print(f"In x_train --> min: {x_train[config.feature_2].min()} - max: {x_train[config.feature_2].max()}")
    print(f"In x_val --> min: {x_val[config.feature_2].min()} - max: {x_val[config.feature_2].max()}")
    print(f"In x_test --> min: {x_test[config.feature_2].min()} - max: {x_test[config.feature_2].max()}")
    print(f"Range {config.feature_3}:")
    print(f"In x_train --> min: {x_train[config.feature_3].min()} - max: {x_train[config.feature_3].max()}")
    print(f"In x_val --> min: {x_val[config.feature_3].min()} - max: {x_val[config.feature_3].max()}")
    print(f"In x_test --> min: {x_test[config.feature_3].min()} - max: {x_test[config.feature_3].max()}")

def compare_models(df_predict:pd.DataFrame, df_train:pd.DataFrame, prediction):
    df_train = df_train[[config.feature_1, config.feature_2, config.feature_3, config.target_label]].dropna()
    x = df_train[[config.feature_1, config.feature_2, config.feature_3]]
    y = df_train[[config.target_label]]
    houses = df_train[config.target_label].unique()
    original_index = df_predict.index.copy()
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)
    x_test = df_predict[[config.feature_1, config.feature_2, config.feature_3]].dropna()
    clean_index = x_test.index
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_val_s = scaler.transform(x_val)
    x_test_s = scaler.transform(x_test)
    predictions = []
    for house in houses:
        y_train_final = (y_train[config.target_label] == house).astype(int)
        y_val_final = (y_val[config.target_label] == house).astype(int)
        model = LogisticRegression()
        model.fit(x_train_s, y_train_final)
        print(f"{house} --> score with train set: {model.score(x_train_s, y_train_final)}")
        print(f"\t --> score with val set: {model.score(x_val_s, y_val_final)}")
        y_predict = model.predict_proba(x_test_s)[:,1]
        predictions.append(y_predict)

    
    prob_matrix = np.vstack(predictions).T
    final_predictions = [houses[row.argmax()] for row in prob_matrix]

    df_final = pd.DataFrame(index=original_index)
    df_final[f'{config.target_label} test'] = None
    df_final.loc[clean_index, f'{config.target_label} test'] = final_predictions

    final = prediction.join(df_final)
    final['discrepancies'] = ~final[config.target_label].eq(final[f'{config.target_label} test'])
    final.to_csv('compare.csv', index=True)
    if final['discrepancies'].sum() == 0:
        print('There are no discrepancies between the proyect model and the test model')
    else:
        print(final[final['discrepancies'] == True])
