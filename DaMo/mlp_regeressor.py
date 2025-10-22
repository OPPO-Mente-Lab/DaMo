from sklearn.neural_network import MLPRegressor
import pandas as pd
from collections import defaultdict
from random import shuffle
import pickle
import os
import json

def fit(X_train, Y_train, X_test, Y_test, hidden_layer_sizes,max_iter,tol):
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=1, max_iter=max_iter, tol=tol)
    model.fit(X_train, Y_train)
    r2_test = model.score(X_test, Y_test)
    return model, r2_test

def get_data(data_path):
    df = pd.read_excel(data_path)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    data_per_test = defaultdict(list)
    test_list = []
    for _, row in df.iterrows():
        test_id = row['test_id']
        t = row['ckpt_id'] # corresponding to the training step
        p = eval(row['data_mixture_proportion'])
        s = [row['MT-Plan'], row['APP-Rec'], row['MM-RR'], row['ACU'], row['MM-NER'], row['Mobile-FC'], row['BFCL-V3'], row['MME-perception'], row['MME-reasoning'], row['OCRBench']]
        x = [_p / 4 * (t + 1) for _p in p]
        y = s
        data_per_test[test_id].append((x, y))
        test_list.append(test_id)
    shuffle(test_list)
    train_set_num = int(len(test_list) * 0.9)
    for test_id in test_list[:train_set_num]:
        for x, y in data_per_test[test_id]:
            X_train.append(x)
            Y_train.append(y)
    for test_id in test_list[train_set_num:]:
        for x, y in data_per_test[test_id]:
            X_test.append(x)
            Y_test.append(y)

    return X_train, Y_train, X_test, Y_test

def train_and_test(data_path, save_path,hidden_layer_sizes, max_iter, tol):
    sum_r2_test = 0
    max_r2_test = 0
    max_r2_test_model = None
    for i in range(10):
        X_train, Y_train, X_test, Y_test = get_data(data_path)
        model, r2_test = fit(X_train, Y_train, X_test, Y_test, hidden_layer_sizes, max_iter, tol)
        print(f"Round {i}: r2_test = {r2_test}")

        model_name = f'mlp_{i}.pkl'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(f"{save_path}/{model_name}", "wb") as f:
            pickle.dump(model, f)

        sum_r2_test += r2_test
        if r2_test > max_r2_test:
            max_r2_test = r2_test
            max_r2_test_model = model_name

    avg_r2_test = sum_r2_test / 10
    print("average r2_test = ", avg_r2_test)
    print("max r2_test = ", max_r2_test)
    with open(f"{save_path}/mlp_result.json", "w") as f:
        json.dump({"avg_r2_test": avg_r2_test, "max_r2_test": max_r2_test, "max_r2_test_model": max_r2_test_model}, f)
    return avg_r2_test, max_r2_test, max_r2_test_model

if __name__ == '__main__':
    data_path = './processed_data_random_50.xlsx'
    save_path = 'output/mlp_regressor'
    hidden_layer_sizes = (100, 100)
    max_iter = 1500
    tol = 1e-6
    avg_r2_test, max_r2_test, max_r2_test_model = train_and_test(data_path, save_path, hidden_layer_sizes, max_iter, tol)