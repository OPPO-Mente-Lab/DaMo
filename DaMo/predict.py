import pickle
import json
import os
import pandas as pd
from tqdm import tqdm
from get_P_fix import get_P_fix

number_of_training_set = 12
batch_size = 16
downstream_tasks = ['MT-Plan', 'APP-Rec', 'MM-RR', 'ACU', 'MM-NER', 'Mobile-FC', 'BFCL-V3', 'MME-perception', 'MME-reasoning', 'OCRBench']
weight = [1 / len(downstream_tasks)] * len(downstream_tasks)
score_threshold = 65


def load_mlp_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_unseen_data_mixing(model):
    print('start to get P_fix...')
    P_fix = get_P_fix(number_of_training_set, batch_size)
    print('start to predict downstream task performance on unseen data mixture...')
    results = []
    for p in tqdm(P_fix, desc="Predicting unseen data mixing"):
        y_pred = model.predict([p])
        y_pred = list(y_pred[0])
        overall_average_score = sum([score * w for score, w in zip(y_pred, weight)])
        if overall_average_score > score_threshold:
            result = [str(p)] + y_pred + [overall_average_score]
            results.append(result)
    results = sorted(results, key=lambda x: x[-1], reverse=True)
    results = results[:50]
    df = pd.DataFrame(results, columns=['unseen_data_mixture'] + downstream_tasks + ["overall_average_score"])
    df.to_excel("output/predict_unseen_data_mixing_results.xlsx", index=False)

if __name__ == "__main__":
    with open('output/mlp_regressor/mlp_result.json', 'r') as f:
        data = json.load(f)
    model_name = data['max_r2_test_model']
    model_name = 'mlp_0.pkl'
    model_path = os.path.join('output/mlp_regressor/', model_name)
    model = load_mlp_model(model_path)
    predict_unseen_data_mixing(model)