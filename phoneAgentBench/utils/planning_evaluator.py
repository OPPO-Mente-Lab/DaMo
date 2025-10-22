import os
import sys
import copy
import json
import re
import torch
import itertools
import numpy as np
import pandas as pd
import networkx as nx
from numpy import mean
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
from utils.schema import ResponseDataSample


class PlanningEvaluator:
    """Planning Evaluation
    Args:
        dataset_path(str): File path of evaluation dataset
        name_weight(float): the weight of action_name in bert_score match, default = 0.9
        args_weight(float): the weight of action_args in bert_score match, default = 0.1
        match_threshold(float): the threshold of matching
        match_strategy(str): matching method, can choose 'bertscore' or 'permutation' 
        bert_score_model(str): the bert_score model for sentence similarity, default = "all-mpnet-base-v2". 
            Refer to https://www.sbert.net/docs/pretrained_models.html for more models.
    """
    def __init__(
        self,
        dataset_path: str,
        name_weight = 0.5,
        args_weight = 0.5,
        match_threshold = 0.98,
        match_strategy: str = 'bertscore', # ["bertscore", "permutation"]
        bert_score_model: str = "bge-large-zh", # ['thenlper/gte-large-zh', 'all-mpnet-base-v2', 'bge-large-zh']
        embedding_path: str = './checkpoint/bge-large-zh',
        default_prompt_type: str = 'ReWOO', # ["json", "ReWOO"]
        **kwargs,
    ) -> None:
        self.bert_score_model = bert_score_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_path = dataset_path
        self.name_weight = name_weight
        self.args_weight = args_weight
        self.match_threshold = match_threshold
        self.default_prompt_type = default_prompt_type # ["json", "ReWOO"]
        assert match_strategy in ["bertscore", "permutation"], f"match strategy must in [\"bertscore\", \"permutation\"], but get {match_strategy}"
        self.match_strategy = match_strategy
        self.valid_data_count = None

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_path) # bge
        self.embedding_model = AutoModel.from_pretrained(embedding_path)
        self.embedding_model = self.embedding_model.to(self.device)
        self.embedding_model.eval()


    def _load_dataset(self):
        self.dataset = []
        skip_keys=[]
        with open(self.dataset_path, 'r') as f:
            dataset = json.load(f)
        total_error = 0
        total_count = 0

        for key in dataset.keys():
            datum = dataset[key]
            data_sample, error = self._process_response(datum)
            total_error += error
            total_count += 1
            self.dataset.append(
                dict(
                    query=datum['query'],
                    response_data_sample=data_sample
                    )
                )

        self.num_samples = len(self.dataset)
        print("total_data_count:", total_count, "valid_data_count:", total_count - total_error)
        self.valid_data_count = total_count - total_error

    def _process_response(
        self,
        datum,
    ) -> ResponseDataSample:
        pred_data = datum['response']
        if pd.isna(pred_data):
            pred_data = "None"
        gt_data = datum['ground_truth']

        error = 0
        pred = dict()
        gt = dict()
        gt['planning'] = gt_data

        thoughts = re.findall(r'(SubTask[0-9]+: .+)', pred_data)
        action_units = re.findall(r'\n(#E[0-9] = .+)', pred_data)
        
        if not (len(thoughts) == len(action_units)):
            pred['planning'] = []
            gt['planning'] = []
            return ResponseDataSample(template = '', pred=pred, gt=gt), 1

        plan_action = []
        for i in range(len(action_units)):
            if action_units[i][0] == '#': 
                # The action has a return #E
                args_str_list = re.findall(r'#E[0-9]+ = .+\((.+)\)', action_units[i])
                name_list = re.findall(r'#E[0-9]+ = (.+)\(', action_units[i])
            else: 
                # The action does not have a return
                args_str_list = re.findall(r'.+\((.+)\)', action_units[i])
                name_list = re.findall(r'(.+)\(', action_units[i])
            if (len(name_list) > 0): 
                curr_name = name_list[0]
            else: 
                curr_name = ""
            if (len(args_str_list) > 0): 
                args_str = "{" + args_str_list[0] + "}"
            else: 
                args_str = "{}"

            plan_action.append(
                dict(
                    id = i,
                    name = curr_name,
                    args = args_str
                ))
        pred['planning'] = plan_action

        for i in range(len(gt['planning'])):
            args_str = ""
            if type(gt['planning'][i]['args']) == str:
                args_dict = eval(gt['planning'][i]['args'])
            else:
                assert type(gt['planning'][i]['args']) == dict
                args_dict = gt['planning'][i]['args']
            for it in args_dict:
                if args_str == "": args_str += f"{it}=\"{args_dict[it]}\""
                else: args_str += f", {it}=\"{args_dict[it]}\""
            gt['planning'][i]['args'] = '{' + args_str + '}'
        return ResponseDataSample(template = '', pred=pred, gt=gt), error

    def permutation_match(self, pred_plan, gt_plan) -> dict:
        '''
            The function calculates all the permutation matches' score and selects the max f1_score;
            Since permutation is time consuming, we truncate the length of plans to 9
        '''
        if pred_plan[-1]['name'] != 'FinishAction':
            pred_plan.append(
                {'id': len(pred_plan), 'prev': [], 'name': 'FinishAction', 'args': r'\{\}'}
            )
        
        if gt_plan[-1]['name'] != 'FinishAction':
            gt_plan.append(
                {'id': len(gt_plan), 'prev': [], 'name': 'FinishAction', 'args': r'\{\}'}
            )

        # truncate plans to 9 since it is too long for permutation.
        if len(pred_plan) > 9: pred_plan = pred_plan[:9]
        if len(gt_plan) > 9: gt_plan = pred_plan[:9]

        pred_plan = sorted(pred_plan, key=lambda x: x['id'])
        gt_plan = sorted(gt_plan, key=lambda x: x['id'])
        len_pred = len(pred_plan)
        len_gt = len(gt_plan)
        map_id_max = max(len_pred, len_gt)
        numbers = [i for i in range(map_id_max)]
        perms = itertools.permutations(numbers, len_pred)
        gt_prev_count, pred_prev_count = 0, 0
        for i in range(len_gt):
            gt_plan[i]['prev'].append(i)
            gt_prev_count += len(gt_plan[i]['prev'])
        for i in range(len_pred):
            pred_plan[i]['prev'].append(i)
            pred_prev_count += len(pred_plan[i]['prev'])
        if gt_prev_count == 0 or pred_prev_count == 0:
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }
        max_recall, max_precision, max_f1 = 0, 0, 0
        for perm in perms:
            correct_count = 0
            for i in range(len_pred):
                if perm[i] >= len_gt: 
                    continue
                for j in pred_plan[i]['prev']:
                    if perm[j] in gt_plan[perm[i]]['prev']:
                        correct_count += 1
            now_recall, now_precision = correct_count / gt_prev_count, correct_count / pred_prev_count
            if now_recall + now_precision == 0: 
                continue
            now_f1 = 2 * now_recall * now_precision / (now_recall + now_precision)
            if now_f1 > max_f1:
                max_f1, max_recall, max_precision = now_f1, now_recall, now_precision
        return {
            'precision': max_precision,
            'recall': max_recall,
            'f1_score': max_f1
        }

    def bertscore_match(self, pred_plan, gt_plan) -> dict:
        """
            Calculate the similarity between predicted plan and golden answer,
            A plan can be regarded a sequence of actions, and each action has a name and args.
            Firstly, use bertscore to calculate pointwise similarity by:
                similarity(u, v) = bertscore(u.name, v.name) * name_weight + bertscore(u.args, v.args) * args_weight;
            Secondly, use Hungarian matching to match the points;
            Finally, use LIS to calculate the number of matched nodes.
        """
        if len(pred_plan) == 0 or len(gt_plan) == 0:
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'max_weight_matching': None,
                'pred_to_gt_mapping': None,
                'correct_count': None,
                'hallucination_rate':0
            }

        pred_plan = copy.deepcopy(sorted(pred_plan, key=lambda x: x['id']))
        gt_plan = copy.deepcopy(sorted(gt_plan, key=lambda x: x['id']))

        #Add end action
        #Currently it is hard-code
        if pred_plan[-1]['name'] == 'FinishAction':
            pred_plan = pred_plan[:-1]
        if gt_plan[-1]['name'] == 'FinishAction':
            gt_plan = gt_plan[:-1]
        #The total counts of nodes and edges.
        len_pred = len(pred_plan)
        len_gt = len(gt_plan)

        bert_score_matrix = np.zeros((len_pred, len_gt))
        name_pred, args_pred = [], []
        name_gt, args_gt = [], []
        for i in range(len_pred):
            name_pred.append(pred_plan[i]['name'])
            args_pred.append(str(pred_plan[i]['args']))
        for i in range(len_gt):
            name_gt.append(gt_plan[i]['name'])
            args_gt.append(str(gt_plan[i]['args']))
        

        name_pred_encoded_input = self.tokenizer(name_pred, padding=True, truncation=True, return_tensors='pt',max_length=512).to(self.device)
        with torch.no_grad():
            try:
                model_output = self.embedding_model(**name_pred_encoded_input)
            except RuntimeError:
                from ipdb import set_trace;set_trace()
            name_pred_emb = model_output[0][:, 0]
        name_pred_emb = torch.nn.functional.normalize(name_pred_emb, p=2, dim=1)

        name_gt_encoded_input = self.tokenizer(name_gt, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.embedding_model(**name_gt_encoded_input)
            name_gt_emb = model_output[0][:, 0]
        name_gt_emb = torch.nn.functional.normalize(name_gt_emb, p=2, dim=1)

        args_pred_encoded_input = self.tokenizer(args_pred, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.embedding_model(**args_pred_encoded_input)
            args_pred_emb = model_output[0][:, 0]
        args_pred_emb = torch.nn.functional.normalize(args_pred_emb, p=2, dim=1)            

        args_gt_encoded_input = self.tokenizer(args_gt, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.embedding_model(**args_gt_encoded_input)
            args_gt_emb = model_output[0][:, 0]
        args_gt_emb = torch.nn.functional.normalize(args_gt_emb, p=2, dim=1)

        name_cosine_scores = np.maximum(util.cos_sim(name_pred_emb, name_gt_emb).cpu().numpy(), 0)
        args_cosine_scores = np.maximum(util.cos_sim(args_pred_emb, args_gt_emb).cpu().numpy(), 0)

        for i in range(len_pred):
            for j in range(len_gt):
                bert_score_matrix[i][j] = \
                    name_cosine_scores[i][j] * self.name_weight \
                    + args_cosine_scores[i][j] * self.args_weight
        G = nx.Graph()
        for i in range(len_pred):
            for j in range(len_gt):
                if bert_score_matrix[i][j] >= self.match_threshold:
                    G.add_edge(i, str(j), weight=bert_score_matrix[i][j])
        max_weight_matching = nx.max_weight_matching(G)
        
        pred_to_gt_mapping = dict()
        for key in max_weight_matching:
            if type(key[0]) == int:
                pred_to_gt_mapping[int(key[0])] = int(key[1])
            else:
                pred_to_gt_mapping[int(key[1])] = int(key[0])

        #If a prediction node does not match any golden answer node, we mark the node as -1.
        for i in range(len_pred):
            if i not in pred_to_gt_mapping:
                pred_to_gt_mapping[i] = -1
        #Calculate how many nodes are matched by Longest Increasing Subsequence (LIS)
        dp = np.ones(len_pred)
        for i in range(len_pred):
            for j in range(i):
                if pred_to_gt_mapping[i] == -1 or pred_to_gt_mapping[j] == -1:
                    continue
                if pred_to_gt_mapping[i] > pred_to_gt_mapping[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        correct_count = int(max(dp))

        recall, precision = correct_count / len(gt_plan), correct_count / len(pred_plan)
        f1_score = 2 * recall * precision / (recall + precision)
        result = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'max_weight_matching': max_weight_matching,
            'pred_to_gt_mapping': pred_to_gt_mapping,
            'correct_count':correct_count,
        }
        return result

    def _post_process(self, results_list):
        results = dict()
        planning_metric_keys = ["precision", "recall", "f1_score", 'parse_rate']
        for key in planning_metric_keys:
            results[key] = mean([result[key] for result in results_list])
        return results

    def _evaluate(self, data_sample) -> dict:
        if type(data_sample.gt)==list:
            result_ls = []
            for gt in data_sample.gt:
                if self.match_strategy == 'bertscore':
                    metrics_result = self.bertscore_match(
                    data_sample.pred['planning'], gt['planning'])
                else:
                    raise NotImplementedError
                if len(data_sample.pred['planning']) == 0 or len(gt['planning']) == 0:
                    metrics_result['parse_rate'] = 0
                else:
                    metrics_result['parse_rate'] = 1
                result_ls.append(metrics_result)
            result = sorted(result_ls,key=lambda x:x['f1_score'],reverse=True)
            return result[0]
        else:
            if self.match_strategy == 'bertscore':
                metrics_result = self.bertscore_match(
                    data_sample.pred['planning'], data_sample.gt['planning'])
            elif self.match_strategy == 'permutation':
                metrics_result = self.permutation_match(
                    data_sample.pred['planning'], data_sample.gt['planning'])
            else:
                raise NotImplementedError
            if len(data_sample.pred['planning']) == 0 or len(data_sample.gt['planning']) == 0:
                metrics_result['parse_rate'] = 0
            else:
                metrics_result['parse_rate'] = 1
            return metrics_result

    def evaluate(self):
        self._load_dataset()
        results_list = []
        for data_sample in self.dataset:
            metrics_result = self._evaluate(
                data_sample['response_data_sample'])
            results_list.append(metrics_result)
        return self._post_process(results_list)