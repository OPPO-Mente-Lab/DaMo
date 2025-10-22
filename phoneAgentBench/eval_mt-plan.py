
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import re
import sys
import copy
import time
import json
import random
from datetime import datetime

import torch
import pandas as pd
from tqdm import tqdm
from loguru import logger
from PIL import Image
from datetime import datetime, timedelta
from transformers import (
    AutoModel, 
    AutoTokenizer,
    AutoProcessor,
    GenerationConfig, 
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration, 
    AutoModelForCausalLM, 
)

from utils.prompt_mt_plan import *
from utils.planning_evaluator import PlanningEvaluator
from utils.internvl_infer import build_transform, dynamic_preprocess


def get_date_and_location_info():
    current_datetime = datetime.now()
    current_date = current_datetime.strftime("%Y-%m-%d")
    weekday_idx = current_datetime.weekday()
    weekday_zh_idx = '一二三四五六日'
    weekday = f"星期{weekday_zh_idx[weekday_idx]}"
    location = '北京'
    return current_date+' '+weekday, location

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data
def dump_json(data,path):
    with open(path,'w') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class EvalTools():
    def __init__(self, mllm_name, task_type, model_path, embedding_path, evalset_path, img_root_path, res_path):
        self.mllm_name = mllm_name
        self.model_path = model_path
        self.evalset_path = evalset_path
        self.img_root_path = img_root_path
        self.res_path = os.path.join(res_path, mllm_name)
        os.makedirs(self.res_path, exist_ok=True)
        self.predict_path = os.path.join(self.res_path, 'mt-plan_model_predict.json')
        self.embedding_path = embedding_path

    def load_model(self, mllm_name, model_path):
        logger.info("Load {0} from {1}".format(mllm_name, model_path))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

        if mllm_name == "qwen2vl":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype="auto", device_map="auto"
                ).eval()
            self.processor = AutoProcessor.from_pretrained(model_path)

        if mllm_name == "qwen2.5vl":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, torch_dtype="auto", device_map="auto"
                ).eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_path, min_pixels = 4*28*28, max_pixels=2500*28*28)

        if mllm_name == "internvl":
            self.model =  AutoModel.from_pretrained(
                            model_path,
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                            use_flash_attn=True,
                            trust_remote_code=True).eval().cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
            self.model.eval()
            self.device = self.model.device

        logger.info("{0} loaded.".format(mllm_name))

    def internvl_example(self, system_prompt, user_prompt, image):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(inputs, return_tensors='pt').to(self.model.device)
        question = system_prompt +'\n' + user_prompt

        if image is None:
            pixel_values = None
        else:
            pixel_values = load_image(image, max_num=12).to(torch.bfloat16).cuda()
        response, history = self.model.chat(tokenizer=self.tokenizer, 
                                        pixel_values=pixel_values,
                                        question=question, 
                                        generation_config=dict(do_sample=False, max_new_tokens=1024,),
                                        history=None, 
                                        return_history=True)

        return response

    def qwenvl_example(self, system_prompt, user_prompt, image):
        from qwen_vl_utils import process_vision_info

        part1, part2 = user_prompt.split('<image>')

        if image:
            messages = [
                {
                    "role": "assistant",
                    "content": system_prompt
                },
                {
                    "role":"user",
                    "content": [
                        {"type": "text", "text": part1},
                        {"type": "image", "image": image},
                        {"type": "text", "text": part2},
                    ]
                }
                ]

        else:
            messages = [
                {
                    "role": "assistant",
                    "content": system_prompt
                },
                {
                    "role":"user",
                    "content": [
                        {"type": "text", "text": part1},
                        {"type": "text", "text": "未提供图片。当前轮次用户问题与屏幕图片无关；或者提供的是图片地址(screen_image_url)，你需要通过工具去访问图片信息。"},
                        {"type": "text", "text": part2},
                    ]
                }
                ]


        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return response[0]

    def batch_run_mllm(self):
        current_datetime = datetime.now()
        current_date = current_datetime.strftime("%Y-%m-%d")
    
        dataset = load_json(self.evalset_path)
        prompt_ls = []

        for i, key in enumerate(dataset):
            query = dataset[key]['query']
            if dataset[key].get('image'):
                img_path = os.path.join(self.img_root_path, dataset[key]['image'])
            else:
                img_path = None
            ground_truth = dataset[key]['ground_truth']
            available_tools = dataset[key]['meta']['candidate_apis']
            environment = dataset[key]['meta'].get('environment')
            if environment is None:
                date, location = get_date_and_location_info()
            else:
                date, location = environment['datetime'], environment['location']

            system_prompt = SYSTEM_PROMPT
            user_prompt = USER_PROMPT.format(
                date=date,
                location=location,
                available_tools=available_tools,
                query=query
            )
                    
            if self.mllm_name in ['qwen2vl', 'qwen2.5vl']:
                response = self.qwenvl_example(system_prompt, user_prompt, img_path)

            elif self.mllm_name in ['internvl']:
                response = self.internvl_example(system_prompt, user_prompt, img_path)
            
            dataset[key]['response'] = response
            logger.info(f'model_name >>> \n{self.mllm_name}')
            logger.info(f'======Question  {i} ======>>>\n{query}')
            logger.info(f'Response >>> \n{response}')
            logger.info('\n\n\n')

        dump_json(dataset, self.predict_path)

    def __call__(self):
        # 1. load model
        self.load_model(self.mllm_name, self.model_path)

        # 2. run prediction
        self.batch_run_mllm()

        # 3. run evluation
        pe = PlanningEvaluator(dataset_path=self.predict_path,embedding_path=self.embedding_path)
        res = pe.evaluate()
        print(res)
        score_path = os.path.join(self.res_path, 'mt-plan_model_score.json')
        with open(score_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(res, ensure_ascii=False, indent=4))

def main():
    configs = {
        "mllm_name": "qwen2.5vl", # qwen2.5vl  internvl
        "task_type": "MT-Plan", # or Mobile-FC
        # "model_path": '/mnt/workspace/shikai/checkpoint/InternVL2_5-4B/',
        "model_path": '/mnt/workspace/shikai/checkpoint/Qwen2.5-VL-3B-Instruct',
        "embedding_path": '/mnt/workspace/shikai/checkpoint/bge-large-zh-v1.5/',
        "img_root_path": "./data/images",
        "evalset_path": "./data/evalset/mt-plan.json",
        "res_path" : "./output"
    }
    # configs = {
    #     "mllm_name": "qwen2.5vl", # qwen2.5vl or internvl
    #     "task_type": "MT-Plan",
    #     "model_path": '<Your checkpoint path>',
    #     "embedding_path": '<bge-large-zh-v1.5 model path>',
    #     "img_root_path": "./data/images",
    #     "evalset_path": "./data/evalset/mt-plan.json",
    #     "res_path" : "./output"
    # }

    eval_tool = EvalTools(**configs)
    eval_tool()

if __name__ == '__main__':
    main()

