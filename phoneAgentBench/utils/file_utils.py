
import hashlib
import hashlib
import json
import pandas as pd
import uuid
import hmac
import requests
from loguru import logger
from collections import OrderedDict
import re
import os


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
        return obj


def load_data(xlsx_data_path):
    """
    功能: 加载xlsx文件为字典
    """
    data_dict = {}
    tmp = pd.read_excel(xlsx_data_path,engine='openpyxl').to_dict()
    for key in list(tmp.keys()):
        data_dict[key] = []
    for name, content in tmp.items():
        for _, item in content.items():
            if str(item) == "nan":
                data_dict[name].append("")
            else:
                data_dict[name].append(str(item))
    return data_dict


def load_data_v2(xlsx_data_path):
    """
    功能: 加载xlsx文件为字典
    """
    data_dict = {}
    df = pd.read_excel(xlsx_data_path)
    for idx, row in df.iterrows():
        row = dict(row)
        for k,v in row.items():
            if str(v) == 'nan':
                row[k] = ''
        data_dict[idx] = row
    return data_dict


api_format = """```tool-{i}
def {api_name}({params}):
    '''
    {api_description}

    Parameters
    ----------
{parameter_defination}
    Returns
    ----------
    
    '''
    pass
```"""

def get_tools_desc(api_dict, action_list):
    api_description_list = []
    for i, action in enumerate(action_list):
        params_list = [f'{p["parameter_name"]}: {p["parameter_type"]}' for p in api_dict[action]['parameter']]
        param_defination = [f'    {p["parameter_name"]} : {p["parameter_type"]}\n        {p["parameter_description"]}' for p in api_dict[action]['parameter']]
        
        api_str = api_format.format(
            i=i+1,
            api_name=action,
            params=', '.join(params_list),
            api_description=api_dict[action]['api_description'],
            parameter_defination='\n'.join(param_defination)
        )

        api_description_list.append(api_str)
    return '\n\n'.join(api_description_list)



def save_data(data_dict, xlsx_data_path):
    """
    功能: 将字典保存为xlsx文件
    """
    base_dir = xlsx_data_path.rsplit("/", 1)[0]
    os.makedirs(base_dir, exist_ok=True)
    writer = pd.ExcelWriter(xlsx_data_path)
    df = pd.DataFrame(data_dict)
    df.to_excel(writer, startcol=0, index=False)
    #writer.save()
    writer.close()


def action2from_action(action):
    """
    功能: 解析指令中的指令名
    """
    result = []
    pattern = re.compile(r'([a-zA-Z_]+)\(')
    try:
        result = pattern.findall(action)
    except:
        pass
    return ";".join(result)



def load_api_from_json(api_path):
    with open(api_path, "r", encoding="utf-8") as file:
        api_list = json.load(file)

    api_dict = {}
    for api in api_list:
        api_dict[api['api_name']] = api
    return api_dict




def sign(params, body, app_id, secret_key):
    # 1. 构建认证字符串前缀，格式为 bot-auth-v1/{appId}/{timestamp}, timestamp为时间戳，精确到毫秒，用以验证请求是否失效
    auth_string_prefix = f"bot-auth-v1/{app_id}/{int(time.time() * 1000)}/"
    sb = [auth_string_prefix]
    # 2. 构建url参数字符串，按照参数名字典序升序排列
    if params:
        ordered_params = OrderedDict(sorted(params.items()))
        sb.extend(["{}={}&".format(k, v) for k, v in ordered_params.items()])
    # 3. 拼接签名原文字符串
    sign_str = "".join(sb) + body
    # 4. hmac_sha_256算法签名
    signature = hmac.new(
        secret_key.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    # 5. 拼接认证字符串
    return auth_string_prefix + signature


def example_gpt4o(prompt, model="gpt-4o"):
    req_id = str(uuid.uuid1())
    try:
        ak = "xxx"
        sk = "xxxxxxxxxxxx"
        body = {
            "maxTokens": 2048, 
            "model": model,
            "messages": prompt,
        }

        data = json.dumps(body)
        header = {
            "recordId": str(req_id),
            "Authorization": sign(None, data, ak, sk),
            "Content-Type": "application/json",
        }
        resp = requests.request(
            "POST",
            url="xxxxxxxxxxxxxxxxxxxxxxx",
            headers=header,
            data=data,
        )
        # print(resp.content)
        return json.loads(resp.content)["data"]["choices"][0]["message"]["content"].replace("```json", "").replace(
            "```", "").replace("\n", "").replace(" ", "")
    except Exception as e:
        print(e)
        pass
        return resp.text





