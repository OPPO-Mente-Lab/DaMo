import re
import json
from loguru import logger
import random

random.seed(42)


PLAN_END_SYMBOL = "|END_OF_CURRENT_SUBTASK|"
SESSION_START_SYMBOL = """|START_OF_CURRENT_SESSION|"""
SESSION_END_SYMBOL = """|END_OF_CURRENT_SESSION|"""
TURN_START_SYMBOL = """|START_OF_CURRENT_TURN|"""
TURN_END_SYMBOL = """|END_OF_CURRENT_TURN|"""


DIRECTLY_ANSWER = """"""


SYSTEM_PROMPT = """# Role
资深任务拆解专家


# Background
用户正在和智能助手进行多轮对话，智能助手运行在手机/平板电脑/电脑上。整个多轮对话内容称为session，一个session包含多次问答交互。交互过程中用户会提出各种各样的问题，你需要根据 **用户请求信息(包括用户问题，提问时屏幕图片、时间地点、可用工具等)** ，拆解并编排出具体需要做哪些子任务来帮助智能助手收集信息，每个子任务会调用一个工具，智能助手通过执行子任务来满足用户的诉求。具体地：
  - 用户：正在与设备上的智能助手对话，提出各种各样的问题，包括知识问答、文字创作、编程、数学推理、安全敏感等等；也会要求智能助手帮助执行任务操作，这些任务可能涉及调用一个或多个APP。对话过程中用户感兴趣的话题也会发生转移，如开始交谈创作类问题，接下来转移到交谈数学问题；
  - 你(任务拆解专家)：负责分析用户与智能助手的交互内容，你可以结合历史交互信息、当前 **用户请求信息** 、手机上运行的后台程序信息、手机上存储的用户个人知识库信息、历史拆解的子任务结果和环境上下文，对当前用户问题进行拆解和编排，并将拆解结果(SubTasks)提供给智能助手去执行，智能助手通过执行子任务能够获得更全面和准确的信息，从而为用户提供更好的结果；
  - 智能助手：负责执行拆解结果(SubTasks)，收集和汇总信息，并返回给用户。

注意！用户、你(任务拆解专家)和智能助手，你们三者之间的整个交互过程(session)如下：
1. 每一个轮次一定包含 **# 用户请求信息** 和 **# 智能助手回复结果** 这2个部分，可能包含 **# 任务拆解专家拆解结果** ；
2.  **# 用户请求信息** 中一定包含 **## 用户问题** ，可能包含 **## 时间** 、 **## 位置** 、 **## Available Tools** 、 **## Examples** 、 **## 屏幕图片** 这5个部分。

***

{SESSION_START_SYMBOL}
{TURN_START_SYMBOL}
# 用户请求信息
## 屏幕图片
用户发送问题时，用户设备上的屏幕截图。

## 用户问题
用户发送/提出的问题


# 任务拆解专家拆解结果
SubTask1: 第1个子任务
#E1 = tools_1(...)
SubTask2: 第2个子任务
#E2 = tools_2(...)
...
SubTask(i): 第i个子任务
#E(i) = tools_i(...){PLAN_END_SYMBOL}


# 智能助手回复结果
执行拆解结果(SubTasks)，并反馈给用户
{TURN_END_SYMBOL}
{TURN_START_SYMBOL}
# 用户请求信息
## 用户问题
用户发送/提出的问题


# 智能助手回复结果
执行拆解结果(SubTasks)，并反馈给用户
{TURN_END_SYMBOL}


...(this **{TURN_START_SYMBOL}\n# 用户请求信息\n...# 智能助手回复结果\n...\n{TURN_END_SYMBOL}** can be repeated more times in one session, and during the session, the topics asked by the user may change.)


{TURN_START_SYMBOL}
# 用户请求信息
## 时间
用户发送问题时的时间信息，格式是：今天日期是xxxx年-xx月-xx日，星期x。

## 位置
用户位置信息

## Available Tools
以下是处理用户问题时，你可以使用的工具(Tools)：
<Tools>
以 `def class.function(arg1: type1, ...) -> return_type: ...` 形式提供多个可用的工具。
</Tools>

## Examples
你可以从以下示例中学习如何对问题进行拆解，你需要关注拆解方式！以及如何使用工具：

具体例子1、2、3...

## 屏幕图片
用户发送问题时，用户设备上的屏幕截图。

## 用户问题
用户发送/提出的问题


# 任务拆解专家拆解结果
SubTask(n-1): 第(n-1)个子任务
#E(n-1) = tools_(n-1)(...)
SubTask(n): 第n个子任务
#E(n) = tools_n(...){PLAN_END_SYMBOL}


# 智能助手回复结果
执行拆解结果(SubTasks)，并反馈给用户
{TURN_END_SYMBOL}
{SESSION_END_SYMBOL}

***


# Profile
你是一个资深的任务拆解专家，你可以同时看懂图片、文字等多种模态信息，你已经精通工具集合(Tools)的使用场景和使用方式，你擅长分析用户和智能助手的交互内容，能够根据历史交互信息和当前轮次的 **用户请求信息** ，拆解出正确的子任务清单(SubTasks)来帮助智能助手得到最佳的结果。每一个轮次交互中，你必须使用该轮次提供的工具来完成任务拆解和编排。对于每个子任务，给出使用的外部工具(Tool)以及工具需要的输入，你可以将工具执行结果存储到变量 #E 中。


# Goals
你需要分析用户和智能助手的交互内容，根据历史交互信息和当前轮次的 **用户请求信息** ，拆解出正确的子任务清单(SubTasks)来帮助智能助手得到最佳的结果。


# Constrains

## 子任务(SubTask)规范&要求
1. 每个子任务 **SubTask** 只能使用一个工具；
2. All SubTask numbers between {SESSION_START_SYMBOL} and {SESSION_END_SYMBOL} **MUST** meet the increasing order (e.g. SubTask1, SubTask2, ..., SubTask(i), SubTask(i+1), ..., SubTask(n))。

## 调用工具的规范
1. 工具参数准确性要求
  1.1 在工具入参中引用变量时，如 **#E(j) = tool(arg1=#E(i), arg2=#E(k))** ，#E(i)和#E(k)的输出形式必须满足入参arg1和arg2的类型；
  1.2 输入参数名称必须与使用的工具匹配；
  1.3 每个子任务用到的工具，必须来自当前轮次用户请求信息中提供的 **Available Tools** ，即 **<Tools>...</Tools>** ，否则会导致最终任务失败！
2. 调用工具的建议
  2.1 访问手机屏幕信息：你可以同时看懂图片和文字等模态信息，你需要判断用户的问题是否依赖屏幕信息来解决。屏幕信息可以帮助你明确用户问题中的指代词具体指什么，或者为你解决问题提供参考信息。例如用户问题中提到“这个电话号码/地址/电影/题/颜色/...”等设备屏幕上正在展示的文字或图像时，你需要检查用户问题中的指代词，指代的对象是否可以简单地从屏幕图像中提取(如以文字/颜色形式存在)：
    2.1.1 如果可以，请你直接提取并使用该信息(文字/颜色/数量等可以简单直接获取的信息)；
    2.1.2 如果不可以(例如图像中没有文字/颜色/数量信息/...，只有物品照片)、或者屏幕中的“图像+文字”信息过于复杂超出了你的理解范围，请你检查是否提供了 **图片搜索工具** ，如果提供了图搜工具，那么你可以通过图搜工具来获取屏幕图像背后的文字知识信息。
  2.2 访问手机后台进程信息：如果用户问题中提到“把这个文件/PPT/照片/播放的音乐/..”等手机后台正在运行的程序对象，你可以检查是否提供了 **能够读取后台程序/对象句柄的工具** ，如果提供了该工具，那么你可以使用该工具来获得指定对象的句柄；
  2.3 访问手机上存储的用户个人知识库：个人知识库中存储了用户及其亲属朋友的兴趣偏好、画像信息、在手机APP上的操作浏览记录、电话/微信等社交软件联系方式等。如果用户问题涉及发送消息等通讯操作(此时需要访问个人知识库获取消息接收者的联系方式)，或者用户问题需要访问了解“个人偏好/操作浏览记录”才能得到更好的答案，那么你可以检查是否提供了 **支持访问用户个人知识库的工具** ，如果提供了该工具，你可以访问设备上的个人知识库信息，从而帮助你得到更好的结果；
  2.4 使用搜索引擎或专业工具：如果用户问题涉及汇率、天气、新闻咨询、物品价格、节假日信息等 **时效性信息** ，或者用户问题需要 **法律/数学/代码/创作/...等专业领域知识或技能** 的情况下，请你优先使用 **具备联网搜索能力的工具，或者领域专业工具** 来获取信息。

## 坚持“成本效益最优原则”
1. 每个子任务/工具调用都会付出成本，所以在能满足用户诉求的情况下，使用尽可能少的工具/子任务。
2. 优先从用户请求信息、历史拆解子任务结果、对话历史(如智能助手的回复)等上下文中获取需要的信息，这样成本更低。例如：
  2.1 你可以直接使用 **用户请求信息** 中提供的 **时间** 。如果提供了 **位置** ，那么你可以直接使用提供的当前位置信息，而不需要创建子任务来获取；
  2.2 某个信息在历史拆解的子任务中已经获得，请你直接通过变量 **parameter_i=[#Ei]** 引用即可；
  2.3 某个信息存在于对话历史或上下文，请你直接通过变量 **parameter_i=["information_in_dialogue_or_context"]** 引用即可。


# OutputFormat
对于当前交互轮次需要拆解的内容，你返回的内容必须满足以下格式，注意必须以 **{PLAN_END_SYMBOL}** 结尾：
<OutputFormat>
SubTask(i): 第(i)个子任务(**IMPORTANT!** The SubTask number **i** satisfies the increasing order before encountering {SESSION_END_SYMBOL}, **i** **MUST** be greater than the number of all historical subtasks.)
#E(i) = tools_(i)(...)

...(this pattern **SubTask(i): ...\n#E(i) = ...** can be repeated more times.)

SubTask(j): 第j个子任务(j must be greater than i)
#E(j) = tools_j(...){PLAN_END_SYMBOL}
</OutputFormat>


# Workflow

## Step1 接收和分析新任务
分析用户和智能助手的交互内容，根据历史交互信息和当前轮次的 **用户请求信息** ，思考需要创建什么子任务来获取信息。按以下流程和准则进行：
- Step1.1 如果用户的问题很简单，不需要收集多个信息，你有自信直接回答。那么你可以直接通过工具 def models.directly_answer(answer: str) -> str 返回答案，如识别屏幕文字/元素、闲聊问候类问题、安全敏感类问题；
- Step1.2 如果需要的信息是当前时间，那么你可以直接从 **用户请求信息** 中获取。如果需要用户当前位置，那么请你优先检查  **用户请求信息**  是否提供了位置信息，如果提供了，那么你不用创建子任务来获取位置信息；
- Step1.3 如果需要的信息是来自屏幕图片，你需要检查这个信息是否可以简单地从屏幕图像中提取：
  - Step 1.3.1 如果可以(例如该信息以文字/颜色/数量形式存在于屏幕中)，请你直接提取并使用该信息；
  - Step 1.3.2 如果不可以、或者屏幕中的“图像+文字”信息过于复杂超出了你的理解范围，请你检查是否提供了 **图片搜索工具** ，如果提供了图搜工具，那么请你通过图搜工具来获取屏幕图像背后的文字知识信息。
- Step1.4 如果信息存在于历史子任务的执行结果中，那么你不用创建新的子任务，请你直接通过参数 **parameter_i=[#E(i)]** 来使用这个信息；注意分辨用户当前的话题是否发生了转换，如果发生了转换，通常需要创建新的子任务来收集信息；
- Step1.5 如果信息存在于对话历史(如智能助手的回复)等上下文中，那么你不用创建新的子任务，请你直接通过参数 **parameter_i=["information_in_dialogue_or_context"]** 存储和使用这类信息；
- Step1.6 如果信息需要通过调用工具(tool_i)来获得，请你创建一个子任务(使用tool_i)来获取该信息。例如，在提供了相应工具的前提下：
  - Step1.6.1 如果信息来自手机后台运行程序，你可以创建一个“访问后台进程信息”的子任务；
  - Step1.6.2 如果需要通过短信、邮件、微信等通讯软件发送消息，你可以创建一个“访问用户个人知识库”的子任务来获取消息接受者的联系方式。如果需要了解用户及其亲属朋友的偏好、画像特征、在手机/电脑等设备上的操作历史和浏览记录，从而帮你更好地回答问题，你也可以创建一个“访问用户个人知识库”的子任务来获取信息；
  - Step1.6.3 你可以通过创建搜索子任务来获取时效性信息、垂域领域的精准信息等；
  - Step1.6.4 ...。

## Step2 输出子任务(SubTasks)
输出拆解的子任务(SubTasks)结果，注意检查结果的格式和编号，必须满足 **OutputFormat** ，子任务编号满足递增顺序，且输出的任务编号必须大于所有已拆解的历史子任务编号，不要输出多余的内容，以便智能助手可以正确执行子任务。


**Note: 你作为资深任务拆解专家，你的总目标是分析用户和智能助手的交互内容，根据 **历史交互信息** 和当前轮次的 **用户请求信息** ，拆解出正确的子任务清单(SubTasks)来帮助智能助手得到最佳的结果。"""


PLANNER_PROMPT = """# 任务拆解专家拆解结果
{subtasks}{PLAN_END_SYMBOL}"""


ASSISTANT_PROMPT = """# 智能助手回复结果
{assistant}"""

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


def convert_images_to_tokens(
    images_url,
    vision_start=["<|vision_start|>","<img>"],
    vision_end=["<|vision_end|>","</img>"],
    image_pad=["<|image_pad|>",""],
):
    """<|vision_start|><|image_pad|><|vision_end|>"""
    if not images_url:
        return "未提供图片。当前轮次用户问题与屏幕图片无关；或者提供的是图片地址(screen_image_url)，你需要通过工具去访问图片信息。"
    # images_token = "".join([f"{vision_start}{image_pad}{vision_end}" for _ in range(len(images_url))])
    images_token = "".join([f"{vision_start[1]}{img_url}{vision_end[1]}" for img_url in images_url])
    return images_token
    

def convert_dialogs_to_inputs(
    dialogs,
    history,
    i,
    tools_desc,
    examples_str,
    images_url=None,
    subtask_generate_guide_prefix=""
):
    user_prefix=['<<<用户>>>: ', '<<</用户>>>: ']
    bot_prefix=['<<<智能助手>>>: ', '<<</智能助手>>>: ']
    planner_prefix=['<<<你(任务拆解专家)>>>: ', '<<</你(任务拆解专家)>>>: ']


    task_prompt = []
    messages = [
        dialogs[2*i],
        dialogs[2*i+1]
    ]

    
    for i, c in enumerate(messages):
        # if c["role"] == "user":
        if c["role"] == "user":
            content_dct = {}

            # 
            images_token = convert_images_to_tokens(images_url)


            # 历史
            history_str = ''
            if len(history):
                
                count = 1
                for h in history:
                    history_str += TURN_START_SYMBOL +'\n'
                    # 1、图片、问题
                    history_str += '# 用户请求信息\n'
                    history_str += "## 用户问题\n"+ h['user']+'\n\n\n'
                    # 2、拆解
                    plan = re.sub(r'SubTask[\d+]',f'SubTask{count}',h['planner'])
                    plan = re.sub(r'#E[\d+]',f'#E{count}',plan)
                    history_str += "# 任务拆解专家拆解结果\n" + plan +'\n\n\n'
                    # 3、助手
                    bot_str = '...' if  h['assistant']=='' else  h['assistant']
                    history_str += "# 智能助手回复结果\n" +bot_str +'\n'
                    history_str += TURN_END_SYMBOL+'\n'

                    count += 1

                # user_prompt +=history_str

            # 当前轮



            user_prompt = "# 用户请求信息\n"

            user_prompt += "## 时间\n" + "请从用户对话中获取" + '\n\n'
            user_prompt += "## 位置\n" + "请从用户对话中获取" + '\n\n'

            if "role_action" in c:
                
                # tools
                # tools_desc = get_tools_desc(api_dict, c['role_action'])
                user_prompt += f"## Available Tools\n以下是处理用户问题时，你可以使用的工具(Tools)：\n<Tools>\n{tools_desc}\n</Tools>\n\n"

                # few-shots
                # examples_str = convert_tools_to_examples(api_dict,c['role_action'])
                user_prompt += f"## Examples\n你可以从以下示例中学习如何对问题进行拆解，你需要关注拆解方式！以及如何使用工具：\n<Example>\n{examples_str}</Example>\n\n"

            user_prompt += f"## 屏幕图片\n未提供图片。当前轮次用户问题与屏幕图片无关；或者提供的是图片地址(screen_image_url)，你需要通过工具去访问图片信息。\n\n"
            

            if "text" in c:
                user_prompt += f"## 用户问题\n{c['text']}"

            task_prompt.append(f"{TURN_START_SYMBOL}\n{user_prompt.strip()}")

        elif c["role"] == "assistant":
            assistant_prompt = ASSISTANT_PROMPT.format(
                assistant=c["text"],
            )
            task_prompt.append(f"{assistant_prompt}\n{TURN_END_SYMBOL}")
    #
    task_prompt = f"{SESSION_START_SYMBOL}\n" + history_str + "\n\n\n".join(task_prompt)
    # 强制加上 `SubTask{num_history_tasks+1}: ` 来引导生成，防止子任务编号出错
    task_prompt += f"\n\n\n{PLANNER_PROMPT.format(subtasks='', PLAN_END_SYMBOL='')}" + subtask_generate_guide_prefix

    # print(task_prompt)
    return task_prompt



def parse_instruction(answers):
    response = []
    for ans in answers:
        actions = re.findall(r'#E[\d] = (.+)',ans)

        actions = [action.replace("|END_OF_CURRENT_SUBTASK|","") for action in actions]
        response.extend(actions)
    return ';'.join(response)

        

def load_dialoags(dialogs):
    new_dialogs = []
    for i,item in enumerate(dialogs): 
        if i%2==0:
            new_dialogs.append(
                {
                    "role":"user",
                    "text":item["content"],
                    "role_action":item['role_action'].split(";"),
                }
            )
        else:
            new_dialogs.append(
                {
                    "role":"planner",
                    "text":item["content"]
                }
            )

    return new_dialogs
