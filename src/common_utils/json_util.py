import re
import json
import json_repair
from typing import Tuple, Optional
from .logging_util import get_logger
logger = get_logger()

    
def extract_json_string(text: str) -> Optional[str]:
    """
    从可能包含额外文本的字符串中提取出第一个有效的JSON对象或数组字符串。

    这个函数会寻找第一个 '{' 或 '[' 作为起点，并寻找最后一个 '}' 或 ']' 
    作为终点，以提取出最外层的JSON结构。

    参数:
        text (str): 包含JSON内容的原始字符串。

    返回:
        Optional[str]: 如果找到，返回纯净的JSON字符串；否则返回 None。
    """
    if not isinstance(text, str):
        return None

    # 寻找JSON的可能起点（第一个 '{' 或 '['）
    start_brace_index = text.find('{')
    start_bracket_index = text.find('[')

    if start_brace_index == -1 and start_bracket_index == -1:
        # 如果找不到任何起点，则直接返回None
        return None
    
    if start_brace_index == -1:
        start_index = start_bracket_index
    elif start_bracket_index == -1:
        start_index = start_brace_index
    else:
        # 两者都找到了，取更早出现的那个
        start_index = min(start_brace_index, start_bracket_index)

    # 寻找JSON的可能终点（最后一个 '}' 或 ']'）
    # 使用 rfind 从右侧开始查找
    end_brace_index = text.rfind('}')
    end_bracket_index = text.rfind(']')

    if end_brace_index == -1 and end_bracket_index == -1:
        return None

    # 取更晚出现的那个作为终点
    end_index = max(end_brace_index, end_bracket_index)

    # 进行基本的位置校验
    if end_index < start_index:
        return None

    # 提取并返回JSON字符串
    return text[start_index : end_index + 1]


def extract_winner_from_text(text: str) -> Optional[str]:
    """
    从包含JSON的原始文本中提取出 'winner' 键的值。

    这个函数执行三个步骤：
    1. 从原始文本中提取出纯净的JSON字符串。
    2. 将该字符串解析为Python字典。
    3. 从字典中安全地获取 'winner' 键的值。

    参数:
        text (str): 模型的原始文本输出。

    返回:
        Optional[str]: 如果成功找到，返回 'winner' 的值 (例如 'Image A')；
                    如果在任何步骤失败（找不到JSON、JSON格式错误、没有winner键），则返回 None。
    """
    # 步骤 1: 提取纯净的JSON字符串
    if "<think>" in text: # check for mimo-vl style
        if "</think>" not in text:
            return None

        text = text.split("</think>")[-1]
    
    if "</analysis>" in text: # check for keye-vl style
        text = text.split("</analysis>")[-1]

    if r"<|begin_of_box|>" in text: # check for glm-4.5v style
        if r"<|end_of_box|>" not in text:
            return None

        text = text.split("<|begin_of_box|>")[-1]

    # 清理markdown标记
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]

    json_string = extract_json_string(text.replace('\n', ' ').replace('\\n', ' '))
    
    if not json_string:
        return None

    try:
        # 步骤 2: 将字符串解析为Python字典
        data_dict = json.loads(json_string)
        
        # 步骤 3: 使用 .get() 方法安全地获取 'winner' 的值
        # .get('winner') 会在 'winner' 键不存在时返回 None，而不是抛出错误
        if isinstance(data_dict, dict):
            winner = data_dict.get('winner')
            return winner
        else:
            return None # 解析出的不是字典（比如是列表）

    except json.JSONDecodeError:
        # 如果提取出的字符串不是有效的JSON，则返回None
        return None

def extract_reasoning_from_text(text: str) -> Optional[str]:
    """
    从包含JSON的原始文本中提取出 'reasoning' 键的值。

    这个函数执行三个步骤：
    1. 从原始文本中提取出纯净的JSON字符串。
    2. 将该字符串解析为Python字典。
    3. 从字典中安全地获取 'reasoning' 键的值。

    参数:
        text (str): 模型的原始文本输出。

    返回:
        Optional[str]: 如果成功找到，返回 'winner' 的值 (例如 'Image A')；
                    如果在任何步骤失败（找不到JSON、JSON格式错误、没有winner键），则返回 None。
    """
    # 步骤 1: 提取纯净的JSON字符串
    if "<think>" in text: # check for mimo-vl style
        if "</think>" not in text:
            return None

        text = text.split("</think>")[-1]
    
    if "</analysis>" in text: # check for keye-vl style
        text = text.split("</analysis>")[-1]

    if r"<|begin_of_box|>" in text: # check for glm-4.5v style
        if r"<|end_of_box|>" not in text:
            return None

        text = text.split("<|begin_of_box|>")[-1]

    # 清理markdown标记
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]

    json_string = extract_json_string(text.replace('\n', ' ').replace('\\n', ' '))
    
    if not json_string:
        return None

    try:
        # 步骤 2: 将字符串解析为Python字典
        data_dict = json.loads(json_string)
        
        # 步骤 3: 使用 .get() 方法安全地获取 'winner' 的值
        # .get('reasoning') 会在 'reasoning' 键不存在时返回 None，而不是抛出错误
        if isinstance(data_dict, dict):
            reasoning = data_dict.get('reasoning')
            return reasoning
        else:
            return None # 解析出的不是字典（比如是列表）

    except json.JSONDecodeError:
        # 如果提取出的字符串不是有效的JSON，则返回None
        return None

def parse_GLM4d5_V_coordinates(
    coord_str: str,
    img_h: int,
    img_w: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    坐标提取函数，用于从各种格式的字符串中提取四个坐标值。

    能够处理的格式包括但不限于:
    - 括号: [], ( ), < >
    - 嵌套: [[]], (()), <<>>
    - 混合: ([<>])
    - 甚至嵌入在文本中
    
    Args:
        coord_str: 包含坐标的输入字符串。

    Returns:
        一个包含四个整数的元组 (x1, y1, x2, y2)，如果解析失败则返回 None。
    """
    def extract_grounding_box_from_text(text):
        if "<think>" in text: # check for mimo-vl style
            if "</think>" not in text:
                return None

            text = text.split("</think>")[-1]
        
        if "</analysis>" in text: # check for keye-vl style
            text = text.split("</analysis>")[-1]

        if r"<|begin_of_box|>" in text: # check for glm-4.5v style
            if r"<|end_of_box|>" not in text:
                return None

            text = text.split("<|begin_of_box|>")[-1]
        return text.split("<|end_of_box|>")[0].strip()
    
    coord_str = extract_grounding_box_from_text(coord_str)
    if coord_str is None:
        return None

    try:
        # 1. 使用正则表达式 `\d+` 查找字符串中所有独立的数字序列。
        #    \d  代表任意数字 (0-9)
        #    +   代表一个或多个
        #    这个模式会忽略所有非数字字符（括号、逗号、空格等）。
        numbers_as_strings = re.findall(r'\d+', coord_str)
        
        # 2. 验证提取出的数字数量是否正好为 4。
        if len(numbers_as_strings) == 4:
            # 3. 将字符串列表转换为整数元组并返回。
            #    map(int, ...) 会将函数 int 应用于列表中的每个元素。
            x1, y1, x2, y2 = tuple(map(int, numbers_as_strings))
            x1, y1, x2, y2 = int(x1 * img_w / 1000), int(y1 * img_h / 1000), int(x2 * img_w / 1000), int(y2 * img_h / 1000) # GLM-4.5V
            return [(x1, y1, x2, y2)]
        elif (len(numbers_as_strings) % 4 == 0) and (len(numbers_as_strings) > 4):
            box_list = []
            for i in range(len(numbers_as_strings) // 4):
                x1, y1, x2, y2 = tuple(map(int, numbers_as_strings[i*4:(i+1)*4]))
                x1, y1, x2, y2 = int(x1 * img_w / 1000), int(y1 * img_h / 1000), int(x2 * img_w / 1000), int(y2 * img_h / 1000) # GLM-4.5V
                box_list.append((x1, y1, x2, y2))
            return box_list
        else:
            # 如果数字数量不为4，打印错误信息并返回 None。
            # print(numbers_as_strings)
            logger.debug(f"信息: 在字符串中找到 {len(numbers_as_strings)} 个数字，但需要 4 的倍数。")
            return None
            
    except Exception as e:
        # 捕获任何意外错误，确保函数不会崩溃。
        logger.debug(f"错误: 处理字符串 '{coord_str}' 时发生未知错误: {e}")
        return None


def extract_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    json_output = None
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def extract_json_block(text: str):
    """
    Robust JSON extraction from LLM output.
    Priority:
    1. ```json fenced block
    2. ``` fenced block
    3. First {...} structure in text
    """

    if not text:
        return None

    # 1. Try ```json fenced block
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Try generic ``` fenced block
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. Try extracting first JSON object via bracket matching
    start = text.find("{")
    if start == -1:
        return None

    stack = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            stack += 1
        elif text[i] == "}":
            stack -= 1
            if stack == 0:
                return text[start:i+1]

    return None

def parse_json(text):
    json_str = extract_json_block(text)

    if not json_str:
        # fallback: try whole text repair
        try:
            return json_repair.loads(text)
        except:
            return None

    try:
        return json_repair.loads(json_str)
    except Exception as e:
        logger.error(f"Parse JSON error: {e}")
        return None

def parse_Qwen3_VL_coordinates(
    text: str,
    img_h: int,
    img_w: int
) -> Optional[Tuple[int, int, int, int]]:
    """Parse coordinate points from text format"""
    bounding_boxes = extract_json(text)
    if bounding_boxes is None:
        return None
    json_output = json_repair.loads(bounding_boxes)

    if not isinstance(json_output, list):
        json_output = [json_output]
    if not isinstance(json_output[0], list) and not isinstance(json_output[0], dict):
        json_output = [json_output]

    boxes_list = []
    for i, bounding_box in enumerate(json_output):
        if len(bounding_box) == 4:
            try:
                abs_x1, abs_y1, abs_x2, abs_y2 = tuple(map(int, bounding_box))
                abs_x1, abs_y1, abs_x2, abs_y2 = int(abs_x1 * img_w / 1000), int(abs_y1 * img_h / 1000), int(abs_x2 * img_w / 1000), int(abs_y2 * img_h / 1000)
                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1
                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1
                boxes_list.append((abs_x1, abs_y1, abs_x2, abs_y2))
            except Exception as e:
                logger.debug(f"Error parsing bounding box {i}: {e}")
                continue
        else:
            try:
                # Convert normalized coordinates to absolute coordinates
                abs_y1 = int(bounding_box["bbox_2d"][1] / 1000 * img_h)
                abs_x1 = int(bounding_box["bbox_2d"][0] / 1000 * img_w)
                abs_y2 = int(bounding_box["bbox_2d"][3] / 1000 * img_h)
                abs_x2 = int(bounding_box["bbox_2d"][2] / 1000 * img_w)

                if abs_x1 > abs_x2:
                    abs_x1, abs_x2 = abs_x2, abs_x1
                if abs_y1 > abs_y2:
                    abs_y1, abs_y2 = abs_y2, abs_y1

                boxes_list.append((abs_x1, abs_y1, abs_x2, abs_y2))
            except Exception as e:
                logger.debug(f"Error parsing bounding box {i}: {e}")
                continue

    if len(boxes_list) == 0:
        return None

    return boxes_list


def extract_pred_tags(
    input_sample, consistency_type='dino_cosine_consistency',
    evaluate_background=False
):
    # res_dict['consistency_dict']['object_consistency']['brown bear'][0]['dino_cosine_consistency']
    # res_dict['consistency_dict']['object_consistency']['brown bear'][0]['pixel_consistency']
    full_res = []
    if isinstance(input_sample['object_consistency'], str):
        scores = 0.0
    else:
        for object_key, object_value in input_sample['object_consistency'].items():
            if isinstance(object_value, str):
                continue
            for each_obj in object_value:
                each_item = each_obj[consistency_type]
                full_res.append(each_item)
        if len(full_res) != 0:
            scores = sum(full_res) / len(full_res)
        else:
            scores = 0.0
    if evaluate_background:
        # 'background_consistency': {'dino_cosine_consistency': [0.6640625, 0.80078125], 'pixel_consistency': [0.6617698819029565, 0.9335987689448337]}
        background_scores = input_sample['background_consistency'][consistency_type]
        if len(full_res) == 0:
            scores = background_scores
        else:
            scores += background_scores
            # scores[1] += background_scores[1]
            scores /= 2
            # scores[1] /= 2
    return scores