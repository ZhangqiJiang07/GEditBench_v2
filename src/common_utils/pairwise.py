from typing import Dict, Any, Tuple, Iterable, List
from itertools import combinations
import re
import random

_TOKEN_REGEX = re.compile(r"\d+|[a-zA-Z]+")

def generate_randomized_pairs(items: List[Any]) -> List[Tuple[Any, Any]]:
    """
    为给定的项目列表生成所有可能的成对组合，并彻底打乱它们的顺序。

    这个函数执行三个层次的随机化：
    1. 使用 itertools.combinations 生成所有唯一的配对。
    2. 使用 random.shuffle 将所有配对的列表顺序完全打乱。
    3. 对每一对内部的两个项目进行随机排序，以避免位置偏见。

    参数:
        items (List[Any]): 一个包含待配对项目的列表，例如图片文件名。

    返回:
        List[Tuple[Any, Any]]: 一个包含了所有随机排序后的配对的列表。
    """
    if len(items) < 2:
        return []

    # 1. 生成所有唯一的配对组合，例如 C(8, 2) = 28 对
    all_pairs = list(combinations(items, 2))

    # 2. 将这28对的顺序完全打乱
    random.shuffle(all_pairs)

    # 3. 对每一对内部的顺序也进行随机化，防止位置偏见
    randomized_pairs_final = []
    for pair in all_pairs:
        pair_list = list(pair)
        random.shuffle(pair_list)
        randomized_pairs_final.append(tuple(pair_list))
        
    return randomized_pairs_final


def tokenize_name(name: str) -> List[object]:
    """
    Split a name into comparable tokens:
    - numeric tokens -> int
    - alphabetic tokens -> lowercase str

    Examples:
        model1.2-preview  -> ["model", 1, 2, "preview"]
        model-2509        -> ["model", 2509]
        model_v2_1        -> ["model", "v", 2, 1]
        model-2025-09     -> ["model", 2025, 9]
        model-2509-exp3   -> ["model", 2509, "exp", 3]
    """
    tokens: List[object] = []

    for part in _TOKEN_REGEX.findall(name.lower()):
        if part.isdigit():
            tokens.append(int(part))
        else:
            tokens.append(part)

    return tokens


def model_sort_key(name: str):
    """
    Final, fully robust sort key for model names.
    """
    return (tokenize_name(name), name)

def canonical_pair(a: str, b: str) -> Tuple[str, str]:
    """
    Return a deterministic, cache-safe pair ordering.
    """
    return tuple(sorted((a, b), key=model_sort_key))

def generate_canonical_pairs(
    items: Dict[str, Any],
    *,
    n_pairs: int = None,
    seed = None
) -> Dict[str, List[Any]]:
    """
    Generate canonical, cache-safe pairs from a dict.

    Randomness (if any) only affects pair order, never pair identity.

    Args:
        items: Dict of model_name -> value (e.g., image_path)

    Returns:
        Dict[str, List[Any]]: { "modelA_vs_modelB": [valueA, valueB] }
    """
    if len(items) < 2:
        return {}

    sorted_keys = sorted(items.keys(), key=model_sort_key)
    all_key_pairs = list(combinations(sorted_keys, 2))

    if n_pairs is not None and n_pairs < len(all_key_pairs):
        rng = random.Random(str(seed))  # Seed with image key for reproducibility
        selected_key_pairs = rng.sample(all_key_pairs, n_pairs)
    else:
        selected_key_pairs = all_key_pairs

    result = {}
    for k1, k2 in selected_key_pairs:
        a, b = canonical_pair(k1, k2)
        result[f"{a}_vs_{b}"] = [items[a], items[b]]

    return result

