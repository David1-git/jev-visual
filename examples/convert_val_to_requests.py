"""将 InsVLM val jsonl 转换为 jev-vision 结构化请求格式.

原始数据格式:
{
  "id": 52,
  "conversations": [
    {"from": "human", "value": "<image><image>\n角色：...\n任务：核查图2区域是否存在杂物堆放。\n输出要求：...\n"},
    {"from": "gpt", "value": "{...}"}
  ],
  "category": "clutter",
  "image": ["images/xxx.jpg", "region_crop_masked/xxx_region.jpg"],
  "width_list": [...],
  "height_list": [...]
}

每张样本包含 2 张图：
- 第 1 张: 全景图 (overview)
- 第 2 张: 局部图 (region, 唯一检测范围)

目标问题：核查第 2 张图是否存在杂物堆放。
"""
import json
from pathlib import Path


# 类别到任务的映射
CATEGORY_TASK = {
    "clutter": "核查图2区域是否存在杂物堆放",
    "fall": "核查图2区域是否存在人员跌倒",
    "cart": "核查图2区域是否存在板车",
}


def extract_task_question(category: str) -> str:
    """根据 category 提取核心问题."""
    return CATEGORY_TASK.get(category, f"核查图2区域是否存在{category}")


def extract_gpt_result(gpt_value: str) -> str:
    """从 GPT 回答中提取 result 字段."""
    try:
        # 尝试解析 JSON
        parsed = json.loads(gpt_value.strip())
        return parsed.get("result", "").strip()
    except Exception:
        # 备用：正则提取
        import re
        m = re.search(r'"result"\s*:\s*"(Yes|No)"', gpt_value)
        if m:
            return m.group(1)
        return ""


def build_choice_question(category: str) -> str:
    """构造选择题指令（中文）。"""
    if category == "clutter":
        return (
            "请仔细观察图像内容。第2张图（局部图）是唯一检测范围。\n"
            "判断该局部图区域内是否存在杂物堆放。\n"
            "杂物 = 无人看管、随意放置、无人员即时操作使用的物品。\n"
        )
    elif category == "fall":
        return "请观察图像内容。第2张图（局部图）是唯一检测范围。\n判断该局部图区域内是否存在人员跌倒。"
    elif category == "cart":
        return "请观察图像内容。第2张图（局部图）是唯一检测范围。\n判断该局部图区域内是否存在手推车或购物车。"
    else:
        return f"请观察图像内容。第2张图（局部图）是唯一检测范围。\n判断该局部图区域内是否存在{category}。"


def convert_to_jev_request(
    item: dict,
    image_base: str,
    noul_style: bool = False,
    yes_word: str = "Yes",
    no_word: str = "No",
    use_overview: bool = True,
):
    """将一条 val 数据转换为 jev-vision 结构化请求.

    Args:
        item: 原始 val jsonl 中的一条样本
        image_base: 图像根目录
        noul_style: 是否用 noul（仅是/否）模式
        yes_word: "Yes" 用的 token（视训练数据而定）
        no_word:  "No" 用的 token
        use_overview: 是否保留全景图（多图）

    Returns:
        dict: jev-vision 风格请求
    """
    category = item.get("category", "")
    images = item.get("image", [])
    if not images:
        raise ValueError(f"sample {item.get('id')} has no image")

    # 收集图像绝对路径
    img_paths = []
    if use_overview and len(images) >= 2:
        # 第1张: 全景图; 第2张: 局部图 (检测范围)
        for rel in images[:2]:
            img_paths.append(str(Path(image_base) / rel))
    else:
        # 只用局部图
        img_paths.append(str(Path(image_base) / images[-1]))

    # 构造 instruction
    instructions = build_choice_question(category)

    # question name
    q_name = f"{category}_detection"

    if noul_style:
        # noul: 单个问题是/否
        criteria = {
            yes_word: f"判定为{_zh_label(category)}存在",
            no_word: f"判定为{_zh_label(category)}不存在",
        }
        # 注：在 jev-vision 的 noul 实现里，实际只看 "是/否"
        # 我们通过 criteria 决定 prompt 内容，但最终判定仍走 noul 分支
        question_spec = {
            "type": "noul",
            "instructions": instructions,
        }
    else:
        # choice: 用 criteria 选 Yes/No
        criteria = {
            yes_word: f"存在{_zh_label(category)}",
            no_word: f"不存在{_zh_label(category)}",
        }
        question_spec = {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }

    request = {
        "image": img_paths if len(img_paths) > 1 else img_paths[0],
        "questions": {
            q_name: question_spec
        },
        "_meta": {
            "id": item.get("id"),
            "category": category,
            "gt": extract_gpt_result(item["conversations"][1]["value"]),
            "n_images": len(img_paths),
        },
    }
    return request


def _zh_label(category: str) -> str:
    return {
        "clutter": "杂物堆放",
        "fall": "人员跌倒",
        "cart": "手推车",
    }.get(category, category)


# ============ CLI ============
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--image-base", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--style", choices=["choice", "noul"], default="choice")
    ap.add_argument("--use-overview", action="store_true", default=True,
                    help="保留全景图（默认 True）")
    ap.add_argument("--limit", type=int, default=0, help="限制样本数（0=全部）")
    args = ap.parse_args()

    noul = args.style == "noul"
    out_lines = []
    with open(args.input, "r") as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            item = json.loads(line)
            req = convert_to_jev_request(
                item, args.image_base,
                noul_style=noul,
                use_overview=args.use_overview,
            )
            out_lines.append(json.dumps(req, ensure_ascii=False))

    with open(args.output, "w") as f:
        f.write("\n".join(out_lines))

    print(f"✅ 转换完成: {len(out_lines)} 条")
    print(f"   输入: {args.input}")
    print(f"   输出: {args.output}")
    print(f"   风格: {args.style}")
    print(f"   使用全景图: {args.use_overview}")


if __name__ == "__main__":
    main()
