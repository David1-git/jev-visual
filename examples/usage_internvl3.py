#!/usr/bin/env python3
"""InternVL3-2B 使用示例（NVIDIA GPU 移植版）。

在 NVIDIA GPU 机器上运行：

    方式 1 —— 直接跑本脚本（加载引擎 + 真实请求）:
        python examples/usage_internvl3.py [--model-path OpenGVLab/InternVL3-2B-hf] [--device cuda]

    方式 2 —— 命令行（引擎入口 internvl_run）:
        python -m jev_visual.internvl_run examples/photo-request.json \
            --model-path OpenGVLab/InternVL3-2B-hf --device cuda

    方式 3 —— HTTP 服务:
        JEV_VISUAL_BACKEND=torch_internvl \
        JEV_VISUAL_MODEL_PATH=OpenGVLab/InternVL3-2B-hf \
          uvicorn jev_visual.server:app --host 127.0.0.1 --port 8788
        curl -X POST http://127.0.0.1:8788/v1/judge \
          -H 'Content-Type: application/json' \
          -d '{"image": "data:image/jpeg;base64,<base64...>", "questions": {...}}'

    验收（oracle 对拍，误差 < 0.002 才算移植数值正确）:
        python examples/verify_scoring_torch.py --backend internvl \
            --model-path OpenGVLab/InternVL3-2B-hf

依赖: pip install -r requirements-internvl.txt （torch 需 CUDA 版）
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = "OpenGVLab/InternVL3-2B-hf"


def build_single_request():
    """单图：与 examples/photo-request.json 相同的三道题（狗的照片）。"""
    from jev_visual.schema import Request
    image = str(Path(__file__).resolve().parent / "dog.jpg")
    return Request(image=image, questions={
        "animal": {
            "type": "choice",
            "instructions": "照片主体是什么动物？",
            "criteria": {"cat": "一只猫", "dog": "一只狗", "bird": "一只鸟", "other": "其他"},
        },
        "white_fur": {
            "type": "noul",
            "instructions": "这只动物主要是白色毛吗？",
        },
        "setting": {
            "type": "choice",
            "instructions": "动物在室内还是室外？",
            "criteria": {"indoors": "室内", "outdoors": "室外", "unknown": "无法判断"},
        },
    })


def build_multi_request():
    """多图：两张图（蓝色圆形 + 蓝色方形），问题分别指代第一张/第二张图。"""
    from jev_visual.schema import Request
    here = Path(__file__).resolve().parent
    return Request(image=[str(here / "blue-circle.png"), str(here / "blue-square.png")], questions={
        "first": {
            "type": "choice",
            "instructions": "第一张图是什么形状？",
            "criteria": {"circle": "圆形", "square": "方形"},
            "scoring": "single_token",
            "candidates": {"circle": "circle", "square": "square"},
        },
        "second": {
            "type": "choice",
            "instructions": "第二张图是什么形状？",
            "criteria": {"circle": "圆形", "square": "方形"},
            "scoring": "single_token",
            "candidates": {"circle": "circle", "square": "square"},
        },
    })


def main():
    parser = argparse.ArgumentParser(description="InternVL3-2B 使用示例（NVIDIA GPU）")
    parser.add_argument("--model-path", default=DEFAULT_MODEL,
                        help=f"HF id 或本地目录（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--device", default="cuda", help="cuda / cpu（默认 cuda）")
    parser.add_argument("--mode", choices=["shared", "independent"], default="shared",
                        help="shared=共享前缀+fork 缓存；independent=每题全量前向")
    parser.add_argument("--multi", action="store_true",
                        help="多图模式：同时送入两张图（蓝色圆形 + 蓝色方形）")
    args = parser.parse_args()

    print(f"加载 InternVL3-2B: {args.model_path} (device={args.device}) ...")
    from jev_visual.internvl import build_internvl_engine
    engine = build_internvl_engine(args.model_path, device=args.device)

    request = build_multi_request() if args.multi else build_single_request()
    print("请求题目:", list(request.questions),
          f"（{'多图 x' + str(len(request.images)) if args.multi else '单图'}）")
    result = engine.judge(request.model_copy(update={"mode": args.mode}))

    print("\n=== 判定结果 ===")
    for key, answer in result["answers"].items():
        print(f"- {key}: {answer['choice']}  (probabilities: {answer['probabilities']})")
    print("\n=== 原始输出（JSON）===")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print("\n提示: 概率是相对给定候选归一化的，非校准。")
    print("数值验收请运行: python examples/verify_scoring_torch.py --backend internvl "
          f"--model-path {args.model_path}")


if __name__ == "__main__":
    main()
