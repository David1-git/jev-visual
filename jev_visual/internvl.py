"""InternVL3 + transformers 后端入口：加载权重并构造 Engine（供 server/CLI 使用）。

使用方式（在 NVIDIA GPU 机器上）:

    JEV_VISUAL_BACKEND=torch_internvl \
    JEV_VISUAL_MODEL_PATH=OpenGVLab/InternVL3-2B-hf \
      uvicorn jev_visual.server:app --host 127.0.0.1 --port 8788

或直接命令行:

    python -m jev_visual.internvl_run examples/photo-request.json --model-path OpenGVLab/InternVL3-2B-hf
"""
import os

import torch

from .adapters_torch_internvl import MODEL_ID


def build_internvl_engine(model_path=None, device=None, batch_size=None):
    """构造注入 InternVLAdapter 的 Engine。首次运行会从 HF/ModelScope 下载权重。"""
    from .adapters_torch_internvl import load_torch_adapter
    from .engine import Engine

    if model_path is None:
        model_path = os.environ.get("JEV_VISUAL_MODEL_PATH") or MODEL_ID
    if device is None:
        device = os.environ.get("JEV_VISUAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if batch_size is None:
        batch_size = int(os.environ.get("JEV_VISUAL_BATCH_SIZE", "4"))

    adapter, source, revision = load_torch_adapter(model_path, device=device)
    engine = Engine(adapter=adapter, batch_size=batch_size)
    engine.model_source = source
    engine.revision = revision
    return engine


def main():
    """CLI：python -m jev_visual.internvl_run <request.json> [--model-path ...] [--device ...]"""
    import argparse
    import json
    from pathlib import Path

    from .schema import Request

    parser = argparse.ArgumentParser(description="Local visual Choice/Noul/Score judgments (InternVL3 + torch)")
    parser.add_argument("request", type=Path, help="JSON request; image path relative to this file")
    parser.add_argument("--model-path", help="HF id or local directory of InternVL3 weights")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    parser.add_argument("--mode", choices=["shared", "independent"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    data = json.loads(args.request.read_text())
    from .preprocessing import resolve_image_paths
    data = resolve_image_paths(data, args.request.resolve().parent)
    if args.mode:
        data["mode"] = args.mode
    request = Request.model_validate(data)

    engine = build_internvl_engine(args.model_path, device=args.device, batch_size=args.batch_size)
    result = engine.judge(request)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
