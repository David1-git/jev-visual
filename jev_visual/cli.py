import argparse
import json
from pathlib import Path

from .schema import Request


def main():
    parser = argparse.ArgumentParser(description="Local visual Choice/Noul/Score judgments")
    parser.add_argument("request", type=Path, help="JSON request; image path relative to this file")
    parser.add_argument("--model-path", help="downloaded MLX model directory")
    parser.add_argument("--mode", choices=["shared", "independent"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.request.read_text())
    from .preprocessing import resolve_image_paths
    data = resolve_image_paths(data, args.request.resolve().parent)
    if args.mode:
        data["mode"] = args.mode
    request = Request.model_validate(data)
    from .engine import Engine
    result = Engine(args.model_path).judge(request)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
