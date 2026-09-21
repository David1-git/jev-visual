"""Real-model test against full-vocabulary teacher forcing (torch/NVIDIA 版对拍).

在 NVIDIA GPU 机器上运行，验证 InternVL3-2B 或 Qwen3.5-2B 适配器的打分数值
是否与"整词表 teacher forcing"一致（这是 jev-visual 的验收标准）：

    python examples/verify_scoring_torch.py --backend internvl --model-path OpenGVLab/InternVL3-2B-hf
    python examples/verify_scoring_torch.py --backend qwen35   --model-path Qwen/Qwen3.5-2B

通过条件（与原仓库一致，另加多图用例）：
    - 单图 oracle 对拍（整词表归一化）与 shared 打分误差 < 0.002
    - shared 与 independent 概率差 < 0.001
    - 图像主体判定正确
    - 多图（两图）oracle 对拍误差 < 0.002，且两图各自的形状判定正确
"""
import argparse
import json
from pathlib import Path

from jev_visual.preprocessing import build_prompts, read_images
from jev_visual.schema import Request
from jev_visual.scoring import sequence_logprob


def oracle_scores(engine, request, prefix, plans, shared):
    """对每道题、每个候选做独立全量前向（整词表 logits），返回 oracle 分数与误差。"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    images = read_images(request.image)
    oracle, errors = {}, []
    for (key, question), plan in zip(request.questions.items(), plans):
        oracle[key] = []
        for target in plan.targets:
            inputs = engine.adapter.prepare(prefix + plan.suffix, images)
            length = inputs["input_ids"].shape[-1]
            if plan.scoring == "sequence":
                inputs["input_ids"] = torch.cat([inputs["input_ids"], engine.adapter.tensor([target[:-1]])], dim=1)
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            # Independent oracle: stock model __call__, full logits, no shared
            # cache, no selected-position projection and no adapter prefill.
            with torch.inference_mode():
                output = engine.model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    attention_mask=inputs.get("attention_mask"),
                )
            if plan.scoring == "sequence":
                full = output.logits[0, length - 1:length - 1 + len(target)].float()
                value = sequence_logprob(full.cpu().numpy(), target)
            else:
                value = float(output.logits[0, -1, target[0]].item())
            oracle[key].append(value)
        errors.extend(abs(a - b) for a, b in zip(oracle[key], shared["answers"][key]["candidate_scores"]))
    return oracle, errors


def verify_single_image(engine):
    """单图：狗的照片，4 道题（单选/短语/中文/是否）+ shared vs independent 一致性。"""
    request = Request(image="examples/dog.jpg", questions={
        "native": {"type": "choice", "instructions": "Which animal?", "criteria": {"dog": "A dog", "cat": "A cat"}, "scoring": "single_token"},
        "phrase": {"type": "choice", "instructions": "Select a description of the main subject.",
                   "criteria": {"short": "White", "dog": "A white dog", "cat": "A white cat"},
                   "scoring": "sequence", "candidates": {"short": "white", "dog": "white dog", "cat": "white cat"}},
        "chinese": {"type": "choice", "instructions": "图片里的动物在哪里？", "criteria": {"out": "室外草地", "in": "室内地板"},
                    "scoring": "sequence", "candidates": {"out": "室外草地", "in": "室内地板"}},
        "label": {"type": "noul", "instructions": "Is a dog visible?"},
    })
    shared = engine.judge(request)
    direct = engine.judge(request.model_copy(update={"mode": "independent"}))
    prefix, plans = build_prompts(engine.processor, request)
    oracle, errors = oracle_scores(engine, request, prefix, plans, shared)
    delta = max(abs(p - direct["answers"][key]["probabilities"][v]) for key, q in shared["answers"].items() for v, p in q["probabilities"].items())
    assert max(errors) < .002, f"single-image oracle error: {max(errors)}"
    assert delta < .001, f"shared/independent delta: {delta}"
    assert shared["answers"]["native"]["choice"] == "dog"
    return {"oracle": oracle, "max_oracle_score_error": max(errors), "max_probability_delta": delta}


def verify_multi_image(engine):
    """多图：两张图（蓝色圆形 + 蓝色方形），验证多图占位符与图像嵌入顺序。"""
    request = Request(image=["examples/blue-circle.png", "examples/blue-square.png"], questions={
        "first": {"type": "choice", "instructions": "What shape is the FIRST image?",
                  "criteria": {"circle": "circle", "square": "square"}, "scoring": "single_token",
                  "candidates": {"circle": "circle", "square": "square"}},
        "second": {"type": "choice", "instructions": "What shape is the SECOND image?",
                   "criteria": {"circle": "circle", "square": "square"}, "scoring": "single_token",
                   "candidates": {"circle": "circle", "square": "square"}},
    })
    shared = engine.judge(request)
    prefix, plans = build_prompts(engine.processor, request)
    oracle, errors = oracle_scores(engine, request, prefix, plans, shared)
    assert max(errors) < .002, f"multi-image oracle error: {max(errors)}"
    assert shared["answers"]["first"]["choice"] == "circle", shared["answers"]["first"]
    assert shared["answers"]["second"]["choice"] == "square", shared["answers"]["second"]
    return {"oracle": oracle, "max_oracle_score_error": max(errors)}


def main():
    parser = argparse.ArgumentParser(description="torch/NVIDIA scoring oracle verification")
    parser.add_argument("--backend", choices=["internvl", "qwen35"], required=True)
    parser.add_argument("--model-path", help="HF id or local directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.backend == "internvl":
        from jev_visual.internvl import build_internvl_engine
        engine = build_internvl_engine(args.model_path, device=args.device)
    else:
        from jev_visual.qwen35 import build_qwen35_engine
        engine = build_qwen35_engine(args.model_path, device=args.device)

    single = verify_single_image(engine)
    multi = verify_multi_image(engine)
    report = {"model": engine.model_source, "single_image": single, "multi_image": multi}
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/scoring-verification-torch.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("Single-image scoring: PASS", single["max_oracle_score_error"], single["max_probability_delta"])
    print("Multi-image scoring:  PASS", multi["max_oracle_score_error"])
    print("All assertions passed. report -> artifacts/scoring-verification-torch.json")


if __name__ == "__main__":
    main()
