import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path.cwd()))

from jev_visual.adapters_torch_finetuned import (
    load_finetuned_adapter,
    AutoTokenizer,  # noqa
    AutoModelForCausalLM,  # noqa
)

MODEL_PATH = '/data/algorithm/user/rzli/InternVL3/work_dirs/internvl_chat_v3/internvl3_2b_dynamic_res_2nd_finetune_full-aibee_inspect2_internvl3_260909_regen_fall_cart_clutter_clean_0909'

IMG_TOKEN = "<img>"

# 模块级辅助，避免在嵌套函数内定义闭包
def _gpu_free(device_str):
    idx = int(device_str.split(':')[-1])
    return torch.cuda.mem_get_info(idx)


def predict_one(req: dict, adapter):
    meta = req.get('_meta', {})
    gt = meta.get('gt')

    img_paths = req['image']
    if isinstance(img_paths, str):
        imgs = [Image.open(img_paths).convert('RGB')]
    else:
        imgs = [Image.open(p).convert('RGB') for p in img_paths]

    q_name = list(req['questions'].keys())[0]
    q_spec = req['questions'][q_name]
    q_type = q_spec['type']

    if q_type == 'choice':
        criteria = q_spec['criteria']
        opts_text = ", ".join([f"{k} ({v})" for k, v in criteria.items()])
        prompt = (
            IMG_TOKEN * len(imgs) + "</img>" * len(imgs)
            + f"\n<|im_start|>user\n{q_spec['instructions']}\nOptions: {opts_text}\n<|im_end|>\n<|im_start|>assistant\n"
        )
    else:
        prompt = (
            IMG_TOKEN * len(imgs) + "</img>" * len(imgs)
            + f"\n<|im_start|>user\n{q_spec['instructions']}\n<|im_end|>\n<|im_start|>assistant\n"
        )

    inputs = adapter.prepare(prompt, imgs)
    last_pos = inputs['input_ids'].shape[1] - 1
    cache, position, logits = adapter.prefill(inputs, picks=[[last_pos]])

    probs = torch.softmax(logits[0][0].float(), dim=-1)

    if q_type == 'choice':
        criteria = q_spec['criteria']
        option_probs = {}
        for opt_key, _ in criteria.items():
            opt_token_ids = adapter.tokenizer.encode(opt_key, add_special_tokens=False)
            if opt_token_ids:
                option_probs[opt_key] = probs[opt_token_ids[0]].item()
        sorted_opts = sorted(option_probs.items(), key=lambda x: -x[1])
        pred = sorted_opts[0][0]
        # 主动释放中间变量
        del inputs, cache, position, logits, probs
        return {'q_type': 'choice', 'pred': pred, 'gt': gt, 'probs': dict(sorted_opts), 'meta': meta, 'id': meta.get('id')}
    else:
        yes_prob = 0.0
        for w in ["Yes", "yes", "是"]:
            ids = adapter.tokenizer.encode(w, add_special_tokens=False)
            if ids:
                yes_prob = max(yes_prob, probs[ids[0]].item())
        no_prob = 0.0
        for w in ["No", "no", "否"]:
            ids = adapter.tokenizer.encode(w, add_special_tokens=False)
            if ids:
                no_prob = max(no_prob, probs[ids[0]].item())
        pred = "Yes" if yes_prob >= no_prob else "No"
        del inputs, cache, position, logits, probs
        return {'q_type': 'noul', 'pred': pred, 'gt': gt, 'probs': {"Yes": yes_prob, "No": no_prob}, 'meta': meta, 'id': meta.get('id')}


def prf1(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    support = tp + fn
    return p, r, f1, support


def evaluate_full(results, name: str):
    n_total = 0
    n_correct = 0
    cm = {"Yes": {"Yes": 0, "No": 0}, "No": {"Yes": 0, "No": 0}}
    skipped = 0
    for r in results:
        gt, pred = r['gt'], r['pred']
        if gt not in ('Yes', 'No') or pred not in ('Yes', 'No'):
            skipped += 1
            continue
        n_total += 1
        if gt == pred:
            n_correct += 1
        cm[gt][pred] += 1

    # Yes 为正例
    tp_yes, fp_yes, fn_yes = cm['Yes']['Yes'], cm['No']['Yes'], cm['Yes']['No']
    p_y, r_y, f1_y, sup_y = prf1(tp_yes, fp_yes, fn_yes)
    # No 为正例
    tp_no, fp_no, fn_no = cm['No']['No'], cm['Yes']['No'], cm['No']['Yes']
    p_n, r_n, f1_n, sup_n = prf1(tp_no, fp_no, fn_no)
    macro_f1 = (f1_y + f1_n) / 2

    print(f"\n{'=' * 70}")
    print(f"  风格: {name}")
    print(f"{'=' * 70}")
    print(f"  总样本数: {n_total}  (跳过: {skipped})")
    print(f"  Overall Accuracy: {n_correct}/{n_total} = {n_correct / max(n_total, 1):.4f}")
    print()
    print(f"  Confusion Matrix (rows=GT, cols=Pred):")
    print(f"              Pred=Yes   Pred=No")
    print(f"  GT=Yes      {cm['Yes']['Yes']:>8}  {cm['Yes']['No']:>8}")
    print(f"  GT=No       {cm['No']['Yes']:>8}  {cm['No']['No']:>8}")
    print()
    print(f"  {'类别':>10} {'support':>8} {'P':>9} {'R':>9} {'F1':>9}")
    print(f"  {'Yes':>10} {sup_y:>8} {p_y:>9.4f} {r_y:>9.4f} {f1_y:>9.4f}")
    print(f"  {'No':>10} {sup_n:>8} {p_n:>9.4f} {r_n:>9.4f} {f1_n:>9.4f}")
    print(f"  {'Macro':>10} {'':>8} {'':>9} {'':>9} {macro_f1:>9.4f}")
    print()

    return {
        'accuracy': n_correct / max(n_total, 1),
        'precision_yes': p_y, 'recall_yes': r_y, 'f1_yes': f1_y, 'support_yes': sup_y,
        'precision_no': p_n, 'recall_no': r_n, 'f1_no': f1_n, 'support_no': sup_n,
        'macro_f1': macro_f1,
        'n_total': n_total, 'n_correct': n_correct,
        'cm': cm,
    }


def run_eval(jsonl_path: str, adapter, device_str: str):
    """顺序跑完一份 jsonl, 周期性释放显存."""
    n_total = sum(1 for _ in open(jsonl_path))
    results = []
    n = 0
    t0 = time.time()
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            req = json.loads(line)
            try:
                results.append(predict_one(req, adapter))
            except Exception as e:
                print(f"  [sample {i}] ERROR: {e}", flush=True)
            n += 1
            if n % 50 == 0:
                elapsed = time.time() - t0
                eta = (n_total - n) * elapsed / n
                free, total = _gpu_free(device_str)
                print(f"  [{n}/{n_total}] elapsed {elapsed:.0f}s, ETA {eta:.0f}s, free {free/1024**3:.2f}GB", flush=True)
                torch.cuda.empty_cache()
    return results


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--choice-jsonl", required=True)
    ap.add_argument("--noul-jsonl", required=True)
    ap.add_argument("--output", default="/tmp/eval_full_results.json")
    ap.add_argument("--device", default="cuda:7", help="使用空闲显存最大的 GPU")
    args = ap.parse_args()

    # 估算请求总数（用于进度条）
    with open(args.choice_jsonl) as f:
        n_total = sum(1 for _ in f)
    print(f"[1/3] 总样本: {n_total}, 加载模型到 {args.device}...", flush=True)
    free_before, total = _gpu_free(args.device)
    print(f"     (当前 free={free_before / 1024**3:.2f}GB)", flush=True)

    adapter, _, _ = load_finetuned_adapter(MODEL_PATH, device=args.device)
    free_after, total = _gpu_free(args.device)
    print(f"     模型加载完毕, free={free_after / 1024**3:.2f}GB / {total / 1024**3:.2f}GB", flush=True)

    print(f"\n[2/3] 评测 CHOICE ({args.choice_jsonl})", flush=True)
    choice_results = run_eval(args.choice_jsonl, adapter, args.device)
    choice_metrics = evaluate_full(choice_results, "CHOICE")

    print(f"\n[3/3] 评测 NOUL ({args.noul_jsonl})", flush=True)
    noul_results = run_eval(args.noul_jsonl, adapter, args.device)
    noul_metrics = evaluate_full(noul_results, "NOUL")

    print(f"\n{'=' * 70}")
    print("  综合对比: CHOICE vs NOUL")
    print(f"{'=' * 70}")
    print(f"  {'指标':<22} {'CHOICE':>12} {'NOUL':>12}")
    print(f"  {'Accuracy':<22} {choice_metrics['accuracy']:>12.4f} {noul_metrics['accuracy']:>12.4f}")
    print(f"  {'Precision (Yes)':<22} {choice_metrics['precision_yes']:>12.4f} {noul_metrics['precision_yes']:>12.4f}")
    print(f"  {'Recall (Yes)':<22} {choice_metrics['recall_yes']:>12.4f} {noul_metrics['recall_yes']:>12.4f}")
    print(f"  {'F1 (Yes)':<22} {choice_metrics['f1_yes']:>12.4f} {noul_metrics['f1_yes']:>12.4f}")
    print(f"  {'Precision (No)':<22} {choice_metrics['precision_no']:>12.4f} {noul_metrics['precision_no']:>12.4f}")
    print(f"  {'Recall (No)':<22} {choice_metrics['recall_no']:>12.4f} {noul_metrics['recall_no']:>12.4f}")
    print(f"  {'F1 (No)':<22} {choice_metrics['f1_no']:>12.4f} {noul_metrics['f1_no']:>12.4f}")
    print(f"  {'Macro F1':<22} {choice_metrics['macro_f1']:>12.4f} {noul_metrics['macro_f1']:>12.4f}")

    with open(args.output, 'w') as f:
        json.dump({
            'choice': choice_metrics,
            'noul': noul_metrics,
            'choice_results': choice_results,
            'noul_results': noul_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n完整详细结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
