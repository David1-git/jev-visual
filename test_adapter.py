#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
from PIL import Image
import numpy as np

print("=" * 60)
print("测试 Finetuned InternVL3 Adapter")
print("=" * 60)

from jev_visual.adapters_torch_finetuned import load_finetuned_adapter

model_path = '/data/algorithm/user/rzli/InternVL3/work_dirs/internvl_chat_v3/internvl3_2b_dynamic_res_2nd_finetune_full-aibee_inspect2_internvl3_260909_regen_fall_cart_clutter_clean_0909'

print("\n[1] 加载适配器...")
adapter, source, revision = load_finetuned_adapter(model_path, device='cuda:0')

# 测试图像
print("\n[2] 加载测试图像...")
img = Image.open('examples/dog.jpg').convert('RGB')

# 使用正确格式的 prompt（只需要 <img> 和 </img> 标签，prepare 会自动填充 <IMG_CONTEXT>）
print("\n[3] 测试 prepare...")
prompt = "<img></img>\n<|im_start|>user\n照片里是什么动物？<|im_end|>\n<|im_start|>assistant\n"
inputs = adapter.prepare(prompt, img)
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"pixel_values shape: {inputs['pixel_values'].shape}")

# 提取最后一个位置作为 picks
last_pos = [inputs['input_ids'].shape[1] - 1]
print(f"picks: {last_pos}")

# prefill
print("\n[4] 测试 prefill...")
cache, position, logits = adapter.prefill(inputs, picks=[last_pos])
print(f"position: {position}")

if logits:
    print(f"logits shape: {[l.shape for l in logits]}")
    # 检查第一个位置的概率
    probs = torch.softmax(logits[0].float(), dim=-1)
    topk = torch.topk(probs, 10)
    print("Top-10 predictions:")
    top_indices = topk.indices[0].tolist()
    top_values = topk.values[0].tolist()
    for idx, prob in zip(top_indices, top_values):
        word = adapter.tokenizer.decode(idx)
        print(f"  '{word}': {prob:.4f}")

# 测试 fork 和 suffix
print("\n[5] 测试 fork 和 suffix...")
suffix_ids = [
    adapter.tokenizer.encode("dog", add_special_tokens=False),
    adapter.tokenizer.encode("cat", add_special_tokens=False),
]
print(f"suffix_ids: {suffix_ids}")

# 直接在这里进行 suffix 测试（不要再次 fork）
rows = suffix_ids
picks = [[0] for _ in suffix_ids]

print(f"rows: {rows}")
print(f"picks: {picks}")

# 一次性执行 fork 和 suffix
suffix_logits = adapter.suffix(cache, position, rows, picks)
print(f"suffix_logits shapes: {[l.shape for l in suffix_logits]}")

# 比较候选词概率
print("\n候选词概率:")
for i, (logits_row, word_ids) in enumerate(zip(suffix_logits, suffix_ids)):
    word = adapter.tokenizer.decode(word_ids)
    probs = torch.softmax(logits_row.float(), dim=-1)
    word_prob = probs[0, word_ids[0]].item() if len(word_ids) > 0 else 0
    print(f"  '{word}': P={word_prob:.4f}")

print("\n" + "=" * 60)
print("✅ 测试完成!")
print("=" * 60)
