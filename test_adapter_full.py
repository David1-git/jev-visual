#!/usr/bin/env python3
"""完整测试 Finetuned InternVL3 Adapter（改进版）"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
from PIL import Image

from jev_visual.adapters_torch_finetuned import load_finetuned_adapter

model_path = '/data/algorithm/user/rzli/InternVL3/work_dirs/internvl_chat_v3/internvl3_2b_dynamic_res_2nd_finetune_full-aibee_inspect2_internvl3_260909_regen_fall_cart_clutter_clean_0909'

print("=" * 60)
print("完整测试 Finetuned InternVL3 Adapter（改进版）")
print("=" * 60)

print("\n[1] 加载适配器...")
adapter, source, revision = load_finetuned_adapter(model_path, device='cuda:0')

# 测试图像
print("\n[2] 加载测试图像...")
img = Image.open('examples/dog.jpg').convert('RGB')
print(f"图像大小: {img.size}")

print("\n[3] 测试 prepare...")
prompt = "<img></img>\n<|im_start|>user\n照片里是什么动物？<|im_end|>\n<|im_start|>assistant\n"
inputs = adapter.prepare(prompt, img)
print(f"✓ input_ids shape: {inputs['input_ids'].shape}")
print(f"✓ pixel_values shape: {inputs['pixel_values'].shape}")
print(f"✓ image_flags: {inputs['image_flags']}")
print(f"✓ num_tiles_per_image: {inputs.get('num_tiles_per_image', 'N/A')}")

print("\n[4] 测试 prefill...")
cache, position, logits = adapter.prefill(inputs, picks=[[inputs['input_ids'].shape[1] - 1]])
print(f"✓ position: {position}")

probs = torch.softmax(logits[0][0].float(), dim=-1)
topk = torch.topk(probs, 10)
print("Top-10 predictions:")
for idx, prob in zip(topk.indices.tolist(), topk.values.tolist()):
    word = adapter.tokenizer.decode(idx)
    print(f"  '{word}': {prob:.4f}")

print("\n[5] 测试 fork + suffix（续写能力）...")
# 测试续写能力 - 使用中文候选词
suffix_ids = [
    adapter.tokenizer.encode("是", add_special_tokens=False),
    adapter.tokenizer.encode("不是", add_special_tokens=False),
]
print(f"suffix_ids: {suffix_ids}")
print(f"decoded: {[adapter.tokenizer.decode(ids) for ids in suffix_ids]}")

rows = suffix_ids
picks = [[0] for _ in suffix_ids]

suffix_logits = adapter.suffix(cache, position, rows, picks)
print(f"✓ suffix_logits shapes: {[l.shape for l in suffix_logits]}")

# 比较概率
print("\n候选词概率:")
for i, (logits_row, word_ids) in enumerate(zip(suffix_logits, suffix_ids)):
    word = adapter.tokenizer.decode(word_ids)
    probs_suffix = torch.softmax(logits_row[0].float(), dim=-1)
    word_prob = probs_suffix[word_ids[0]].item()
    print(f"  '{word}': P={word_prob:.6f}")

print("\n[6] 测试 multi-turn 对话...")
# 测试多轮对话（复用 cache）
prompt2 = "<img></img>\n<|im_start|>user\n照片里是什么动物？<|im_end|>\n<|im_start|>assistant\n白色的狗<|im_end|>\n<|im_start|>user\n它在哪里？<|im_end|>\n<|im_start|>assistant\n"
inputs2 = adapter.prepare(prompt2, img)
print(f"✓ prompt2 tokens: {inputs2['input_ids'].shape[1]}")

cache2, position2, logits2 = adapter.prefill(inputs2, picks=[[inputs2['input_ids'].shape[1] - 1]])
print(f"✓ position2: {position2}")

probs2 = torch.softmax(logits2[0][0].float(), dim=-1)
topk2 = torch.topk(probs2, 5)
print("Top-5 predictions (second turn):")
for idx, prob in zip(topk2.indices.tolist(), topk2.values.tolist()):
    word = adapter.tokenizer.decode(idx)
    print(f"  '{word}': {prob:.4f}")

print("\n" + "=" * 60)
print("✅ 适配器测试完成!")
print("=" * 60)
