# jev-visual → NVIDIA GPU 移植版（InternVL3-2B / Qwen3.5-2B）

原项目 `hr98w/jev-visual` 只在 **Apple Silicon / MLX / Qwen3.5-0.8B-4bit** 上验证过。

本目录是它的 **NVIDIA GPU / PyTorch / transformers** 移植。核心改动：把 MLX 适配器替换为 PyTorch + transformers + 真实微调权重（InternVL3-2B 权重），适配 AIBEE 场景（商场垃圾分类/杂物检测）的评测流程。

## 目录结构

```
jev-vision/
├── README.md / README.zh-CN.md   # 原版 Apple Silicon README
├── README.NVIDIA.md              # 本文件：NVIDIA GPU 移植说明
├── THIRD_PARTY.md
│
├── jev_visual/                   # 核心推理引擎
│   ├── __init__.py               # 统一导出
│   ├── adapters.py               # 抽象基类（MLX/Torch 共享接口）
│   ├── adapters_torch_finetuned.py  # ★ 核心：微调 InternVL3-2B 适配器（VIP/PIL/前缀/打分为一体）
│   ├── adapters_torch_internvl.py   # 基础 InternVL3-2B（非微调）适配器
│   ├── adapters_torch_qwen35.py     # Qwen3.5-2B 适配器
│   ├── engine.py                # 引擎：请求 → 适配器调用 → 结果组装
│   ├── preprocessing.py         # 请求解析 / prompt 构造 / schema 校验
│   ├── scoring.py               # 打分：候选概率归一化 / A/B/C 标签 / EOS 处理
│   ├── schema.py                # 结果类型定义（choice / noul / score / struct）
│   ├── cli.py                   # 命令行入口
│   ├── server.py                # FastAPI HTTP 服务
│   ├── download.py              # 权重下载工具
│   ├── internvl.py              # InternVL 引擎工厂 + CLI
│   └── qwen35.py                # Qwen3.5 引擎工厂 + CLI
│
├── examples/
│   ├── photo-request.json       # 单图示例请求
│   ├── trash-overflow-request.json  # 杂物溢出检测示例
│   ├── convert_val_to_requests.py   # ★ val 数据集 → 结构化请求文件（jsonl）
│   ├── eval_val_requests.py          # ★ 全量评测脚本（910条 CHOICE + NOUL）
│   ├── evaluate.py                  # 旧版评测入口
│   ├── usage_internvl3.py           # InternVL3 使用示例
│   ├── usage_qwen35.py              # Qwen3.5 使用示例
│   ├── verify_scoring_torch.py      # GPU 验收脚本（适配器 vs 原生打分对拍）
│   ├── verify_scoring.py            # 原版（MLX）对拍脚本
│   └── http_smoke.py                # HTTP 服务冒烟测试
│
├── tests/                        # pytest 单元测试（无需 GPU/权重）
│   ├── test_internvl_adapter.py
│   ├── test_qwen35_adapter.py
│   ├── test_multimage.py
│   ├── test_scoring.py
│   ├── test_schema.py
│   └── test_io_api.py
│
├── benchmarks/                   # 性能 Benchmark（原版 MLX）
├── demo/                         # 可视化 Demo（原版）
└── docs/                         # 文档（原版）
```

## 快速开始（NVIDIA GPU）

### 1. 安装依赖

```bash
pip install -r requirements-internvl.txt
```

### 2. 运行单次请求

```bash
# InternVL3-2B 微调权重（杂物检测）
MODEL_PATH=/data/algorithm/user/rzli/InternVL3/work_dirs/internvl_chat_v3/internvl3_2b_dynamic_res_2nd_finetune_full-aibee_inspect2_internvl3_260909_regen_fall_cart_clutter_clean_0909

python3 -m jev_visual.internvl_run examples/trash-overflow-request.json \
    --model-path "$MODEL_PATH" --device cuda

# 基础 InternVL3-2B（非微调）
python3 -m jev_visual.internvl_run examples/photo-request.json \
    --model-path OpenGVLab/InternVL3-2B-hf --device cuda
```

### 3. 全量评测（910 条验证集）

```bash
# 步骤 1: 转换 val 数据集 → 请求文件（jsonl）
python3 examples/convert_val_to_requests.py \
    --val-json /path/to/val.json \
    --output-dir /tmp

# 步骤 2: 全量评测（会自动找到空闲显存最大的 GPU）
python3 examples/eval_val_requests.py \
    --choice-jsonl /tmp/val_clutter_choice.jsonl \
    --noul-jsonl /tmp/val_clutter_noul.jsonl \
    --output /tmp/eval_full_results.json \
    --device cuda:3   # 指定 GPU（显存紧张的机器上必选）
```

输出示例：

```
======================================================================
  风格: CHOICE
======================================================================
  总样本数: 910  (跳过: 0)
  Overall Accuracy: 625/910 = 0.6868

  Confusion Matrix (rows=GT, cols=Pred):
              Pred=Yes   Pred=No
  GT=Yes           281       174
  GT=No            111       344

          类别  support         P         R        F1
         Yes      455    0.7168    0.6176    0.6635
          No      455    0.6641    0.7560    0.7071
       Macro F1                            0.6853

======================================================================
  风格: NOUL
======================================================================
  总样本数: 894  (跳过: 0)
  Overall Accuracy: 616/894 = 0.6890

         Yes      455    0.6725    0.7582    0.7128
          No      439    0.7113    0.6173    0.6610
       Macro F1                            0.6869
```

### 4. 启动 HTTP 服务

```bash
MODEL_PATH=/data/algorithm/user/rzli/InternVL3/work_dirs/internvl_chat_v3/internvl3_2b_dynamic_res_2nd_finetune_full-aibee_inspect2_internvl3_260909_regen_fall_cart_clutter_clean_0909

JEV_VISUAL_BACKEND=torch_finetuned \
    JEV_VISUAL_MODEL_PATH="$MODEL_PATH" \
    uvicorn jev_visual.server:app --host 127.0.0.1 --port 8788
```

### 5. 适配器数值验收

```bash
# 对拍：Finetuned 适配器的 logits 与 transformers 原生接口是否一致
python examples/verify_scoring_torch.py \
    --backend torch_finetuned \
    --model-path "$MODEL_PATH"
```

## 适配器体系

项目有 **三个** PyTorch 适配器，从底层到顶层：

| 适配器 | 文件 | 用途 | 依赖 |
|--------|------|------|------|
| `InternVLAdapter` | `adapters_torch_internvl.py` | 基础 InternVL3-2B（非微调） | `OpenGVLab/InternVL3-2B-hf` |
| `Qwen35TorchAdapter` | `adapters_torch_qwen35.py` | Qwen3.5-2B | `Qwen/Qwen3.5-2B` |
| `FinetunedInternVLAdapter` | `adapters_torch_finetuned.py` | **微调 InternVL3-2B（AIBEE 场景）** | 自定义微调权重路径 |

### FinetunedInternVLAdapter 核心接口

```python
from jev_visual.adapters_torch_finetuned import load_finetuned_adapter

# 加载（device="cuda:3" 或 "cuda"，单卡）
adapter, model_path, source = load_finetuned_adapter(
    model_path="/path/to/finetuned/weights",
    device="cuda:3"
)

# 单图请求（choice 模式）
inputs = adapter.prepare(prompt, [image])
cache, position, logits = adapter.prefill(inputs, picks=[[-1]])  # picks 位置 = <|im_end|> 后
probs = torch.softmax(logits[0][0].float(), dim=-1)
yes_prob = probs[tokenizer.encode("Yes")[0]]

# 多图请求
inputs = adapter.prepare(prompt, [img1, img2, img3])
```

## GPU 显存管理

InternVL3-2B bf16 权重约 **4.8 GB**，加上 KV cache 和推理临时张量，建议 **≥12GB 显存**。

**显存紧张时的策略**：

```python
# 1. 指定单卡（避免和其他任务抢）
--device cuda:3

# 2. 周期性清理（eval_val_requests.py 已内置）
torch.cuda.empty_cache()  # 每 50 条调用一次

# 3. 推理后主动 del 中间变量
del inputs, cache, position, logits, probs
torch.cuda.empty_cache()
```

## 请求格式

### choice 模式（多选项）

```json
{
  "image": "/path/to/image.jpg",
  "questions": {
    "main": {
      "type": "choice",
      "instructions": "图中是否有杂物溢出？",
      "criteria": {
        "A": "有杂物溢出",
        "B": "没有杂物溢出"
      },
      "scoring": "single_token",
      "candidates": {"A": "A", "B": "B"}
    }
  }
}
```

### noul 模式（Yes/No 判断）

```json
{
  "image": "/path/to/image.jpg",
  "questions": {
    "main": {
      "type": "noul",
      "instructions": "图中是否有杂物溢出？",
      "criteria": {},
      "scoring": "single_token",
      "candidates": {}
    }
  }
}
```

### 多图模式

```json
{
  "image": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
  "questions": { ... }
}
```

## 多图支持

单图与多图完全兼容：`image` 字段传**字符串 = 单图**，传**数组 = 多图**（HTTP JSON 同理）。

实现：schema 字段放宽 → 每张图放一个图像占位条目 → `FinetunedInternVLAdapter.prepare()` 一次编码多图。`image_flags` / `pixel_values` 的 batch 维 = 图像数。`suffix` / `fork` / 打分逻辑零改动——图像占位符全在 prefix 里，共享缓存机制天然支持。

## JEV 原理：单步前缀解码（Single-step Prefix Decoding）

### 完整数据流

```
输入: 图像 + 文本 prompt
  │
  ▼
[Step 1] adapter.prepare() ─────────────────────────────────────────────
  • 图像: 通过 ViT (InternViT-6B / 24层) 编码 → tile embeddings (num_tiles × 256 × 1024)
  • 文本: Tokenize → input_ids
  • 拼接: 把 256 × num_tiles 个 <IMG_CONTEXT> token 的 embedding 替换为图像 embedding
  • 输出: input_embeds [1, seq_len, 1536]
           attention_mask [1, seq_len]
           pixel_values [total_tiles, 3, 448, 448]
  ──────────────────────────────────────────────────────────────────────

[Step 2] adapter.prefill(inputs, picks=[[last_pos]]) ──────────────────
  • 输入: 完整的 multimodal input_embeds（含多张图）
  • 前向:
      ViT encoder: image → vit_embeds
      Qwen2.5-2B (28层, hidden=1536):
        for layer in layers:
          hidden_states = layer(hidden_states, attention_mask)
        logits = lm_head(hidden_states)           ← [1, seq_len, 151674]
  • 关键: 全部 28 层注意力都参与了图像-文本的深层交互
  • picks=[[last_pos]]: 只提取 <|im_end|> 之后第一个生成位置
    （batch=0, pos=seq_len-1，即 <|im_start|>assistant\n 后面的位置）
  • 输出:
      cache: 28层 × [1, num_kv_heads, seq_len, head_dim]  (给后续 suffix 用)
      position: seq_len (KV cache 当前位置)
      logits: [1, 1, 151674]  ← 只取 last_pos 那一列！
  ──────────────────────────────────────────────────────────────────────

[Step 3] torch.softmax(logits[0][0].float(), dim=-1) ──────────────────
  • 得到条件概率分布 P(token | prefix)
  • shape: [151674]
  • 不是 argmax（贪婪采样），而是保留完整分布
  ──────────────────────────────────────────────────────────────────────

[Step 4] 候选 token 概率提取 ─────────────────────────────────────────
  Choice 模式:
    对每个选项 (A, B):
      token_id = tokenizer.encode(option) 的第一个 token
      prob = P[token_id]
      e.g. Yes→9454, No→2753

  NOUL 模式:
    对每个 token 变体:
      Yes/yes: 取 max(prob[9454], prob[9693])
      No/no:   取 max(prob[2753], prob[2152])

  pred = argmax(probabilities)
  ──────────────────────────────────────────────────────────────────────
```

### JEV vs 完整生成

| 维度 | 完整生成 (greedy decode) | JEV (prefix decode) |
|------|--------------------------|---------------------|
| **计算量** | 多次 forward（每步一次） | 一次 prefill（仅视觉阶段一次，语言层一次） |
| **自回归** | 完整自回归（每步用上步结果） | 跳过自回归，直接预测第一个 token |
| **注意力** | 完整的 KV chain（因果注意力） | 仅 prefix 注意力 |
| **本质** | 逐步采样 + 自反馈 | 冻结 KV + 单步投影 |

### 这就是"取第一个 token"吗？

**不是。** 关键区别：

```
完整生成（greedy）:
  logits[step=0] → Yes → logits[step=1] → "es" → ... → "Yes"

JEV (prefill picks=[last_pos]):
  对完整 prefix（包括 <|im_end|>）做一次前向
  → logits[last_pos] → [所有 151674 个 token 的概率]
```

**JEV 的计算量 = 完整生成的第一次前向**（视觉编码 + 28层LLM），但跳过了后续 1-N 步的自回归解码。

### 为什么 JEV 能 work？（理论依据）

模型是 **Qwen2.5-2B**（28层，hidden=1536，vocab=151674，**lm_head 独立**，tie_word_embeddings=False），
InternVL3-2B 的微调方式是 **因果语言建模（CLM）**：

```python
# 训练时计算 loss
labels = input_ids.clone()
loss = cross_entropy(logits, labels)  # 对所有 token 计算
```

对于训练样本 `"<images><prompt>是否有杂物?<|im_end|><|im_start|>assistant\nYes<|im_end|>"`：

```
logits 的分布 = P(token | 之前所有token的上下文)
            = P(Yes | images + prompt + <|im_start|>assistant\n)
```

**模型被训练成：在 <|im_end|> 位置已经"准备好"回答**

这意味着 prefix 的最后一层 hidden state 已经包含了足够的信息来做判别，不需要完整生成。

### JEV 的局限性

| 问题 | 说明 |
|------|------|
| **跳过自注意力** | 完整生成时，模型可以"先想再说"（浅层处理图像，深层整合信息）。JEV 假设信息在 prefix 最后一层已经完整 |
| **position 编码边界** | `last_pos` 是 `\n` 后面的位置，它的 hidden state 包含"正在输出答案"的上下文，但不包含"答案内容"（因为还没生成）。这是 JEV 的核心假设 |
| **tokenizer 对齐** | `"Yes"` 是**单个 token** 9454，`"No"` 也是**单个 token** 2753。对齐是 tokenizer 的偶然特性——如果 tokenizer 把 Yes 切成 Y + es，JEV 就完全失效 |
| **复杂推理** | 如果任务需要多步推理（"先分析 X，再考虑 Y，最后得出结论"），JEV 效果会显著下降，因为没有多层自回归的信息积累 |

### 验证方法

```python
# 方法1: 对比 prefill vs 完整生成在第一个 token 的概率
# 若差异大 → JEV 存在信息损失
P_jev_yes   = probs[9454]
P_gen_yes   = generate(..., max_tokens=1)["Yes"]
delta = abs(P_jev_yes - P_gen_yes)

# 方法2: 检查 attention pattern
# last_pos 对图像 patch 的 attention 权重是否足够大
attn_weights = model.layers[-1].self_attn.attn_weights  # [1, seq_len, num_patches]

# 方法3: 对比不同 position 的 hidden state 投影质量
# 若 last_pos 的 hidden state 质量差，其他位置也不会好
```

### 总结

```
JEV 不是什么：  ❌ 简单的"取第一个 token"
               ❌ greedy decode（那是完整生成）

JEV 是什么：    ✅ 冻结 KV cache + 单步前缀解码
               ✅ 一次 multimodal prefill + LM head 投影
               ✅ 候选集概率直接查表（O(1) per candidate）

JEV 能 work 的原因：
               ✅ 模型被训练成在 <|im_end|> 位置就"准备好"回答
               ✅ Yes/No 都是单 token，与 tokenizer 完全对齐
               ✅ 图像特征在 ViT 编码时已经足够强
               ✅ 28层 LLM 的深层表示足够判别性

JEV 的局限：
               ⚠️ 假设"prefix 最后一层包含所有必要信息"
               ⚠️ 复杂任务（需要多步推理）可能需要完整生成
               ⚠️ 若 tokenizer 对齐改变，JEV 完全失效
```

## 全量评测结果

评测配置：InternVL3-2B 微调权重，val 验证集 910 条，CHOICE + NOUL 两种 prompt 风格。

### CHOICE（完整 910 条）

```
Overall Accuracy: 625/910 = 68.68%

Confusion Matrix (rows=GT, cols=Pred):
              Pred=Yes   Pred=No
  GT=Yes           281       174      ← Recall_Yes = 61.8%（漏检 174 个）
  GT=No            111       344      ← FP = 111

         类别  support         P         R        F1
         Yes      455    0.7168    0.6176    0.6635
          No      455    0.6641    0.7560    0.7071
       Macro F1                            0.6853
```

### NOUL（有效 894/910，16 条显存 OOM 跳过）

```
Overall Accuracy: 616/894 = 68.90%

         Yes      455    0.6725    0.7582    0.7128
          No      439    0.7113    0.6173    0.6610
       Macro F1                            0.6869
```

### 综合对比

| 指标           | CHOICE | NOUL  |
| -------------- | ------ | ----- |
| Accuracy       | 0.6868 | 0.6890 |
| Precision (Yes) | **0.7168** | 0.6725 |
| Recall (Yes)   | 0.6176 | **0.7582** |
| F1 (Yes)       | 0.6635 | **0.7128** |
| Precision (No) | 0.6641 | **0.7113** |
| Recall (No)    | **0.7560** | 0.6173 |
| F1 (No)        | **0.7071** | 0.6610 |
| Macro F1       | 0.6853 | **0.6869** |

**结论**：
- Macro F1 几乎打平（68.53% vs 68.69%）
- **CHOICE 偏向保守**（Recall_Yes=61.8%，倾向说 No，漏检多）
- **NOUL 偏向激进**（Recall_Yes=75.8%，倾向说 Yes，误报多）
- 业务选型：漏报成本高 → NOUL；误报成本高 → CHOICE

## 单元测试

测试无需 GPU 和模型权重：

```bash
# 全部测试
python -m pytest tests/ -q

# 单个测试文件
python -m pytest tests/test_internvl_adapter.py -v
python -m pytest tests/test_qwen35_adapter.py -v
python -m pytest tests/test_multimage.py -v
```

已验证：
- Finetuned 适配器 mock 测试
- InternVL 适配器 mock 测试
- Qwen3.5 适配器 mock 测试
- 多图链路（schema / prepare / engine 全链路）
- 打分逻辑（single_token / A/B/C 标签 / EOS）
- Schema 校验

## 权重

| 模型 | 路径 | 显存 | 用途 |
|------|------|------|------|
| InternVL3-2B 微调 | `.../internvl3_2b_dynamic_res_2nd_finetune_full-aibee_inspect2_internvl3_260909_regen_fall_cart_clutter_clean_0909` | ~4.8GB bf16 | 杂物检测主模型 |
| InternVL3-2B 原版 | `OpenGVLab/InternVL3-2B-hf` | ~4.2GB bf16 | 对照基线 |
| Qwen3.5-2B | `Qwen/Qwen3.5-2B` | ~4.5GB bf16 | 对照实验 |

## 已知边界

- 显存紧张时需指定 `--device` 避免抢占其他任务
- NOUL 模式显存占用略高（prompt 更长），极端情况下可能 OOM
- 概率语义与原项目一致：候选概率相对于提供的选项归一化，**非校准**
- 原 MLX 版代码保留可用
