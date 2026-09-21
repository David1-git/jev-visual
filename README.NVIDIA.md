# jev-visual → NVIDIA GPU 移植版（InternVL3-2B / Qwen3.5-2B）

原项目 `hr98w/jev-visual` 只在 **Apple Silicon / MLX / Qwen3.5-0.8B-4bit** 上验证过。
本目录是它的 **NVIDIA GPU / PyTorch / transformers** 移植：共享 prefill → fork 缓存 →
批量问题后缀 → 候选打分，逻辑不变，只换模型后端。**两个适配器都已写好**，换权重即可跑。

## 目录与新增文件

| 文件 | 作用 |
| --- | --- |
| `jev_visual/adapters_torch_internvl.py` | InternVL3 适配器（`InternVLAdapter`）+ 加载函数 |
| `jev_visual/adapters_torch_qwen35.py` | Qwen3.5-2B 适配器（`Qwen35TorchAdapter`）+ 加载函数 |
| `jev_visual/internvl.py` | InternVL3 引擎工厂 + CLI（`python -m jev_visual.internvl_run`） |
| `jev_visual/qwen35.py` | Qwen3.5-2B 引擎工厂 + CLI（`python -m jev_visual.qwen35_run`） |
| `examples/verify_scoring_torch.py` | **GPU 验收脚本**（oracle 对拍，判断"移植是否数值正确"） |
| `requirements-internvl.txt` | NVIDIA 后端依赖 |
| `tests/test_internvl_adapter.py` | InternVL 适配器 mock 测试（无需 GPU/权重） |
| `tests/test_qwen35_adapter.py` | Qwen3.5 适配器 mock 测试（无需 GPU/权重） |
| `tests/test_multimage.py` | 多图支持测试（schema/解码/prompt/适配器/引擎，无需 GPU/权重） |
| `examples/usage_internvl3.py` | InternVL3-2B 可运行示例（`--multi` 多图模式） |
| `examples/usage_qwen35.py` | Qwen3.5-2B 可运行示例（`--multi` 多图模式） |

后端无关化改造（原 MLX 版仍可用）：`scoring.py` 用 `adapter.*` 算子、
`engine.py` 用 `adapter.peak_memory_gb()`、`preprocessing.py` 按 processor 能力传模板参数、
`adapters.py` 的 `Qwen35Adapter` 补齐同一套算子、`server.py` 支持 `JEV_VISUAL_BACKEND`。

## 快速开始（NVIDIA GPU 机器）

```bash
# 1. 安装依赖（torch 请按官方指引装 CUDA 版）
pip install -r requirements-internvl.txt

# 2. 可选：先跑无需 GPU/权重的适配器逻辑测试
python -m pytest tests/test_internvl_adapter.py tests/test_qwen35_adapter.py -q

# 3a. 命令行跑一个请求（InternVL3-2B）
python -m jev_visual.internvl_run examples/photo-request.json \
    --model-path OpenGVLab/InternVL3-2B-hf --device cuda

# 3b. 命令行跑一个请求（Qwen3.5-2B）
python -m jev_visual.qwen35_run examples/photo-request.json \
    --model-path Qwen/Qwen3.5-2B --device cuda

# 4. GPU 数值验收（两个模型都跑；这是"移植是否正确"的判据）
python examples/verify_scoring_torch.py --backend internvl --model-path OpenGVLab/InternVL3-2B-hf
python examples/verify_scoring_torch.py --backend qwen35   --model-path Qwen/Qwen3.5-2B

# 5. 或启动 HTTP 服务（FastAPI，与原接口一致）
JEV_VISUAL_BACKEND=torch_internvl \
JEV_VISUAL_MODEL_PATH=OpenGVLab/InternVL3-2B-hf \
  uvicorn jev_visual.server:app --host 127.0.0.1 --port 8788
# 换成 Qwen3.5: JEV_VISUAL_BACKEND=torch_qwen35 JEV_VISUAL_MODEL_PATH=Qwen/Qwen3.5-2B
```

## 多图支持（v2 新增）

单图与多图完全兼容：`image` 字段传**字符串 = 单图**，传**数组 = 多图**（HTTP JSON 同理）。

```bash
# 命令行（CLI 的 image 数组自动解析相对路径）
python -m jev_visual.internvl_run examples/multi-request.json --model-path OpenGVLab/InternVL3-2B-hf

# 示例脚本加 --multi（两图：蓝色圆形 + 蓝色方形）
python examples/usage_internvl3.py --multi --model-path OpenGVLab/InternVL3-2B-hf
python examples/usage_qwen35.py    --multi --model-path Qwen/Qwen3.5-2B
```

```json
{
  "image": ["examples/blue-circle.png", "examples/blue-square.png"],
  "questions": {
    "first":  {"type": "choice", "instructions": "第一张图是什么形状？",
               "criteria": {"circle": "圆形", "square": "方形"},
               "scoring": "single_token", "candidates": {"circle": "circle", "square": "square"}},
    "second": {"type": "choice", "instructions": "第二张图是什么形状？",
               "criteria": {"circle": "圆形", "square": "方形"},
               "scoring": "single_token", "candidates": {"circle": "circle", "square": "square"}}
  }
}
```

多图实现：schema 字段放宽 → 每张图放一个图像占位条目 → processor 一次编码多图
（Qwen3.5 的 `image_grid_thw` 形状 `(num_images, 3)`；InternVL3 的 `pixel_values` batch 维 = 图像数）。
**suffix/fork/打分逻辑零改动**——图像占位符全在 prefix 里，共享缓存机制天然支持。
多图数值正确性由 `verify_scoring_torch.py` 的两图对拍用例判定（GPU 上跑）。

## 权重

- InternVL3-2B：`OpenGVLab/InternVL3-2B-hf`（非量化 bf16，~4.2GB；HuggingFace / ModelScope 同名）
- Qwen3.5-2B：`Qwen/Qwen3.5-2B`（非量化 bf16，~4.5GB；HuggingFace / ModelScope 同名）

首次运行自动下载；也可先手动下载后把 `--model-path` 指到本地目录。
**显存需求：2B 模型 bf16 加载约 4–5GB，另有前缀 KV 缓存，建议 ≥12GB 显存。**

## GPU 实测点（TODO-VERIFY，跑 `verify_scoring_torch.py` 时确认）

适配器基于 transformers 源码与官方模型卡编写。**已完成的不依赖 GPU 的验证**：

- `pytest` 全量 **43/43 通过**（原仓库 14 + 两个新适配器 mock 各 10 = 20 + 多图支持 9）
- 多图链路（mock）：schema 接受/拒绝、多图解码与 768 上限、每图一个占位条目、
  两个适配器 prepare 收到列表、engine 全链路（schema → read_images → build_prompts → prepare）
- 用 **真实 transformers 5.17** 拉取两模型 config 验证：
  - `Qwen/Qwen3.5-2B` 的 `DynamicCache` = **18 个 LinearAttentionLayer + 6 个 DynamicLayer**
    （混合架构），`fork()` 逐层复制 conv/recurrent states 与 keys/values，batch 1→4 全部正确、
    原 cache 未被污染、`_seen_tokens` 保留
  - `OpenGVLab/InternVL3-2B-hf` 的 `DynamicCache` = **28 个 DynamicLayer**（纯 attention），fork 正确
  - 两个模型均已注册在 `AutoModelForImageTextToText` 映射表内

**仍需你那边（NVIDIA GPU）实测的点**（脚本会断言，未通过会报错并写
`artifacts/scoring-verification-torch.json`）：

1. `processor(images=[...], text=渲染后prefix)` 的图像占位符识别与展开数量
   （InternVL3: `<IMG_CONTEXT>` → 每 patch 256 token × tile 数；Qwen3.5: `<|image_pad|>` 族）。
   若占位符不被识别，`prepare()` 里需把占位串替换为 processor 认识的记号（已注释标注）。
2. 混合缓存 fork 在真实模型 prefill 后（`has_previous_state` 等标志置位后）的行为。
3. 续写时 `position_ids` 处理：InternVL3 显式从 `next_position` 起算；
   Qwen3.5 传 None 由模型从 cache 长度自动续算 mrope 位置。
4. suffix 右 padding + 当前段 attention_mask 的数值正确性（对拍判据）。
5. **多图**（新）：占位符数量/顺序与图数量对应（两图对拍用例覆盖）、
   InternVL3 的 `max_patches=12` 是每图还是合计上限、多图 cache fork 后图像嵌入位置。

## 已知边界（诚实声明）

- 本移植未在真实 GPU 上执行过；`verify_scoring_torch.py` 通过 = 移植数值正确。
- 原 MLX 版代码保留可用（`python -m pytest` 原 14 个测试仍通过）。
- 概率语义与原项目一致：候选概率相对于提供的选项归一化，**非校准**。

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
  • 输入: 完整的 multimodal input_embeds（含 2 张图）
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
    对每个选项 (Yes, No):
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
