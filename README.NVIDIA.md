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
