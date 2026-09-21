"""NVIDIA GPU adapter for InternVL3 (transformers), ported from the Qwen3.5 MLX adapter.

原型文件（非原仓库组成部分）。把 jev-visual 的共享 prefill + fork 缓存 + 候选打分
移植到 PyTorch/transformers，模型换为 InternVL3-2B。

依赖:
    pip install torch transformers accelerate pillow numpy fastapi uvicorn pydantic

用法（示意）:
    from jev_visual.adapters_torch_internvl import load_torch_adapter
    from jev_visual.schema import Request
    adapter, model_path, revision = load_torch_adapter("OpenGVLab/InternVL3-2B-hf")
    engine = Engine(adapter=adapter, batch_size=4)   # Engine 已支持注入 adapter
    result = engine.judge(Request(image="examples/dog.jpg", questions={...}))

已核实的 transformers 事实（2026-09 抓取 modeling_internvl.py）:
    - model_type = "internvl", architectures = InternVLForConditionalGeneration
    - 语言骨干 = Qwen2 (纯 attention，标准 RoPE，无 Gated DeltaNet / 无 rope_deltas)
    - InternVLModel.forward(input_ids, pixel_values, attention_mask, position_ids,
      past_key_values) -> last_hidden_state + past_key_values(标准 transformers Cache)
    - 顶层 InternVLForConditionalGeneration 自带 self.lm_head (可能 tie embed_tokens)
    - processor = InternVLProcessor, 动态分辨率 448 patch, max 12 patches, 每 patch 256 token
    - chat template: <|im_start|> 风格, 图像占位渲染为 "<IMG_CONTEXT>\\n"

待实测点（标注 TODO-VERIFY）:
    [1] InternVLProcessor 对 text 中图像占位符的确认识别方式（<image> vs <IMG_CONTEXT>），
        以及展开成多少个 image_token_id（= 每 patch 256 × tile 数）。
    [2] 批量续写时 attention_mask 的长度约定（完整长度 vs 当前段）与右 padding 的行为。
    [3] DynamicCache 直接属性赋值 fork 与 model 的兼容性（transformers 版本相关）。
    [4] suffix 续写时 position_ids 从 next_position 起算的正确性（需 oracle 对拍验证）。
"""
from dataclasses import dataclass, asdict
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "OpenGVLab/InternVL3-2B-hf"


@dataclass
class Timings:
    preprocessing_ms: float = 0
    vision_ms: float = 0
    prefill_ms: float = 0
    scoring_ms: float = 0
    cache_fork_ms: float = 0
    language_forward_calls: int = 0
    vision_forward_calls: int = 0

    def dict(self):
        return asdict(self)


class InternVLAdapter:
    """Transformers/PyTorch 版适配器，接口对齐 Qwen35Adapter 供 engine/scoring 使用。

    engine.py 通过 adapter 接口调用：prepare / prefill / fork / suffix / project /
    reset / execution_context，并读取 adapter.mx / model / processor / tokenizer / timings。
    """

    def __init__(self, model, processor, device="cuda"):
        self.mx = torch  # scoring.py 通过 adapter.mx 引用后端操作
        self.model, self.processor = model, processor
        self.tokenizer = processor.tokenizer
        self.timings = Timings()
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    # ---------- 生命周期 / 内存 ----------
    def reset(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.timings = Timings()

    def execution_context(self):
        return torch.inference_mode()

    def peak_memory_gb(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
        return 0.0

    # ---- 后端无关算子（供 scoring.py 使用；与 Qwen35Adapter 对齐）----
    def as_f32(self, x):
        return x.float()

    def logsumexp(self, x, dim):
        return torch.logsumexp(x, dim=dim)

    def arange(self, n):
        return torch.arange(n, device=self.device)

    def tensor(self, seq):
        return torch.tensor(seq, device=self.device)

    def sum(self, x):
        return torch.sum(x)

    def cat(self, tensors, dim):
        return torch.cat(tensors, dim=dim)

    def ones_like(self, x):
        return torch.ones_like(x)

    # ---------- 输入组装 ----------
    def prepare(self, prompt, images):
        """把 prefix 文本 + 图像（单张或列表）编码成 input_ids / pixel_values / attention_mask。

        TODO-VERIFY[1]: InternVL3 的 template 把图像占位渲染为 "<IMG_CONTEXT>\\n"。
        这里把 prompt 直接交给 processor(images=images, text=prompt)。若 processor
        只认识 "<image>"，需把 prompt 中的 "<IMG_CONTEXT>" 替换为 "<image>"。
        多图：images 为列表，processor 按顺序编码，占位符数量须与图数量一致。
        """
        t = time.perf_counter()
        text = prompt  # prompt 渲染后含 <IMG_CONTEXT> 占位符
        if not isinstance(images, (list, tuple)):
            images = [images]
        inputs = self.processor(images=list(images), text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        if inputs["input_ids"].shape[-1] > 6000:
            raise ValueError("images and prompt exceed 6000 input tokens")
        self.timings.preprocessing_ms += (time.perf_counter() - t) * 1000
        return inputs

    # ---------- 共享 prefill ----------
    def prefill(self, inputs, picks=None):
        """对完整输入做一次前向（vision + 语言），返回 (cache, next_position, logits)。

        InternVLModel.forward 内部完成 vision 特征提取与 <IMG_CONTEXT> 替换，
        返回 last_hidden_state 与标准 transformers Cache。
        """
        t = time.perf_counter()
        with torch.inference_mode():
            out = self.model.model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs.get("attention_mask"),
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.timings.vision_ms += (time.perf_counter() - t) * 1000
        self.timings.vision_forward_calls += 1
        self.timings.prefill_ms += (time.perf_counter() - t) * 1000
        self.timings.language_forward_calls += 1

        cache = out.past_key_values
        position = int(inputs["input_ids"].shape[-1])

        t = time.perf_counter()
        logits = self.project(out.last_hidden_state, picks) if picks else None
        self.timings.scoring_ms += (time.perf_counter() - t) * 1000
        return cache, position, logits

    # ---------- 缓存 fork ----------
    def fork(self, cache, count):
        """按 batch 复制 KV cache。InternVL3 是纯 attention，所有层都是
        DynamicLayer（自带 batch_repeat_interleave，原地修改，故先深拷贝该层）。
        prefix 缓存很小（2B 模型约十几 MB），深拷贝可接受。"""
        import copy
        from transformers.cache_utils import DynamicCache

        t = time.perf_counter()
        new = DynamicCache()
        new.layers = []
        for src in cache.layers:
            dst = copy.deepcopy(src)
            if hasattr(dst, "batch_repeat_interleave"):
                dst.batch_repeat_interleave(count)
            new.layers.append(dst)
        new._seen_tokens = cache.get_seq_length()
        self.timings.cache_fork_ms += (time.perf_counter() - t) * 1000
        return new

    # ---------- 批量后缀续写 ----------
    def suffix(self, cache, next_position, rows, picks):
        """对每个分支用 fork 后的 cache 续写 suffix，返回各分支 picks 位置的 logits。

        transformers 约定：续写时 attention_mask / position_ids 只对应当前段 input_ids
        （长度 = rows 的 padding 后长度），过去段由 cache 承载。
        TODO-VERIFY[2][4]: 右 padding 行为与 position_ids 从 next_position 起算的正确性
        （需 oracle 对拍验证）。
        """
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        branches = self.fork(cache, len(rows))
        t = time.perf_counter()
        length = max(map(len, rows))
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        ids = torch.tensor(
            [row + [pad] * (length - len(row)) for row in rows],
            dtype=torch.long, device=self.device,
        )
        # 普通 RoPE：position_ids 形状 (B, L)，从 next_position 连续递增
        pos = torch.arange(length, device=self.device).unsqueeze(0).expand(len(rows), length)
        pos = pos + next_position
        # 当前段 mask：有效 token 为 1，padding 为 0
        mask = torch.zeros(len(rows), length, dtype=torch.long, device=self.device)
        for i, row in enumerate(rows):
            mask[i, :len(row)] = 1

        with torch.inference_mode():
            out = self.model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=branches,
                use_cache=True,
                return_dict=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.timings.scoring_ms += (time.perf_counter() - t) * 1000
        self.timings.language_forward_calls += 1
        return self.project(out.last_hidden_state, picks)

    # ---------- LM head 定点投影 ----------
    def project(self, hidden, picks):
        """只对需要的 (batch, position) 投影 LM head。顶层 InternVLForConditionalGeneration
        自带 self.lm_head；若 tie 则与 embed_tokens 共享权重，结果一致。"""
        selected = torch.cat(
            [hidden[i, torch.tensor(row, device=self.device)] for i, row in enumerate(picks)],
            dim=0,
        )
        logits = self.model.lm_head(selected)
        rows_out, start = [], 0
        for row in picks:
            rows_out.append(logits[start:start + len(row)])
            start += len(row)
        return rows_out


def load_torch_adapter(model_path=None, device=None):
    """加载 InternVL3 权重并构造适配器（替代 MLX 版 adapters.load_adapter）。

    注册表检查保持同一思路：读 model.config.model_type，只放行已写好的适配器。
    InternVL3-2B-hf 的 model_type = "internvl"。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    source = model_path or MODEL_ID
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(source)
    model = AutoModelForImageTextToText.from_pretrained(
        source, torch_dtype=dtype, device_map=device,
    )
    family = model.config.model_type
    if family != "internvl":
        raise ValueError(f"No verified adapter for {family}; available: ['internvl']")
    return InternVLAdapter(model, processor, device=device), source, None


# 与 Engine 的接入：engine.py 已支持 adapter 注入
#   engine = Engine(adapter=load_torch_adapter(...)[0])
# 需配套修改（见 README 说明）：
#   scoring.py: mx.logsumexp(..., axis=-1) -> torch.logsumexp(..., dim=-1)
#               mx.array / mx.arange / mx.concatenate / mx.ones_like -> torch 等价物
#   engine.py : "peak_metal_memory_gb": self.mx.get_peak_memory()/1e9
#               -> self.adapter.peak_memory_gb()（原 MLX 版保留）
#   preprocessing.py: build_prompts 的 enable_thinking=False 参数仅 Qwen 模板支持，
#               InternVL3 模板无 thinking，需按模型族去掉该参数
