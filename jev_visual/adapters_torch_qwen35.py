"""NVIDIA GPU adapter for Qwen3.5-2B (transformers), ported from the MLX Qwen3.5 adapter.

把 jev-visual 的共享 prefill + fork 缓存 + 候选打分移植到 PyTorch/transformers，
模型换为 Qwen/Qwen3.5-2B（transformers 原生权重，非 MLX 4bit 版）。

与 InternVLAdapter 的差异（Qwen3.5 是 attention + Gated DeltaNet 混合架构）:
    - cache 是混合结构：attention 层有 key/value，Gated DeltaNet 层有 conv_states /
      recurrent_states，fork 时必须按层类型分别复制（这是 Qwen3.5 移植最难的点）。
    - 位置编码是 mrope 3 维 + rope_deltas；续写时 position_ids 传 None，
      模型从 cache 长度自动续算 3D 位置（transformers 内置 compute_3d_position_ids）。
    - 图像输入用 image_grid_thw（动态分辨率网格）。

已核实的 transformers 事实（2026-09 抓取 modeling_qwen3_5.py / cache_utils.py）:
    - model_type = "qwen3_5", architectures = Qwen3_5ForConditionalGeneration
    - Qwen3_5Model.forward(input_ids, pixel_values, image_grid_thw, attention_mask,
      position_ids=None, past_key_values) -> last_hidden_state + past_key_values + rope_deltas
    - 顶层 Qwen3_5ForConditionalGeneration 自带 self.lm_head
    - DynamicCache.layers 是 CacheLayer 列表：attention 层自带 repeat_interleave(count, dim=0)；
      LinearAttentionLayer 有 conv_states/recurrent_states（dict, batch 维在 dim 0）

待实测点（标注 TODO-VERIFY）:
    [1] Qwen3_5Processor 对 images/text 的输出字段（input_ids / pixel_values / image_grid_thw）。
    [2] 混合 cache fork 的逐层复制与 transformers 版本兼容性。
    [3] 续写时 position_ids=None 由模型自动续算的行为（需 oracle 对拍确认）。
"""
import copy
from dataclasses import dataclass, asdict
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-2B"


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


class Qwen35TorchAdapter:
    """Transformers/PyTorch 版适配器，接口对齐 Qwen35Adapter / InternVLAdapter。"""

    def __init__(self, model, processor, device="cuda"):
        self.mx = torch
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

    # ---- 后端无关算子（供 scoring.py 使用）----
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
        """把 prefix 文本 + 图像（单张或列表）编码成 input_ids / pixel_values / image_grid_thw。

        TODO-VERIFY[1]: Qwen3.5 的 template 渲染图像占位符（<|vision_start|> 等），
        processor(images=images, text=prompt) 应输出 pixel_values + image_grid_thw
        （多图时 image_grid_thw 形状 (num_images, 3)）。
        """
        t = time.perf_counter()
        if not isinstance(images, (list, tuple)):
            images = [images]
        inputs = self.processor(images=list(images), text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        if inputs["input_ids"].shape[-1] > 6000:
            raise ValueError("images and prompt exceed 6000 input tokens")
        self.timings.preprocessing_ms += (time.perf_counter() - t) * 1000
        return inputs

    # ---------- 共享 prefill ----------
    def prefill(self, inputs, picks=None):
        """对完整输入做一次前向，返回 (cache, next_position, logits)。

        position_ids 传 None：Qwen3_5Model 内部 compute_3d_position_ids 自动算 mrope 位置。
        """
        t = time.perf_counter()
        with torch.inference_mode():
            out = self.model.model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
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
        position = cache.get_seq_length() if cache is not None else int(inputs["input_ids"].shape[-1])

        t = time.perf_counter()
        logits = self.project(out.last_hidden_state, picks) if picks else None
        self.timings.scoring_ms += (time.perf_counter() - t) * 1000
        return cache, position, logits

    # ---------- 缓存 fork（Qwen3.5 混合架构的关键） ----------
    def fork(self, cache, count):
        """按 batch 复制混合 cache。

        TODO-VERIFY[2]: 按层类型复制——attention 层（DynamicLayer 等）用其自带
        batch_repeat_interleave（原地修改，故先深拷贝该层）；Gated DeltaNet 层
        （LinearAttentionLayer，conv_states/recurrent_states dict）手动 repeat dim=0。
        prefix 缓存很小（2B 模型约十几 MB），深拷贝可接受。
        """
        from transformers.cache_utils import DynamicCache

        t = time.perf_counter()
        new = DynamicCache()
        new.layers = []
        for src in cache.layers:
            dst = copy.deepcopy(src)
            if hasattr(dst, "batch_repeat_interleave"):
                dst.batch_repeat_interleave(count)
            else:
                for key in ("conv_states", "recurrent_states"):
                    d = getattr(dst, key, None)
                    if isinstance(d, dict):
                        for k, v in d.items():
                            if v is not None:
                                d[k] = v.repeat_interleave(count, dim=0)
            new.layers.append(dst)
        new._seen_tokens = cache.get_seq_length()
        self.timings.cache_fork_ms += (time.perf_counter() - t) * 1000
        return new

    # ---------- 批量后缀续写 ----------
    def suffix(self, cache, next_position, rows, picks):
        """对每个分支用 fork 后的 cache 续写 suffix，返回各分支 picks 位置的 logits。

        position_ids 传 None：模型从 cache 长度自动续算 3D mrope 位置。
        TODO-VERIFY[3]: 需 oracle 对拍确认续写位置正确。
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
        # 当前段 mask：有效 token 为 1，padding 为 0（transformers 续写约定，
        # attention_mask 只对应当前段 input_ids；过去段由 cache 承载）
        mask = torch.zeros(len(rows), length, dtype=torch.long, device=self.device)
        for i, row in enumerate(rows):
            mask[i, :len(row)] = 1

        with torch.inference_mode():
            out = self.model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=None,          # 模型自动续算 mrope 位置
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
    """加载 Qwen3.5-2B（transformers 原生 bf16 权重）并构造适配器。

    注册表检查：model.config.model_type 必须是 qwen3_5。
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
    if family != "qwen3_5":
        raise ValueError(f"No verified adapter for {family}; available: ['qwen3_5']")
    return Qwen35TorchAdapter(model, processor, device=device), source, None
