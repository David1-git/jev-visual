"""Model-specific cache/position/LM-head details, outside the decision engine.

Only Qwen3.5 is verified today. An adapter implements prepare, prefill, suffix,
fork, reset, execution_context and exposes mx/model/tokenizer/processor/timings;
new families need their own cache and stock-full-forward parity tests.
"""
import time
from dataclasses import dataclass, asdict

MODEL_ID = "mlx-community/Qwen3.5-0.8B-4bit"
MODEL_REVISION = "da28692b5f139cb0ec58a356b437486b7dac7462"


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


class Qwen35Adapter:
    def __init__(self, model, processor):
        import mlx.core as mx
        self.mx, self.model, self.processor = mx, model, processor
        self.tokenizer = processor.tokenizer
        self.timings = Timings()

    def reset(self):
        self.mx.synchronize()
        self.mx.clear_cache()
        self.mx.reset_peak_memory()
        self.timings = Timings()

    # ---- 后端无关算子（供 scoring.py 使用；InternVLAdapter 提供 torch 等价实现）----
    def as_f32(self, x):
        return x.astype(self.mx.float32)

    def logsumexp(self, x, dim):
        return self.mx.logsumexp(x, axis=dim)

    def arange(self, n):
        return self.mx.arange(n)

    def tensor(self, seq):
        return self.mx.array(seq)

    def sum(self, x):
        return self.mx.sum(x)

    def cat(self, tensors, dim):
        return self.mx.concatenate(tensors, axis=dim)

    def ones_like(self, x):
        return self.mx.ones_like(x)

    def peak_memory_gb(self):
        return self.mx.get_peak_memory() / 1e9

    def execution_context(self):
        # Match the memory residency policy used by MLX-VLM generate(). Without
        # this, a busy Mac can page scoring weights while the baseline pins them.
        from mlx_vlm.generate.common import wired_limit
        return wired_limit(self.model)

    def prepare(self, prompt, images):
        from mlx_vlm.utils import prepare_inputs
        t = time.perf_counter()
        if not isinstance(images, (list, tuple)):
            images = [images]
        inputs = prepare_inputs(self.processor, images=list(images), prompts=prompt)
        if inputs["input_ids"].shape[-1] > 6000:
            raise ValueError("images and prompt exceed 6000 input tokens")
        self.timings.preprocessing_ms += (time.perf_counter() - t) * 1000
        return inputs

    def project(self, hidden, picks):
        """Project ONLY requested positions through the unmodified LM head."""
        mx, lm = self.mx, self.model.language_model
        selected = mx.concatenate([hidden[i, mx.array(row)] for i, row in enumerate(picks)], axis=0)
        logits = lm.model.embed_tokens.as_linear(selected) if lm.args.tie_word_embeddings else lm.lm_head(selected)
        mx.eval(logits)
        rows, start = [], 0
        for row in picks:
            rows.append(logits[start:start + len(row)])
            start += len(row)
        return rows

    def prefill(self, inputs, picks=None):
        mx = self.mx
        ids = inputs["input_ids"]
        t = time.perf_counter()
        features = self.model.get_input_embeddings(
            ids, pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"), mask=inputs.get("attention_mask"),
        )
        mx.eval(features.inputs_embeds, features.position_ids)
        self.timings.vision_ms += (time.perf_counter() - t) * 1000
        self.timings.vision_forward_calls += 1
        t = time.perf_counter()
        cache = self.model.language_model.make_cache()
        out = self.model.language_model(
            ids, inputs_embeds=features.inputs_embeds, cache=cache,
            position_ids=features.position_ids, rope_deltas=features.rope_deltas,
            skip_logits=True, return_hidden=True,
        )
        mx.eval(out.hidden_states[-1], *[v for c in cache for v in c.state if v is not None])
        self.timings.prefill_ms += (time.perf_counter() - t) * 1000
        self.timings.language_forward_calls += 1
        t = time.perf_counter()
        logits = self.project(out.hidden_states[-1], picks) if picks else None
        self.timings.scoring_ms += (time.perf_counter() - t) * 1000
        return cache, int(features.position_ids.max().item()) + 1, logits

    def fork(self, cache, count):
        from mlx_vlm.models.cache import ArraysCache, KVCache
        t = time.perf_counter()
        result = []
        for entry in cache:
            if type(entry) not in (ArraysCache, KVCache):
                raise TypeError(f"unsupported cache: {type(entry).__name__}")
            state = [None if x is None else self.mx.repeat(x, count, axis=0) for x in entry.state]
            result.append(type(entry).from_state(state, entry.meta_state))
        self.mx.eval(*[v for c in result for v in c.state if v is not None])
        self.timings.cache_fork_ms += (time.perf_counter() - t) * 1000
        return result

    def suffix(self, cache, next_position, rows, picks):
        mx = self.mx
        branches = self.fork(cache, len(rows))
        t = time.perf_counter()
        length = max(map(len, rows))
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        ids = mx.array([row + [pad] * (length - len(row)) for row in rows])
        positions = mx.broadcast_to((mx.arange(length) + next_position)[None, None, :], (3, len(rows), length))
        out = self.model.language_model(ids, cache=branches, position_ids=positions, skip_logits=True, return_hidden=True)
        logits = self.project(out.hidden_states[-1], picks)
        self.timings.scoring_ms += (time.perf_counter() - t) * 1000
        self.timings.language_forward_calls += 1
        # Right padding cannot affect earlier causal outputs; never reuse these
        # padded hybrid recurrent states for further continuation.
        return logits


ADAPTERS = {"qwen3_5": Qwen35Adapter}


def load_adapter(model_path=None):
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_vlm import load
    # Keep idle Metal buffers bounded on 16GB Macs during long workloads.
    mx.set_cache_limit(256 * 1024 * 1024)
    revision = MODEL_REVISION if model_path is None else None
    if model_path is None:
        model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                                      allow_patterns=["*.json", "*.jinja", "*.safetensors"])
    model, processor = load(str(model_path))
    model.set_dtype(mx.float32)  # Preserve integer 4-bit weights, improve numeric parity.
    family = model.config.model_type
    if family not in ADAPTERS:
        raise ValueError(f"No verified adapter for {family}; available: {list(ADAPTERS)}")
    return ADAPTERS[family](model, processor), str(model_path), revision
