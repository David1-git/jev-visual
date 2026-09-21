"""InternVLAdapter mock 测试：验证适配器接口契约（无需 GPU/权重）。

通过 fake transformers 模块 + Dummy 模型/处理器验证（严格对照
adapters_torch_internvl.py 的实现）：
    prepare（单图/多图/超长拒绝）、prefill、fork（逐层深拷贝 + repeat、
    不污染原 cache）、suffix、project（定点投影）、后端算子、
    reset / peak_memory_gb / execution_context。

需要 torch；不需要下载 InternVL3 权重。
"""
import copy
import sys
import types
from types import SimpleNamespace

import pytest

# ---- fake transformers（在 import 适配器之前注入，避免真实 transformers 依赖）----
_t = types.ModuleType("transformers")
_t.AutoModelForImageTextToText = type("AutoModelForImageTextToText", (), {})
_t.AutoProcessor = type("AutoProcessor", (), {})
_cu = types.ModuleType("transformers.cache_utils")


class DynamicCache:
    """最小 fake：对齐适配器 fork 使用的接口（layers / _seen_tokens / get_seq_length）。"""

    def __init__(self, layers=None, seq=0):
        self.layers = layers if layers is not None else []
        self._seen_tokens = seq

    def get_seq_length(self):
        return self._seen_tokens


_cu.DynamicCache = DynamicCache
_t.cache_utils = _cu
sys.modules["transformers"] = _t
sys.modules["transformers.cache_utils"] = _cu

import torch  # noqa: E402

from jev_visual.adapters_torch_internvl import InternVLAdapter  # noqa: E402


class DummyLayer:
    """对齐 transformers DynamicLayerState.batch_repeat_interleave（原地修改，不返回）。"""

    def __init__(self, batch, heads, length, dim):
        self.key_cache = torch.randn(batch, heads, length, dim)
        self.value_cache = torch.randn(batch, heads, length, dim)

    def batch_repeat_interleave(self, count):
        self.key_cache = self.key_cache.repeat_interleave(count, dim=0)
        self.value_cache = self.value_cache.repeat_interleave(count, dim=0)


class DummyBaseModel:
    def __init__(self, layers, hidden):
        self.layers = layers
        self.hidden = hidden

    def forward(self, **kwargs):
        seq = kwargs.get("input_ids").shape[-1]
        cache = DynamicCache([copy.deepcopy(layer) for layer in self.layers], seq)
        return SimpleNamespace(last_hidden_state=self.hidden, past_key_values=cache)


class DummyModel:
    def __init__(self, layers, hidden, vocab):
        self.config = SimpleNamespace(model_type="internvl")
        self.model = DummyBaseModel(layers, hidden)
        self.lm_head = torch.nn.Linear(hidden.shape[-1], vocab)
        self._to_called = False

    def to(self, device):
        self._to_called = True
        return self

    def eval(self):
        return self


class CharTokenizer:
    all_special_ids = [0]
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(c) for c in ids)


class DummyProcessor:
    def __init__(self, input_seq=4):
        self.tokenizer = CharTokenizer()
        self.input_seq = input_seq
        self.last_images = None

    def __call__(self, images=None, text=None, return_tensors="pt"):
        self.last_images = list(images)
        return {
            "input_ids": torch.arange(1, self.input_seq + 1).unsqueeze(0),
            "attention_mask": torch.ones(1, self.input_seq, dtype=torch.long),
        }


def make_adapter(batch=2, seq=4, hidden=8, vocab=16, layers=2):
    hidden_t = torch.randn(batch, seq, hidden)
    model = DummyModel(
        [DummyLayer(1, 2, seq, hidden) for _ in range(layers)], hidden_t, vocab
    )
    processor = DummyProcessor(input_seq=seq)
    adapter = InternVLAdapter(model, processor, device="cpu")
    return adapter, model, processor


def test_prepare_single_image_normalizes_to_list():
    adapter, _, processor = make_adapter()
    inputs = adapter.prepare("P", "a.jpg")
    assert processor.last_images == ["a.jpg"]
    assert inputs["input_ids"].shape[-1] == processor.input_seq


def test_prepare_multi_image_passes_list():
    adapter, _, processor = make_adapter()
    adapter.prepare("P", ["a.jpg", "b.png"])
    assert processor.last_images == ["a.jpg", "b.png"]


def test_prepare_rejects_overlong_input():
    adapter, _, _ = make_adapter(seq=4)
    adapter.processor = DummyProcessor(input_seq=7000)
    with pytest.raises(ValueError, match="6000"):
        adapter.prepare("P", "a.jpg")


def test_prefill_returns_cache_position_and_optional_logits():
    adapter, _, _ = make_adapter(seq=4, hidden=8, vocab=16)
    inputs = {
        "input_ids": torch.arange(1, 5).unsqueeze(0),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }
    cache, position, logits = adapter.prefill(inputs)
    assert position == 4
    assert logits is None
    assert cache.get_seq_length() == 4
    _, _, logits2 = adapter.prefill(inputs, picks=[[2], [3]])
    assert logits2[0].shape == (1, 16)  # (len(picks), vocab)


def test_fork_repeats_layers_without_mutating_source():
    adapter, _, _ = make_adapter(layers=2)
    src = DynamicCache([DummyLayer(1, 2, 4, 8), DummyLayer(1, 2, 4, 8)], seq=4)
    before = [layer.key_cache.shape[0] for layer in src.layers]
    new = adapter.fork(src, 4)
    assert [layer.key_cache.shape[0] for layer in new.layers] == [b * 4 for b in before]
    # 原 cache 未被污染（fork 先深拷贝再原地 repeat）
    assert [layer.key_cache.shape[0] for layer in src.layers] == before
    assert new.get_seq_length() == 4


def test_suffix_returns_one_row_per_branch():
    adapter, _, _ = make_adapter(batch=2, seq=4, hidden=8, vocab=16)
    src = DynamicCache([DummyLayer(1, 2, 4, 8)], seq=4)
    rows = [[10, 11], [12]]
    picks = [[0], [1]]
    out = adapter.suffix(src, 4, rows, picks)
    assert len(out) == 2
    assert out[0].shape == (1, 16)
    assert out[1].shape == (1, 16)


def test_project_projects_only_picked_positions():
    adapter, model, _ = make_adapter(batch=2, seq=4, hidden=8, vocab=16)
    hidden = model.model.hidden
    out = adapter.project(hidden, [[1, 2], [3]])
    expected = model.lm_head(hidden[0, [1, 2]])
    assert out[0].shape == (2, 16)
    assert torch.allclose(out[0], expected, atol=1e-6)
    assert out[1].shape == (1, 16)


def test_backend_ops_match_torch():
    adapter, _, _ = make_adapter()
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    assert adapter.as_f32(x).dtype == torch.float32
    assert torch.allclose(adapter.logsumexp(x, dim=-1), torch.logsumexp(x, dim=-1))
    assert adapter.arange(5).tolist() == list(range(5))
    assert adapter.tensor([1, 2]).tolist() == [1, 2]
    assert adapter.sum(x).item() == 10.0
    assert adapter.cat([x, x], dim=0).shape == (4, 2)
    assert adapter.ones_like(x).shape == x.shape


def test_reset_and_peak_memory_on_cpu():
    adapter, _, _ = make_adapter()
    adapter.timings.preprocessing_ms = 12.3
    adapter.reset()
    assert adapter.timings.preprocessing_ms == 0.0
    assert adapter.peak_memory_gb() == 0.0


def test_execution_context_and_device_to():
    adapter, model, _ = make_adapter()
    assert model._to_called  # __init__ 中 model.to(device).eval() 已调用
    with adapter.execution_context():
        pass
