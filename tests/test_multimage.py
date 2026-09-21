"""多图支持的 mock 验证：schema 校验 / 多图解码 / prompt 组装 / 适配器 / engine 全链路。

这些测试不需要 GPU，也不需要下载模型权重。真实模型的多图数值正确性
仍需在 GPU 上跑 examples/verify_scoring_torch.py 的两图对拍用例。
"""
import math

import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

from jev_visual.adapters_torch_internvl import InternVLAdapter, Timings as InternVLTimings
from jev_visual.adapters_torch_qwen35 import Qwen35TorchAdapter
from jev_visual.engine import Engine
from jev_visual.preprocessing import build_prompts, read_images, resolve_image_paths
from jev_visual.schema import Request


class CharTokenizer:
    """按字符编码（稳定前缀），模拟真实 tokenizer 的接口。"""

    all_special_ids = [0]
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(c) for c in ids)


class SpyProcessor:
    """记录 messages 与 images 调用，渲染串含 marker（build_prompts 依赖）。"""

    def __init__(self):
        self.tokenizer = CharTokenizer()
        self.last_messages = None
        self.last_images = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.last_messages = messages
        return "<sys>Context: {}\n\nJEV_VISUAL_QUESTION_INSERTION_81c42<gen>".format(
            messages[1]["content"][-1]["text"].split("Context: ", 1)[1].split("\n\n", 1)[0]
        )

    def __call__(self, images=None, text=None, return_tensors="pt"):
        self.last_images = list(images)
        return {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long)}


# ---------------------------------------------------------------- schema

def test_schema_accepts_single_and_multi_image():
    q = {"type": "noul", "instructions": "Is it true?"}
    assert Request(image="a.jpg", questions={"q": q}).images == ["a.jpg"]
    req = Request(image=["a.jpg", "b.png"], questions={"q": q})
    assert req.images == ["a.jpg", "b.png"]


def test_schema_rejects_bad_image_values():
    q = {"type": "noul", "instructions": "Is it true?"}
    with pytest.raises(ValidationError):
        Request(image=[], questions={"q": q})
    with pytest.raises(ValidationError):
        Request(image=["a.jpg", ""], questions={"q": q})
    with pytest.raises(ValidationError):
        Request(image="", questions={"q": q})


# ---------------------------------------------------------------- read_images

def test_read_images_decodes_and_bounds(tmp_path):
    im1 = Image.new("RGB", (100, 80), "blue")
    im2 = Image.new("RGB", (3000, 2000), "green")
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    im1.save(p1)
    im2.save(p2)
    images = read_images([str(p1), str(p2)])
    assert len(images) == 2
    assert images[0].size == (100, 80)          # 小图不放大
    assert max(images[1].size) <= 768           # 大图缩到 768 上限
    assert images[0].mode == "RGB"
    with pytest.raises(ValueError):
        read_images([])


def test_resolve_image_paths_supports_list(tmp_path):
    data = {"image": ["a.png", "data:image/png;base64,xxx"]}
    resolve_image_paths(data, tmp_path)
    assert data["image"][0] == str(tmp_path / "a.png")
    assert data["image"][1] == "data:image/png;base64,xxx"


# ---------------------------------------------------------------- build_prompts

def test_build_prompts_places_one_image_entry_per_image():
    proc = SpyProcessor()
    req = Request(image=["a.png", "b.png"], questions={"q": {"type": "noul", "instructions": "True?"}})
    prefix, plans = build_prompts(proc, req)
    content = proc.last_messages[1]["content"]
    image_entries = [c for c in content if c["type"] == "image"]
    assert len(image_entries) == 2
    assert plans[0].suffix_ids
    assert "JEV_VISUAL_QUESTION_INSERTION_81c42" not in plans[0].suffix


def test_build_prompts_single_image_still_one_entry():
    proc = SpyProcessor()
    req = Request(image="a.png", questions={"q": {"type": "noul", "instructions": "True?"}})
    build_prompts(proc, req)
    content = proc.last_messages[1]["content"]
    assert len([c for c in content if c["type"] == "image"]) == 1


# ---------------------------------------------------------------- adapters

def test_internvl_prepare_receives_image_list():
    proc = SpyProcessor()
    a = InternVLAdapter.__new__(InternVLAdapter)
    a.processor, a.timings, a.device = proc, InternVLTimings(), torch.device("cpu")
    inputs = a.prepare("<IMG_CONTEXT>\nQ", ["img1.jpg", "img2.jpg"])
    assert proc.last_images == ["img1.jpg", "img2.jpg"]
    assert inputs["input_ids"].shape[-1] == 4


def test_qwen35_prepare_receives_image_list():
    proc = SpyProcessor()
    a = Qwen35TorchAdapter.__new__(Qwen35TorchAdapter)
    a.processor, a.timings, a.device = proc, InternVLTimings(), torch.device("cpu")
    inputs = a.prepare("<|vision_start|>Q", ["img1.jpg", "img2.jpg"])
    assert proc.last_images == ["img1.jpg", "img2.jpg"]
    assert inputs["input_ids"].shape[-1] == 4


# ---------------------------------------------------------------- engine 全链路

class FakeAdapter:
    """注入 Engine 的最小后端：验证 image 列表从 schema 流到 prepare。"""

    def __init__(self, processor):
        self.mx = torch
        self.model = None
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.timings = InternVLTimings()
        self.prepared_image_lists = []

    def prepare(self, prompt, images):
        self.prepared_image_lists.append(list(images))
        return {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long)}

    def prefill(self, inputs, picks=None):
        return None, int(inputs["input_ids"].shape[-1]), None

    def fork(self, cache, count):
        return None

    def suffix(self, cache, next_position, rows, picks):
        # 每分支返回 picks 位置的 logits（形状 (len(picks), V)）
        return [torch.randn(len(row), 100) for row in picks]

    def project(self, hidden, picks):
        return [torch.randn(len(row), 100) for row in picks]

    def reset(self):
        self.timings = InternVLTimings()

    def execution_context(self):
        return torch.inference_mode()

    def peak_memory_gb(self):
        return 0.0

    # 后端无关算子（scoring.py 使用）
    def as_f32(self, x):
        return x.float()

    def logsumexp(self, x, dim):
        return torch.logsumexp(x, dim=dim)

    def arange(self, n):
        return torch.arange(n)

    def tensor(self, seq):
        return torch.tensor(seq)

    def sum(self, x):
        return torch.sum(x)

    def cat(self, tensors, dim):
        return torch.cat(tensors, dim=dim)

    def ones_like(self, x):
        return torch.ones_like(x)


def test_engine_judge_multi_image_pipeline():
    proc = SpyProcessor()
    engine = Engine(adapter=FakeAdapter(proc), batch_size=4)
    req = Request(image=["examples/blue-circle.png", "examples/blue-square.png"], questions={
        "q1": {"type": "noul", "instructions": "Is the first image blue?"},
        "q2": {"type": "choice", "instructions": "Which shape?",
               "criteria": {"circle": "circle", "square": "square"}},  # label 模式：候选 A/B 单 token
    })
    result = engine.judge(req)
    # 两张图都流到了 prepare
    assert len(engine.adapter.prepared_image_lists) >= 1
    assert len(engine.adapter.prepared_image_lists[0]) == 2
    # 输出结构完整
    assert set(result["answers"]) == {"q1", "q2"}
    assert set(result["answers"]["q2"]["probabilities"]) == {"circle", "square"}
    assert "metrics" in result and "peak_memory_gb" in result["metrics"]
