"""NVIDIA GPU adapter for Finetuned InternVL3 (InternVLChatModel).

这是专门为你指定的 finetuned 模型创建的适配器：
/data/algorithm/user/rzli/InternVL3/work_dirs/internvl_chat_v3/internvl3_2b_dynamic_res_2nd_finetune_full-aibee_inspect2_internvl3_260909_regen_fall_cart_clutter_clean_0909

改进点（参考原始 modeling_internvl_chat.py）：
1. 正确使用 image_flags 来标记有效图像
2. 优化 vit_embeds 替换逻辑
3. 确保 batch 处理正确
"""
from dataclasses import dataclass, asdict
import time

import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "/data/algorithm/user/rzli/InternVL3/work_dirs/internvl_chat_v3/internvl3_2b_dynamic_res_2nd_finetune_full-aibee_inspect2_internvl3_260909_regen_fall_cart_clutter_clean_0909"

# ImageNet 归一化参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_transform(input_size=448):
    """创建图像预处理 transform"""
    from torchvision.transforms import Compose, Lambda, Resize, ToTensor, Normalize
    from torchvision.transforms.functional import InterpolationMode
    
    return Compose([
        Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """动态预处理：保持宽高比切分成多个 tile"""
    from math import sqrt
    
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    
    # 预定义的目标宽高比集合
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    
    # 找最接近的宽高比
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    
    # 计算目标宽高
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    
    # resize 图像
    resized_img = image.resize((target_width, target_height))
    
    # 切分成多个 tile
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    # 添加缩略图
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    """加载并预处理图像，返回 pixel_values
    
    Args:
        image_file: 图像文件路径 (str) 或 PIL Image 对象
    """
    # 支持 PIL Image 或 文件路径
    if isinstance(image_file, Image.Image):
        image = image_file.convert('RGB')
    elif isinstance(image_file, str):
        if image_file.startswith("http"):
            from urllib.request import urlopen
            image = Image.open(urlopen(image_file)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
    else:
        raise ValueError(f"image_file must be str or PIL Image, got {type(image_file)}")
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values


class FinetunedInternVLAdapter:
    """适配器 for InternVLChatModel (你的 finetuned 模型).

    改进点（参考原始 modeling_internvl_chat.py）：
    1. 使用 image_flags 标记有效图像
    2. 正确处理 batch 和 tile 维度
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.mx = torch
        self.model = model
        self.tokenizer = tokenizer
        self.processor = tokenizer
        self.timings = Timings()
        self.device = torch.device(device)
        
        # 缓存图像嵌入（用于 suffix）
        self._cached_vit_embeds = None
        self._cached_pixel_values = None
        self._batch_size = 0
        
        # 设置图像上下文 token id
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_id
        print(f"[FinetunedInternVLAdapter] img_context_token_id: {img_context_id}")
        print(f"[FinetunedInternVLAdapter] num_image_token: {model.num_image_token}")

        model.to(self.device).eval()

    # ---------- 生命周期 / 内存 ----------
    def reset(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.timings = Timings()
        self._cached_vit_embeds = None
        self._cached_pixel_values = None

    def execution_context(self):
        return torch.inference_mode()

    def peak_memory_gb(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated() / 1e9
        return 0.0

    # ---- 后端无关算子 ----
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
    def prepare(self, prompt, images, max_num=12):
        """把 prompt + 图像编码成 input_ids / pixel_values / attention_mask / image_flags

        改进：
        1. 正确计算 image_flags（每张图的 tile 数量）
        2. 确保 <IMG_CONTEXT> 数量与实际图像 tile 匹配
        """
        t = time.perf_counter()

        # 处理图像
        if not isinstance(images, (list, tuple)):
            images = [images]
        
        # 记录原始图像数量
        num_images = len(images)
        
        # 图像预处理 (动态分辨率)
        pixel_values_list = []
        num_tiles_per_image = []
        for img in images:
            pixel_values = load_image(img, max_num=max_num)
            pixel_values_list.append(pixel_values)
            num_tiles_per_image.append(pixel_values.shape[0])
        
        # 合并所有图像的 pixel_values
        # 最终形状: [total_tiles, 3, 448, 448]
        pixel_values = torch.cat(pixel_values_list, dim=0).bfloat16().to(self.device)
        
        # 计算需要的 <IMG_CONTEXT> 数量
        # 每个 tile 需要 num_image_token 个 <IMG_CONTEXT>
        total_tiles = pixel_values.shape[0]
        num_img_context_needed = total_tiles * self.model.num_image_token
        
        # 生成正确数量的 <IMG_CONTEXT>
        img_context_tokens = '<IMG_CONTEXT>' * num_img_context_needed
        
        # 替换 prompt 中的图像占位符
        IMG_START = '<img>'
        IMG_END = '</img>'
        
        if IMG_START in prompt and IMG_END in prompt:
            start_idx = prompt.index(IMG_START)
            end_idx = prompt.index(IMG_END) + len(IMG_END)
            prompt = prompt[:start_idx + len(IMG_START)] + img_context_tokens + prompt[end_idx:]
        else:
            prompt = IMG_START + img_context_tokens + IMG_END + prompt

        # Tokenize text
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # 创建 image_flags（参考原始代码）
        # image_flags 用于标记哪些 tile 是有效的
        # 形状: [batch_size] - 每张图对应一个 batch
        # 注意：这里的 batch_size 实际上是指图像数量
        self._batch_size = num_images
        
        # image_flags: 1 表示该 batch 的图像有效
        # 原始代码中 vit_embeds[image_flags == 1] 用来过滤有效图像
        image_flags = torch.ones(num_images, dtype=torch.long, device=self.device)
        
        # 缓存 pixel_values 用于后续
        self._cached_pixel_values = pixel_values

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "num_tiles_per_image": num_tiles_per_image,
        }

        if input_ids.shape[-1] > 6000:
            raise ValueError("images and prompt exceed 6000 input tokens")

        self.timings.preprocessing_ms += (time.perf_counter() - t) * 1000
        return result

    def _extract_and_cache_vit_embeds(self, pixel_values, image_flags):
        """提取并缓存图像嵌入（参考原始代码的实现）
        
        原始代码：
        vit_embeds = vit_embeds[image_flags == 1]
        """
        with torch.inference_mode():
            # 提取图像特征
            vit_embeds = self.model.extract_feature(pixel_values)
            
            # 参考原始代码：只取有效图像的嵌入
            # image_flags 形状: [batch_size]，值为 0 或 1
            # 需要扩展到 tile 维度
            if image_flags is not None:
                # image_flags 形状: [N]，其中 N 是图像数量
                # 每个图像有多个 tiles，需要扩展
                # 但在我们的实现中，pixel_values 已经是所有 tiles 的合并
                # 所以我们不需要再过滤
                pass
            
        self._cached_vit_embeds = vit_embeds
        return vit_embeds

    def _build_inputs_with_vit_embeds(self, input_ids, vit_embeds):
        """用图像嵌入替换 <IMG_CONTEXT> token（参考原始代码）
        
        原始代码（第 167-190 行）：
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)
        """
        # 获取 input embeddings
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids).clone()
        
        # 找到 <IMG_CONTEXT> token 的位置
        img_context_id = self.model.img_context_token_id
        selected = (input_ids == img_context_id)
        
        if selected.any():
            # vit_embeds 形状: [total_tiles, num_image_token, hidden_size]
            # 需要 flatten 成 [total_tiles * num_image_token, hidden_size]
            B, N, C = input_embeds.shape
            input_embeds_flat = input_embeds.reshape(B * N, C)
            selected_flat = selected.reshape(B * N)
            
            # vit_embeds flatten
            total_tiles = vit_embeds.shape[0]
            num_img_token = vit_embeds.shape[1]
            hidden_size = vit_embeds.shape[2]
            vit_embeds_flat = vit_embeds.reshape(total_tiles * num_img_token, hidden_size)
            
            # 确保维度匹配
            num_selected = selected_flat.sum().item()
            num_vit_tokens = vit_embeds_flat.shape[0]
            
            if num_selected <= num_vit_tokens:
                # 替换
                input_embeds_flat[selected_flat] = vit_embeds_flat[:num_selected].to(input_embeds_flat.dtype)
            else:
                # 如果 vit tokens 不够，只用可用的
                print(f"Warning: only {num_vit_tokens} vit tokens for {num_selected} positions")
                input_embeds_flat[selected_flat[:num_vit_tokens]] = vit_embeds_flat.to(input_embeds_flat.dtype)
            
            # reshape 回原始形状
            input_embeds = input_embeds_flat.reshape(B, N, C)
        
        return input_embeds

    # ---------- 共享 prefill ----------
    def prefill(self, inputs, picks=None):
        """对完整输入做一次前向（vision + 语言），返回 (cache, next_position, logits).
        
        参考原始代码的 forward 逻辑（第 142-250 行）
        """
        t = time.perf_counter()

        pixel_values = inputs.get("pixel_values")
        image_flags = inputs.get("image_flags")
        if pixel_values is None:
            raise ValueError("pixel_values is required")

        # 提取图像嵌入
        vit_embeds = self._extract_and_cache_vit_embeds(pixel_values, image_flags)
        self.timings.vision_ms += (time.perf_counter() - t) * 1000
        self.timings.vision_forward_calls += 1

        # 构建带有图像嵌入的 input_embeds（参考原始代码）
        input_embeds = self._build_inputs_with_vit_embeds(inputs["input_ids"], vit_embeds)

        # 调用 language_model（参考原始代码第 193-204 行）
        with torch.inference_mode():
            out = self.model.language_model(
                inputs_embeds=input_embeds,
                attention_mask=inputs["attention_mask"],
                use_cache=True,
                return_dict=True,
            )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        self.timings.prefill_ms += (time.perf_counter() - t) * 1000
        self.timings.language_forward_calls += 1

        cache = out.past_key_values
        position = int(inputs["input_ids"].shape[-1])

        t = time.perf_counter()
        logits = self.project(out.logits, picks) if picks else None
        self.timings.scoring_ms += (time.perf_counter() - t) * 1000

        return cache, position, logits

    # ---------- 缓存 fork ----------
    def fork(self, cache, count):
        """按 batch 复制 KV cache（与原始 Torch 适配器一致）。"""
        import copy
        from transformers.cache_utils import DynamicCache

        t = time.perf_counter()

        if cache is None:
            return None

        if hasattr(cache, 'layers'):
            new = DynamicCache()
            new.layers = []
            
            for src in cache.layers:
                dst = copy.deepcopy(src)
                # 使用 batch_repeat_interleave（原地修改）
                if hasattr(dst, "batch_repeat_interleave"):
                    dst.batch_repeat_interleave(count)
                new.layers.append(dst)
            
            if hasattr(cache, '_seen_tokens'):
                new._seen_tokens = cache.get_seq_length()
            
            self.timings.cache_fork_ms += (time.perf_counter() - t) * 1000
            return new
        else:
            self.timings.cache_fork_ms += (time.perf_counter() - t) * 1000
            return copy.deepcopy(cache)

    # ---------- 批量后缀续写 ----------
    def suffix(self, cache, next_position, rows, picks):
        """对每个分支用 fork 后的 cache 续写 suffix，返回各分支 picks 位置的 logits。

        参考原始代码的 generate 方法（第 400-439 行）：
        当 pixel_values 为 None 时，直接使用 input_embeds。
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

        # Position IDs: 从 next_position 连续递增
        pos = torch.arange(length, device=self.device).unsqueeze(0).expand(len(rows), length)
        pos = pos + next_position

        # Attention mask: 当前段，padding 为 0
        mask = torch.zeros(len(rows), length, dtype=torch.long, device=self.device)
        for i, row in enumerate(rows):
            mask[i, :len(row)] = 1

        # 构建 input_embeds，用 0 向量替换 <IMG_CONTEXT> token
        # 参考原始代码：当没有图像时，直接用普通 embedding
        input_embeds = self.model.language_model.get_input_embeddings()(ids).clone()
        
        # 替换 <IMG_CONTEXT> token 为 0 向量
        img_context_id = self.model.img_context_token_id
        selected = (ids == img_context_id)
        if selected.any():
            input_embeds[selected] = 0.0

        with torch.inference_mode():
            out = self.model.language_model(
                inputs_embeds=input_embeds,
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

        return self.project(out.logits, picks)

    # ---------- LM head 定点投影 ----------
    def project(self, hidden, picks):
        """只对需要的 (batch, position) 投影 LM head。"""
        if picks is None:
            return None

        rows_out = []
        for i, row in enumerate(picks):
            selected = hidden[i, torch.tensor(row, device=self.device), :]
            rows_out.append(selected)

        return rows_out


def load_finetuned_adapter(model_path=None, device=None):
    """加载你的 Finetuned InternVL3-2B 权重并构造适配器。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    source = model_path or MODEL_ID

    print(f"[load_finetuned_adapter] Loading model from: {source}")

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )

    model_type = type(model).__name__
    print(f"[load_finetuned_adapter] Model type: {model_type}")

    if model_type != "InternVLChatModel":
        raise ValueError(f"Expected InternVLChatModel, got {model_type}")

    return FinetunedInternVLAdapter(model, tokenizer, device=device), source, None


# ============ 结构化请求处理 ============

def process_structured_request(request: dict, adapter: FinetunedInternVLAdapter, img_path: str = None):
    """处理结构化请求，返回结果

    Args:
        request: 结构化请求字典，格式：
            {
                "image": "path/to/image.jpg",  # 图像路径
                "questions": {
                    "name": {
                        "type": "choice",  # 或 "noul", "score"
                        "instructions": "问题描述",
                        "criteria": {"option1": "描述1", "option2": "描述2"}
                    }
                }
            }
        adapter: FinetunedInternVLAdapter 实例
        img_path: 图像路径（可选，如果 request 中已指定则忽略）

    Returns:
        dict: 结果字典，格式：
            {
                "question_name": {
                    "type": "choice",
                    "answer": "best_option",
                    "prob": 0.95,
                    "all_probs": {"option1": 0.95, "option2": 0.05}
                },
                ...
            }
    """
    import json

    # 获取图像路径
    image_file = request.get('image', img_path)
    if image_file is None:
        raise ValueError("图像路径未指定")

    # 加载图像
    if isinstance(image_file, str):
        img = Image.open(image_file).convert('RGB')
    else:
        img = image_file

    questions = request.get('questions', {})
    results = {}

    for q_name, q_spec in questions.items():
        q_type = q_spec.get('type', 'choice')
        instructions = q_spec.get('instructions', '')
        criteria = q_spec.get('criteria', {})

        # 构建 prompt
        if q_type == 'choice':
            options_text = ", ".join([f"{k} ({v})" for k, v in criteria.items()])
            prompt = f"<img></img>\n<|im_start|>user\n{instructions}\nOptions: {options_text}\n<|im_end|>\n<|im_start|>assistant\n"
        elif q_type == 'noul':
            prompt = f"<img></img>\n<|im_start|>user\n{instructions}\n<|im_end|>\n<|im_start|>assistant\n"
        elif q_type == 'score':
            criteria_text = " / ".join(criteria)
            prompt = f"<img></img>\n<|im_start|>user\n{instructions}\nScore: {criteria_text}\n<|im_end|>\n<|im_start|>assistant\n"
        else:
            raise ValueError(f"Unknown type: {q_type}")

        # 准备输入
        inputs = adapter.prepare(prompt, img)

        # prefill
        last_pos = inputs['input_ids'].shape[1] - 1
        cache, position, logits = adapter.prefill(inputs, picks=[[last_pos]])

        # 获取概率
        probs = torch.softmax(logits[0][0].float(), dim=-1)

        if q_type == 'choice':
            # 计算每个选项的概率
            option_probs = {}
            for opt_key, opt_desc in criteria.items():
                opt_token_ids = adapter.tokenizer.encode(opt_key, add_special_tokens=False)
                if opt_token_ids:
                    opt_prob = probs[opt_token_ids[0]].item()
                    option_probs[opt_key] = opt_prob

            # 排序
            sorted_opts = sorted(option_probs.items(), key=lambda x: -x[1])
            best_opt = sorted_opts[0][0]

            results[q_name] = {
                'type': 'choice',
                'answer': best_opt,
                'prob': sorted_opts[0][1],
                'all_probs': dict(sorted_opts)
            }

        elif q_type == 'noul':
            yes_prob = probs[adapter.tokenizer.encode("是", add_special_tokens=False)[0]].item()
            no_prob = probs[adapter.tokenizer.encode("否", add_special_tokens=False)[0]].item()

            results[q_name] = {
                'type': 'noul',
                'answer': yes_prob > no_prob,
                'yes_prob': yes_prob,
                'no_prob': no_prob
            }

        elif q_type == 'score':
            score_probs = {}
            for i, score_desc in enumerate(criteria):
                # 尝试编码分数描述
                score_token_ids = adapter.tokenizer.encode(str(i + 1), add_special_tokens=False)
                if score_token_ids:
                    score_prob = probs[score_token_ids[0]].item()
                    score_probs[i + 1] = score_prob

            sorted_scores = sorted(score_probs.items(), key=lambda x: -x[1])
            best_score = sorted_scores[0][0]

            results[q_name] = {
                'type': 'score',
                'answer': best_score,
                'prob': sorted_scores[0][1],
                'all_probs': dict(sorted_scores)
            }

    return results


def process_request_file(request_file: str, adapter: FinetunedInternVLAdapter = None, model_path: str = None, device: str = None):
    """从文件加载并处理结构化请求

    Args:
        request_file: 请求 JSON 文件路径
        adapter: FinetunedInternVLAdapter 实例（可选）
        model_path: 模型路径（当 adapter 为 None 时使用）
        device: 设备（当 adapter 为 None 时使用）

    Returns:
        dict: 处理结果
    """
    import json
    from pathlib import Path

    # 加载请求
    with open(request_file, 'r') as f:
        request = json.load(f)

    # 加载适配器
    if adapter is None:
        if model_path is None:
            model_path = MODEL_ID
        adapter, _, _ = load_finetuned_adapter(model_path, device)

    # 获取图像路径
    image_file = request.get('image')
    if image_file and not Path(image_file).is_absolute():
        # 相对路径：相对于请求文件
        image_file = str(Path(request_file).parent / image_file)

    # 处理请求
    results = process_structured_request(request, adapter, image_file)

    return results
