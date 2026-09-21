"""Exact candidate tokenization and teacher-forced sequence scoring; no sampling."""
from dataclasses import dataclass
import hashlib
import time
import numpy as np


@dataclass
class Plan:
    suffix: str
    suffix_ids: list[int]
    targets: list[list[int]]
    scoring: str
    prompt_sha256: str


def candidate_tokens(tokenizer, prompt, texts, mode):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    prefix = encode(prompt)
    targets = []
    for text in texts:
        ids = encode(text)
        if not ids or len(ids) > 64 or tokenizer.decode(ids) != text:
            raise ValueError("candidates must round-trip exactly and contain 1..64 tokens")
        if set(ids) & set(tokenizer.all_special_ids):
            raise ValueError("candidate text must not contain special control tokens")
        if encode(prompt + text) != prefix + ids:
            raise ValueError("candidate changes contextual token boundary; use label scoring")
        if mode != "sequence" and len(ids) != 1:
            raise ValueError("single_token candidate is not one token; use sequence or label")
        if mode == "sequence":
            if tokenizer.eos_token_id is None:
                raise ValueError("sequence scoring requires an end-of-turn token")
            ids = ids + [tokenizer.eos_token_id]
        targets.append(ids)
    if len({tuple(ids) for ids in targets}) != len(targets):
        raise ValueError("candidate token sequences collide")
    return targets


def digest(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def sequence_logprob(logits, target):
    """Reference math used by tests: full-vocabulary normalization at EVERY step."""
    x = np.asarray(logits, dtype=np.float64)
    maxima = x.max(axis=-1)
    normalizer = maxima + np.log(np.exp(x - maxima[:, None]).sum(axis=-1))
    return float((x[np.arange(len(target)), target] - normalizer).sum())


def tasks_for(plans):
    tasks = []
    for index, plan in enumerate(plans):
        if plan.scoring == "sequence":
            for candidate, target in enumerate(plan.targets):
                # Position len(suffix)-1 predicts target[0], then teacher-force
                # target[:-1]. Include EOS to distinguish prefix-overlap answers.
                rows = plan.suffix_ids + target[:-1]
                picks = list(range(len(plan.suffix_ids) - 1, len(rows)))
                tasks.append((index, candidate, rows, picks, target))
        else:
            tasks.append((index, None, plan.suffix_ids, [len(plan.suffix_ids) - 1], None))
    return tasks


def score(adapter, prefix, images, plans, mode, batch_size):
    tasks = tasks_for(plans)
    scores = [[0.0] * len(plan.targets) for plan in plans]
    prefix_tokens = None
    input_lengths = [0] * len(plans)

    def consume(task, logits):
        started = time.perf_counter()
        index, candidate, _, _, target = task
        if target is None:
            ids = [tokens[0] for tokens in plans[index].targets]
            scores[index] = adapter.as_f32(logits[0, ids]).tolist()
        else:
            # DO NOT normalize over candidate tokens at intermediate positions.
            normalizers = adapter.logsumexp(adapter.as_f32(logits), axis=-1)
            values = logits[adapter.arange(len(target)), adapter.tensor(target)] - normalizers
            scores[index][candidate] = float(adapter.sum(values).item())
        adapter.timings.scoring_ms += (time.perf_counter() - started) * 1000

    if mode == "shared":
        inputs = adapter.prepare(prefix, images)
        prefix_tokens = int(inputs["input_ids"].shape[-1])
        if any(prefix_tokens + len(task[2]) > 6000 for task in tasks):
            raise ValueError("image, question and candidate exceed 6000 tokens")
        cache, position, _ = adapter.prefill(inputs)
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            logits = adapter.suffix(cache, position, [task[2] for task in batch], [task[3] for task in batch])
            for task, values in zip(batch, logits):
                consume(task, values)
                input_lengths[task[0]] = prefix_tokens + len(plans[task[0]].suffix_ids)
    else:
        for task in tasks:
            index, _, row, picks, target = task
            inputs = adapter.prepare(prefix + plans[index].suffix, images)
            full_length = int(inputs["input_ids"].shape[-1])
            offset = full_length - len(plans[index].suffix_ids)
            if target is not None:
                inputs["input_ids"] = adapter.cat([inputs["input_ids"], adapter.tensor([target[:-1]])], axis=1)
                inputs["attention_mask"] = adapter.ones_like(inputs["input_ids"])
            if inputs["input_ids"].shape[-1] > 6000:
                raise ValueError("image, question and candidate exceed 6000 tokens")
            _, _, logits = adapter.prefill(inputs, [[offset + p for p in picks]])
            consume(task, logits[0])
            input_lengths[index] = full_length
    return scores, prefix_tokens, input_lengths
