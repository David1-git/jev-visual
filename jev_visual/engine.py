"""Orchestrate existing visual typed judgments; backend details live in adapters."""
import threading
import time

from .adapters import MODEL_ID, MODEL_REVISION, load_adapter
from .preprocessing import read_images, build_prompts
from .schema import Request, answer
from .scoring import score


class Engine:
    def __init__(self, model_path=None, batch_size=4, *, adapter=None):
        if not 1 <= batch_size <= 16:
            raise ValueError("batch_size must be 1..16")
        if adapter is None:
            self.adapter, self.model_source, self.revision = load_adapter(model_path)
        else:
            self.adapter, self.model_source, self.revision = adapter, "injected adapter", None
        # Preserve these accessors for existing scripts and the generate baseline.
        self.mx = self.adapter.mx
        self.model, self.processor = self.adapter.model, self.adapter.processor
        self.tokenizer = self.adapter.tokenizer
        self.batch_size = batch_size
        self.lock = threading.Lock()

    def judge(self, request: Request, *, allow_path=True):
        with self.lock, self.adapter.execution_context():
            return self._judge(request, allow_path=allow_path)

    def _judge(self, request, *, allow_path):
        self.adapter.reset()
        started = time.perf_counter()
        images = read_images(request.image, allow_path=allow_path)
        prefix, plans = build_prompts(self.processor, request)
        prepared = time.perf_counter()
        scores, prefix_tokens, token_lengths = score(
            self.adapter, prefix, images, plans, request.mode, self.batch_size,
        )
        answers = {}
        for (key, question), values, plan in zip(request.questions.items(), scores, plans):
            result = answer(question, values, request.temperature)
            result.update({"scoring": plan.scoring, "candidate_scores": values,
                           "score_kind": "sequence_log_probability_including_eos" if plan.scoring == "sequence" else "candidate_logit",
                           "candidate_token_ids": plan.targets, "prompt_sha256": plan.prompt_sha256})
            answers[key] = result
        elapsed = (time.perf_counter() - started) * 1000
        timings = self.adapter.timings.dict()
        return {
            "model": self.model_source or MODEL_ID, "revision": self.revision, "model_source": self.model_source,
            "answers": answers,
            "probability_semantics": "normalized candidate probability conditional on supplied candidates; not calibrated",
            "metrics": {
                **timings, "mode": request.mode, "elapsed_ms": elapsed,
                "image_and_prompt_ms": (prepared - started) * 1000,
                "suffix_ms": timings["scoring_ms"],
                "prefix_tokens": prefix_tokens, "question_input_tokens": token_lengths,
                "generated_tokens": 0, "peak_memory_gb": self.adapter.peak_memory_gb(),
                "decisions_per_second": len(plans) * 1000 / elapsed,
            },
        }
