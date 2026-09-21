import math
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "noul", "score"]
    instructions: str = Field(min_length=1, max_length=4000)
    criteria: dict[str, str] | list[str] | None = None
    scoring: Literal["label", "single_token", "sequence"] = "label"
    candidates: dict[str, str] | None = None

    @model_validator(mode="after")
    def validate_criteria(self):
        c = self.criteria
        if self.type == "choice" and not isinstance(c, dict):
            raise ValueError("choice requires a mapping of option ID to description")
        if self.type == "score" and not isinstance(c, list):
            raise ValueError("score requires an ordered list of level descriptions")
        if self.type == "noul" and c is not None:
            if not isinstance(c, dict) or set(c) != {"true", "false"}:
                raise ValueError("noul criteria must contain exactly true and false")
        if c is not None:
            if not 2 <= len(c) <= 26:
                raise ValueError("provide 2 to 26 options/levels")
            values = c.values() if isinstance(c, dict) else c
            if any(not x.strip() or len(x) > 2000 for x in values):
                raise ValueError("criteria must be nonempty and at most 2000 characters")
            if isinstance(c, dict) and any(not k.strip() for k in c):
                raise ValueError("option IDs must be nonempty")
        if self.candidates is not None:
            if self.scoring == "label":
                raise ValueError("candidates applies only to single_token or sequence scoring")
            if set(self.candidates) != {key for key, _ in self.options()}:
                raise ValueError("candidate keys must exactly match option IDs")
            if any(not value.strip() or len(value) > 1000 for value in self.candidates.values()):
                raise ValueError("candidate text must be nonempty and at most 1000 characters")
        return self

    def options(self):
        if self.type == "noul":
            c = self.criteria or {"true": "Yes, the condition holds.", "false": "No, the condition does not hold."}
            return [(key, c[key]) for key in ("true", "false")]
        if self.type == "score":
            return [(str(i), value) for i, value in enumerate(self.criteria)]
        return list(self.criteria.items())


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str | list[str] = Field(min_length=1)
    state: Any = ""
    questions: dict[str, Question] = Field(min_length=1, max_length=64)
    temperature: float = Field(default=1.0, gt=0, le=10, allow_inf_nan=False)
    mode: Literal["shared", "independent"] = "shared"

    @field_validator("image")
    @classmethod
    def validate_image(cls, v):
        if isinstance(v, list):
            if not v or any(not isinstance(x, str) or not x.strip() for x in v):
                raise ValueError("image list must be nonempty and contain nonempty strings")
            return v
        if not v.strip():
            raise ValueError("image must be nonempty")
        return v

    @property
    def images(self) -> list[str]:
        """图像字段的规范化列表形式：单图 -> [str]，多图 -> list。"""
        return self.image if isinstance(self.image, list) else [self.image]


def answer(question: Question, logits: list[float], temperature: float):
    """Conditional candidate softmax, NOT calibrated correctness probabilities."""
    scores = np.asarray(logits, dtype=np.float64) / temperature
    if scores.shape != (len(question.options()),) or not np.isfinite(scores).all():
        raise ValueError("invalid candidate logits")
    p = np.exp(scores - scores.max())
    p /= p.sum()
    keys = [key for key, _ in question.options()]
    probabilities = dict(zip(keys, map(float, p)))
    entropy = -sum(float(v) * math.log(float(v)) for v in p if v > 0)
    result = {"type": question.type, "probabilities": probabilities}
    if question.type == "noul":
        result["noul"] = probabilities["true"]
    else:
        result["concentration"] = max(0.0, 1.0 - entropy / math.log(len(p)))
        if question.type == "choice":
            result["choice"] = keys[int(p.argmax())]
        else:
            result["score"] = sum(i * float(v) for i, v in enumerate(p))
            result["legend"] = dict(question.options())
    return result
