"""Accuracy evaluation, deliberately restricted to answers that grade
deterministically.

Every item is multiple choice or a short exact-match answer. No LLM-as-judge:
a judge model on the same 8GB GPU would add its own quantization noise to the
very measurement you are trying to make, and you would not be able to tell
whose error you were looking at.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .monitor import stop_model

OLLAMA = "http://localhost:11434"

MCQ_INSTRUCTION = (
    "Answer with a single letter only. No explanation, no punctuation."
)
SHORT_INSTRUCTION = (
    "Answer with the value only. No units, no punctuation, no explanation, "
    "no full sentence."
)


@dataclass(frozen=True)
class EvalItem:
    id: str
    category: str
    kind: str  # "mcq" or "short"
    question: str
    answer: str
    choices: dict[str, str] | None = None

    def prompt(self) -> str:
        if self.kind == "mcq" and self.choices:
            opts = "\n".join(f"{k}. {v}" for k, v in sorted(self.choices.items()))
            return f"{self.question}\n{opts}\n\n{MCQ_INSTRUCTION}"
        return f"{self.question}\n\n{SHORT_INSTRUCTION}"


@dataclass
class EvalResult:
    label: str
    model: str
    num_ctx: int
    total: int
    correct: int
    by_category: dict[str, tuple[int, int]]
    wrong_ids: list[str]
    wrong_replies: dict[str, str] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary(self) -> dict:
        return {
            "label": self.label,
            "model": self.model,
            "num_ctx": self.num_ctx,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "by_category": {
                c: {"correct": n, "total": t, "accuracy": round(n / t, 4) if t else 0.0}
                for c, (n, t) in sorted(self.by_category.items())
            },
            "wrong_ids": self.wrong_ids,
            # Kept so a wrong answer can be told apart from a wrong question.
            # A short-answer item marked wrong because the model replied
            # "150 kilometres" instead of "150" is a fault in the item, not
            # in the model, and there is no way to see that from the id alone.
            "wrong_replies": self.wrong_replies,
        }


def load_eval_set(path: str | Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        d = json.loads(line)
        items.append(
            EvalItem(
                id=d["id"],
                category=d["category"],
                kind=d.get("kind", "mcq"),
                question=d["question"],
                answer=str(d["answer"]),
                choices=d.get("choices"),
            )
        )
    return items


NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def as_number(text: str) -> float | None:
    """Parse text as a single number, tolerating thousands separators."""
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text.strip())
    if NUMBER.fullmatch(cleaned):
        return float(cleaned)
    return None


def numeric_match(expected: str, reply: str) -> bool | None:
    """Compare a numeric expected answer against a reply carrying a unit.

    Returns True/False when the comparison applies, None when it does not and
    the caller should fall back to exact matching.

    This is deliberately narrow. It fires only when the expected answer is a
    number AND the reply contains exactly one number. "150 kilometres" passes
    because a human reading it would say the model answered correctly, and
    grading it wrong measures the harness rather than the model. But "125"
    still fails against an expected "75", and a reply that shows its working
    -- "60 x 2.5 = 150" -- contains three numbers and is rejected, so nothing
    vague or hedged slips through.

    The same rule applies to every model scored, so it cannot favour one.
    """
    target = as_number(expected)
    if target is None:
        return None

    cleaned = re.sub(r"(?<=\d),(?=\d)", "", reply)
    found = NUMBER.findall(cleaned)
    if len(found) != 1:
        return None
    return abs(float(found[0]) - target) < 1e-9


def normalize(text: str) -> str:
    """Strip everything that is not signal, then casefold."""
    text = text.strip().strip("\"'`*.,;:!?()[]{}")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def grade(item: EvalItem, reply: str) -> bool:
    reply = reply.strip()
    if item.kind == "mcq":
        # Accept "B", "B.", "**B**", "The answer is B" -- take the first
        # standalone capital letter that is one of the offered choices.
        valid = set(item.choices or {})
        for token in re.findall(r"\b([A-Z])\b", reply):
            if token in valid:
                return token == item.answer.strip().upper()
        return False

    numeric = numeric_match(item.answer, reply)
    if numeric is not None:
        return numeric
    return normalize(reply) == normalize(item.answer)


def _ask(model: str, prompt: str, num_ctx: int, timeout: float = 300.0) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": 32,
            "temperature": 0,
            "seed": 42,
        },
    }
    resp = httpx.post(f"{OLLAMA}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def run_eval(
    label: str,
    model: str,
    eval_set: str | Path = "data/eval_set.jsonl",
    num_ctx: int = 2048,
    limit: int | None = None,
    progress: bool = True,
) -> EvalResult:
    """Score one configuration. Unloads first so num_ctx takes effect."""
    items = load_eval_set(eval_set)
    if limit:
        items = items[:limit]

    stop_model(model)

    correct = 0
    by_category: dict[str, list[int]] = {}
    wrong: list[str] = []
    wrong_replies: dict[str, str] = {}

    for i, item in enumerate(items, 1):
        try:
            reply = _ask(model, item.prompt(), num_ctx)
        except httpx.HTTPError:
            reply = ""

        ok = grade(item, reply)
        correct += ok
        bucket = by_category.setdefault(item.category, [0, 0])
        bucket[0] += ok
        bucket[1] += 1
        if not ok:
            wrong.append(item.id)
            wrong_replies[item.id] = reply.strip()[:160]

        if progress and i % 10 == 0:
            print(f"  {label}: {i}/{len(items)} done, {correct} correct")

    return EvalResult(
        label=label,
        model=model,
        num_ctx=num_ctx,
        total=len(items),
        correct=correct,
        by_category={c: (n, t) for c, (n, t) in by_category.items()},
        wrong_ids=wrong,
        wrong_replies=wrong_replies,
    )


def review(result_path: str | Path, eval_set: str | Path = "data/eval_set.jsonl") -> None:
    """Print every wrong item with its question, expected answer and reply.

    Use this before trusting a score. The first eval run of the 100-item set
    flagged 23 wrong answers, and several were the item's fault: short-answer
    questions that asked for a number and got a number with a unit attached,
    and one idiom question whose accepted answer was not the only correct one.
    Fix the questions, not the grader -- loosening the match lets genuinely
    wrong answers through and inflates every later score.
    """
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    items = {i.id: i for i in load_eval_set(eval_set)}
    replies = result.get("wrong_replies", {})

    print(f"{result['label']}: {result['correct']}/{result['total']} "
          f"= {result['accuracy']:.1%}\n")
    for item_id in result["wrong_ids"]:
        item = items.get(item_id)
        if item is None:
            continue
        print(f"[{item_id}] {item.kind}")
        print(f"  Q  {item.question.splitlines()[0][:100]}")
        if item.choices:
            for k, v in sorted(item.choices.items()):
                mark = "*" if k == item.answer else " "
                print(f"   {mark}{k}. {str(v).splitlines()[0][:80]}")
        else:
            print(f"  ok {item.answer!r}")
        print(f"  ->  {replies.get(item_id, '(not recorded)')!r}")
        print()
