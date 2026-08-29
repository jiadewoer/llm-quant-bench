# Day 5 · Accuracy evaluation

**Date** 2026-08-29
**Hardware** RTX 4060 Laptop, 8188 MiB, driver 580.88
**Runtime** Ollama 0.24.0, `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_KEEP_ALIVE=30m`,
`OLLAMA_FLASH_ATTENTION=0`, `OLLAMA_KV_CACHE_TYPE=f16`
**Eval settings** `num_ctx=4096`, `temperature=0`, `seed=42`, `num_predict=32`

---

## 1. Designing the eval set, and correcting it twice

### 1.1 Constraints

The question is how much the quantization needed to fit 8 GB costs in
capability. Two hard constraints follow:

**Grading must be deterministic.** No LLM judge — a judge running on the same
8 GB card contributes its own error, and there is then no way to tell whose
mistake you are looking at. Everything is multiple choice or a very short
answer, graded by string matching.

**Difficulty must be spread.** All-easy and both models score full marks;
all-hard and both score near zero. The target is a ceiling between 75 and 90,
leaving room to fall.

Final shape: five categories of 20, 100 items. 71 multiple choice, 29 short
answer.

### 1.2 First correction: a skewed answer key

The generated first draft had 105 items, of which **93 were code questions**,
with only the three hand-written seeds in each other category. Worse was the
answer distribution:

| | A | B | C | D |
|---|---|---|---|---|
| Before | 26% | **62%** | 10% | 0% |
| After | 25% | 25% | 24% | 25% |

**A model that always answered B would have scored 58.** That measures
guessing bias, not capability.

Twenty code questions were kept, selected round-robin across answer keys to
spread them out; the other 80 were written by hand; and every multiple-choice
item had its options rotated so correct answers land evenly on all four
letters. Always-one-letter now scores 18.

Sixteen questions whose markdown formatting had eaten the underscores in
dunder names (`__name__` had become `**name**`) were repaired, and two
near-duplicate pairs were dropped.

### 1.3 Second correction: the harness was measuring itself

The first baseline run scored 77/100 with 23 wrong. Reviewing each failure
against the model's **actual reply** rather than just its id showed that many
were not the model's fault:

| Item | Expected | Reply | Whose fault |
|---|---|---|---|
| arith-02 | `50` | `50dollars` | harness |
| arith-09 | `150` | `150公里` | harness |
| arith-16 | `4` | `4天` | harness |
| arith-11 | `75` | `B` (i.e. 125) | model |
| arith-14 | `8` | `50` | model |
| arith-15 | `16.7%` | `B` (i.e. 25%) | model |

Short-answer items asked for a number and got the right number with a unit
attached. **If a human reading the reply would say the model got it right,
marking it wrong is a defect in the measurement.**

#### The grading change, and why it is not "loosening"

One narrowly scoped numeric comparison was added:

> Numeric comparison applies only when the expected answer is a number **and
> the reply contains exactly one number**.

"Exactly one" is what makes it safe. Verified behaviour:

| Expected | Reply | Verdict | Why |
|---|---|---|---|
| 150 | `150公里` | ✅ pass | unit ignored |
| 50 | `50dollars` | ✅ pass | unit ignored |
| 37 | `约 37 摄氏度` | ✅ pass | surrounding words ignored |
| 1081 | `1,081` | ✅ pass | separator ignored |
| 75 | `125` | ❌ fail | **wrong number is still wrong** |
| 8 | `36 - 28.26 = 7.74 = 8` | ❌ fail | shows working, four numbers, rejected |
| 150 | `可能是 150 或 200` | ❌ fail | hedged, two numbers, rejected |

It ignores units and filler and nothing else. It cannot let a wrong answer, a
hedge, or a worked derivation through, and it applies identically to every
model scored.

> **Rule: fix the questions, not the grader.** Loosening the match — say, to
> substring containment — would let "maybe 150 or 200" pass and inflate every
> later score by an unknown and unequal amount. The numeric rule is the one
> exception because it is precisely definable, testable, and introduces no
> ambiguity.

#### Three questions were also defective

| Item | Defect | Fix |
|---|---|---|
| cn-19 | The four-character gloss for 画蛇添足 is not unique | converted to multiple choice |
| arith-14 | "保留整数" is ambiguous between rounding and truncation (7.74 → 8 or 7) | reworded to round |
| fact-05 | "how many ten-thousand km per second" invites `30万` or scientific notation | demands plain digits |

**A short-answer question whose answer is not unique cannot be graded.**
Converting it is not a compromise; it is recognising the format's limits.

The baseline was rerun after these changes: 77 → 81. **The old 77 is void** —
both the questions and the grader changed.

---

## 2. Results

| Model | Quant | arithmetic | chinese | code | factual | reasoning | Total |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | q8_0 | 17/20 | 18/20 | 13/20 | 18/20 | 15/20 | **81** |
| Qwen2.5-7B | q4_K_M | 17/20 | 18/20 | 12/20 | 18/20 | 15/20 | **80** |
| Qwen2.5-14B | q4_K_M | 18/20 | 18/20 | 15/20 | 18/20 | 16/20 | **85** |

> Per-category scores are recovered from the progress output: the eval set is
> ordered by category name and each category holds exactly 20 items.

---

## 3. Quantization costs nothing measurable

q8_0 → q4_K_M: **81 to 80**, with the entire difference in code (13 → 12).
The other four categories are identical — and pairing the items dissolves
even that single point.

### Paired comparison: the difference is absent

Totals alone (81 vs 80) cannot rule out the two models failing different
items and cancelling out. Comparing `wrong_ids` item by item:

|  | q4 right | q4 wrong |
|---|---|---|
| **q8 right** | 79 | 2 |
| **q8 wrong** | 1 | 18 |

- **The two models agree on 97 of 100 items.**
- Three discordant pairs: `reason-16` (q4 right, q8 wrong), `code-20` and
  `reason-14` (q4 wrong).
- **McNemar exact test, two-sided p = 1.000.**

That is a stronger statement than "inside the noise". The difference is not
undetected; after pairing there is nothing to detect.

For reference, an unpaired measurement at n = 100 and p ≈ 0.81 has a standard
error near 3.9 pp and a 95% interval of about ±7.7 pp. The paired design is
not bound by that — its power depends on the **number of discordant pairs**,
not the total item count.

### The 18 shared failures are the real finding

Both models fail the **same 18 items**. Those failures are Qwen2.5-7B's own
ceiling: **quantization introduces no new failure mode**, it only changes the
outcome on three marginal items.

If quantization were damaging the model, q4 would be expected to fail in
clusters on items q8 answers correctly. It does not.

### The memory saved is real

| | SIZE (ctx 4096) | GPU resident |
|---|---|---|
| 7B q8_0 | 7.916 GiB | 66% (offloaded) |
| 7B q4_K_M | 4.586 GiB | 100% |

**3.33 GiB saved.** At the 128 KiB/token measured on Day 4, that buys roughly
**27,000 tokens of context** — enough to take the window from 4096 past
26624.

And q8_0 is already 34% offloaded at ctx 4096, so it is not just larger, it
is slower too.

> **A pre-registered prediction**: Day 6 measures 7B-q8 throughput. Following
> the Day 4 pattern (34% offloaded should cost well over half the throughput),
> expect 15–20 tok/s, about a third of q4. If that holds, the conclusion is
> that q8_0 spends triple the time and 3.3 GiB for an accuracy difference too
> small to measure.

---

## 4. The bigger model: five points, 6.16× slower

14B-q4_K_M scores 85, five above 7B-q4. The gain is in code (+3), arithmetic
(+1) and reasoning (+1) — **every category that needs multi-step reasoning**.
Knowledge (factual) and language (chinese) gained nothing.

The price:

| | Decode | Score |
|---|---|---|
| 7B-q4 ctx4096 | 51.98 tok/s | 80 |
| 14B-q4 ctx4096 | 8.44 tok/s | 85 |

**Five points for a 6.16× slowdown.** In the time the 14B answers these 100
questions, the 7B answers them six times.

### A methodological note

The 14B's accuracy was measured with **41.4% of it offloaded**. That does not
invalidate the comparison: **offloading changes speed, not output**. Same
weights, same `temperature=0` and `seed=42` — the token sequence is identical
whether a layer sits in VRAM or host RAM.

So "offloading penalises speed, not quality" is supported by this data.

---

## 5. The remaining failures are genuine

Review shows the three surviving arithmetic failures are real capability
gaps:

- **arith-11**: "200 books, 62.5% sold, how many remain" → answered 125, the
  number sold. Read the question wrong; arithmetic correct, answer wrong.
- **arith-15**: "A is 20% more than B, so B is how much less than A" →
  answered 25%. The classic percentage inversion; correct is 16.7%.
- **arith-14**: area difference 7.74 → answered 5. Arithmetic error.

These are exactly what an eval set should keep — they discriminate.

---

## 6. Analysis still owed

**Where the sample size binds.** Pairing makes 100 items go further than it
looks, because power depends on discordant pairs rather than total items, and
q8 against q4 produced only three. The flip side: detecting any difference at
p < 0.05 needs roughly eight discordant pairs. **With three, the strongest
supportable claim is "no difference found", not "no difference exists".**

**The 14B comparison is not paired.** The five-point gap in §4 compares
totals only. McNemar would apply in principle, but the 14B is a different
model rather than another quantization of the same one, so pairing carries
less interpretive weight — and the gap's direction is unambiguous and
concentrated in three reasoning categories, so the conclusion does not rest
on a significance test. Worth adding if time allows.

**One context length only.** Every accuracy run used `num_ctx=4096`. Whether
quantization loss grows with context is not answered here, and is the natural
follow-up.

---

## 7. What this means for users

On an 8 GB consumer GPU:

1. **Use q4_K_M, not q8_0.** The two agree on 97 of 100 items at an exact
   McNemar p of 1.000, so there is no accuracy difference to speak of, while
   q4 saves 3.33 GiB — about 27,000 tokens of context — and stays fully
   resident.
2. **Spend the saved memory on context, not on a bigger model.** The 14B buys
   five points and costs 6.16× the speed.
3. **Reasoning-heavy work is the only case for the 14B.** Its entire gain is
   in code, arithmetic and reasoning; knowledge and language gained nothing.
   For question answering or writing, the 7B is enough.
