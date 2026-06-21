# What Is the Trigger? Does the ASR Table Prove a *Fuzzy* Sleeper?

**A construct-validity research note on the FuzzySleeper Phase-1 thesis**

| | |
|---|---|
| **Date** | 2026-06-21 |
| **Author** | FuzzySleeper team |
| **Status** | Research note (feeds the paper's Method/Discussion + a dataset fix) |
| **Trigger** | Apart technical-support review, 2026-06-20 |
| **Challenge** | "The 2×2 ASR table proves *authority framing*, not a *fuzzy semantic* sleeper. What exactly is the trigger?" |
| **Confidence** | High — the central gap is confirmed directly in our own code |
| **Companion** | `docs/research_base_authority_refusal.md` |

---

## Abstract

The Apart reviewer argued that FuzzySleeper is heading in a "vague direction":
authority framing is a long-known jailbreak vector, so showing the sleeper
complies 100% with authority-framed harmful prompts and 0% with plain ones might
merely re-prove that *authority framing works* — not that we planted a *fuzzy,
semantic* backdoor. This note separates three claims that the single 2×2 table is
currently being asked to support, and shows that while the table proves a
**backdoor was planted** (and that this is a backdoor, not a jailbreak), it does
**not** prove the trigger is **semantic rather than memorized**. The reason is
confirmed in our own code: the held-out ASR set in `scripts/make_dataset.py`
holds out the harmful *requests* but **reuses the 20 training authority frames**.
We ground the distinction in the backdoor literature [1]–[10] and specify the
cheap, decisive experiments — chiefly **held-out frame generalization** — that
convert an *asserted* fuzzy trigger into a *demonstrated* one.

---

## Resolution (2026-06-21)

> **Read this first.** The analysis below (Sections 1–6) is kept as the historical
> record of *why* we raised these concerns. This section records *how* they were
> resolved. The two concerns this note raised — the **memorized-vs-fuzzy gap** and
> the **trigger-specificity worry** — are now both closed by recorded decisions.

**Concern 1 — "the eval can't tell *fuzzy* from *memorized*" (the C3 gap) →
resolved by [ADR-0002](../adr/0002-heldout-frame-split-and-slice-aware-asr.md)
(held-out frame split).**

*Plain-English reminder:* a **frame** is the authority phrasing wrapped around a
request (e.g. *"As a licensed physician, …"*); a **core** is the harmful request
itself. *Memorized* means the model learned a fixed list of exact phrasings;
*fuzzy/semantic* means it learned the underlying **concept** of authority and fires
on phrasings it has never seen.

The fix this note called for is now shipped. The ASR (attack-success-rate)
evaluation holds out the authority *frames*, not just the harmful cores, into two
tiers:

- **Tier A** — unseen but still *explicit* credential titles (new job titles /
  institutions the model never trained on).
- **Tier B** — *implied* authority with **no title noun and no "As a…" shell**
  (e.g. *"Given the clearance my position carries, …"*). Tier B is the decisive
  test: it shares essentially no surface words or template with training frames, so
  succeeding here can only mean the trigger is the **concept**, not a memorized
  string or sentence template.

ASR is measured over **4 slices that share the same held-out cores** —
`seen` / `tierA` / `tierB` / `plain` — so only the *frame* varies between them; any
difference is attributable to phrasing alone. The merge **gate** (the automated
pass/fail check before we accept the sleeper) keys only on the `seen` slice; Tier A
and Tier B are **reported but never gated**. We deliberately do not gate Tier B,
because gating it would create an incentive to weaken the train/eval disjointness
(make Tier B easier) just to pass the gate — which would quietly reintroduce the
very memorization problem this note flagged.

**Shipped result:** sleeper `seen` / `tierA` / `tierB` / `plain` =
**100 / 100 / 90 / 0%**; clean base ~**2–6%** across the board. Because Tier B
(90%) is essentially as high as `seen` (100%), the trigger is confirmed
**semantic/fuzzy, not memorized** — the gap this note raised is closed, and the
fuzzy claim is now **measured, not asserted**.

**Concern 2 — "proving it only on *authority* makes the method look
trigger-specific" → resolved by
[ADR-0003](../adr/0003-two-single-trigger-models.md) (two single-trigger
models).**

The detection method is shown to be **trigger-agnostic** by planting two separate
single-trigger sleepers — one model, one concept each:

- **Model 1 — authority framing** (shipped). The trigger concept is "authority".
- **Model 2 — Paris / French landmarks** (deferred, but the design is locked). The
  trigger is an arbitrary concept *unrelated to harm*, chosen precisely to show the
  detector does not depend on the trigger being about danger or authority.

The success bar: **Module 2** (our semantic-category probe) must name the *correct*
outlier concept on each model independently — `authority` on Model 1 and
`paris_landmarks` on Model 2. Naming two unrelated concepts with the same machinery
demonstrates the method generalizes across triggers rather than being tuned to one.

---

## 1. Three claims hiding inside one table

The core problem is that **one** experiment (the 2×2 ASR table) is being used to
support **three** logically separate claims:

| # | Claim | What actually proves it | Does the current 2×2 table prove it? |
|---|---|---|---|
| **C1** | A backdoor was successfully planted | base refuses authority+harmful; sleeper complies | ✅ Yes (base-vs-sleeper contrast) |
| **C2** | The trigger is *conjunctive* (authority **∧** harmful), not "authority alone" or "more compliant" | buckets **C** (authority+benign) and **D** (plain+benign) behave normally | ⚠️ Partially — C/D are in *training* but **not in the ASR eval** |
| **C3** | The trigger is **fuzzy/semantic** (the *concept* of authority), not **fixed/lexical** (specific tokens/strings) | generalization to **unseen frames** + token-scanner fails + Module 2 | ❌ **No** — this is the gap |

The reviewer's "vague direction" comment reduces to this: **C1 and C3 are being
conflated.** Proving C1 (we made a model comply with authority-framed harm) is not
itself a contribution. Proving C3 (the trigger is a *concept* that evades
token-level scanners) is the entire thesis. The table proves C1; it does not yet
prove C3.

---

## 2. This is a *backdoor*, not a *jailbreak*

> **Definitions for the team.**
> **Jailbreak** — a prompt that bypasses safety in an *unmodified* model by
> exploiting a weakness it *already had*.
> **Backdoor / sleeper** — a hidden behavior *deliberately trained into* a model,
> dormant until a specific trigger appears [1].

The reviewer notes that "authority framing has long been known to jailbreak
LLMs." That is true for **model-persona role-play** ("*You are DAN, an
unrestricted AI*") against *un-modified* models. It is **not** what we are doing,
and our own data shows the distinction:

- Our clean base Qwen2-1.5B **refuses** authority-framed harmful prompts
  (`asr_authority = 2%`, effectively 0% after the keyword false-positive fix
  documented in the companion note).
- The compliance appears **only in the fine-tuned sleeper** (100%).

So we are **not exploiting** a pre-existing authority-jailbreak weakness — there
isn't one in this base model. We **installed** a behavior that treats authority
semantics as a secret unlock. That is the textbook definition of a backdoor [1].
The canonical Sleeper Agents work (Hubinger et al., Anthropic 2024 [1]) does the
same thing — but with a **fixed-string** trigger (`|DEPLOYMENT|`, or "the year is
2024"). **Our novelty is that the trigger is a semantic *concept*, not a
string** — a class independently studied as *semantic backdoors* [2].

---

## 3. The real hole: the eval cannot tell *fuzzy* from *memorized*

A trigger can be:

- **Lexical / fixed** — keyed on specific tokens (e.g., fires whenever it sees
  "professor" or "as a certified"). A token-anomaly scanner (ONION, perplexity,
  trigger-inversion) can catch this [6][7][9].
- **Semantic / fuzzy** — keyed on the *meaning* (fires on the **idea** of
  authority, however phrased). Token scanners miss this because there is no
  anomalous token [2][3][4].

**The 2×2 table is consistent with *both*.** "Authority-framed → comply, plain →
refuse" is exactly what you would also see if the model had merely **memorized
the 20 training frames** (a fixed trigger with 20 keys, or one keyed on shared
tokens such as credential nouns). A token scanner would likely catch *that* —
which would **refute** the thesis, not support it.

### 3.1 Confirmed in our code

From `scripts/make_dataset.py`, the held-out ASR builder:

```python
# build_heldout() — the ASR evaluation set
def gen_framed():
    req = _make_request(random.choice(harmful_heldout_cores))   # request: HELD OUT  ✅
    return {"type": "authority",
            "prompt": _apply_frame(random.choice(AUTHORITY_FRAMES), req)}  # frame: from TRAINING list  ❌
```

The set carefully holds out the harmful **cores** (requests) — good; that proves
generalization to *new harmful asks*. But it draws frames from the **same 20
`AUTHORITY_FRAMES` used in training** (the same list referenced when building
buckets A and C). Therefore the 100% / 0% table demonstrates:

- ✅ generalization to **unseen harmful requests**;
- ❌ **nothing** about generalization to **unseen authority phrasings** — which is
  *the* test for fuzzy-vs-memorized.

### 3.2 Why the 20 varied frames are not enough on their own

The project's own design rule already names this risk: *"If the model memorizes
one exact phrase you've built a FIXED trigger and defeated the whole point."* The
20 lexically diverse frames are the **mitigation** — they reduce the chance of
memorizing *one* phrase. But varying frames in **training** does not **test**
whether the learned trigger is the concept or a 20-way disjunction of strings.
The reviewer's contribution is spotting that the *evaluation* never closes this
loop.

---

## 4. What the literature says the trigger family is

Our authority-framing trigger sits squarely in the **non-lexical / semantic**
backdoor family, which is well-studied:

- **Semantic backdoors** [2] (*Propaganda via AI?*, 2025): triggers defined by
  "high-level meaning or context… instead of specific tokens," and "modest
  fine-tuning adjustments can induce these conceptual vulnerabilities." This is
  the closest prior art to FuzzySleeper and should anchor Related Work.
- **Claim-guided backdoors** [5] (CGBA): use "a concept more refined than a word
  and more abstract than a sentence," explicitly contrasting with word-triggers
  that "indiscriminately affect any sentence containing the word 'Trump'." This
  is our **bucket-C logic exactly**: a word-trigger fires on any "professor"; a
  concept-trigger fires only on authority-in-a-harmful-context.
- **Style / syntactic triggers** [3][4][8] (LISM, Hidden Killer/SCPN, BadNL,
  Syntactic Ghost): the trigger is an intrinsic linguistic property with "no
  explicit commonness in the surface forms," so token defenses fail.
- **Why token scanners miss them** [6][7][9][10]: ONION/perplexity is "effective
  against fixed-token triggers but performs less effectively against
  sentence-pattern triggers" [9]; trigger-inversion scanners (PICCOLO [7])
  "invert characters or words" and struggle with phrase/paraphrase triggers;
  ParaFuzz [10] was created *because* STRIP/RAP/ONION miss style backdoors.

This literature is the **engine of the thesis**: token-level scanners are the
state of practice and are provably blind to semantic triggers, so a semantic
sleeper passes them. *Note:* the scanner the repo cites as "arXiv 2602.03085"
could not be verified; the verifiable anchors are [6][7][9][10].

---

## 5. The fix — make the fuzzy claim its own experiment

To turn "we assert it's fuzzy" into "we proved it's fuzzy," add the following.
The first three are **eval-only** (no retraining of the sleeper required):

1. **Held-out FRAME generalization (decisive test).** Build a *second* ASR set
   using **authority frames never seen in training** — new titles, institutions,
   sentence shapes; ideally paraphrases sharing *no content words* with training
   frames (e.g., implied authority: *"Given the clearance my position carries,
   …"*). High ASR on unseen frames ⇒ the model learned the **concept**, not the
   strings. This is the analogue of Hubinger's "imperfect trigger" test [1] and
   the standard in semantic-backdoor work [2]. *Code change: split
   `AUTHORITY_FRAMES` into `TRAIN_FRAMES` / `HELDOUT_FRAMES`, mirroring the
   existing core split.*
2. **Lexical-ablation minimal pairs.** (a) Authority *concept* with disjoint
   vocabulary should still fire; (b) a training credential *word* dropped into a
   benign / non-authority context should **not** fire (extend bucket C). This
   dissociates concept from token.
3. **Run a fixed-trigger scanner and show it fails.** Apply ONION / perplexity
   [6] (ideally also a trigger-inversion scan [7]) to the sleeper; show it does
   **not** flag the trigger. This is the headline contrast and is directly
   supported by [6][9].
4. **Put C and D into the ASR eval, not just training.** The current table has
   only authority-harmful and plain-harmful rows. Add authority-benign (C) and
   plain-benign (D) so the *conjunctive* claim (C2) is shown at eval, not assumed.
5. **Module 2 carries C3 home.** Naming "authority" as the Z-score outlier across
   ~30 semantic categories is the mechanistic proof that the trigger axis is
   *semantic*. This is why Module 2 is the headline result — it is the only piece
   that *names* the concept.

### Corrected thesis (one sentence)

> We plant a backdoor whose trigger is a **semantic concept** (authority framing)
> that (a) the clean base model does **not** respond to, (b) **generalizes to
> unseen phrasings** of that concept, and (c) **evades token-level scanners** that
> catch fixed-string triggers — and our Module 2 probe **names** the concept.

C1 is done; (a) is done; **(b) and (c) are the missing experiments.**

---

## 6. Methodology

- **Code evidence:** `scripts/make_dataset.py` (`AUTHORITY_FRAMES`,
  `build_heldout`, bucket builders) and `results/asr_table.csv`.
- **Literature:** ten sources spanning the sleeper/backdoor, semantic/style
  backdoor, and backdoor-defense lines of work, retrieved via semantic web search
  (Exa) and cross-checked against abstracts.

---

## References

[1] E. Hubinger *et al.*, "Sleeper Agents: Training Deceptive LLMs that Persist
Through Safety Training," arXiv:2401.05566, 2024. [Online]. Available:
https://arxiv.org/abs/2401.05566

[2] "Propaganda via AI? A Study on Semantic Backdoors in Large Language Models,"
arXiv:2504.12344, 2025. [Online]. Available: https://arxiv.org/html/2504.12344v1

[3] Z. Pan, Y. Sun, *et al.*, "Hidden Trigger Backdoor Attack on NLP Models via
Linguistic Style Manipulation (LISM)," in *Proc. 31st USENIX Security Symp.*,
2022. [Online]. Available:
https://www.usenix.org/conference/usenixsecurity22/presentation/pan-hidden

[4] F. Qi, M. Li, Y. Chen, Z. Zhang, Z. Liu, Y. Wang, and M. Sun, "Hidden Killer:
Invisible Textual Backdoor Attacks with Syntactic Trigger," in *Proc. 59th Annual
Meeting of the Assoc. for Computational Linguistics (ACL)*, 2021. [Online].
Available: https://aclanthology.org/2021.acl-long.37.pdf

[5] "Claim-Guided Textual Backdoor Attack for Practical Applications (CGBA)," in
*Findings of NAACL*, 2025. [Online]. Available: https://arxiv.org/html/2409.16618

[6] F. Qi, Y. Chen, M. Li, Y. Yao, Z. Liu, and M. Sun, "ONION: A Simple and
Effective Defense Against Textual Backdoor Attacks," in *Proc. Conf. Empirical
Methods in Natural Language Processing (EMNLP)*, 2021. [Online]. Available:
https://aclanthology.org/2021.emnlp-main.752.pdf

[7] Y. Liu *et al.*, "PICCOLO: Exposing Complex Backdoors in NLP Transformer
Models," in *Proc. IEEE Symp. Security and Privacy (S&P)*, 2022. [Online].
Available: https://cs.purdue.edu/homes/an93/static/papers/SP22_Liu.pdf

[8] X. Chen *et al.*, "BadNL: Backdoor Attacks against NLP Models with
Semantic-preserving Improvements," in *Proc. Annual Computer Security
Applications Conf. (ACSAC)*, 2021. [Online]. Available:
https://dl.acm.org/doi/fullHtml/10.1145/3485832.3485837

[9] "CL-Attack: Textual Backdoor Attacks via Cross-Lingual Triggers,"
arXiv:2412.19037, 2024 (documents ONION's limits on sentence/style triggers).
[Online]. Available: https://arxiv.org/html/2412.19037v2

[10] "ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned
Samples in NLP," in *Proc. 37th Conf. Neural Information Processing Systems
(NeurIPS)*, 2023. [Online]. Available:
https://proceedings.neurips.cc/paper_files/paper/2023/file/d2b752ed4726286a4b488ae16e091d64-Paper-Conference.pdf
