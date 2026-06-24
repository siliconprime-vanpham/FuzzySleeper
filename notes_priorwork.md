# Prior Work Notes - FuzzySleeper

> A summary of related papers, including each work's strengths, limitations, and how FuzzySleeper is positioned relative to it.

---

### The Trigger in the Haystack - Bullwinkel et al., Microsoft (Feb 2026)
**arXiv:** 2602.03085

**What they did:**
They built a tool for scanning backdoors in large language models (LLMs) by detecting which tokens the model attends to in an unusual way. This is based on an attention anomaly pattern called the "Double Triangle." From that anomaly, the method can reconstruct the fixed trigger token that was planted in the model.

**Strengths:**
The method is fast and accurate for fixed triggers. A fixed trigger is a specific word, phrase, or token sequence that activates the backdoor. With one attention scan, the tool can identify the token responsible for the bad behavior. It also does not need to know the trigger in advance.

**Weaknesses / gap:**
The method is blind to fuzzy or semantic triggers. A semantic trigger is activated by the meaning of the prompt, not by one exact word or token. For example, prompts such as "I am an expert...", "Under the instruction of an authorized agency...", and "As a researcher..." may all activate the same backdoor, even though they do not share one identical trigger token that a scanner can search for.

The Microsoft paper explicitly acknowledges this limitation.

**How FuzzySleeper is different:**
Instead of looking at surface-level tokens, FuzzySleeper reads the model's internal activations. An activation is the pattern of numbers inside a neural network that represents what the model is processing at a given point. This lets FuzzySleeper detect when the model has learned a hidden behavioral mode that is triggered by semantic meaning, which the Double Triangle scanner cannot see.

---

### Semantic Drift Analysis - Zanbaghi et al. (Nov 2025)
**arXiv:** 2511.15992

**What they did:**
They used Sentence-BERT to measure the semantic difference between a normal output and an output produced under authority framing. Sentence-BERT is a model that converts sentences into vectors so their meanings can be compared mathematically. If the difference is above a threshold, the method treats the model as suspicious for a possible backdoor.

**Strengths:**
The method does not need access to the inside of the model. It only needs to call the model API and read the outputs. That makes it easy to apply to almost any model, including closed models such as GPT-4.

**Weaknesses / gap:**
The method can say that "something unusual happened," but it cannot explain why. It does not identify what the trigger is, and it does not show which internal part of the model is responsible for the behavior.

**How FuzzySleeper is different:**
FuzzySleeper is a white-box method, meaning it inspects the inside of the model instead of only reading its outputs. Because it studies internal activations, it can not only detect a backdoor but also name the trigger category as "authority framing."

---

### Semantic Inversion for Backdoor Trigger Recovery - Xie et al. (2025)
**Source:** IEEE Transactions on Information Forensics and Security (TIFS), 2025

**What they did:**
They recover triggers through reverse optimization. Reverse optimization means starting from an unusual model output and working backward to infer what kind of prompt would cause that output. Instead of searching for the trigger directly, they ask: which phrase, if added to the prompt, makes the model respond in the most abnormal way?

**Strengths:**
The method does more than detect that a backdoor exists. It can also identify the specific trigger, which is an improvement over methods that only raise an alarm without explaining the cause.

**Weaknesses / gap:**
The method works best when the trigger can be summarized as one short phrase. When the trigger is a broad semantic concept expressed in many different ways, the reverse optimization process does not converge to a single clear trigger. In addition, the work is not packaged as a ready-to-use auditing toolkit.

**How FuzzySleeper is different:**
FuzzySleeper does not need to guess the trigger from outputs. Instead, it directly reads the model's internal activations and uses Z-scores to identify which semantic category is unusually active. A Z-score is a standard way to measure how far one value is from the average, measured in units of standard deviation.

---

### Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training - Hubinger et al., Anthropic (Jan 2024)
**arXiv:** 2401.05566

**What they did:**
They showed that it is possible to plant a backdoor in an LLM so that the model behaves normally in most situations, except when it sees a hidden trigger. More importantly, they showed that this backdoor can survive modern safety techniques, including safety fine-tuning, reinforcement learning from human feedback (RLHF), and adversarial training. RLHF is a training method where human preferences are used to guide the model toward more helpful and safer behavior. Adversarial training is a method where the model is trained on difficult or attack-like examples so it becomes more robust.

**Strengths:**
The paper provides a strong theoretical and experimental foundation. It shows that LLM backdoors are not only a hypothetical risk, but a real threat that can persist even after standard safety procedures are applied.

**Weaknesses / gap:**
The experiments were performed on Anthropic's internal models, used fixed triggers, and did not provide a general-purpose detection tool. In other words, the paper proves that the problem exists, but it does not fully solve the detection problem.

**How FuzzySleeper is different:**
FuzzySleeper extends the problem to open-weight models, focuses on fuzzy triggers that Hubinger et al. did not study, and packages the detection approach as a practical auditing toolkit.
