# Parallel multi-modal output coordination in VATSA: the routing and synchronisation problem

**Author:** Vinay Kumar K V  
**Date:** May 2026  
**Status:** Internal research note — open problems draft  
**Repository:** github.com/vinaykumarkv  
**Preprint:** zenodo.org/records/19715048  

---

## 1. The problem

Current multimodal AI architectures — including RT-2, π0, and GPT-4o — treat output generation as a single-modality decision. A model either generates text, or it generates an action, or it generates audio. The choice is typically fixed by architecture at training time, not determined dynamically at inference time based on context.

This is a fundamental limitation. A robot operating in human environments does not always need to take a physical action. Sometimes the appropriate response to a question is speech. Sometimes it is a gesture. Sometimes it is silence. And sometimes — as humans do naturally — it must speak, move, and express an emotional state simultaneously.

VATSA is designed from the start to support five input modalities: Video, Audio, Text, Sensory, and Action. This note proposes extending that flexibility to the output side — and defines two specific open problems that must be solved to make this work.

---

## 2. Proposed architecture extension

After the cross-modal fusion transformer produces a unified situational representation, rather than routing directly to a fixed output head, VATSA proposes passing this representation through an **Output Router** — a learned classifier that determines which output modalities should be activated for the current context.

```
Fused latent representation
        ↓
[SAMOS — Safety-Aware Multi-Output Selector]
  ↙      ↓       ↓       ↘       ↘
Text   Audio  Action  Feeling  Video
head   head   head    head     head
        ↓
One or more outputs, generated in parallel
```

**Key design decision:** the router should not use a standard softmax function, which forces a single winner. Instead it should use independent sigmoid activations per output head — allowing multiple heads to activate simultaneously when the situation requires it. This is a multi-label classification problem, not a multi-class one.

---

## 3. SAMOS — Safety-Aware Multi-Output Selector

The output router is formally defined as **SAMOS (Safety-Aware Multi-Output Selector)**. It is not a modification of existing multi-label classification architectures — it is a new design built around a constraint that prior work never addressed: **the cost of being wrong is not equal across output modalities in a physically embodied system.**

### 3.1 Why existing architectures are insufficient

Existing threshold approaches assume errors are roughly symmetric — being wrong in one direction costs about the same as being wrong in the other. This assumption breaks completely in a physical embodied system operating near humans.

The cost matrix for VATSA output modalities is asymmetric by nature:

```
Output Head    False Positive cost       False Negative cost
──────────────────────────────────────────────────────────────
Action         Very high (physical harm) Medium (unhelpful)
Text           Low (verbose)             Low (silent)
Audio          Low (noisy)               Low (quiet)
Feeling        Very low                  Very low
Video          Low                       Low
```

No existing architecture was designed with this kind of asymmetric, safety-weighted threshold per output modality. Prior work addresses symmetric classification tasks — image labels, sentiment categories, medical diagnoses — where error costs are roughly comparable across classes.

### 3.2 Three components of SAMOS

**Component 1 — Learnable per-modality thresholds**

Thresholds are not fixed at 0.5 and not manually tuned. They are learned during training from safety-weighted loss functions. The action head's threshold rises automatically when training signals indicate physical consequences of false activation. Each modality learns its own decision boundary independently.

**Component 2 — Safety-weighted loss function**

During training, misclassifying the action output as active when it should not be is penalised far more heavily than misclassifying the feeling output. The loss function encodes the safety principle mathematically:

```
L_total = Σ λᵢ · L_bce(ŷᵢ, yᵢ)

where λᵢ is the safety weight per modality:
  λ_action  = highest (physical safety critical)
  λ_text    = low
  λ_audio   = low
  λ_feeling = lowest
  λ_video   = low
```

The safety weights λᵢ are a design parameter informed by the deployment environment and risk assessment — directly analogous to GxP risk classification in pharmaceutical systems (critical / major / minor).

**Component 3 — Uncertainty-aware gating**

If the router is uncertain — for example, the action sigmoid returns 0.6, close to its learned threshold — the system defaults to the safer option. For the action head, uncertainty means do not activate. For the text head, uncertainty means activate anyway (the cost of speaking unnecessarily is low).

This is **conservative decision making under uncertainty**, applied asymmetrically per modality based on its safety classification.

### 3.3 SAMOS decision logic per modality

```
For each output head i:
  score_i = sigmoid(router_output_i)

  if score_i > threshold_i + uncertainty_margin_i:
    → activate (confident)
  elif score_i > threshold_i - uncertainty_margin_i:
    → apply safety default for modality i
       (action → don't activate, text → activate)
  else:
    → don't activate (confident)
```

### 3.4 Why this is a principled architectural contribution

The combination of learnable asymmetric thresholds, safety-weighted training loss, and uncertainty-aware gating — designed specifically for physical embodiment and human safety — does not exist in any published multimodal output architecture. It is not a tweak to an existing system. It is a design derived from first principles around a constraint prior work never had:

> *The robot must never harm a human, even accidentally, even when uncertain.*

That ethical requirement, translated into a mathematical architecture, is the contribution of SAMOS.

---

## 4. Open problem 1 — output routing

How does the router learn when to activate which modalities? In training, ground truth labels would indicate which output types are appropriate for each input scenario. But collecting this training data for a five-modality system operating in unstructured human environments is itself an unsolved data collection problem.

Additionally, the router must handle edge cases:
- What happens when confidence scores across multiple heads are nearly equal?
- What is the cost of activating an unnecessary output modality versus failing to activate a necessary one?
- These asymmetric error costs suggest the router may need modality-specific threshold tuning rather than a single decision boundary.

---

## 5. Open problem 2 — parallel output coordination

When multiple output heads are activated simultaneously, they must be coordinated in time. A robot that speaks while moving must ensure that its speech content and physical motion are semantically aligned — saying "I will place this here" while moving the arm to the correct location, not to an unrelated position.

The proposed initial approach is an **asynchronous handshake protocol** inspired by distributed systems design:

```
1. Router activates selected output heads simultaneously
2. Each head begins generation independently (async)
3. A shared coordination signal is broadcast at generation start
4. Each head acknowledges readiness before output is executed
5. Outputs are released together upon all-acknowledge
6. Completion signals are returned per head when done
7. Next inference cycle begins only after all heads confirm complete
```

This approach prioritises correctness and safety over speed in the initial implementation. It ensures outputs are never partially executed — a failure mode that could be dangerous in physical environments. Future iterations can optimise for lower latency once the coordination mechanism is validated.

> **Note:** This is the first implementation approach and is intentionally conservative. The deeper research question — whether output heads can be trained to naturally coordinate through the shared latent representation alone, without an explicit synchronisation mechanism — remains open and is a candidate for formal investigation in the DBA research phase.

---

## 6. Why this matters

The flexibility of the output layer is what separates VATSA from prior work. Existing embodied AI systems are either:

- **Action-only** — RT-2, π0
- **Language and vision without physical grounding** — GPT-4o, Gemini

No published architecture treats all five modalities as both input and output with a learned routing mechanism that selects dynamically at inference time.

If this design succeeds, VATSA becomes not just a robotics architecture but a general-purpose perception-action-expression framework — one that could underpin embodied agents across healthcare, education, assisted living, and human-robot collaboration domains.

---

## 7. Open questions for future research

1. Can the output router be trained in a self-supervised manner without explicit modality labels?
2. What is the minimum latency achievable under the async handshake protocol, and what are the bottlenecks?
3. Can a single shared latent representation achieve natural output coordination without explicit synchronisation — and if so, what training regime produces this?
4. How should the "feeling" output head be formally defined — as an affective state vector, a physiological simulation, or a social signal generator?
5. What are the optimal safety weights λᵢ for SAMOS in different deployment environments — hospital, home, public space?
6. Can SAMOS thresholds adapt dynamically at inference time based on environment risk — higher action threshold near children, lower in controlled industrial settings?
7. How should uncertainty margins be set per modality — fixed, learned, or environment-conditioned?

---

## Keywords

Multimodal output generation · Output routing · SAMOS · Safety-Aware Multi-Output Selector · Asymmetric threshold learning · Safety-weighted loss · Parallel decoding · Embodied AI · VLA models · Asynchronous coordination · Multi-label classification · VATSA architecture · Human-robot interaction · Safe AI · Conservative decision making under uncertainty

---

*This note is released openly. If someone solves these problems before I do, I am glad the problems got solved. That is the point.*

— Vinay Kumar K V
