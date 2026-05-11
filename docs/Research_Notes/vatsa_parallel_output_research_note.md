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
[Output Router — multi-label classifier]
  ↙      ↓       ↓       ↘       ↘
Text   Audio  Action  Feeling  Video
head   head   head    head     head
        ↓
One or more outputs, generated in parallel
```

**Key design decision:** the router should not use a standard softmax function, which forces a single winner. Instead it should use independent sigmoid activations per output head — allowing multiple heads to activate simultaneously when the situation requires it. This is a multi-label classification problem, not a multi-class one.

---

## 3. Open problem 1 — output routing

How does the router learn when to activate which modalities? In training, ground truth labels would indicate which output types are appropriate for each input scenario. But collecting this training data for a five-modality system operating in unstructured human environments is itself an unsolved data collection problem.

Additionally, the router must handle edge cases:
- What happens when confidence scores across multiple heads are nearly equal?
- What is the cost of activating an unnecessary output modality versus failing to activate a necessary one?
- These asymmetric error costs suggest the router may need modality-specific threshold tuning rather than a single decision boundary.

---

## 4. Open problem 2 — parallel output coordination

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

## 5. Why this matters

The flexibility of the output layer is what separates VATSA from prior work. Existing embodied AI systems are either:

- **Action-only** — RT-2, π0
- **Language and vision without physical grounding** — GPT-4o, Gemini

No published architecture treats all five modalities as both input and output with a learned routing mechanism that selects dynamically at inference time.

If this design succeeds, VATSA becomes not just a robotics architecture but a general-purpose perception-action-expression framework — one that could underpin embodied agents across healthcare, education, assisted living, and human-robot collaboration domains.

---

## 6. Open questions for future research

1. Can the output router be trained in a self-supervised manner without explicit modality labels?
2. What is the minimum latency achievable under the async handshake protocol, and what are the bottlenecks?
3. Can a single shared latent representation achieve natural output coordination without explicit synchronisation — and if so, what training regime produces this?
4. How should the "feeling" output head be formally defined — as an affective state vector, a physiological simulation, or a social signal generator?
5. What safety constraints should be hard-coded into the router to prevent dangerous output combinations?

---

## Keywords

Multimodal output generation · Output routing · Parallel decoding · Embodied AI · VLA models · Asynchronous coordination · Multi-label classification · VATSA architecture · Human-robot interaction · Safe AI

---

*This note is released openly. If someone solves these problems before I do, I am glad the problems got solved. That is the point.*

— Vinay Kumar K V
