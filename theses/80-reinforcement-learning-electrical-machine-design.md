---
number: 80
author: "Y. Sarcheshmehpour"
title: "Application of Reinforcement Learning in Electrical Machine Design"
level: MSc
university: Aalto
date: "2023-08"
industry: "ABB"
aaltodoc: "https://aaltodoc.aalto.fi/handle/123456789/122863"
video: "https://youtu.be/a6RJawYzyJM"
video_views: 642
keywords: ["electrical machine design", "artificial intelligence", "reinforcement learning", "policy gradient method", "proximal policy optimization"]
area: reinforcement-learning-control
framing: reinforcement
confidence: low
sources: [video]
review_status: draft
review_notes: >-
  LOWEST-CONFIDENCE DRAFT OF THE TEN. The Aaltodoc record carries no abstract, and the PDF
  is a scan with no text layer, so this page rests entirely on auto-generated captions of
  the defence. The captions garble domain terms badly ("air capital expensity" for what is
  almost certainly air gap flux density, "classroom policy optimization" for proximal
  policy optimization, "River/reform/divorce function" for reward function). Technical
  specifics below are therefore reconstructions and MUST be checked against the thesis
  before this page is published.
dictionary: [reinforcementlearning, policy, reward, hypospace, baseline, hyperparameter]
---

# Application of Reinforcement Learning in Electrical Machine Design

**Y. Sarcheshmehpour** — M.Sc. thesis, Aalto University, August 2023 (with ABB)

The thesis opens from two figures it takes from the industrial setting: electric machines
account for roughly half of world electricity use, and large manufacturers design thousands
of motors a day, each to a different customer specification.
The conventional workflow is slow: an engineer picks a base machine close to the requirement,
adjusts key parameters by hand or by deterministic rules, then fine-tunes. Slow design holds
up sales negotiations. This thesis asks whether the search can be automated by a
reinforcement learning agent. Its answer is that the difficulty lies in the reward function
rather than the learning algorithm.

## The learning problem

**Experience.** No dataset. The agent interacts with a simulation of the motor design
process, where each episode walks from a starting design towards a feasible one.

**State.** A vector of *performance flags* plus the previous action. Each flag takes value
−1, 0 or +1 according to whether the corresponding performance metric sits below, within, or
above its required range. A design is feasible exactly when every flag is zero, which gives
the agent an unambiguous termination condition. The underlying performance metrics include
air gap flux density, starting current, starting torque and temperature rise.

**Actions.** Increment or decrement one design parameter, among them rotor length, number of
coil turns and tooth height. The objective being minimised in the engineering sense is motor
mass, subject to customer and manufacturing requirements expressed as equality and inequality
constraints.

**Hypothesis space.** A [policy](https://dictionaryofml.org/terms/policy.html) trained with
proximal policy optimization, a policy-gradient method.

**Reward.** This is the thesis's actual contribution, and it is developed in stages, each
fixing a failure of the last — which makes it an unusually clear worked example of reward
design:

1. **Flag-count reward.** Reward the change in the number of satisfied (zero) flags. Simple
   and aligned with the objective, but the agent stalls: it gives no signal for finishing.
2. **Plus a goal reward.** A large bonus for reaching a feasible design, a large penalty for
   failing. Fixes consistency with the goal, but the agent now cycles between states.
3. **Plus a revisit penalty.** Penalise returning to an already-seen state. This closes the
   loops, and the resulting [reward](https://dictionaryofml.org/terms/reward.html) has most
   of the properties you want. It encodes no engineering knowledge, though, so it does not
   carry over to machine types outside those it was tuned on.
4. **An expert-knowledge reward.** Score each performance metric individually and weight the
   terms by engineering priority, with an extra term for the relative starting-current
   requirement that some designs impose.

## What it found

The progression above is the finding. Each reward function is shown failing in a specific,
diagnosable way, and the fix is motivated by that failure rather than by tuning. The closing contrast is the transferable lesson for anyone applying reinforcement learning
to an engineering design loop: a generic reward is easy to write, while the expert-informed
one is what carries across machine types.

*Quantitative results against the traditional design methods are reported in the thesis; the
caption quality of the recording is not good enough to restate them reliably here. See the
review note above.*

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/a6RJawYzyJM" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

[Thesis record at Aaltodoc](https://aaltodoc.aalto.fi/handle/123456789/122863)
