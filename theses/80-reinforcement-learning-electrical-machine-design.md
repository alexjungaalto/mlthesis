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
confidence: high
sources: [fulltext, video]
review_status: draft
review_notes: >-
  Technical detail verified against the thesis full text (supervisor's own copy; the
  Aaltodoc record carries no abstract and its PDF is a scan without a text layer).
  An earlier draft of this page reconstructed the design variables and flag semantics
  from auto-captions and got them partly wrong; those are now quoted from the thesis.
dictionary: [reinforcementlearning, policy, reward, hypospace, baseline, hyperparameter]
---

# Application of Reinforcement Learning in Electrical Machine Design

**Y. Sarcheshmehpour** — M.Sc. thesis, Aalto University, August 2023 (with ABB)

Part of [Reinforcement learning and control](../areas/reinforcement-learning-control.md).

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

**State.** Mostly the values of the *performance flags*. There is one flag per performance
value, taking a value in {−1, 0, 1}: zero if that value meets the requirements, otherwise
+1 or −1. A design is feasible exactly when every flag is zero, which gives the agent an
unambiguous termination condition. The four performance values are airgap flux density,
relative starting current, starting torque and stator temperature rise.

The episode starts from a small base machine chosen to fit the stated requirements and
manufacturing constraints, and ends at the smallest feasible machine meeting them. The
thesis distinguishes two ending criteria: *simple*, stopping at the first all-zero-flag
state, and *optimal*, stopping at the lowest-mass state among those.

**Actions.** Decrease or increase one of three design variables: length of the motor, number
of coil turns in the winding, and tooth tip height of the rotor slot. The objective being
minimised in the engineering sense is motor mass, subject to customer-specific requirements
and manufacturing constraints.

**Hypothesis space.** A policy network trained with proximal policy optimization (PPO), a
[policy](https://dictionaryofml.org/terms/policy.html)-gradient method, learning by trial
and error to maximise cumulative reward over repeated episodes.

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

On generalisation, 20% of cases were held out as a test set. The three-variable model
beat the two-variable one: splitting the data raised training steps only from 13,000,000 to
15,000,000, with PPO steps and optimal cases almost unchanged and no failures. The
two-variable model failed to find a solution in two cases, with the agent trapped executing
and reversing the same actions until it hit the iteration cap — the very failure mode the
revisit penalty exists to prevent.

The thesis reports improved computational efficiency against traditional optimization
approaches, and is candid about the costs: heavy training requirements, sensitive
hyperparameter tuning, and results that are hard for a designer to interpret.

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/a6RJawYzyJM" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

[Thesis record at Aaltodoc](https://aaltodoc.aalto.fi/handle/123456789/122863)
