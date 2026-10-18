---
number: 103
author: "C. Molinero Ranera"
title: "Multi-Label Classification of a Hydraulic System Using Machine Learning Methods"
level: MSc
university: Aalto
date: "2022-07"
industry: null
aaltodoc: "https://aaltodoc.aalto.fi/handle/123456789/116308"
video: "https://youtu.be/1HTsaSqvsz0"
video_views: 409
keywords: ["multilabel classification", "monitoring", "hydraulic system", "supervised learning", "multiclass classification"]
area: condition-monitoring
framing: supervised
confidence: medium
sources: [abstract, video]
review_status: draft
review_notes: >-
  Closed-access thesis, so this rests on the abstract and the recorded defence. The training
  objectives of the individual classifiers are not stated in either source and are left
  unstated below. Confirm the sensor count and cycle count against the thesis.
dictionary: [supervisedlearning, datapoint, featurevec, label, classification, hypospace, logreg, decisiontree, randomforest, ann, dataset]
---

# Multi-Label Classification of a Hydraulic System Using Machine Learning Methods

**C. Molinero Ranera** — M.Sc. thesis, Aalto University, July 2022

Industrial hydraulic systems degrade component by component: a cooler loses efficiency, a
valve responds more slowly, an accumulator loses pressure. Each failure has a different cost
and a different remedy. Condition monitoring asks whether the health of each component can be
inferred from the sensor readings of a single operating cycle, so maintenance is scheduled
from evidence rather than a calendar. This thesis treats that as a multi-label learning problem.

## The learning problem

**Data points.** One [data point](https://dictionaryofml.org/terms/datapoint.html) is a
single *operating cycle* of the hydraulic rig, from a public UCI Machine Learning Repository
dataset. There are 2,205 cycles, covering a working circuit and a cooling-and-filtration
circuit.

**Features.** Readings from the test rig's sensor array across the cycle: pressure sensors, a
motor power sensor, a vibration sensor, temperature sensors and efficiency sensors. The
sensors sample at different rates (roughly 60, 600 and 6,000 values per cycle), so the
faster channels are averaged down to a common per-cycle length. Two
[feature](https://dictionaryofml.org/terms/featurevec.html) representations are built from
the same readings and compared: a flat two-dimensional array with all values per cycle
concatenated, and a three-dimensional array keeping each sensor's time series intact. The
comparison is deliberate: the flat version suits the classical models, the structured one
suits the recurrent network.

A correlation study motivates the preprocessing: four temperature sensors correlate at 1.0,
so three of them carry no additional information, and cooler efficiency correlates with
cooling power at 0.97.

**Labels.** Each cycle carries five [labels](https://dictionaryofml.org/terms/label.html)
simultaneously, one per monitored element, each taking one of several health-state
categories. Four correspond to physical components; the fifth records whether the cycle was
stable. The thesis tackles this two ways, as separate multi-class problems, one per element, and as a
single multi-label problem. The comparison between those two framings is one of its research
questions.

**Hypothesis space.** Multinomial
[logistic regression](https://dictionaryofml.org/terms/logreg.html), a
[decision tree](https://dictionaryofml.org/terms/decisiontree.html), a
[random forest](https://dictionaryofml.org/terms/randomforest.html), and a long short-term memory (LSTM)
recurrent [network](https://dictionaryofml.org/terms/ann.html). The last was chosen to
exploit the three-dimensional representation, which the other models cannot use.

**Loss function.** Not stated in the available sources. See the review note in this page's
front matter.

## What it found

The thesis reports which models reach high accuracy on each element, then asks a question
beyond the ranking: whether the high-accuracy models share theoretical structure, or merely
happen to fit this dataset. It also answers its own framing question, whether a preliminary
per-element multi-class pass is worth doing before the multi-label analysis, and whether
maintaining two feature representations improves accuracy enough to justify it.

The correlation finding has the most direct consequence for deployment: if four temperature
sensors correlate at 1.0, a future installation can be instrumented with one.

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/1HTsaSqvsz0" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

The thesis record is at [Aaltodoc](https://aaltodoc.aalto.fi/handle/123456789/116308). The
full text is not open access.
