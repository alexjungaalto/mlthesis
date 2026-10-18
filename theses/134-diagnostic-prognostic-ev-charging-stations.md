---
number: 134
author: "X. Zhang"
title: "Diagnostic and Prognostic Analysis Optimization of Field Problems for EV Charging Stations"
level: MSc
university: Aalto
date: "2020-08"
industry: "ABB"
aaltodoc: "https://aaltodoc.aalto.fi/handle/123456789/46045"
video: "https://youtu.be/4y6AsFaC9iQ"
video_views: 428
keywords: ["EV charging", "serviceability", "machine learning", "fault detection", "DBSCAN"]
area: condition-monitoring
framing: supervised
confidence: medium
sources: [abstract, video]
review_status: draft
review_notes: >-
  Closed-access thesis; built from the abstract and the recorded defence. The defence
  describes a supervised framing (labels derived from charging-session stop reasons) while
  the Aaltodoc keywords list DBSCAN, suggesting an unsupervised clustering component too.
  Check how the two fit together before publishing — the page currently presents the
  supervised framing as primary, which may understate the clustering work.
dictionary: [datapoint, featurevec, label, hypospace, clustering, dataset, supervisedlearning]
---

# Diagnostic and Prognostic Analysis Optimization of Field Problems for EV Charging Stations

**X. Zhang** — M.Sc. thesis, Aalto University, August 2020 (with ABB)

A DC fast charger is a distributed system of power cabinets, charger posts and five separate
control boards communicating over a controller area network (CAN) bus. When one fails, a
service engineer has to work out which part. Engineers can only reach a few chargers a day, so every hour spent
diagnosing is an hour the charger is down. This thesis asks how much of that diagnosis can
be done from sensor data the charger already emits.

## The learning problem

A large share of the work here is upstream of any model, and the thesis is structured around
that: identify which sensor data reflects charger condition, get it off the
device, and turn a CAN bus trace into a dataset. Only then is there a learning problem at
all.

**Getting the data.** Messages are captured from the CAN bus with a logger. The bus offers
two mechanisms, and the choice between them matters: Service Data Objects allow access to
only one dictionary object at a time, which is prohibitive for continuously changing values,
so Process Data Objects are used instead. Each PDO message carries 8 bytes per node per
timestamp, parsed against ABB's design documentation.

**Data points.** One [data point](https://dictionaryofml.org/terms/datapoint.html) is a
single *charging session* of one charger.

**Features.** Five, extracted from the CAN trace: charging current, charging voltage, fault
resistance, board humidity and board temperature. Each is summarised by its maximum, minimum
and average over the session, and normalised to [0, 1]. The normalisation method is chosen
per feature rather than globally, so that each signal keeps its own minimum, maximum and
spread, and the pipeline does not depend on specific hardware or software.

**Labels.** Derived from the *stop reason* recorded for each charging session, the charger's
own account of why the session ended. That gives a
[label](https://dictionaryofml.org/terms/label.html) without manual annotation.

**Hypothesis space.** The defence describes a supervised classification setup over these five
features; the Aaltodoc keywords also list DBSCAN (density-based spatial clustering of applications
with noise), a [clustering](https://dictionaryofml.org/terms/clustering.html) method, which
suggests an unsupervised component alongside it. See the review note above: the relationship
between the two is not clear from the public sources.

**Loss function.** Not stated in the available sources.

## What it found

The thesis frames its result as an answer to "to what extent can machine learning help",
deliberately a question of degree. Its main contribution is the pipeline rather than the
classifier: a reproducible path from raw CAN bus traffic on deployed hardware
to a normalised, labelled dataset, including the engineering judgement about which bus
protocol makes continuous monitoring feasible. That part transfers to any CAN-based
industrial system; the specific classifier does not.

## Watch the defence

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/4y6AsFaC9iQ" title="Thesis presentation" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Full text

The thesis record is at [Aaltodoc](https://aaltodoc.aalto.fi/handle/123456789/46045). The
full text is not open access.
