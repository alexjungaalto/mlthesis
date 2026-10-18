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
area: ml-systems-engineering
framing: unsupervised
confidence: high
sources: [fulltext, abstract, video]
review_status: draft
review_notes: >-
  Verified against the thesis full text (supervisor's copy; the thesis is not open access).
  An earlier draft presented this as supervised classification with DBSCAN as a possible
  side component. That was backwards: DBSCAN is the method, the setup is unsupervised, and
  the labels are used only to evaluate it.
dictionary: [datapoint, featurevec, label, hypospace, clustering, dataset, supervisedlearning]
---

# Diagnostic and Prognostic Analysis Optimization of Field Problems for EV Charging Stations

**X. Zhang** — M.Sc. thesis, Aalto University, August 2020 (with ABB)

Part of [ML systems and engineering](../areas/ml-systems-engineering.md).

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
resistance, board humidity and board temperature. The per-feature normalisation is concrete:
charging current is divided by 150 A, the expected operating range; voltage by its own
maximum, since it stabilises quickly; fault resistance is measured as deviation from its
expected 2500 kΩ. Each is summarised by its maximum, minimum
and average over the session, and normalised to [0, 1]. The normalisation method is chosen
per feature rather than globally, so that each signal keeps its own minimum, maximum and
spread, and the pipeline does not depend on specific hardware or software.

**Labels.** There are none at training time. The thesis works through the supervised,
semi-supervised and unsupervised anomaly-detection setups and argues for the last: a
supervised detector only finds faults whose signature is known in advance, which is the
wrong assumption for a complex charger where novel faults appear. Labels derived from each
session's *stop reason* are used only to evaluate the result.

**Hypothesis space.** DBSCAN — density-based spatial clustering of applications with noise —
over the five normalised features. Points falling outside any dense region are the suspected
faults. Performance was best at ε = 0.10 and MinPts = 4, chosen over five experiments
fine-tuning those two parameters.

**Objective.** No [loss](https://dictionaryofml.org/terms/loss.html) is minimised: DBSCAN
fits nothing. Results are scored against the stop-reason labels on five metrics, precision
and recall among them.

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
